"""Fixed-window augmented EKF-SLAM coefficient fitting with frozen decisions."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from benchmarks.milestone4b_smooth import gradient_validation, least_squares_approximation, localization_metrics, write_json
from benchmarks.milestone4c_reporting import completion_checks, write_report
from benchmarks.slam_window_plots import plot_experiment
from estimation.coefficient_optimization import CheckedScipyObjective, disturbance_metrics, gradient_norms, optimize_coefficients
from estimation.fixed_window_objective import InvalidCandidateError, checked_evaluation
from estimation.propagation import rk4_stage_times
from estimation.slam import INITIALIZED, UPDATED, SlamState, create_state, filter_slam
from estimation.slam_window_objective import make_slam_window_objective
from simulation.slam_window import sensor_settings, simulate_slam_window
from simulation.smooth_unicycle import smooth_disturbance
from splines import UniformBasis, disturbance_from_coefficients


def configuration(kind='constant'):
    cfg = json.loads(Path('artifacts/milestone4b/results.json').read_text())['configuration'].copy()
    cfg.update(initialization_seconds=2., total_duration_seconds=7., capacity=16,
               disturbance_kind=kind, optimization_observations='strictly after t0=2.0 s',
               arrival_a='complete posterior from the disturbance-free pre-window EKF-SLAM',
               initial_coefficient_vector=[0.]*14, coefficient_bounds=[-.5, .5])
    cfg['sensor'] = sensor_settings('regression')
    if kind == 'smooth':
        del cfg['true_disturbance_v_omega']
        cfg['true_disturbance_functions'] = ['0.15*sin(2*pi*sigma/5)','0.05*cos(2*pi*sigma/5)']
    return cfg


def prepare_a(cfg, data):
    """Run preserved Baseline B through the t0 measurement; freeze its posterior."""
    split = round(cfg['initialization_seconds']/cfg['integration_step_seconds'])
    adapter = UniformBasis(cfg['degree'], cfg['segments'])
    command = lambda t: jnp.asarray(cfg['command_v_omega'])
    ids = np.broadcast_to(np.arange(len(data.landmark_map), dtype=np.int64), data.mask.shape).copy()
    prefix_times = data.times[:split+1]
    prefix_basis = adapter.precompute_jax(rk4_stage_times(prefix_times), cfg['initialization_seconds'])
    prefix = jax.jit(lambda: filter_slam(
        create_state(cfg['initial_pose'], cfg['arrival_covariance'], cfg['capacity']),
        prefix_times, command, prefix_basis, jnp.zeros(14),
        cfg['continuous_input_noise_spectral_density_Qc'], data.measurements[:split+1],
        ids[:split+1], data.measurement_covariance, data.mask[:split+1]))()
    jax.block_until_ready(prefix)
    arrival = SlamState(*(value[-1].copy() for value in prefix.states))
    window = data._replace(times=data.times[split:], truth=data.truth[split:],
        measurements=data.measurements[split:], landmarks_at_times=data.landmarks_at_times[split:],
        mask=data.mask[split:].copy())
    window.mask[0] = False  # t0 observations already consumed by the posterior.
    basis = adapter.precompute_jax(rk4_stage_times(window.times)-window.times[0], cfg['window_length_seconds'])
    return arrival, window, ids[split:], adapter, basis, prefix


def evaluator_and_baseline(cfg, arrival, data, ids, basis):
    command = lambda t: jnp.asarray(cfg['command_v_omega'])
    evaluator = make_slam_window_objective(arrival, data.times, command, basis,
        cfg['continuous_input_noise_spectral_density_Qc'], data.measurements, ids,
        data.measurement_covariance, data.mask)
    baseline = jax.jit(lambda: filter_slam(arrival, data.times, command, basis, jnp.zeros(14),
        cfg['continuous_input_noise_spectral_density_Qc'], data.measurements, ids,
        data.measurement_covariance, data.mask))()
    jax.block_until_ready(baseline)
    return evaluator, baseline


def prepare_b(cfg):
    """Command-only robot prior at t0: no map-building or window observations.

    Propagate the original robot prior and Qc with zero disturbance. This is a
    separate, observation-free construction, not A's posterior robot marginal.
    The original t=0 pose prior anchors the map gauge in both configurations.
    """
    times = np.linspace(0., cfg['initialization_seconds'],
                        round(cfg['initialization_seconds']/cfg['integration_step_seconds'])+1)
    adapter = UniformBasis(cfg['degree'], cfg['segments'])
    basis = adapter.precompute_jax(rk4_stage_times(times), cfg['initialization_seconds'])
    initial = create_state(cfg['initial_pose'], cfg['arrival_covariance'], cfg['capacity'])
    # An observation-free invocation of the preserved filter uses exactly its
    # propagation equations without importing any measured or true map data.
    result = jax.jit(lambda: filter_slam(initial, times, lambda t:jnp.asarray(cfg['command_v_omega']),
        basis, jnp.zeros(14), cfg['continuous_input_noise_spectral_density_Qc'],
        np.zeros((len(times),1,2)), np.zeros((len(times),1),dtype=np.int64),
        cfg['measurement_covariance'], np.zeros((len(times),1),dtype=bool)))()
    jax.block_until_ready(result)
    return SlamState(*(value[-1].copy() for value in result.states))


def covariance_summary(states):
    minimum, asymmetry, inactive = [], [], []
    for x, P, active, ids in zip(*(np.asarray(value) for value in states)):
        selected = np.r_[np.arange(3), np.flatnonzero(np.repeat(active, 2))+3]
        block = P[np.ix_(selected, selected)]
        if not np.isfinite(x).all() or not np.isfinite(P).all():
            raise ValueError('Nonfinite state/covariance')
        minimum.append(float(np.linalg.eigvalsh((block+block.T)/2).min()))
        asymmetry.append(float(np.max(np.abs(block-block.T))))
        unused = np.setdiff1d(np.arange(len(x)), selected)
        inactive.append(float(max(np.max(np.abs(x[unused]), initial=0.),
                                  np.max(np.abs(P[unused]), initial=0.),
                                  np.max(np.abs(P[:, unused]), initial=0.))))
    summary = {'minimum_active_subspace_eigenvalue': min(minimum),
               'maximum_covariance_asymmetry': max(asymmetry), 'maximum_inactive_storage': max(inactive)}
    if min(minimum) < -1e-10 or max(asymmetry) > 1e-10 or max(inactive) != 0:
        raise ValueError(f'Covariance validation failed: {summary}')
    return summary


def metrics(result, coefficients, dense_basis, dense_truth, data):
    if not bool(result.candidate_valid):
        raise InvalidCandidateError('Invalid SLAM candidate cannot supply metrics')
    localization, _ = localization_metrics(np.asarray(result.states.x)[:, :3], data.truth)
    active, ids = np.asarray(result.final_state.active), np.asarray(result.final_state.truth_ids)
    estimated_map = np.asarray(result.final_state.x)[3:].reshape(-1, 2)[active]
    e_map = estimated_map-data.landmark_map[ids[active]]
    mask = np.asarray(result.valid_mask)
    e, S = np.asarray(result.innovations)[mask], np.asarray(result.innovation_covariances)[mask]
    independent_J = float(np.sum(e*np.linalg.solve(S, e[..., None])[..., 0]))
    independent_logdet = float(np.linalg.slogdet(S)[1].sum())
    return {'objective': float(result.innovation_quadratic),
            'logdet_diagnostic': float(result.logdet_diagnostic),
            'gaussian_diagnostic': float(result.gaussian_diagnostic),
            'objective_numpy_absolute_error': abs(independent_J-float(result.innovation_quadratic)),
            'logdet_numpy_absolute_error': abs(independent_logdet-float(result.logdet_diagnostic)),
            'localization': localization, 'final_landmark_rmse_m': float(np.sqrt(np.mean(np.sum(e_map**2, axis=1)))) if len(e_map) else None,
            'disturbance': disturbance_metrics(dense_basis, coefficients, dense_truth),
            'requested_observations': int(result.requested_count), 'valid_observations': int(result.valid_count),
            'initialization_observations': int(result.initialization_count),
            'correction_observations': int(result.correction_count),
            'active_landmarks_at_arrival': int(np.asarray(result.states.active)[0].sum()),
            'active_landmarks_final': int(active.sum()), 'invalid_observations': int(result.rejected_count),
            'covariance': covariance_summary(result.states),
            'mean_innovation_covariance_diagonal': np.diagonal(S, axis1=1, axis2=2).mean(axis=0).tolist() if len(e) else None,
            'unweighted_residual_sum_squares_range_bearing': np.sum(e**2, axis=0).tolist()}


def fit_configuration(cfg, arrival, data, ids, adapter, basis, output, label):
    evaluator, baseline = evaluator_and_baseline(cfg, arrival, data, ids, basis)
    elapsed = np.linspace(0., cfg['window_length_seconds'], 501)
    dense_basis = adapter.evaluate(elapsed, cfg['window_length_seconds'])
    dense_truth = (np.broadcast_to([.15, -.05], (len(elapsed), 2)) if cfg['disturbance_kind'] == 'constant'
                   else smooth_disturbance(elapsed))
    zero = np.zeros(14)
    true_coefficients = (np.repeat([.15, -.05], 7) if cfg['disturbance_kind'] == 'constant'
                         else least_squares_approximation(dense_basis, dense_truth)[0])
    reference_key = 'true_coefficient_diagnostic' if cfg['disturbance_kind'] == 'constant' else 'least_squares_coefficient_diagnostic'
    bridge = CheckedScipyObjective(evaluator, lambda c: c, zero)
    bridge.reference_check(zero)
    zero_result = checked_evaluation(bridge.compiled_evaluate, zero)
    for left, right in zip(zero_result.states, baseline.states):
        np.testing.assert_allclose(left, right, rtol=0, atol=2e-14)
    record = {'configuration': cfg, 'label': label,
              'arrival': {name: np.asarray(value).tolist() for name, value in zip(arrival._fields, arrival)},
              'dataset_sha256': hashlib.sha256(data.truth.tobytes()+data.measurements.tobytes()+data.mask.tobytes()).hexdigest(),
              'baseline_b': metrics(zero_result, zero, dense_basis, dense_truth, data),
              reference_key: metrics(checked_evaluation(bridge.compiled_evaluate, true_coefficients),
                                                    true_coefficients, dense_basis, dense_truth, data)}
    records = []
    for repeat in range(2):
        fitted = optimize_coefficients(evaluator, zero, coefficients_per_channel=7)
        records.append(fitted)
        print(json.dumps({'experiment': label, 'repeat': repeat, 'status': fitted['status'],
                          'J': fitted['objective_final'], 'iterations': fitted['iterations']}), flush=True)
        if fitted['invalid_trials']:
            write_json(output/f'{label}_failure.json', fitted)
            raise InvalidCandidateError(f'{label}: stopped on invalid candidate; no domain adjustment')
    fitted, repeat = records
    coefficients = np.asarray(fitted['coefficients'])
    optimized = checked_evaluation(bridge.compiled_evaluate, coefficients)
    record.update(optimization=fitted, repeat_optimization=repeat,
        optimized_mhe=metrics(optimized, coefficients, dense_basis, dense_truth, data),
        reproducibility={'coefficient_max_absolute_difference': float(np.max(np.abs(coefficients-repeat['coefficients']))),
                         'objective_absolute_difference': abs(fitted['objective_final']-repeat['objective_final']),
                         'accepted_objective_history_identical': [r['objective'] for r in fitted['history']]
                             == [r['objective'] for r in repeat['history']]})
    record['total_fitting_seconds_including_preparation'] = fitted['timing']['preparation_seconds']+fitted['timing']['optimization_seconds']
    record['gradient_validation'] = gradient_validation(bridge.compiled_evaluate, bridge.compiled_value_gradient,
        {'zero': zero, 'nonconstant': np.linspace(-.04, .06, 14), 'optimized': coefficients})
    for _ in range(10):
        bridge(zero, 'timing')
    timing_rows = [r for r in bridge.evaluation_history if r['phase'] == 'timing']
    record['baseline_timing'] = {**bridge.timings,
        **{field+suffix: float(function([r[field] for r in timing_rows]))
           for field in ('validity_check_seconds', 'value_gradient_seconds')
           for suffix, function in [('_mean', np.mean), ('_p95', lambda a: np.percentile(a, 95))]}}
    record['baseline_gradient_norms'] = gradient_norms(zero, np.asarray(bridge.compiled_value_gradient(zero)[1]), np.tile([-.5,.5],(14,1)))
    record['validation_checks'] = {
        'both_repeats_converged': fitted['success'] and repeat['success'],
        'baseline_matches_preserved_filter': True,
        'objective_matches_numpy': all(record[name]['objective_numpy_absolute_error'] < 2e-10 for name in ('baseline_b','optimized_mhe',reference_key)),
        'all_coefficient_gradients': record['gradient_validation']['passed'],
        'exact_reproducibility': record['reproducibility']['coefficient_max_absolute_difference'] == 0.
                                  and record['reproducibility']['accepted_objective_history_identical'],
        'all_requested_records_valid': int(optimized.valid_count) == int(optimized.requested_count)}
    write_json(output/f'{label}_results.json', record)
    arrays = {'times': data.times, 'truth': data.truth, 'measurements': data.measurements,
        'mask': data.mask, 'observation_ids': ids, 'landmark_truth': data.landmark_map,
        'elapsed': elapsed, 'dense_basis': dense_basis, 'disturbance_truth': dense_truth,
        'optimized_disturbance': np.asarray(disturbance_from_coefficients(dense_basis, coefficients)),
        'optimization_iteration': np.asarray([r['iteration'] for r in fitted['history']]),
        'optimization_objective': np.asarray([r['objective'] for r in fitted['history']])}
    for name, result in [('baseline',zero_result),('optimized',optimized)]:
        arrays.update({name+'_'+field: np.asarray(value) for field, value in zip(result.states._fields, result.states)})
        arrays.update({name+'_statuses': np.asarray(result.statuses), name+'_slots': np.asarray(result.slots),
                       name+'_correction_mask': np.asarray(result.valid_mask),
                       name+'_initialization_mask': np.asarray(result.initialization_mask),
                       name+'_innovations': np.asarray(result.innovations),
                       name+'_innovation_covariances': np.asarray(result.innovation_covariances)})
    np.savez_compressed(output/f'{label}_histories.npz', **arrays)
    write_json(output/f'{label}_optimization_history.json',
        {key:fitted[key] for key in ('history','evaluation_history','invalid_trials')})
    return record


def run_a(output, kind='constant'):
    jax.config.update('jax_enable_x64', True)
    output.mkdir(parents=True, exist_ok=True)
    cfg = configuration(kind)
    data, truth_check = simulate_slam_window(cfg)
    arrival, window, ids, adapter, basis, prefix = prepare_a(cfg, data)
    covariance_summary(prefix.states)
    np.savez_compressed(output/f'dataset_{kind}.npz', **data._asdict())
    np.savez_compressed(output/f'initialization_{kind}_history.npz',
        **{key: np.asarray(value) for key, value in prefix.states._asdict().items()},
        observation_status=np.asarray(prefix.observations.status), times=data.times[:len(prefix.states.x)])
    result = fit_configuration(cfg, arrival, window, ids, adapter, basis, output, 'a_'+kind)
    result['truth_validation'] = truth_check
    result['pre_window_initializations'] = int(np.sum(np.asarray(prefix.observations.status) == INITIALIZED))
    result['pre_window_corrections'] = int(np.sum(np.asarray(prefix.observations.status) == UPDATED))
    write_json(output/f'a_{kind}_results.json', result)
    return result


def run_b(output, kind='constant'):
    jax.config.update('jax_enable_x64', True)
    output.mkdir(parents=True, exist_ok=True)
    cfg = configuration(kind)
    cfg['arrival_b'] = 'command-only propagation of original robot prior/Qc to t0; no measurements, empty map'
    data, truth_check = simulate_slam_window(cfg)
    _, window, ids, adapter, basis, _ = prepare_a(cfg, data)
    arrival = prepare_b(cfg)
    result = fit_configuration(cfg, arrival, window, ids, adapter, basis, output, 'b_'+kind)
    result['truth_validation'] = truth_check
    write_json(output/f'b_{kind}_results.json', result)
    return result


def finalize(output,test_record=None):
    """Generate plots and report from actual completed, serialized experiments."""
    results={}
    for kind in ('constant','smooth'):
        with np.load(output/f'dataset_{kind}.npz') as full:
            full_times,full_truth=full['times'],full['truth']
        for config in ('a','b'):
            name=config+'_'+kind
            result=json.loads((output/f'{name}_results.json').read_text())
            with np.load(output/f'{name}_histories.npz') as saved:
                arrays={key:saved[key] for key in saved.files}
            result['figures']=plot_experiment(output,name,arrays,full_truth,full_times)
            write_json(output/f'{name}_results.json',result)
            results[name]=result
    manifest=Path('artifacts/milestone4c/preservation.json')
    preservation=json.loads(manifest.read_text())
    differences=[name for name,digest in preservation['sha256'].items()
                 if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
    preservation.update(changed=differences,all_preserved=not differences)
    write_json(output/'preservation.json',preservation)
    payload={'experiments':results,
        'configuration_a_gate':json.loads((output/'configuration_a_gate.json').read_text()),
        'configuration_b_constant_gate':json.loads((output/'configuration_b_constant_gate.json').read_text()),
        'current_test_record':json.loads(test_record.read_text()) if test_record else None,
        'preservation':preservation,
        'environment':{'python':platform.python_version(),'jax':jax.__version__,'jax_backend':jax.default_backend(),
                       'jax_enable_x64':jax.config.x64_enabled,'numpy':np.__version__,'scipy':scipy.__version__}}
    payload['completion_checks']=completion_checks(payload)
    write_json(output/'configuration.json',{name:r['configuration'] for name,r in results.items()})
    write_json(output/'results.json',payload)
    write_report(output,payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/milestone4c'))
    parser.add_argument('--configuration', choices=['A','B','all'], default='all')
    parser.add_argument('--disturbance', choices=['constant','smooth','all'], default='all')
    parser.add_argument('--test-record',type=Path)
    args = parser.parse_args()
    for kind in (('constant','smooth') if args.disturbance == 'all' else (args.disturbance,)):
        if kind == 'smooth' and not (args.output/'configuration_b_constant_gate.json').exists():
            parser.error('constant-disturbance validation is required before smooth experiments')
        if args.configuration in ('A','all'):
            run_a(args.output,kind)
        if args.configuration in ('B','all'):
            gate = args.output/'configuration_a_gate.json'
            if not gate.exists() or not all(json.loads(gate.read_text())['checks'].values()):
                parser.error('Configuration A intermediate validation record is required before Configuration B')
            run_b(args.output,kind)
    if args.configuration == args.disturbance == 'all':
        finalize(args.output,args.test_record)


if __name__ == '__main__':
    main()
