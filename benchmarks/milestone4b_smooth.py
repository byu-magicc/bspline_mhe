"""Smooth, outside-spline-family Stage 4B experiment; no estimator changes.

Run with PYTHONPATH=build/bsplines_library python3 -B -m
benchmarks.milestone4b_smooth --test-record path/to/test_record.json.
"""

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

from benchmarks.milestone4a import summarize_candidate
from estimation.coefficient_optimization import (
    CheckedScipyObjective, disturbance_metrics, gradient_norms,
    innovation_sensitivity, optimize_coefficients,
)
from estimation.fixed_window_objective import (
    InvalidCandidateError, checked_evaluation, make_fixed_window_objective,
)
from estimation.propagation import rk4_stage_times
from simulation.smooth_unicycle import simulate_smooth_landmarks, smooth_disturbance
from splines import UniformBasis, disturbance_from_coefficients


CONSTANT_RESULTS = Path('artifacts/milestone4b/results.json')
FIT_NAMES = ('constant_constrained', 'full_spline')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def configuration():
    cfg = json.loads(CONSTANT_RESULTS.read_text())['configuration'].copy()
    del cfg['true_disturbance_v_omega']
    cfg.update(true_disturbance_functions=['0.15*sin(2*pi*sigma/5)', '0.05*cos(2*pi*sigma/5)'],
               dense_sample_count=501, coefficient_bounds=[-.5, .5],
               association_policy='known fixed landmark order',
               truth_process='deterministic analytic inputs; Qc is an estimator assumption')
    return cfg


def make_evaluator(cfg, data):
    adapter = UniformBasis(cfg['degree'], cfg['segments'])
    basis = adapter.precompute_jax(rk4_stage_times(data.times)-data.times[0],
                                   cfg['window_length_seconds'])
    # One immutable evaluator captures the common prior and data for every fit.
    return adapter, make_fixed_window_objective(
        (np.asarray(cfg['initial_pose']), np.asarray(cfg['arrival_covariance'])), data.times,
        lambda t: jnp.asarray(cfg['command_v_omega']), basis,
        cfg['continuous_input_noise_spectral_density_Qc'], data.measurements,
        data.landmarks_at_times, data.measurement_covariance, data.mask,
        coefficient_count=2*adapter.num_coefficients)


def least_squares_approximation(basis, truth):
    """Unconstrained sampled LS diagnostic; never an optimizer initial point."""
    basis, truth = np.asarray(basis), np.asarray(truth)
    solution, _, rank, singular = np.linalg.lstsq(basis, truth, rcond=None)
    coefficients = solution.T.reshape(-1)  # c=[c_v; c_omega]
    return coefficients, {'sample_count': len(basis), 'basis_rank': int(rank),
                           'basis_singular_values': singular.tolist(),
                           'coefficients': coefficients.tolist(),
                           'within_estimator_bounds': bool(np.all(np.abs(coefficients) <= .5)),
                           'disturbance_metrics': disturbance_metrics(basis, coefficients, truth),
                           'normal_equation_residual_max': float(np.max(np.abs(
                               basis.T@(basis@solution-truth))))}


def localization_metrics(poses, truth):
    """Uniform-node posterior-pose RMSE, including the arrival node."""
    error = np.asarray(poses)-np.asarray(truth)
    error = error.copy()
    error[:, 2] = np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2]))
    position = np.linalg.norm(error[:, :2], axis=1)
    return {'position_rmse_m': float(np.sqrt(np.mean(position**2))),
            'heading_rmse_rad': float(np.sqrt(np.mean(error[:, 2]**2))),
            'final_position_error_m': float(position[-1]),
            'final_heading_error_rad': float(error[-1, 2]),
            'final_absolute_heading_error_rad': float(abs(error[-1, 2])),
            'sample_count': len(error)}, error


def candidate_metrics(result, basis, coefficients, truth_dense, data):
    summary = summarize_candidate(result, data.truth)
    localization, error = localization_metrics(result.poses, data.truth)
    return {'final_candidate': summary, 'localization': localization,
            'disturbance_metrics': disturbance_metrics(basis, coefficients, truth_dense)}, error


def gradient_validation(evaluate, value_gradient, candidates, steps=(1e-4, 1e-5)):
    """Checked float64 central differences in both coefficient channels."""
    records = {}
    for name, coefficients in candidates.items():
        coefficients = np.asarray(coefficients)
        checked_evaluation(evaluate, coefficients)
        value, actual = value_gradient(coefficients)
        actual = np.asarray(actual)
        records[name] = {}
        for step in steps:
            reference = []
            for perturbation in np.eye(len(coefficients))*step:
                plus = checked_evaluation(evaluate, coefficients+perturbation)
                minus = checked_evaluation(evaluate, coefficients-perturbation)
                reference.append((float(plus.innovation_quadratic)-float(minus.innovation_quadratic))/(2*step))
            reference = np.asarray(reference)
            difference = np.abs(actual-reference)
            records[name][str(step)] = {
                'objective': float(value), 'jax_gradient': actual.tolist(),
                'central_difference_gradient': reference.tolist(), 'all_candidates_valid': True,
                'max_absolute_error': float(difference.max()),
                'passed': bool(np.isfinite(actual).all() and np.allclose(actual, reference, atol=2e-5, rtol=2e-5)
                               and difference.max() < 2e-5)}
    return {'precision': 'float64', 'steps': list(steps), 'atol': 2e-5, 'rtol': 2e-5,
            'maximum_absolute_discrepancy_limit': 2e-5, 'results': records,
            'passed': all(row['passed'] for steps in records.values() for row in steps.values())}


def covariance_audit(result, reference):
    """Keep K fixed; diagnose residual reduction with reference S held fixed."""
    mask = np.asarray(result.valid_mask)
    if not np.array_equal(mask, np.asarray(reference.valid_mask)):
        raise RuntimeError('Candidate changed the accepted observation set')
    e = np.asarray(result.innovations)[mask]
    S = np.asarray(result.innovation_covariances)[mask]
    S0 = np.asarray(reference.innovation_covariances)[mask]
    trace_ratios = np.trace(S, axis1=1, axis2=2)/np.trace(S0, axis1=1, axis2=2)
    return {'requested_count': int(result.requested_count), 'valid_count': int(result.valid_count),
            'rejected_count': int(result.rejected_count), 'identical_accepted_mask': True,
            'covariance_mean_diagonal': np.diagonal(S, axis1=1, axis2=2).mean(axis=0).tolist(),
            'mean_diagonal_ratio_to_zero': (np.diagonal(S, axis1=1, axis2=2).mean(axis=0)
                                           /np.diagonal(S0, axis1=1, axis2=2).mean(axis=0)).tolist(),
            'per_record_trace_ratio_min_mean_p95_max': [float(f(trace_ratios)) for f in
                (np.min, np.mean, lambda x: np.percentile(x, 95), np.max)],
            'unweighted_residual_sum_squares_range_bearing': np.sum(e**2, axis=0).tolist(),
            'quadratic_with_zero_candidate_covariances': float(np.sum(
                e*np.linalg.solve(S0, e[..., None])[..., 0]))}


def sensitivity_analysis(evaluator, candidates, basis, true_dense, approximation, elapsed):
    summaries, arrays = {}, {}
    error_coefficients = np.asarray(candidates['optimized'])-approximation
    for name, coefficients in candidates.items():
        summary, D, W = innovation_sensitivity(evaluator, np.asarray(coefficients))
        _, _, Vh = np.linalg.svd(W, full_matrices=False)
        projections = Vh@error_coefficients
        weak_coefficients = Vh[-4:].T@projections[-4:]
        weak_function = np.asarray(disturbance_from_coefficients(basis, weak_coefficients))
        estimation_error = np.asarray(disturbance_from_coefficients(basis, error_coefficients))
        total_error = np.asarray(disturbance_from_coefficients(basis, candidates['optimized']))-true_dense
        boundary = (elapsed <= .5) | (elapsed >= elapsed[-1]-.5)
        summary.update(
            coefficient_error_reference='independent sampled LS approximation, not exact true coefficients',
            whitened_right_singular_vectors=Vh.tolist(),
            coefficient_error_projections=projections.tolist(),
            weakest_four_coefficient_error_energy_fraction=float(np.sum(projections[-4:]**2)/np.sum(projections**2))
                if np.any(projections) else 0.,
            boundary_definition='first and last 0.5 s, inclusive',
            boundary_total_error_rmse_v_omega=np.sqrt(np.mean(total_error[boundary]**2, axis=0)).tolist(),
            interior_total_error_rmse_v_omega=np.sqrt(np.mean(total_error[~boundary]**2, axis=0)).tolist(),
            weak_four_function_rmse_v_omega=np.sqrt(np.mean(weak_function**2, axis=0)).tolist(),
            weak_four_boundary_function_rmse_v_omega=np.sqrt(np.mean(weak_function[boundary]**2, axis=0)).tolist(),
            weak_four_interior_function_rmse_v_omega=np.sqrt(np.mean(weak_function[~boundary]**2, axis=0)).tolist(),
            weak_four_error_cosine_v_omega=[float(np.dot(weak_function[:, i], estimation_error[:, i]) /
                (np.linalg.norm(weak_function[:, i])*np.linalg.norm(estimation_error[:, i])))
                if np.any(weak_function[:, i]) and np.any(estimation_error[:, i]) else None for i in range(2)])
        summaries[name] = summary
        arrays.update({name+'_raw_jacobian': D, name+'_whitened_jacobian': W,
                       name+'_whitened_right_singular_vectors': Vh,
                       name+'_weak_four_reconstruction_error': weak_function})
    return summaries, arrays


def plot_results(output, elapsed, curves, data, poses, errors, fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'truth': dict(color='black', linewidth=2.1),
              'zero': dict(color='0.5', linestyle=':'),
              'constant_constrained': dict(color='#D55E00', linestyle='--'),
              'full_spline': dict(color='#0072B2', linewidth=1.8),
              'least_squares': dict(color='#009E73', linestyle='-.')}
    labels = {'truth': 'Analytic truth', 'zero': 'Zero reference',
              'constant_constrained': 'Constant fit', 'full_spline': 'MHE spline fit',
              'least_squares': 'Independent spline approximation'}
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300}):
        def save(fig, name):
            fig.tight_layout()
            for extension in ('png', 'pdf'):
                fig.savefig(output/f'{name}.{extension}', bbox_inches='tight')
            plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
        for channel, axis in enumerate(axes):
            for name, curve in curves.items():
                axis.plot(elapsed, curve[:, channel], label=labels[name], **styles[name])
            axis.set_ylabel('$w_v$ (m/s)' if channel == 0 else r'$w_\omega$ (rad/s)')
            axis.grid(alpha=.2)
        axes[0].legend(loc='upper right', fontsize=8)
        axes[1].set_xlabel('Elapsed window time (s)')
        save(fig, 'disturbances')
        # Show representation error separately: the approximation overlaps truth
        # closely on the main axes, so a residual plot makes it inspectable.
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True)
        for channel, axis in enumerate(axes):
            axis.plot(elapsed, curves['least_squares'][:, channel]-curves['truth'][:, channel], color='#009E73')
            axis.set_ylabel('LS error (m/s)' if channel == 0 else 'LS error (rad/s)')
            axis.grid(alpha=.2)
        axes[-1].set_xlabel('Elapsed window time (s)')
        save(fig, 'approximation_error')
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for name, fit in fits.items():
            iterations = [row['iteration'] for row in fit['history']]
            values = np.asarray([row['objective'] for row in fit['history']])
            axes[0].plot(iterations, values, label=labels[name], **styles[name])
            axes[1].semilogy(iterations, np.maximum(values-fit['objective_final'], 1e-12), **styles[name])
        axes[0].set_ylabel('Innovation quadratic $J$')
        axes[1].set_ylabel('$J-J_{final}$ (display floor $10^{-12}$)')
        for axis in axes:
            axis.set_xlabel('Accepted iteration (0 = initial)')
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        save(fig, 'objective_history')
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].plot(data.truth[:, 0], data.truth[:, 1], label='Truth', **styles['truth'])
        for name, pose in poses.items():
            axes[0].plot(pose[:, 0], pose[:, 1], label=labels[name], **styles[name])
            axes[1].plot(data.times, np.linalg.norm(errors[name][:, :2], axis=1), **styles[name])
        axes[0].set(xlabel='$p_x$ (m)', ylabel='$p_y$ (m)', aspect='equal')
        axes[0].legend(fontsize=8)
        axes[1].set(xlabel='Elapsed window time (s)', ylabel='Position error (m)')
        for axis in axes:
            axis.grid(alpha=.2)
        save(fig, 'localization')


def validation_checks(payload):
    fits = payload['experiments']
    return {
        'independent_truth_refinement': payload['truth_integration']['refinement']['passed'],
        'all_primary_candidates_valid': all(row['final_candidate']['candidate_valid'] for row in fits.values()),
        'fits_converged_without_invalid_trials': all(fits[name]['success'] and not fits[name]['invalid_trials'] for name in FIT_NAMES),
        'same_requested_observations': all(row['identical_accepted_mask'] and row['rejected_count'] == 0
                                          and row['valid_count'] == 300 for row in payload['covariance_audit'].values()),
        'objective_numpy_consistency': all(row['final_candidate']['numpy_consistency']['passed'] for row in fits.values()),
        'gradients': payload['gradient_validation']['passed'],
        'sampled_ls_normal_equations': payload['least_squares_diagnostic']['normal_equation_residual_max'] < 1e-11,
        'full_spline_reconstructs_both_channels_better_than_zero': bool(np.all(
            np.asarray(fits['full_spline']['disturbance_metrics']['rmse_v_omega'])
            < np.asarray(fits['zero']['disturbance_metrics']['rmse_v_omega']))),
        'full_spline_reduces_residual_cost_with_reference_covariances':
            payload['covariance_audit']['full_spline']['quadratic_with_zero_candidate_covariances']
            < fits['zero']['objective_final'],
        'paired_repeat_reproducibility': all(row['coefficient_max_abs_difference'] < 1e-10
                                            and row['objective_max_abs_difference'] < 1e-10
                                            and row['repeat_success'] and row['accepted_objective_history_identical']
                                            for row in payload['reproducibility'].values()),
        'current_complete_test_suite': payload['current_test_record'] is not None
            and payload['current_test_record'].get('exit_code') == 0
            and payload['current_test_record'].get('scope') == 'complete repository suite',
    }


def run(output, test_record=None):
    jax.config.update('jax_enable_x64', True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cfg = configuration()
    data, integration = simulate_smooth_landmarks(cfg)
    adapter, evaluator = make_evaluator(cfg, data)
    elapsed = np.linspace(0., cfg['window_length_seconds'], cfg['dense_sample_count'])
    basis = adapter.evaluate(elapsed, cfg['window_length_seconds'])
    truth_dense = smooth_disturbance(elapsed, cfg['window_length_seconds'])
    n = adapter.num_coefficients
    zero = np.zeros(2*n)
    bridge = CheckedScipyObjective(evaluator, lambda c: c, zero)
    bridge.reference_check(zero)
    value, gradient = bridge(zero, 'reference')
    for _ in range(10):
        bridge(zero, 'reference_timing')
    reference_timing = dict(bridge.timings)
    samples = [row for row in bridge.evaluation_history if row['phase'] == 'reference_timing']
    for field in ('validity_check_seconds', 'value_gradient_seconds'):
        reference_timing[field+'_mean'] = float(np.mean([row[field] for row in samples]))
        reference_timing[field+'_p95'] = float(np.percentile([row[field] for row in samples], 95))
    reference_timing['optimization_seconds'] = 0.
    fits = {'zero': {'status': 'reference_not_optimized', 'success': None,
                     'message': 'Fixed zero coefficients; no optimization requested',
                     'initial_parameters': zero.tolist(), 'coefficients': zero.tolist(),
                     'objective_initial': value, 'objective_final': value, 'objective_reduction': 0.,
                     'iterations': 0, 'function_evaluations': 0, 'gradient_evaluations': 0,
                     'timing': reference_timing, 'gradient_norm_interpretation': 'diagnostic in full coefficient space',
                     **gradient_norms(zero, gradient, np.tile([-.5, .5], (2*n, 1)))}}
    repetitions = {}

    def checked_fit(evaluate, constant, phase):
        result = optimize_coefficients(evaluate, np.zeros(2 if constant else 2*n),
                                       coefficients_per_channel=n, constant=constant)
        print(json.dumps({'phase': phase, 'status': result['status'],
                          'J': result['objective_final'], 'iterations': result['iterations']}), flush=True)
        if result['invalid_trials']:
            failure = {'phase': phase, 'run': result, 'configuration': cfg,
                       'status': 'STOPPED: invalid candidate; no automatic retry or domain change'}
            write_json(output/'failure.json', failure)
            (output/'report.md').write_text('# Stage 4B smooth experiment stopped\n\n'
                f'Invalid candidate during {phase}. See failure.json for the checked policy record. '
                'Smooth reconstruction is not validated; Stage 4C remains deferred.\n')
            raise InvalidCandidateError(f'Invalid candidate in {phase}; recorded in {output}/failure.json')
        return result

    for name, constant in zip(FIT_NAMES, (True, False)):
        fits[name] = checked_fit(evaluator, constant, name)
        repeat = checked_fit(evaluator, constant, name+'_repeat')
        repetitions[name] = {
            'repeat_run': repeat, 'repeat_success': repeat['success'],
            'coefficient_max_abs_difference': float(np.max(np.abs(np.asarray(fits[name]['coefficients'])-repeat['coefficients']))),
            'objective_max_abs_difference': abs(fits[name]['objective_final']-repeat['objective_final']),
            'accepted_objective_history_identical': [row['objective'] for row in fits[name]['history']]
                == [row['objective'] for row in repeat['history']]}
    approximation, ls = least_squares_approximation(basis, truth_dense)
    results, poses, errors = {}, {}, {}
    for name, fit in fits.items():
        result = checked_evaluation(bridge.compiled_evaluate, np.asarray(fit['coefficients']))
        metrics, errors[name] = candidate_metrics(result, basis, fit['coefficients'], truth_dense, data)
        fit.update(metrics)
        results[name], poses[name] = result, np.asarray(result.poses)
    ls_result = checked_evaluation(bridge.compiled_evaluate, approximation)
    ls['candidate_diagnostics'] = summarize_candidate(ls_result, data.truth)
    gradients = gradient_validation(bridge.compiled_evaluate, bridge.compiled_value_gradient,
                                    {'zero': zero, 'full_optimized': fits['full_spline']['coefficients']})
    sensitivities, sensitivity_arrays = sensitivity_analysis(
        evaluator, {'zero': zero, 'optimized': fits['full_spline']['coefficients']},
        basis, truth_dense, approximation, elapsed)

    # Separate truth/geometry-matched noiseless control helps diagnose noise
    # fitting without feeding truth inputs or LS coefficients to the EKF.
    delta = data.landmarks_at_times-data.truth[:, None, :2]
    clean = np.stack((np.linalg.norm(delta, axis=-1),
                      np.arctan2(delta[..., 1], delta[..., 0])-data.truth[:, None, 2]), axis=-1)
    clean[..., 1] = (clean[..., 1]+np.pi) % (2*np.pi)-np.pi
    clean_data = data._replace(measurements=np.where(data.mask[..., None], clean, 0.))
    _, clean_evaluator = make_evaluator(cfg, clean_data)
    clean_fit = checked_fit(clean_evaluator, False, 'noiseless_full_spline_diagnostic')
    clean_result = checked_evaluation(jax.jit(clean_evaluator), np.asarray(clean_fit['coefficients']))
    clean_fit.update(candidate_metrics(clean_result, basis, clean_fit['coefficients'], truth_dense, clean_data)[0])

    previous = json.loads(CONSTANT_RESULTS.read_text())
    comparison = {
        'source': str(CONSTANT_RESULTS),
        'source_sha256': hashlib.sha256(CONSTANT_RESULTS.read_bytes()).hexdigest(),
        'constant_experiments': {name: {key: previous['experiments'][name][key] for key in
            ('objective_initial', 'objective_final', 'disturbance_metrics', 'final_candidate')}
            for name in FIT_NAMES},
        'constant_sensitivities': previous['local_sensitivities'],
        'interpretation': 'Different true inputs and resulting observations; within-dataset paired fits isolate parameterization effects.'}
    curves = {'truth': truth_dense, **{name: np.asarray(disturbance_from_coefficients(basis, fit['coefficients']))
                                      for name, fit in fits.items()},
              'least_squares': np.asarray(disturbance_from_coefficients(basis, approximation))}
    payload = {'configuration': cfg, 'truth_integration': integration, 'experiments': fits,
               'reproducibility': repetitions, 'least_squares_diagnostic': ls,
               'gradient_validation': gradients, 'local_sensitivities': sensitivities,
               'covariance_audit': {name: covariance_audit(result, results['zero']) for name, result in results.items()},
               'noiseless_full_spline_diagnostic': clean_fit, 'constant_experiment_comparison': comparison,
               'dataset_sha256': hashlib.sha256(data.times.tobytes()+data.truth.tobytes()+data.measurements.tobytes()
                    +data.mask.tobytes()+data.landmarks_at_times.tobytes()+data.measurement_covariance.tobytes()).hexdigest(),
               'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                   'scipy': scipy.__version__, 'jax': jax.__version__, 'jax_backend': jax.default_backend(),
                   'jax_enable_x64': jax.config.x64_enabled, 'platform': platform.platform()},
               'current_test_record': json.loads(test_record.read_text()) if test_record else None,
               'likelihood_diagnostic_definition': 'gaussian_diagnostic = J + sum(logdet(S)); omits dimension*log(2*pi)',
               'localization_metric_grid': '201 posterior integration nodes including arrival, dt=0.025 s',
               'disturbance_metric_grid': '501 uniform times including both endpoints; maxima are sampled',
               'feasibility_probes_do_not_certify_entire_box': True}
    payload['validation_checks'] = validation_checks(payload)
    write_json(output/'configuration.json', cfg)
    write_json(output/'optimization_histories.json', {name: {key: fit[key] for key in ('history', 'evaluation_history')}
                                                    for name, fit in fits.items() if name != 'zero'})
    write_json(output/'sensitivity_diagnostics.json', sensitivities)
    np.savez_compressed(output/'dataset.npz', times=data.times, truth=data.truth, measurements=data.measurements,
        landmarks=data.landmarks_at_times, mask=data.mask, R=data.measurement_covariance,
        arrival_mean=cfg['initial_pose'], arrival_covariance=cfg['arrival_covariance'],
        Qc=cfg['continuous_input_noise_spectral_density_Qc'], command=cfg['command_v_omega'],
        noiseless_control_measurements=clean_data.measurements)
    np.savez_compressed(output/'disturbance_curves.npz', elapsed=elapsed, basis=basis, **curves)
    np.savetxt(output/'disturbance_curves.csv', np.column_stack((elapsed, *(curves[name] for name in curves))),
        delimiter=',', header='sigma,'+','.join(name+'_'+channel for name in curves for channel in ('v', 'omega')), comments='')
    np.savez_compressed(output/'sensitivities.npz', **sensitivity_arrays)
    np.savez_compressed(output/'localization.npz', times=data.times, truth=data.truth,
                        **{name+'_poses': pose for name, pose in poses.items()},
                        **{name+'_errors': error for name, error in errors.items()})
    plot_results(output, elapsed, curves, data, poses, errors, {name: fits[name] for name in FIT_NAMES})
    payload['total_experiment_seconds_excluding_final_report_write'] = time.perf_counter()-started
    write_json(output/'results.json', payload)
    write_report(output/'report.md', payload)
    return payload


def write_report(path, payload):
    fits = payload['experiments']
    checks = validation_checks(payload)
    valid = all(checks.values())
    constant, full = (fits[name] for name in FIT_NAMES)
    zero = fits['zero']
    reduction = constant['objective_final']-full['objective_final']
    ratios = np.asarray(full['disturbance_metrics']['rmse_v_omega'])/constant['disturbance_metrics']['rmse_v_omega']
    ls = payload['least_squares_diagnostic']
    lines = ['# Stage 4B extension — smooth time-varying disturbance reconstruction', '',
        f"Controlled smooth reconstruction validation: **{'PASSED' if valid else 'NOT YET PASSED'}**.", '',
        f"Full spline / constant disturbance RMSE ratios (linear, angular): `{ratios.tolist()}`. "
        f"Full spline reduces J by {reduction:.9g} ({100*reduction/constant['objective_final']:.5g}%) beyond the constant fit, "
        f"and by {zero['objective_final']-full['objective_final']:.9g} relative to zero.", '',
        'This is one seed-7, fixed-window, known-landmark experiment. Experiment labels zero/constant/full below '
        'are parameterization comparisons, separate from the project’s preserved SLAM Baselines A/B.', '',
        '## Reproduction and fixed configuration', '',
        '```sh', 'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4b_smooth --test-record artifacts/milestone4b_smooth/test_record.json', '```', '',
        'Configuration: [configuration.json](configuration.json). L=5 s, cubic degree 3, M=4, seven coefficients per channel, '
        'c=[c_v;c_omega], alpha=M/L=0.8. Commands [1.0,0.2], estimator RK4 step 0.025 s, observations every 0.1 s, '
        'all six landmarks in fixed order, no observation at arrival: 300 updates and 600 innovation components. '
        'The arrival mean/covariance, R and Qc are copied from the validated constant configuration. '
        'One immutable Stage 4A evaluator and one dataset are shared by all primary comparisons. '
        'Qc remains a nonzero estimator uncertainty assumption; truth is deterministic apart from measurement noise.', '',
        f"Dataset SHA-256 (canonical concatenated arrays): `{payload['dataset_sha256']}`.", '',
        'True disturbances are 0.15 sin(2 pi sigma/5) m/s and 0.05 cos(2 pi sigma/5) rad/s. '
        'SciPy DOP853 integrates these analytic functions independently of JAX, RK4 and the C++ spline. '
        'Primary max step 0.05 s, rtol=1e-11, atol=1e-13; refinement halves the step and tightens tolerances tenfold. '
        f"Maximum pose component differences: `{payload['truth_integration']['refinement']['max_absolute_pose_difference']}` "
        '(acceptance <1e-10). Both refinements are independent of estimator propagation.', '',
        '## Objective, constraints and optimization', '',
        'Only the unchanged innovation quadratic J=sum(r.T solve(S,r)) is minimized. No regularization, log determinant '
        'or finite failure penalty is added. The constant fit uses z=[a,b] and c=[a ones(7);b ones(7)]. '
        'Both fits start at zero, use the existing checked L-BFGS-B/JAX gradients and bounds [-0.5,0.5]. '
        'Every candidate retains all requested measurements; an invalid preflight/trial stops the experiment with a failure record. '
        'Deterministic feasibility probes do not certify the whole box.', '',
        '| Model | Initial J | Final J | Status | Iterations | Function / gradient evaluations | Gradient L2 / inf | Projected gradient inf |',
        '|---|---:|---:|---|---:|---|---|---:|']
    for name, fit in fits.items():
        lines.append(f"| {name} | {fit['objective_initial']:.12g} | {fit['objective_final']:.12g} | {fit['status']} | {fit['iterations']} | "
                     f"{fit['function_evaluations']} / {fit['gradient_evaluations']} | {fit['gradient_l2']:.6g} / {fit['gradient_inf']:.6g} | {fit['projected_gradient_inf']:.6g} |")
    lines += ['', 'Zero has no optimizer calls; its full-coefficient gradient norms are diagnostic only. '
        'Raw and projected gradient infinity norms are reported because SciPy function-reduction convergence can precede '
        'the configured gtol=1e-6. The other unchanged primary stopping settings include ftol=1e-12 and maxiter=500. '
        'A successful function-reduction termination does not establish strict stationarity or global optimality.', '']
    for name in FIT_NAMES:
        lines += [f"**{name}:** {fits[name]['message']}. Coefficients: `{fits[name]['coefficients']}`.", '']
    lines += ['## Reconstruction and localization', '',
        'Disturbance errors use the same 501 uniform samples, including endpoints. Maximum errors are sampled maxima. '
        'Pose RMSE uses all 201 posterior integration nodes (including arrival); heading differences wrap to [-pi,pi]. '
        'Position RMSE is sqrt(mean(dx²+dy²)); heading RMSE is sqrt(mean(wrapped heading error²)).', '',
        '| Model | w_v RMSE (m/s) | w_omega RMSE (rad/s) | Max abs w_v / w_omega | Position RMSE (m) | Heading RMSE (rad) | Final position (m) | Final signed heading (rad) |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for name, fit in fits.items():
        d, p = fit['disturbance_metrics'], fit['localization']
        lines.append(f"| {name} | {d['rmse_v_omega'][0]:.9g} | {d['rmse_v_omega'][1]:.9g} | {d['max_absolute_error_v_omega']} | "
                     f"{p['position_rmse_m']:.9g} | {p['heading_rmse_rad']:.9g} | {p['final_position_error_m']:.9g} | {p['final_heading_error_rad']:.9g} |")
    lines += ['', f"Relative to the constant constraint, full-spline linear/angular disturbance RMSE changes by "
        f"`{(100*(ratios-1)).tolist()}` percent. Position/heading RMSE changes by "
        f"`{[100*(full['localization'][key]/constant['localization'][key]-1) for key in ('position_rmse_m', 'heading_rmse_rad')]}` percent. "
        'Disturbance recovery and localization therefore need separate assessment; a lower disturbance RMSE '
        'does not ensure every pose metric improves.', '']
    lines += ['', '## Independent spline approximation', '',
        f"Unconstrained least squares on the dense C++ basis gives RMSE `{ls['disturbance_metrics']['rmse_v_omega']}`, "
        f"maximum errors `{ls['disturbance_metrics']['max_absolute_error_v_omega']}`, rank {ls['basis_rank']}, "
        f"and normal-equation residual {ls['normal_equation_residual_max']:.3g}. "
        f"Its diagnostic innovation J is {ls['candidate_diagnostics']['innovation_quadratic']:.12g}. "
        'This is the best sampled linear least-squares approximation on the specified grid, not an estimator, '
        'a continuous-time minimax bound or optimized EKF result. Its coefficients never initialize an optimization. '
        'The analytic functions are outside the finite cubic spline family.', '',
        '## Fixed measurements, covariance and likelihood diagnostics', '',
        'Log determinant and Gaussian diagnostic J+sum(logdet S) are preserved diagnostics only; the Gaussian diagnostic '
        'omits the dimension-dependent constant. The fixed-S column recomputes each candidate residual quadratic using '
        'the zero-reference innovation covariance blocks, isolating residual changes from S changes.', '',
        '| Model | Requested / valid / rejected | logdet sum | J + logdet | Mean S diagonal / zero | Fixed-S quadratic | Unweighted residual square sum (range, bearing) |',
        '|---|---|---:|---:|---|---:|---|']
    for name, audit in payload['covariance_audit'].items():
        c = fits[name]['final_candidate']
        lines.append(f"| {name} | {audit['requested_count']} / {audit['valid_count']} / {audit['rejected_count']} | "
                     f"{c['logdet_diagnostic']:.10g} | {c['gaussian_diagnostic']:.10g} | {audit['mean_diagonal_ratio_to_zero']} | "
                     f"{audit['quadratic_with_zero_candidate_covariances']:.10g} | {audit['unweighted_residual_sum_squares_range_bearing']} |")
    fixed_s_improvement = payload['covariance_audit']['full_spline']['quadratic_with_zero_candidate_covariances'] < zero['objective_final']
    lines += ['', f"Residual quadratic improves with S fixed at zero reference: **{fixed_s_improvement}**. "
        'All priors/posteriors and innovation covariances pass the unchanged domain checks. '
        'Full per-record covariance trace ratio min/mean/p95/max is retained in results.json. '
        'These checks expose covariance effects without treating them as an extra optimization objective.', '',
        '## Sensitivity and boundary errors', '',
        'Raw D stacks innovation derivatives; W=blockdiag(cholesky(S)^-1)D freezes S for whitening at the evaluated point. '
        'The objective gradient still includes dS/dc. Numerical rank uses max(rows,cols)*epsilon64*sigma_max; '
        'practical rank uses 1e-3*sigma_max. These are local diagnostics and do not establish global identifiability.', '',
        '| Experiment / point | Matrix | Shape | Numerical / practical rank | Condition number | Singular values |',
        '|---|---|---|---|---:|---|']
    for experiment, summaries in [('smooth', payload['local_sensitivities']),
                                  ('constant', payload['constant_experiment_comparison']['constant_sensitivities'])]:
        for point, summary in summaries.items():
            for kind in ('raw', 'whitened'):
                d = summary[kind]
                lines.append(f"| {experiment} / {point} | {kind} | {d['shape']} | {d['numerical_rank']} / {d['practical_rank']} | {d['condition_number']:.8g} | {d['singular_values']} |")
    s = payload['local_sensitivities']['optimized']
    clean_rmse = np.asarray(payload['noiseless_full_spline_diagnostic']['disturbance_metrics']['rmse_v_omega'])
    approximation_rmse = np.asarray(ls['disturbance_metrics']['rmse_v_omega'])
    lines += ['', f"Relative to the LS approximation coefficients, {100*s['weakest_four_coefficient_error_energy_fraction']:.5g}% "
        'of fitted coefficient error energy lies in the weakest four whitened right singular directions. '
        f"Total reconstruction RMSE at the first/last 0.5 s is `{s['boundary_total_error_rmse_v_omega']}` "
        f"versus `{s['interior_total_error_rmse_v_omega']}` in the interior. "
        f"The weak-four reconstructed error RMSE is `{s['weak_four_boundary_function_rmse_v_omega']}` at boundaries "
        f"and `{s['weak_four_interior_function_rmse_v_omega']}` in the interior. "
        'Boundary errors and the boundary contribution of weak directions are larger here. The weak-direction function '
        'does not explain the entire error: its cosine with the fit-minus-LS function error is '
        f"`{s['weak_four_error_cosine_v_omega']}`. This examines function errors as well as coefficient projections; "
        'coefficient energy alone is parameterization dependent.', '',
        '## Noise diagnostic and constant-experiment comparison', '',
        f"A separately labelled zero-start noiseless full-spline control gives disturbance RMSE "
        f"`{payload['noiseless_full_spline_diagnostic']['disturbance_metrics']['rmse_v_omega']}` "
        f"and J={payload['noiseless_full_spline_diagnostic']['objective_final']:.10g} "
        f"({payload['noiseless_full_spline_diagnostic']['status']}). It preserves geometry, truth and configured R/Qc, "
        'but removes sampled measurement noise. It does not replace the noisy primary comparisons. '
        f"The noiseless/LS RMSE ratios are `{(clean_rmse/approximation_rmse).tolist()}`, "
        f"while noisy/noiseless full-fit RMSE ratios are "
        f"`{(np.asarray(full['disturbance_metrics']['rmse_v_omega'])/clean_rmse).tolist()}`. "
        'Near-approximation-level noiseless recovery supports measurement noise as a substantial contributor to the '
        'larger primary reconstruction errors, consistent with the weak-direction/boundary diagnostics. '
        'This does not prove any one factor is the only error source.', '',
        '| Truth experiment | Fit | Final J | Disturbance RMSE (linear, angular) |',
        '|---|---|---:|---|']
    for experiment, records in [('constant', payload['constant_experiment_comparison']['constant_experiments']),
                                 ('smooth', {name: fits[name] for name in FIT_NAMES})]:
        for name, record in records.items():
            lines.append(f"| {experiment} | {name} | {record['objective_final']:.12g} | {record['disturbance_metrics']['rmse_v_omega']} |")
    lines += ['', 'The constant experiment favors its correctly specified constant constraint; its full spline fits noise '
        'with a slightly lower objective and worse reconstruction. The smooth experiment tests a restricted model that '
        'cannot follow the injected variation. Cross-experiment objective differences are descriptive because truth and '
        'observations differ. Within each experiment, data and prior are paired.', '',
        '## Timing and reproducibility', '',
        '| Model | Evaluation compile (s) | Value/gradient compile (s) | Checked cold reference incl. auxiliary compilation (s) | Warm optimization (s) | Steady value/gradient mean / p95 (s) | Steady checked evaluation mean / p95 (s) |',
        '|---|---:|---:|---:|---:|---|---|']
    for name, fit in fits.items():
        t = fit['timing']
        lines.append(f"| {name} | {t['evaluation_compile_seconds']:.6g} | {t['value_gradient_compile_seconds']:.6g} | "
            f"{t['checked_reference_first_call_seconds_including_auxiliary_compilation']:.6g} | {t['optimization_seconds']:.6g} | "
            f"{t['value_gradient_seconds_mean']:.6g} / {t['value_gradient_seconds_p95']:.6g} | "
            f"{t['validity_check_seconds_mean']:.6g} / {t['validity_check_seconds_p95']:.6g} |")
    lines += ['', 'Explicit lower/compile times exclude execution; the cold checked reference may include auxiliary VJP '
        'compilation. Warm optimization includes checked trial evaluations and gradients, after preparation. '
        'Zero timing uses ten repeated evaluations. These are fixed-window offline timings, not moving-horizon latency. '
        f"Environment: `{payload['environment']}`. Both fitted models repeat from zero on the identical dataset; "
        f"maximum coefficient differences: `{[payload['reproducibility'][name]['coefficient_max_abs_difference'] for name in FIT_NAMES]}`. "
        'Repeats are deterministic checks, not Monte Carlo trials or statistical confidence intervals.', '',
        '## Figures and saved evidence', '',
        '![Disturbance reconstruction](disturbances.png)', '',
        '![Independent approximation error](approximation_error.png)', '',
        '![Optimization history](objective_history.png)', '',
        '![Robot trajectories and position errors](localization.png)', '',
        'Each figure also has a vector PDF. results.json contains complete metrics, histories, timings, diagnostics '
        'and the previous constant sensitivity comparison. Additional files: dataset.npz, configuration.json, '
        'optimization_histories.json, disturbance_curves.npz/.csv, localization.npz, sensitivities.npz '
        'and sensitivity_diagnostics.json. Validation/test and preservation records are retained separately.', '',
        '## Validation gate and limitations', '', '```json', json.dumps(checks, indent=2), '```', '',
        f"Finite-difference checks use float64 and steps 1e-4/1e-5, with atol=rtol=2e-5 and maximum absolute discrepancy <2e-5; "
        f"all perturbations must remain valid. Actual gradient checks passed: {payload['gradient_validation']['passed']}.", '',
        'No test requires full-spline reconstruction to beat the constant fit. The validation checks establish '
        'numerical and experimental correctness; the measured RMSEs determine the performance conclusion.', '',
        ('Smooth time-varying reconstruction is validated for this controlled scenario. Stage 4B is ready to transition '
         'to planning Stage 4C within this scope; Stage 4C implementation is not started. '
         if valid else 'The controlled validation gate has outstanding checks; Stage 4C remains deferred. ')+
        'General reliability across seeds, abrupt disturbances, noise levels and sparse measurements remains unvalidated. '
        'Monte Carlo trials are required for final comparisons. Stage 4C must first define initialization/arrival '
        'measurement accounting; moving windows, nearest-neighbor association and Victoria Park remain outside this task.', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/milestone4b_smooth'))
    parser.add_argument('--test-record', type=Path)
    args = parser.parse_args()
    if args.output.resolve() == CONSTANT_RESULTS.parent.resolve():
        parser.error('smooth artifacts must use a separate directory from constant Stage 4B')
    run(args.output, args.test_record)


if __name__ == '__main__':
    main()
