"""Stage 4B paired constant/full fits, local sensitivities and saved diagnostics."""

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
    disturbance_metrics, innovation_sensitivity, optimize_coefficients,
)
from estimation.fixed_window_objective import checked_evaluation, make_fixed_window_objective
from estimation.propagation import rk4_stage_times
from simulation.continuous_unicycle import simulate_known_landmarks
from splines import UniformBasis, disturbance_from_coefficients


CONFIG_PATH = Path('artifacts/milestone4a_hardening/results.json')


def completion_checks(payload):
    """Evidence for the controlled constant-disturbance scope, not global claims."""
    fits = payload['experiments']
    baseline = np.asarray(payload['baseline_zero_disturbance_metrics']['rmse_v_omega'])
    return {
        'required_experiments_and_diagnostics_present':
            {'constant_constrained', 'full_spline'} <= fits.keys()
            and {'constant_constrained', 'full_spline'} <= payload['reproducibility'].keys()
            and {'zero', 'optimized'} <= payload['local_sensitivities'].keys()
            and {'constant_constrained', 'full_spline'} <= payload['noiseless_diagnostic_controls'].keys(),
        'primary_fits_converged': all(r['success'] for r in fits.values()),
        'valid_trials_and_final_candidates': all(
            not r['invalid_trials'] and r.get('final_candidate', {}).get('candidate_valid', False)
            and bool(r['feasibility_probes']) and all(p['valid'] for p in r['feasibility_probes'])
            for r in fits.values()),
        'objective_reduction': all(r['objective_reduction'] is not None and r['objective_reduction'] > 0 for r in fits.values()),
        'both_channels_reconstructed_better_than_zero': all(
            'disturbance_metrics' in r and np.all(np.asarray(r['disturbance_metrics']['rmse_v_omega']) < baseline)
            for r in fits.values()),
        'paired_repeat_reproducibility': all(
            len(r['runs']) >= 2 and r['coefficient_max_abs_difference'] is not None
            and r['coefficient_max_abs_difference'] < 1e-10
            and r['objective_max_abs_difference'] < 1e-10 for r in payload['reproducibility'].values()),
        'local_sensitivity_dimensions_and_rank': all(
            s[kind]['shape'] == [600, 14] and s[kind]['numerical_rank'] == 14
            for s in payload['local_sensitivities'].values() for kind in ('raw', 'whitened')),
        'noiseless_controls': all(r['success'] and 'disturbance_metrics' in r
                                 and np.all(np.asarray(r['disturbance_metrics']['rmse_v_omega']) < baseline)
                                 for r in payload['noiseless_diagnostic_controls'].values()),
        'current_complete_test_suite': payload['current_test_record'] is not None
                                     and payload['current_test_record'].get('exit_code') == 0
                                     and payload['current_test_record'].get('scope') == 'complete repository suite',
    }


def create_experiment(*, noiseless=False):
    cfg = json.loads(CONFIG_PATH.read_text())['configuration']
    data = simulate_known_landmarks(
        L=cfg['window_length_seconds'], dt=cfg['integration_step_seconds'],
        measurement_interval=cfg['measurement_interval_seconds'],
        command=cfg['command_v_omega'], disturbance=cfg['true_disturbance_v_omega'],
        initial_pose=cfg['initial_pose'], landmark_map=cfg['landmark_coordinates'],
        measurement_covariance=cfg['measurement_covariance'], seed=cfg['seed'])
    if noiseless:
        delta = data.landmarks_at_times-data.truth[:, None, :2]
        clean = np.stack((np.linalg.norm(delta, axis=-1),
                          np.arctan2(delta[..., 1], delta[..., 0])-data.truth[:, None, 2]), axis=-1)
        clean[..., 1] = np.arctan2(np.sin(clean[..., 1]), np.cos(clean[..., 1]))
        data = data._replace(measurements=np.where(data.mask[..., None], clean, 0.))
    adapter = UniformBasis(cfg['degree'], cfg['segments'])
    stages = adapter.precompute_jax(rk4_stage_times(data.times)-data.times[0], cfg['window_length_seconds'])
    evaluator = make_fixed_window_objective(
        (np.array(cfg['initial_pose']), np.array(cfg['arrival_covariance'])), data.times,
        lambda t: jnp.asarray(cfg['command_v_omega']), stages,
        cfg['continuous_input_noise_spectral_density_Qc'], data.measurements,
        data.landmarks_at_times, data.measurement_covariance, data.mask,
        coefficient_count=2*adapter.num_coefficients)
    truth = np.repeat(cfg['true_disturbance_v_omega'], adapter.num_coefficients)
    return cfg, data, adapter, evaluator, truth


def attach_metrics(run, basis, true_disturbance, evaluator, data):
    if run['coefficients'] is None:
        return
    c = np.asarray(run['coefficients'])
    run['disturbance_metrics'] = disturbance_metrics(basis, c, true_disturbance)
    run['final_candidate'] = summarize_candidate(checked_evaluation(jax.jit(evaluator), c), data.truth)


def plot_results(output, elapsed, basis, truth, runs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for channel, axis in enumerate(axes):
        axis.axhline(truth[channel], color='black', linestyle='--', label='True constant disturbance')
        axis.axhline(0., color='gray', linewidth=.7, label='Zero initial guess')
        for name, run in runs.items():
            if run['coefficients'] is not None:
                curve = np.asarray(disturbance_from_coefficients(basis, run['coefficients']))
                axis.plot(elapsed, curve[:, channel], label=name.replace('_', ' '))
        axis.set_ylabel('Linear disturbance (m/s)' if channel == 0 else 'Angular disturbance (rad/s)')
        axis.grid(alpha=.25)
    axes[0].legend()
    axes[1].set_xlabel('Elapsed window time (s)')
    fig.tight_layout()
    fig.savefig(output/'disturbances.png', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    for name, run in runs.items():
        axis.plot([row['iteration'] for row in run['history']],
                  [row['objective'] for row in run['history']], label=name.replace('_', ' '))
    axis.set(xlabel='Accepted optimization iteration (0 = initial)', ylabel='Innovation quadratic J')
    axis.legend()
    axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output/'objective_history.png', dpi=180)
    plt.close(fig)


def run(output, repeats=2, test_record=None):
    jax.config.update('jax_enable_x64', True)
    output.mkdir(parents=True, exist_ok=True)
    cfg, data, adapter, evaluator, c_true = create_experiment()
    n = adapter.num_coefficients
    elapsed = np.linspace(0., cfg['window_length_seconds'], 501)
    basis = adapter.precompute_jax(elapsed, cfg['window_length_seconds'])
    truth = np.asarray(cfg['true_disturbance_v_omega'])
    runs, repeat_records = {}, {}
    for name, constant in [('constant_constrained', True), ('full_spline', False)]:
        records = []
        for repeat in range(repeats):
            result = optimize_coefficients(evaluator, np.zeros(2 if constant else 2*n),
                                           coefficients_per_channel=n, constant=constant)
            attach_metrics(result, basis, truth, evaluator, data)
            records.append(result)
            print(json.dumps({'experiment': name, 'repeat': repeat, 'status': result['status'],
                              'J': result['objective_final'], 'iterations': result['iterations'],
                              'metrics': result.get('disturbance_metrics')}), flush=True)
        runs[name] = records[0]
        repeat_records[name] = {
            'runs': records,
            'coefficient_max_abs_difference': max(float(np.max(np.abs(np.asarray(r['coefficients'])-np.asarray(records[0]['coefficients'])))) for r in records) if all(r['coefficients'] is not None for r in records) else None,
            'objective_max_abs_difference': max(abs(r['objective_final']-records[0]['objective_final']) for r in records) if all(r['objective_final'] is not None for r in records) else None,
        }
    diagnostics = {}
    for name, c in {'zero': np.zeros(2*n), 'true': c_true,
                    'constant_optimized': runs['constant_constrained']['coefficients'],
                    'full_optimized': runs['full_spline']['coefficients']}.items():
        if c is not None:
            diagnostics[name] = summarize_candidate(checked_evaluation(jax.jit(evaluator), c), data.truth)
    sensitivities, arrays = {}, {}
    for name, c in {'zero': np.zeros(2*n), 'optimized': runs['full_spline']['coefficients']}.items():
        if c is None:
            continue
        summary, D, W = innovation_sensitivity(evaluator, np.asarray(c))
        sensitivities[name] = summary
        arrays[name+'_jacobian'] = D
        arrays[name+'_whitened_jacobian'] = W
        # Inspect coefficient error along local right singular directions.
        _, singular_values, Vh = np.linalg.svd(W, full_matrices=False)
        if runs['full_spline']['coefficients'] is not None:
            error = np.asarray(runs['full_spline']['coefficients'])-c_true
            projections = Vh@error
            summary['full_fit_coefficient_error_in_whitened_right_singular_directions'] = projections.tolist()
            summary['weakest_four_coefficient_error_energy_fraction'] = float(np.sum(projections[-4:]**2)/np.sum(projections**2)) if np.any(projections) else 0.
        summary['whitened_right_singular_vectors'] = Vh.tolist()
        summary['inverse_singular_values_local_unit_noise_scales'] = (1/singular_values).tolist()
    # A noise-free version of the identical truth/geometry isolates the effect
    # of fitting measurement noise. It does not replace either requested run.
    controls = {}
    _, clean_data, _, clean_evaluator, _ = create_experiment(noiseless=True)
    for name, constant in [('constant_constrained', True), ('full_spline', False)]:
        result = optimize_coefficients(clean_evaluator, np.zeros(2 if constant else 2*n),
                                       coefficients_per_channel=n, constant=constant)
        attach_metrics(result, basis, truth, clean_evaluator, clean_data)
        controls[name] = result
        print(json.dumps({'control': 'noiseless_'+name, 'status': result['status'],
                          'J': result['objective_final'], 'metrics': result.get('disturbance_metrics')}), flush=True)
    refinement = optimize_coefficients(evaluator, np.zeros(2*n), coefficients_per_channel=n,
                                        options={'ftol': 1e-15})
    attach_metrics(refinement, basis, truth, evaluator, data)
    np.savez_compressed(output/'sensitivities.npz', **arrays)
    np.savez_compressed(output/'dataset.npz', times=data.times, truth=data.truth, measurements=data.measurements,
                        landmarks=data.landmarks_at_times, mask=data.mask, R=data.measurement_covariance,
                        noiseless_control_measurements=clean_data.measurements)
    np.savez_compressed(output/'disturbance_curves.npz', elapsed=elapsed, basis=np.asarray(basis),
                        true_disturbance=np.broadcast_to(truth, (len(elapsed), 2)),
                        **{name: np.asarray(disturbance_from_coefficients(basis, r['coefficients']))
                           for name, r in runs.items() if r['coefficients'] is not None})
    plot_results(output, elapsed, basis, truth, runs)
    payload = {'configuration': cfg, 'exploratory_parameter_bounds': [-.5, .5],
               'feasibility_probes_do_not_certify_entire_box': True,
               'environment': {'jax': jax.__version__, 'scipy': scipy.__version__, 'numpy': np.__version__,
                               'python': platform.python_version(), 'jax_backend': jax.default_backend()},
               'experiments': runs, 'reproducibility': repeat_records,
               'likelihood_diagnostics': diagnostics, 'local_sensitivities': sensitivities,
               'noiseless_diagnostic_controls': controls,
               'tighter_ftol_full_spline_control': refinement,
               'baseline_zero_disturbance_metrics': disturbance_metrics(basis, np.zeros(2*n), truth),
               'dataset_sha256': hashlib.sha256(data.measurements.tobytes()+data.truth.tobytes()+data.mask.tobytes()).hexdigest(),
               'current_test_record': json.loads(test_record.read_text()) if test_record else None}
    payload['completion_checks'] = completion_checks(payload)
    (output/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    write_report(output/'report.md', payload)
    return payload


def write_report(path, payload):
    # Detailed conclusions are assembled after the checked experiments, not
    # inferred from a successful return code alone.
    lines = ['# Stage 4B — B-Spline Coefficient Optimization', '',
             f"Controlled constant-disturbance completion gate: **{'SATISFIED' if all(completion_checks(payload).values()) else 'NOT YET SATISFIED'}**. Scope: the documented seed-7 known-landmark noisy experiment, repeat runs, and noiseless controls. This is not a claim about smooth disturbances or other seeds.", '',
             'The objective is the unchanged Stage 4A innovation quadratic over a fixed requested set. Both requested experiments use the identical saved Stage 4A seed-7 truth/observations and zero initial coefficients. All priors, commands, associations and measurement data remain fixed during each fit.', '',
             '## Configuration and policy', '', '```json', json.dumps(payload['configuration'], indent=2), '```', '',
             'L-BFGS-B parameter bounds are [-0.5,0.5]. Preflight checks include the initial point, axis endpoints, sign patterns and eight seed-7 interior points. These deterministic probes do not prove validity throughout the box. Every SciPy request is checked; an invalid candidate aborts the run without returning a partial cost, NaN gradient or finite penalty. No automatic domain adjustment is implemented.', '',
             'Only J is minimized. Log determinant and Gaussian sums are diagnostic. Constant parameters z=[a,b] map to c=[a*ones(7); b*ones(7)], and JAX differentiates this composition.', '',
             '## Files added or changed', '',
             '- `estimation/coefficient_optimization.py`: checked SciPy interface, constant mapping, feasibility probes, optimization histories, reconstruction metrics and local sensitivities.',
             '- `benchmarks/milestone4b.py`: paired reproducible experiments, diagnostic controls, plots and reports.',
             '- `tests/test_coefficient_optimization.py`, `estimation/STAGE4B.md`, `project.md`: validation and documented scope/status.',
             '- `artifacts/milestone4b/`: report, validation record, JSON results, saved datasets, sensitivities, dense disturbance curves and figures.', '',
             '## Optimization results', '',
             '| Fit | Status | Initial J | Final J | Iterations | Function / gradient calls | Raw gradient L2 | Projected gradient infinity norm |',
             '|---|---|---:|---:|---:|---|---:|---:|']
    for name, result in payload['experiments'].items():
        lines.append(f"| {name} | {result['status']}: {result['message']} | {result['objective_initial']} | {result['objective_final']} | {result['iterations']} | {result['function_evaluations']} / {result['gradient_evaluations']} | {result.get('gradient_l2')} | {result.get('projected_gradient_inf')} |")
        lines += ['', f"**{name} coefficients:** `{result['coefficients']}`.", '',
                  f"Objective reduction: `{result['objective_reduction']}`.", '',
                  f"Disturbance RMSE / maximum absolute error, [linear, angular]: `{result.get('disturbance_metrics')}`.", '',
                  f"Final position error norm / signed wrapped heading error: `{result.get('final_candidate', {}).get('final_position_error_norm')}` m / `{result.get('final_candidate', {}).get('final_pose_error_x_y_heading', [None]*3)[2]}` rad.", '',
                  f"Timing (seconds): `{result['timing']}`.", '',
                  f"Invalid trials: `{result['invalid_trials']}`. Feasibility probes: {len(result['feasibility_probes'])}.", '']
        if name == 'constant_constrained' and result['parameters'] is not None:
            errors = np.asarray(result['parameters'])-np.asarray(payload['configuration']['true_disturbance_v_omega'])
            lines += [f"Estimated [a,b]: `{result['parameters']}`; signed errors relative to truth: `{errors.tolist()}`.", '']
    lines += ['## Likelihood diagnostics', '', '| Candidate | J | Sum log determinant | J + sum log determinant |', '|---|---:|---:|---:|']
    for name, result in payload['likelihood_diagnostics'].items():
        lines.append(f"| {name} | {result['innovation_quadratic']:.12g} | {result['logdet_diagnostic']:.12g} | {result['gaussian_diagnostic']:.12g} |")
    lines += ['', '## Local identifiability', '',
              'D is the Jacobian of stacked prior innovations with respect to c, evaluated by JAX forward differentiation. Each 2-row block is whitened as solve(cholesky(S_k), D_k), with S fixed at that candidate for this diagnostic. The optimization itself retains dS/dc. These local linearized sensitivities do not prove global identifiability or provide calibrated uncertainty for this experiment.', '',
              'Numerical rank threshold: max(rows,columns)*float64_epsilon*sigma_max. A separate practical rank uses 1e-3*sigma_max; it is a scale-relative conditioning diagnostic, not a statistical identifiability threshold.', '',
              '| Evaluation | Matrix | Shape | Numerical rank | Rank tolerance | Practical rank | Condition number | Singular values |', '|---|---|---|---:|---:|---:|---:|---|']
    for name, result in payload['local_sensitivities'].items():
        for kind in ('raw', 'whitened'):
            item = result[kind]
            lines.append(f"| {name} | {kind} | {item['shape']} | {item['numerical_rank']} | {item['rank_tolerance']:.3e} | {item['practical_rank']} | {item['condition_number']} | {item['singular_values']} |")
    lines += ['', 'Full matrices and dense reconstruction samples are saved in sensitivities.npz and disturbance_curves.npz. Accepted-iteration and trial-evaluation histories, convergence status, coefficients, final poses/covariances/errors, repetition results and all timings are in results.json.', '',
              '## Plots', '', '![True and fitted disturbances](disturbances.png)', '', '![Objective history](objective_history.png)', '',
              '## Timing interpretation', '',
              'Explicit lower/compile times contain compilation without execution. The checked-reference first-call time may include auxiliary JAX VJP compilation and is labelled separately. Preparation includes compilation, warmup and probes. Optimization runtime starts only after preparation, using the warmed compiled pair and host validity checks. These measurements cover a fixed-window fit, not moving-horizon update latency.', '',
              '## Current test record', '',
              f"Completion evidence checks: `{completion_checks(payload)}`.", '',
              'This benchmark does not execute pytest. The following record is supplied separately; null means no current test record was supplied.', '',
              '```json', json.dumps(payload['current_test_record'], indent=2), '```', '',
              '## Scope and remaining validation', '',
              'Additional interpretation, noise-free controls, completion-gate assessment and preservation audit are recorded in validation.md. Stage 4C, moving windows, nearest-neighbor association and Victoria Park are not implemented.', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/milestone4b'))
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--test-record', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args.output, args.repeats, args.test_record)


if __name__ == '__main__':
    main()
