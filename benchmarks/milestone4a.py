"""Reproducible Stage 4A diagnostics; this command does not execute pytest."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from estimation.fixed_window_objective import (
    innovation_quadratic, make_fixed_window_objective,
)
from estimation.propagation import rk4_stage_times
from simulation.continuous_unicycle import simulate_known_landmarks
from splines import UniformBasis, disturbance_from_coefficients


GRADIENT_ATOL = 2e-5
GRADIENT_RTOL = 2e-5


def summarize_candidate(result, truth):
    """Invalid candidates retain status/counts, never success statistics."""
    summary = {
        'candidate_valid': bool(result.candidate_valid),
        'requested_observations': int(result.requested_count),
        'valid_observations': int(result.valid_count),
        'masked_observations': int(result.masked_count),
        'rejected_observations': int(result.rejected_count),
        'invalid_geometry_observations': int(result.invalid_geometry_count),
        'numerical_rejections': int(result.numerical_rejection_count),
        'states_finite': bool(result.states_finite),
        'covariances_symmetric': bool(result.covariances_symmetric),
        'covariances_psd': bool(result.covariances_psd),
        'positive_definite_innovation_covariances': bool(result.positive_definite),
        'innovation_statistics': None,
    }
    if not summary['candidate_valid']:
        summary['status'] = 'INVALID: objective undefined; success statistics suppressed'
        return summary
    valid = np.asarray(result.valid_mask)
    e = np.asarray(result.innovations)[valid]
    S = np.asarray(result.innovation_covariances)[valid]
    error = np.asarray(result.final_pose)-truth[-1]
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    summary.update({
        'status': 'valid',
        'innovation_quadratic': float(result.innovation_quadratic),
        'logdet_diagnostic': float(result.logdet_diagnostic),
        'gaussian_diagnostic': float(result.gaussian_diagnostic),
        'final_pose': np.asarray(result.final_pose).tolist(),
        'final_covariance': np.asarray(result.final_covariance).tolist(),
        'final_pose_error_x_y_heading': error.tolist(),
        'final_position_error_norm': float(np.linalg.norm(error[:2])),
    })
    if len(e):
        summary['innovation_statistics'] = {
            'count': len(e), 'mean_range_bearing': e.mean(axis=0).tolist(),
            'std_range_bearing': e.std(axis=0).tolist(),
            'covariance_mean_diagonal': np.diagonal(S, axis1=1, axis2=2).mean(axis=0).tolist(),
            'covariance_minimum_eigenvalue': float(np.linalg.eigvalsh(S).min()),
            'covariance_maximum_eigenvalue': float(np.linalg.eigvalsh(S).max()),
        }
    quadratic = sum(float(x @ np.linalg.solve(cov, x)) for x, cov in zip(e, S))
    logdet = sum(float(np.linalg.slogdet(cov)[1]) for cov in S)
    summary['numpy_consistency'] = {
        'quadratic_abs_error': abs(quadratic-float(result.innovation_quadratic)),
        'logdet_abs_error': abs(logdet-float(result.logdet_diagnostic)),
        'passed': bool(np.isclose(quadratic, result.innovation_quadratic, rtol=2e-13, atol=2e-12)
                       and np.isclose(logdet, result.logdet_diagnostic, rtol=2e-13, atol=2e-12)),
    }
    return summary


def gradient_check(record):
    gradient = np.asarray(record['jax_gradient'], dtype=np.float64)
    fd = np.asarray(record['central_difference_gradient'], dtype=np.float64)
    if gradient.shape != fd.shape or gradient.size == 0:
        return False
    absolute = np.abs(gradient-fd)
    return bool(record['all_candidates_valid'] and np.isfinite(gradient).all()
                and np.isfinite(fd).all() and np.all(absolute <= GRADIENT_ATOL+GRADIENT_RTOL*np.abs(fd))
                and absolute.max() < GRADIENT_ATOL)


def validation_summary(payload):
    candidates = payload['candidate_results']
    checks = payload['gradient_validation']['results']
    return {
        'candidates': bool(candidates) and all(x['candidate_valid'] for x in candidates.values()),
        'numpy_consistency': bool(candidates) and all(x.get('numpy_consistency', {}).get('passed', False)
                                                      for x in candidates.values()),
        'constant_reconstruction': bool(np.isfinite(np.asarray(payload['basis_constant_reconstruction_max_abs_error'], dtype=float))
                                       and np.asarray(payload['basis_constant_reconstruction_max_abs_error'], dtype=float) <= 2e-15),
        'gradients': bool(checks) and all(bool(steps) and all(gradient_check(r) for r in steps.values())
                                        for steps in checks.values()),
    }


def _json_safe(value):
    """Undefined numerical results use JSON null, never nonstandard NaN tokens."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _number(value):
    return f'{value:.6e}' if value is not None and np.isfinite(value) else 'undefined'


def run(output, test_record=None):
    jax.config.update('jax_enable_x64', True)
    output.mkdir(parents=True, exist_ok=True)
    L, dt, interval = 5., .025, .1
    degree, M, seed = 3, 4, 7
    command_values, disturbance = (1., .2), (.15, -.05)
    R = np.diag([.05**2, np.deg2rad(1.)**2])
    Qc = np.diag([.01, .0025])
    P0 = np.diag([.05**2, .05**2, np.deg2rad(2.)**2])
    data = simulate_known_landmarks(measurement_covariance=R, seed=seed)
    adapter = UniformBasis(degree, M)
    B = adapter.precompute_jax(rk4_stage_times(data.times), L)
    n = adapter.num_coefficients
    candidates = {'zero': jnp.zeros(2*n), 'true': jnp.r_[jnp.full(n, .15), jnp.full(n, -.05)]}
    evaluator = make_fixed_window_objective(
        (jnp.zeros(3), P0), data.times, lambda t: jnp.asarray(command_values), B,
        Qc, data.measurements, data.landmarks_at_times, R, data.mask, coefficient_count=2*n)
    evaluate = jax.jit(evaluator)
    results = {name: summarize_candidate(evaluate(c), data.truth) for name, c in candidates.items()}
    reconstructed = disturbance_from_coefficients(adapter.precompute_jax(np.linspace(0., L, 101), L), candidates['true'])
    reconstruction_error = float(np.max(np.abs(np.asarray(reconstructed)-np.asarray(disturbance))))
    scalar = lambda c: innovation_quadratic(evaluator, c)
    gradient = jax.jit(jax.grad(scalar))
    batch = jax.jit(jax.vmap(lambda c: (scalar(c), evaluator(c).candidate_valid)))
    fd = {}
    for name, c in {**candidates, 'nonconstant': jnp.linspace(-.08, .11, 2*n)}.items():
        actual = np.asarray(gradient(c))
        fd[name] = {}
        for step in (1e-4, 1e-5):
            eye = jnp.eye(2*n)*step
            values, valid = batch(jnp.concatenate((c[None]+eye, c[None]-eye)))
            reference = np.asarray((values[:2*n]-values[2*n:])/(2*step))
            error = np.abs(actual-reference)
            record = {
                'jax_gradient': actual.tolist(), 'central_difference_gradient': reference.tolist(),
                'all_candidates_valid': bool(evaluate(c).candidate_valid) and bool(jnp.all(valid)),
                'max_absolute_error': float(error.max()),
                'max_relative_error': float((error/np.maximum(np.abs(reference), 1e-5)).max()),
            }
            record['passed'] = gradient_check(record)
            fd[name][str(step)] = record
    configuration = {
        'window_length_seconds': L, 'integration_step_seconds': dt,
        'measurement_interval_seconds': interval, 'degree': degree, 'segments': M,
        'coefficients_per_channel': n, 'command_v_omega': list(command_values),
        'true_disturbance_v_omega': list(disturbance), 'seed': seed,
        'initial_pose': [0., 0., 0.], 'arrival_covariance': P0.tolist(),
        'measurement_covariance': R.tolist(), 'continuous_input_noise_spectral_density_Qc': Qc.tolist(),
        'landmark_coordinates': data.landmark_map.tolist(), 'covariance_tolerance': 1e-10,
        'observation_policy': 'all six landmarks at positive 0.1 s multiples; no arrival-time observation',
        'jax_enable_x64': bool(jax.config.x64_enabled),
    }
    payload = {'configuration': configuration, 'candidate_results': results,
               'basis_constant_reconstruction_max_abs_error': reconstruction_error,
               'gradient_validation': {'finite_difference_steps': [1e-4, 1e-5],
                                       'atol': GRADIENT_ATOL, 'rtol': GRADIENT_RTOL, 'results': fd},
               'test_execution': {'executed_by_benchmark': False,
                                  'historical_record': 'Original Stage 4A run: 7 focused tests and 205 total tests passed; see artifacts/milestone4a/report.md.',
                                  'current_run_record': None}}
    baseline = Path('artifacts/milestone4a/results.json')
    if baseline.exists():
        old = json.loads(baseline.read_text())['candidate_results']
        payload['original_experiment_differences'] = {
            name: {key: results[name][key]-old[name][key]
                   for key in ('innovation_quadratic', 'logdet_diagnostic', 'gaussian_diagnostic', 'final_position_error_norm')}
            for name in candidates if results[name]['candidate_valid']}
    if test_record is not None:
        payload['test_execution']['current_run_record'] = json.loads(test_record.read_text())
    payload['validation_checks'] = validation_summary(payload)
    payload = _json_safe(payload)
    (output/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    write_report(output/'report.md', payload)
    print(json.dumps({'validation_checks': payload['validation_checks'], 'candidate_results': results}, indent=2))
    return payload


def write_report(path, payload):
    checks = validation_summary(payload)
    passed = all(checks.values())
    cfg = payload['configuration']
    lines = ['# Stage 4A hardening report', '',
             f"Numerical benchmark validation: **{'PASSED' if passed else 'FAILED'}**. This verdict uses candidate status, independent NumPy accumulation, constant reconstruction, and the recorded gradient vectors.", '',
             '## Files changed', '',
             '- `estimation/fixed_window_objective.py`: guarded orchestration using the preserved RK4 and Joseph update; fixed requested-set domain, node/posterior checks, observation masks/counts, checked Python interfaces.',
             '- `simulation/continuous_unicycle.py`: stable midpoint/sinc truth and stricter time-grid checks.',
             '- `benchmarks/milestone4a.py`: validity-aware statistics and evidence-based reporting.',
             '- `tests/test_fixed_window_objective.py`, `tests/test_stage4a_hardening.py`, `tests/test_milestone4a_reporting.py`: domain, independent analytic, decomposition, simulation, reporting, and preservation checks.',
             '- `estimation/STAGE4A.md`, `project.md`: domain requirements and hardening record.',
             '- `artifacts/milestone4a_hardening/report.md`, `results.json`, `validation.md`, `validation.json`: configuration, numerical results, current test record, independent checks and limitations.', '',
             '## Objective and valid domain', '',
             '`J(c) = sum_{k in K} e_k.T @ solve(S_k, e_k)` uses the requested set K fixed before evaluation. Every requested observation must be accepted for J to be defined. No regularization, log determinant, finite invalid-candidate penalty, covariance projection or jitter is added. Log determinant and J + log determinant remain diagnostics only.', '',
             'The JAX evaluator returns explicit status and NaN costs outside the domain. Checked Python evaluation/value-gradient functions raise `InvalidCandidateError` before an invalid trial can be used. RK4 propagation and the original Joseph updates remain unmodified. Numerical failures quarantine subsequent arithmetic; internal placeholders from invalid results are not estimates.', '',
             'Every integration-node prior and sequential posterior is checked for finite pose/covariance, absolute covariance asymmetry <= 1e-10 and minimum eigenvalue >= -1e-10. Innovation covariances must be symmetric and strictly positive definite. The absolute tolerance matches the validated Stage 3B.2 numerical policy, admits float64 roundoff, and does not repair a matrix. Eigenvalue scaling is diagnostic only.', '',
             '## Reproducible configuration', '', '```json', json.dumps(cfg, indent=2), '```', '',
             'The truth is a deterministic analytic mean trajectory. Qc is the estimator’s configured uncertainty assumption; it does not inject stochastic noise into the truth. Measurement noise is seeded independently.', '',
             '## Candidate results', '',
             '| Candidate | Status | Requested / valid / masked / rejected | J | Log determinant | Gaussian diagnostic |',
             '|---|---|---|---:|---:|---:|']
    for name, result in payload['candidate_results'].items():
        counts = ' / '.join(str(result[key]) for key in ('requested_observations', 'valid_observations', 'masked_observations', 'rejected_observations'))
        values = [f"{result[key]:.12g}" if result['candidate_valid'] else 'undefined'
                  for key in ('innovation_quadratic', 'logdet_diagnostic', 'gaussian_diagnostic')]
        lines.append(f"| {name} | {result['status']} | {counts} | {' | '.join(values)} |")
    lines += ['', 'Invalid and masked slots never enter innovation statistics. Statistics are suppressed entirely for invalid candidates. Final states/covariances, wrapped errors and valid-only innovation statistics are stored in `results.json`.', '',
              f"Constant reconstruction maximum error: `{_number(payload['basis_constant_reconstruction_max_abs_error'])}`.", '',
              '## Gradient checks', '', '| Vector | FD step | Max absolute error | Max relative error* | Status |', '|---|---:|---:|---:|---|']
    for name, records in payload['gradient_validation']['results'].items():
        for step, record in records.items():
            actual = np.asarray(record['jax_gradient'], dtype=float)
            reference = np.asarray(record['central_difference_gradient'], dtype=float)
            error = np.abs(actual-reference)
            absolute = float(error.max()) if error.size else None
            relative = float((error/np.maximum(np.abs(reference), 1e-5)).max()) if error.size else None
            lines.append(f"| {name} | {step} | {_number(absolute)} | {_number(relative)} | {'passed' if gradient_check(record) else 'FAILED'} |")
    lines += ['', '*Relative denominator is max(|FD|, 1e-5). Checks cover every coefficient in both channels with float64, steps 1e-4/1e-5, absolute/relative tolerances 2e-5 and maximum absolute discrepancy < 2e-5. All base and perturbed candidates must be valid.', '',
              f'Actual validation checks: `{checks}`.', '', '## Preservation of the original experiment', '',
              'Differences from the original saved Stage 4A results (new minus original):', '', '```json',
              json.dumps(payload.get('original_experiment_differences', {}), indent=2), '```', '',
              '## Test execution provenance', '',
              'This benchmark command does not execute pytest. A benchmark numerical verdict is not evidence that the test suite ran.', '',
              payload['test_execution']['historical_record'], '',
              'Current run test record supplied separately:', '', '```json',
              json.dumps(payload['test_execution']['current_run_record'], indent=2), '```', '',
              '## Remaining limitations', '',
              'The checked interfaces are the Stage 4B boundary. A future optimizer must reject invalid trials or stay in the valid domain, without changing K or inventing a finite penalty. Angular wrap and geometry boundaries remain nonsmooth. Invalid candidates have no defined objective gradient; checked evaluation rejects them before differentiation. Checks cover RK4 integration nodes and sequential posteriors, not every internal RK4 stage or continuous time between nodes. Step refinement remains necessary for accuracy.', '',
              'No optimization, moving window, association search, or SLAM augmentation is implemented here. Stage 4B remains planned. See `validation.md` for the hardening test outcomes and review-issue resolution.', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/milestone4a_hardening'))
    parser.add_argument('--test-record', type=Path)
    args = parser.parse_args()
    run(args.output, args.test_record)


if __name__ == '__main__':
    main()
