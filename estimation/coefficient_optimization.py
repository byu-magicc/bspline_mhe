"""Stage 4B bounded coefficient fitting around the unchanged Stage 4A EKF.

Only the innovation quadratic is minimized. Invalid candidates abort a run;
feasibility probes are deterministic samples, not certification of a box.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from estimation.fixed_window_objective import (
    InvalidCandidateError, checked_evaluation, checked_value_and_grad,
)


DEFAULT_OPTIONS = dict(ftol=1e-12, gtol=1e-6, maxiter=500, maxfun=2000, maxls=40, maxcor=10)


def _record_parameters(parameters):
    """Undefined rejected parameters use JSON null, not a numerical penalty."""
    return [float(value) if np.isfinite(value) else None for value in np.asarray(parameters)]


def constant_to_spline(parameters, coefficients_per_channel):
    """Differentiable c=[a*ones(n); b*ones(n)] from z=[a,b]."""
    parameters = jnp.asarray(parameters, dtype=jnp.float64)
    if parameters.shape != (2,) or coefficients_per_channel < 1:
        raise ValueError('constant mapping requires z=(a,b) and n >= 1')
    return jnp.repeat(parameters, coefficients_per_channel)


def feasibility_probes(bounds, random_count=8, seed=7):
    """Axis endpoints, four sign patterns, and seeded interior samples."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('bounds must be finite (parameters,2) with lower < upper')
    if random_count < 0:
        raise ValueError('random_count must be nonnegative')
    lower, upper = bounds.T
    center = (lower+upper)/2
    points = [center]
    for axis in range(len(center)):
        for endpoint in (lower[axis], upper[axis]):
            point = center.copy()
            point[axis] = endpoint
            points.append(point)
    alternating = np.arange(len(center)) % 2 == 0
    points += [lower, upper, np.where(alternating, lower, upper), np.where(alternating, upper, lower)]
    points += list(np.random.default_rng(seed).uniform(lower, upper, size=(random_count, len(center))))
    return np.asarray(list({tuple(point): point for point in points}.values()))


def gradient_norms(x, gradient, bounds):
    lower, upper = np.asarray(bounds).T
    gradient = np.asarray(gradient)
    projected = gradient.copy()
    projected[((x <= lower+1e-10) & (gradient > 0)) | ((x >= upper-1e-10) & (gradient < 0))] = 0.
    return {'gradient_l2': float(np.linalg.norm(gradient)),
            'gradient_inf': float(np.max(np.abs(gradient))),
            'projected_gradient_inf': float(np.max(np.abs(projected)))}


def candidate_status(result):
    fields = ('candidate_valid', 'numerical_valid', 'states_finite', 'covariances_symmetric',
              'covariances_psd', 'positive_definite')
    counts = ('requested_count', 'valid_count', 'masked_count', 'rejected_count',
              'invalid_geometry_count', 'numerical_rejection_count')
    return {**{key: bool(getattr(result, key)) for key in fields},
            **{key: int(getattr(result, key)) for key in counts}}


class CheckedScipyObjective:
    """Prepared SciPy value/gradient pair with host-side checks on every trial.

    Explicit lower/compile timings exclude execution. A separate cold checked
    reference call may compile JAX's auxiliary VJP; that combined first-call
    time is reported separately and is excluded from optimization runtime.
    """

    def __init__(self, evaluator, mapping, initial):
        self.mapping = mapping
        self.evaluate = jax.jit(lambda z: evaluator(mapping(z)))
        self.value_gradient = jax.jit(jax.value_and_grad(lambda z: evaluator(mapping(z)).innovation_quadratic))
        start = time.perf_counter()
        self.compiled_evaluate = self.evaluate.lower(jnp.asarray(initial)).compile()
        evaluate_compile = time.perf_counter()-start
        start = time.perf_counter()
        self.compiled_value_gradient = self.value_gradient.lower(jnp.asarray(initial)).compile()
        self.timings = {'evaluation_compile_seconds': evaluate_compile,
                        'value_gradient_compile_seconds': time.perf_counter()-start}
        self.invalid_trials = []
        self.evaluation_history = []
        self.cache = {}
        self.gradient_returns = 0

    def check(self, x, phase):
        x = np.asarray(x, dtype=np.float64)
        try:
            return checked_evaluation(self.compiled_evaluate, x)
        except InvalidCandidateError as error:
            result = self.compiled_evaluate(x)
            self.invalid_trials.append({'phase': phase, 'parameters': _record_parameters(x),
                                        'coefficients': _record_parameters(self.mapping(x)),
                                        'diagnostics': candidate_status(result), 'message': str(error)})
            raise

    def reference_check(self, initial):
        self.check(initial, 'initial')
        start = time.perf_counter()
        try:
            value, gradient = checked_value_and_grad(self.evaluate, jnp.asarray(initial))
        except InvalidCandidateError as error:
            self.invalid_trials.append({'phase': 'initial_gradient', 'parameters': initial.tolist(),
                                        'diagnostics': candidate_status(self.compiled_evaluate(initial)),
                                        'message': str(error)})
            raise
        value, gradient = float(value), np.asarray(gradient)
        self.timings['checked_reference_first_call_seconds_including_auxiliary_compilation'] = time.perf_counter()-start
        start = time.perf_counter()
        actual_value, actual_gradient = self.compiled_value_gradient(initial)
        actual_value, actual_gradient = float(actual_value), np.asarray(actual_gradient)
        self.timings['compiled_value_gradient_warmup_seconds'] = time.perf_counter()-start
        if not np.allclose(actual_gradient, gradient, rtol=1e-10, atol=1e-10) or not np.isclose(actual_value, value, rtol=1e-12, atol=1e-12):
            raise RuntimeError('Compiled SciPy pair disagrees with checked Stage 4A reference')
        self.reference_errors = {'value_absolute': abs(actual_value-value),
                                 'gradient_max_absolute': float(np.max(np.abs(actual_gradient-gradient)))}

    def __call__(self, x, phase='optimization'):
        x = np.asarray(x, dtype=np.float64)
        start = time.perf_counter()
        result = self.check(x, phase)  # Always validate, including a repeated point.
        check_seconds = time.perf_counter()-start
        start = time.perf_counter()
        value, gradient = self.compiled_value_gradient(x)
        value, gradient = float(value), np.asarray(gradient, dtype=np.float64)
        gradient_seconds = time.perf_counter()-start
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            message = 'Nonfinite compiled objective/gradient; optimization must stop'
            self.invalid_trials.append({'phase': phase, 'parameters': x.tolist(),
                                        'diagnostics': candidate_status(result), 'message': message})
            raise InvalidCandidateError(message)
        if not np.isclose(value, float(result.innovation_quadratic), rtol=1e-12, atol=1e-12):
            raise RuntimeError('Scalar and checked objective values disagree')
        self.gradient_returns += int(phase == 'optimization')
        self.evaluation_history.append({'phase': phase, 'parameters': x.tolist(), 'objective': value,
                                        'validity_check_seconds': check_seconds,
                                        'value_gradient_seconds': gradient_seconds})
        self.cache[tuple(x)] = (value, gradient.copy())
        return value, gradient


def optimize_coefficients(evaluator, initial, *, coefficients_per_channel=7, constant=False,
                          bounds=None, options=None, random_probe_count=8):
    """L-BFGS-B fit with clean stop on the first invalid probe or SciPy trial.

    The returned point after an invalid trial is the last accepted iterate,
    never an invalid trial or a claimed solution. No objective/gradient is
    returned to SciPy for an invalid request. No automatic domain revision.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('Stage 4B requires JAX float64')
    initial = np.asarray(initial, dtype=np.float64)
    size = 2 if constant else 2*coefficients_per_channel
    if initial.shape != (size,) or not np.isfinite(initial).all():
        raise ValueError(f'initial must be a finite vector of length {size}')
    bounds = np.tile([-.5, .5], (size, 1)) if bounds is None else np.asarray(bounds, dtype=np.float64)
    probes = feasibility_probes(bounds, random_probe_count)
    if bounds.shape != (size, 2) or np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ValueError('initial must lie inside matching parameter bounds')
    mapping = ((lambda z: constant_to_spline(z, coefficients_per_channel)) if constant else
               (lambda z: jnp.asarray(z, dtype=jnp.float64)))
    preparation_start = time.perf_counter()
    bridge = CheckedScipyObjective(evaluator, mapping, initial)
    output = {'method': 'L-BFGS-B', 'parameterization': 'constant' if constant else 'full_spline',
              'bounds': bounds.tolist(), 'options': {**DEFAULT_OPTIONS, **(options or {})},
              'initial_parameters': initial.tolist(), 'success': False,
              'feasibility_probes': [], 'invalid_trials': bridge.invalid_trials,
              'history': [], 'evaluation_history': bridge.evaluation_history,
              'timing': bridge.timings, 'reference_consistency': None,
              'parameters': None, 'coefficients': None, 'objective_initial': None,
              'objective_final': None, 'objective_reduction': None,
              'iterations': 0, 'function_evaluations': 0, 'gradient_evaluations': 0,
              'scipy_status': None}
    try:
        bridge.reference_check(initial)
        output['reference_consistency'] = bridge.reference_errors
        initial_value, initial_gradient = bridge(initial, 'initial')
        output['objective_initial'] = initial_value
        output.update(parameters=initial.tolist(), coefficients=np.asarray(mapping(initial)).tolist(),
                      objective_final=initial_value, objective_reduction=0.,
                      **gradient_norms(initial, initial_gradient, bounds))
        for point in probes:
            try:
                result = bridge.check(point, 'feasibility_probe')
            except InvalidCandidateError:
                output['feasibility_probes'].append({'parameters': point.tolist(), 'valid': False})
                raise
            output['feasibility_probes'].append({'parameters': point.tolist(), 'valid': True,
                                                'objective': float(result.innovation_quadratic)})
    except InvalidCandidateError as error:
        output.update(status='stopped_invalid_preflight', message=str(error))
        bridge.timings['preparation_seconds'] = time.perf_counter()-preparation_start
        bridge.timings['optimization_seconds'] = 0.
        return output
    bridge.timings['preparation_seconds'] = time.perf_counter()-preparation_start
    optimization_start = time.perf_counter()
    last_accepted = initial.copy()

    def record(x, value, gradient, iteration):
        return {'iteration': iteration, 'parameters': x.tolist(),
                'coefficients': np.asarray(mapping(x)).tolist(), 'objective': float(value),
                'elapsed_optimization_seconds': time.perf_counter()-optimization_start,
                **gradient_norms(x, gradient, bounds)}

    output['history'].append(record(initial, initial_value, initial_gradient, 0))

    def callback(intermediate_result):
        nonlocal last_accepted
        x = np.asarray(intermediate_result.x).copy()
        value, gradient = bridge.cache[tuple(x)]
        output['history'].append(record(x, value, gradient, len(output['history'])))
        last_accepted = x

    try:
        result = minimize(bridge, initial, method='L-BFGS-B', jac=True,
                          bounds=bounds, callback=callback, options=output['options'])
        last_accepted = np.asarray(result.x)
        output.update(status='converged' if result.success else 'scipy_not_converged',
                      success=bool(result.success), message=str(result.message),
                      iterations=int(result.nit), scipy_status=int(result.status),
                      function_evaluations=int(result.nfev), gradient_evaluations=int(result.njev))
    except InvalidCandidateError as error:
        output.update(status='stopped_invalid_trial', success=False,
                      message=f'{error}. Search region or line-search strategy requires revision.',
                      iterations=len(output['history'])-1,
                      function_evaluations=sum(r['phase']=='optimization' for r in bridge.evaluation_history)+1,
                      gradient_evaluations=bridge.gradient_returns)
    bridge.timings['optimization_seconds'] = time.perf_counter()-optimization_start
    for field in ('validity_check_seconds', 'value_gradient_seconds'):
        samples = [row[field] for row in bridge.evaluation_history if row['phase'] == 'optimization']
        bridge.timings[field+'_mean'] = float(np.mean(samples)) if samples else None
        bridge.timings[field+'_p95'] = float(np.percentile(samples, 95)) if samples else None
    final_value, final_gradient = bridge.cache[tuple(last_accepted)]
    output.update(parameters=last_accepted.tolist(), coefficients=np.asarray(mapping(last_accepted)).tolist(),
                  objective_final=final_value, objective_reduction=initial_value-final_value,
                  **gradient_norms(last_accepted, final_gradient, bounds))
    return output


def disturbance_metrics(basis, coefficients, truth):
    """Dense-grid sample RMSE and max absolute error for each channel."""
    from splines import disturbance_from_coefficients
    estimated = np.asarray(disturbance_from_coefficients(basis, coefficients))
    truth = np.broadcast_to(np.asarray(truth), estimated.shape)
    error = estimated-truth
    return {'rmse_v_omega': np.sqrt(np.mean(error**2, axis=0)).tolist(),
            'max_absolute_error_v_omega': np.max(np.abs(error), axis=0).tolist()}


def singular_value_diagnostics(matrix, practical_relative_tolerance=1e-3):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not min(matrix.shape) or not np.isfinite(matrix).all():
        raise ValueError('sensitivity matrix must be finite, nonempty and two-dimensional')
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    maximum = singular_values[0]
    tolerance = max(matrix.shape)*np.finfo(np.float64).eps*maximum
    practical = practical_relative_tolerance*maximum
    return {'shape': list(matrix.shape), 'singular_values': singular_values.tolist(),
            'rank_tolerance': float(tolerance), 'numerical_rank': int(np.sum(singular_values > tolerance)),
            'rank_rule': 'max(rows,columns)*float64_epsilon*largest_singular_value',
            'practical_relative_tolerance': practical_relative_tolerance,
            'practical_absolute_tolerance': float(practical),
            'practical_rank': int(np.sum(singular_values > practical)),
            'condition_number': float(maximum/singular_values[-1]) if singular_values[-1] > 0 else None}


def innovation_sensitivity(evaluator, coefficients):
    """Jacobian of fixed requested innovations and per-block Cholesky whitening.

    Whitening holds S at this evaluation; its derivatives are not included in
    this local mean-sensitivity diagnostic. The optimized J retains dS/dc.
    """
    evaluate = jax.jit(evaluator)
    result = checked_evaluation(evaluate, coefficients)
    indices = np.flatnonzero(np.asarray(result.valid_mask).ravel())
    if len(indices) == 0:
        raise ValueError('identifiability requires at least one requested observation')
    stacked = lambda c: evaluator(c).innovations.reshape(-1, 2)[indices].reshape(-1)
    jacobian = jax.jit(jax.jacfwd(stacked))
    start = time.perf_counter()
    compiled = jacobian.lower(jnp.asarray(coefficients)).compile()
    compile_seconds = time.perf_counter()-start
    start = time.perf_counter()
    D = np.asarray(compiled(coefficients))
    execution_seconds = time.perf_counter()-start
    S = np.asarray(result.innovation_covariances).reshape(-1, 2, 2)[indices]
    whitened = np.linalg.solve(np.linalg.cholesky(S), D.reshape(len(indices), 2, -1)).reshape(D.shape)
    return {'raw': singular_value_diagnostics(D), 'whitened': singular_value_diagnostics(whitened),
            'compile_seconds': compile_seconds, 'execution_seconds': execution_seconds}, D, whitened
