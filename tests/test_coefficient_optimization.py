"""Stage 4B mapping, checked SciPy bridge, optimization and local sensitivity."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.coefficient_optimization import (
    CheckedScipyObjective, constant_to_spline, disturbance_metrics, feasibility_probes,
    gradient_norms, innovation_sensitivity, optimize_coefficients, singular_value_diagnostics,
)
from estimation.fixed_window_objective import make_fixed_window_objective
from estimation.propagation import rk4_stage_times
from splines import UniformBasis


@pytest.fixture(scope='module', autouse=True)
def precision():
    with jax.experimental.enable_x64():
        yield


class TestResult(NamedTuple):
    __test__ = False
    innovation_quadratic: object
    candidate_valid: object
    numerical_valid: object
    states_finite: object
    covariances_symmetric: object
    covariances_psd: object
    positive_definite: object
    requested_count: object
    valid_count: object
    masked_count: object
    rejected_count: object
    invalid_geometry_count: object
    numerical_rejection_count: object
    innovations: object
    innovation_covariances: object
    valid_mask: object


def linear_evaluator(A, target, covariance=None, invalid=None):
    A, target = jnp.asarray(A), jnp.asarray(target)
    count = A.shape[0]//2
    noise = jnp.broadcast_to(jnp.eye(2) if covariance is None else jnp.asarray(covariance), (count, 2, 2))

    def evaluate(c):
        e = (A@c-target).reshape(count, 2)
        J = jnp.sum(e*jnp.linalg.solve(noise, e[..., None])[..., 0])
        valid = jnp.asarray(True) if invalid is None else ~invalid(c)
        return TestResult(J, valid, True, True, True, True, True, count,
                          jnp.where(valid, count, 0), 0, jnp.where(valid, 0, count),
                          jnp.where(valid, 0, count), 0, e, noise, jnp.ones(count, dtype=bool))
    return evaluate


def test_constant_mapping_values_and_jacobian():
    z = np.array([.15, -.05])
    actual = jax.jit(lambda z: constant_to_spline(z, 7))(z)
    np.testing.assert_array_equal(actual, np.r_[np.full(7, .15), np.full(7, -.05)])
    expected = np.zeros((14, 2))
    expected[:7, 0] = 1.
    expected[7:, 1] = 1.
    np.testing.assert_array_equal(jax.jacfwd(lambda z: constant_to_spline(z, 7))(z), expected)


def test_chain_rule_and_scipy_jax_consistency_on_real_ekf():
    from benchmarks.milestone4b import create_experiment
    _, _, _, evaluator, _ = create_experiment()
    z = np.array([.12, -.04])
    c = constant_to_spline(z, 7)
    full_gradient = jax.jit(jax.grad(lambda c: evaluator(c).innovation_quadratic))(c)
    composed = jax.jit(jax.value_and_grad(lambda z: evaluator(constant_to_spline(z, 7)).innovation_quadratic))
    value, gradient = composed(z)
    expected = np.array([np.sum(full_gradient[:7]), np.sum(full_gradient[7:])])
    np.testing.assert_allclose(gradient, expected, rtol=2e-12, atol=2e-12)
    bridge = CheckedScipyObjective(evaluator, lambda z: constant_to_spline(z, 7), z)
    bridge.reference_check(z)
    actual_value, actual_gradient = bridge(z)
    np.testing.assert_allclose(actual_value, value, rtol=0, atol=1e-11)
    np.testing.assert_allclose(actual_gradient, gradient, rtol=1e-11, atol=1e-11)
    step = 1e-5
    fd = np.array([(float(composed(z+direction)[0])-float(composed(z-direction)[0]))/(2*step)
                   for direction in np.eye(2)*step])
    np.testing.assert_allclose(actual_gradient, fd, rtol=2e-6, atol=2e-6)
    sensitivity, D, W = innovation_sensitivity(evaluator, np.asarray(c))
    assert D.shape == W.shape == (600, 14)
    assert sensitivity['raw']['numerical_rank'] == sensitivity['whitened']['numerical_rank'] == 14


def test_scipy_fit_objective_reduction_and_reproducibility():
    A = np.diag([1., 2., 3., 4.])
    truth = np.array([.15, -.1, .2, -.05])
    evaluator = linear_evaluator(A, A@truth)
    first = optimize_coefficients(evaluator, np.zeros(4), coefficients_per_channel=2)
    second = optimize_coefficients(evaluator, np.zeros(4), coefficients_per_channel=2)
    assert first['success'] and second['success']
    assert first['objective_reduction'] > .1 and first['objective_final'] < 1e-12
    np.testing.assert_allclose(first['parameters'], truth, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(first['parameters'], second['parameters'])
    np.testing.assert_array_equal([row['objective'] for row in first['history']],
                                  [row['objective'] for row in second['history']])
    assert first['function_evaluations'] == first['gradient_evaluations']
    assert first['history'][0]['iteration'] == 0
    assert len(first['history']) == first['iterations']+1
    assert all(row['valid'] for row in first['feasibility_probes'])


def test_constant_fit_uses_composed_gradient():
    truth = np.r_[np.full(7, .15), np.full(7, -.05)]
    evaluator = linear_evaluator(np.eye(14), truth)
    result = optimize_coefficients(evaluator, np.zeros(2), constant=True)
    assert result['success']
    np.testing.assert_allclose(result['parameters'], [.15, -.05], rtol=0, atol=1e-8)
    assert result['objective_final'] < 1e-14


def test_preflight_invalidity_prevents_scipy_call(monkeypatch):
    import estimation.coefficient_optimization as module
    evaluator = linear_evaluator(np.eye(4), np.zeros(4), invalid=lambda c: c[0] > .3)
    monkeypatch.setattr(module, 'minimize', lambda *args, **kwargs: pytest.fail('SciPy must not run after an invalid probe'))
    result = optimize_coefficients(evaluator, np.zeros(4), coefficients_per_channel=2)
    assert not result['success'] and result['status'] == 'stopped_invalid_preflight'
    assert result['invalid_trials'][0]['phase'] == 'feasibility_probe'
    assert result['timing']['optimization_seconds'] == 0.


def test_invalid_initial_has_no_objective_or_scipy_call(monkeypatch):
    import estimation.coefficient_optimization as module
    evaluator = linear_evaluator(np.eye(4), np.zeros(4), invalid=lambda c: jnp.asarray(True))
    monkeypatch.setattr(module, 'minimize', lambda *args, **kwargs: pytest.fail('Invalid initial point cannot enter SciPy'))
    result = optimize_coefficients(evaluator, np.zeros(4), coefficients_per_channel=2)
    assert result['objective_initial'] is None and result['coefficients'] is None
    assert not result['success'] and len(result['invalid_trials']) == 1


def test_nonfinite_initial_gradient_stops_before_scipy(monkeypatch):
    import estimation.coefficient_optimization as module

    @jax.custom_jvp
    def bad_derivative(c):
        return jnp.sum(c**2)

    @bad_derivative.defjvp
    def derivative(primals, tangents):
        return bad_derivative(primals[0]), jnp.nan*jnp.sum(tangents[0])

    base = linear_evaluator(np.eye(4), np.zeros(4))
    evaluator = lambda c: base(c)._replace(innovation_quadratic=bad_derivative(c))
    monkeypatch.setattr(module, 'minimize', lambda *args, **kwargs: pytest.fail('NaN gradient cannot enter SciPy'))
    result = optimize_coefficients(evaluator, np.zeros(4), coefficients_per_channel=2)
    assert not result['success'] and result['status'] == 'stopped_invalid_preflight'
    assert result['invalid_trials'][0]['phase'] == 'initial_gradient'


def test_invalid_scipy_trial_stops_without_returning_any_cost(monkeypatch):
    import estimation.coefficient_optimization as module
    times = np.array([0., 1.])
    basis = UniformBasis(3, 4).precompute_jax(rk4_stage_times(times), 1.)
    measurements = np.ones((2, 2, 2))
    landmarks = np.broadcast_to([[.123, 0.], [3., 1.]], (2, 2, 2))
    evaluator = make_fixed_window_objective(
        (np.zeros(3), np.eye(3)*.01), times, lambda t: jnp.zeros(2), basis,
        np.zeros((2, 2)), measurements, landmarks, np.eye(2)*.1,
        np.array([[False, False], [True, True]]), coefficient_count=14)
    returned = []

    def requested_trial(fun, initial, **kwargs):
        returned.append(fun(np.array([.123, 0.])))  # Landmark collision after propagation.
        pytest.fail('Invalid trial returned a value or gradient to SciPy')

    monkeypatch.setattr(module, 'minimize', requested_trial)
    result = optimize_coefficients(evaluator, np.zeros(2), constant=True)
    assert not returned
    assert result['status'] == 'stopped_invalid_trial' and not result['success']
    assert result['gradient_evaluations'] == 0 and result['function_evaluations'] == 1
    assert 'requires revision' in result['message']
    assert result['invalid_trials'][0]['diagnostics']['invalid_geometry_count'] == 1
    assert result['invalid_trials'][0]['diagnostics']['valid_count'] == 1
    np.testing.assert_array_equal(result['parameters'], np.zeros(2))
    assert all(row['valid'] for row in result['feasibility_probes'])


def test_probe_reproducibility_and_bound_coverage():
    bounds = np.tile([-.5, .5], (14, 1))
    a, b = feasibility_probes(bounds), feasibility_probes(bounds)
    np.testing.assert_array_equal(a, b)
    assert np.all(a >= -.5) and np.all(a <= .5)
    assert np.all(np.min(a, axis=0) == -.5) and np.all(np.max(a, axis=0) == .5)
    # Finite sampled feasibility cannot rule out an invalid interior point.
    assert not np.any(np.all(a == .123, axis=1))


def test_nonfinite_rejected_trial_has_strict_json_diagnostics(monkeypatch):
    import json
    import estimation.coefficient_optimization as module
    evaluator = linear_evaluator(np.eye(4), np.zeros(4),
                                 invalid=lambda c: ~jnp.all(jnp.isfinite(c)))

    def request_nonfinite(fun, initial, **kwargs):
        fun(np.full(4, np.nan))
        pytest.fail('Nonfinite trial must not return a pair to SciPy')

    monkeypatch.setattr(module, 'minimize', request_nonfinite)
    result = optimize_coefficients(evaluator, np.zeros(4), coefficients_per_channel=2)
    assert not result['success'] and result['status'] == 'stopped_invalid_trial'
    assert result['invalid_trials'][0]['parameters'] == [None]*4
    assert result['invalid_trials'][0]['coefficients'] == [None]*4
    json.dumps(result, allow_nan=False)


def test_disturbance_metrics_are_per_channel_and_use_function_values():
    sigma = np.linspace(0., 5., 101)
    B = UniformBasis(3, 4).precompute_jax(sigma, 5.)
    ramp = (5/4)*(np.arange(7)-1)
    c = np.r_[.1*ramp, -.02*ramp]
    result = disturbance_metrics(B, c, [0., 0.])
    expected = np.stack((.1*sigma, -.02*sigma), axis=-1)
    np.testing.assert_allclose(result['rmse_v_omega'], np.sqrt(np.mean(expected**2, axis=0)), rtol=0, atol=2e-16)
    np.testing.assert_allclose(result['max_absolute_error_v_omega'], [.5, .1], rtol=0, atol=2e-16)


def test_rank_thresholds_and_zero_singular_values():
    result = singular_value_diagnostics(np.diag([3., 1., 1e-16]))
    assert result['numerical_rank'] == 2 and result['practical_rank'] == 2
    assert result['rank_tolerance'] == 3*np.finfo(float).eps*3
    assert singular_value_diagnostics(np.zeros((6, 4)))['numerical_rank'] == 0
    with pytest.raises(ValueError):
        singular_value_diagnostics(np.array([[np.nan]]))


def test_sensitivity_dimensions_whitening_and_rank():
    A = np.linalg.qr(np.random.default_rng(7).normal(size=(600, 14)))[0]
    R = np.array([[.04, .006], [.006, .01]])
    evaluator = linear_evaluator(A, np.linspace(-.1, .1, 600), covariance=R)
    summary, D, W = innovation_sensitivity(evaluator, np.zeros(14))
    assert D.shape == W.shape == (600, 14)
    assert summary['raw']['numerical_rank'] == summary['whitened']['numerical_rank'] == 14
    np.testing.assert_allclose(D, A, rtol=0, atol=1e-15)
    blocks = A.reshape(300, 2, 14)
    gram = sum(block.T@np.linalg.solve(R, block) for block in blocks)
    np.testing.assert_allclose(W.T@W, gram, rtol=3e-14, atol=3e-14)


def test_projected_gradient_distinguishes_active_bound_optimum():
    norms = gradient_norms(np.array([-.5, .5]), np.array([2., -3.]), np.tile([-.5, .5], (2, 1)))
    assert norms['projected_gradient_inf'] == 0.
    assert norms['gradient_l2'] == pytest.approx(np.sqrt(13))


def test_completion_cannot_be_claimed_from_scipy_success_and_cost_alone():
    from benchmarks.milestone4b import completion_checks
    fit = {'success': True, 'invalid_trials': [], 'final_candidate': {'candidate_valid': True},
           'feasibility_probes': [{'valid': True}], 'objective_reduction': 10.,
           'disturbance_metrics': {'rmse_v_omega': [.2, .07]}}
    payload = {'experiments': {'fit': fit},
               'baseline_zero_disturbance_metrics': {'rmse_v_omega': [.15, .05]},
               'reproducibility': {'fit': {'runs': [1, 2], 'coefficient_max_abs_difference': 0., 'objective_max_abs_difference': 0.}},
               'local_sensitivities': {'zero': {'raw': {'shape': [600, 14], 'numerical_rank': 14},
                                               'whitened': {'shape': [600, 14], 'numerical_rank': 14}}},
               'noiseless_diagnostic_controls': {'fit': fit},
               'current_test_record': {'exit_code': 0, 'scope': 'complete repository suite'}}
    checks = completion_checks(payload)
    assert checks['primary_fits_converged'] and checks['objective_reduction']
    assert not checks['both_channels_reconstructed_better_than_zero']
    assert not all(checks.values())
