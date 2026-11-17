"""Independent truth, paired experiments and preserved Stage 4B contracts."""

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.integrate import quad

from benchmarks.milestone4b_smooth import (
    configuration, covariance_audit, gradient_validation, least_squares_approximation,
    localization_metrics, make_evaluator, validation_checks,
)
from estimation.coefficient_optimization import disturbance_metrics, optimize_coefficients
from estimation.fixed_window_objective import checked_evaluation
from estimation.propagation import rk4_stage_times
from simulation.smooth_unicycle import integrate_truth, simulate_smooth_landmarks, smooth_disturbance
from splines import UniformBasis, disturbance_from_coefficients


@pytest.fixture(scope='module', autouse=True)
def precision():
    with jax.experimental.enable_x64():
        yield


@pytest.fixture(scope='module')
def experiment():
    cfg = configuration()
    data, integration = simulate_smooth_landmarks(cfg)
    adapter, evaluator = make_evaluator(cfg, data)
    return cfg, data, adapter, evaluator, integration


def test_sinusoidal_channels_phase_amplitude_and_endpoints():
    times = np.array([0., 1.25, 2.5, 3.75, 5.])
    expected = np.array([[0., .05], [.15, 0.], [0., -.05], [-.15, 0.], [0., .05]])
    np.testing.assert_allclose(smooth_disturbance(times), expected, rtol=0, atol=4e-17)
    assert smooth_disturbance(1.).shape == (2,)


def test_truth_does_not_use_estimator_or_spline(monkeypatch):
    import estimation.propagation as propagation
    import estimation.known_landmark_ekf as ekf

    def forbidden(*args, **kwargs):
        pytest.fail('Truth must not invoke the estimator or spline basis')

    monkeypatch.setattr(propagation, 'rk4_step', forbidden)
    monkeypatch.setattr(ekf, 'measurement_update', forbidden)
    monkeypatch.setattr(UniformBasis, 'evaluate', forbidden)
    truth, info = integrate_truth(np.linspace(0., 5., 21), [0., 0., 0.], [1., .2])
    assert info['method'] == 'SciPy DOP853'
    assert truth.shape == (21, 3) and np.isfinite(truth).all()


def test_truth_matches_independent_quadrature_and_analytic_heading():
    times = np.linspace(0., 5., 21)
    initial = np.array([.2, -.3, .4])
    truth, _ = integrate_truth(times, initial, [1., .2])
    heading = lambda t: initial[2]+.2*t+.05*5/(2*np.pi)*np.sin(2*np.pi*t/5)
    reference = []
    for t in times:
        x = initial[0]+quad(lambda s: (1+.15*np.sin(2*np.pi*s/5))*np.cos(heading(s)),
                             0, t, epsabs=1e-12, epsrel=1e-12)[0]
        y = initial[1]+quad(lambda s: (1+.15*np.sin(2*np.pi*s/5))*np.sin(heading(s)),
                             0, t, epsabs=1e-12, epsrel=1e-12)[0]
        reference.append([x, y, heading(t)])
    np.testing.assert_allclose(truth, reference, rtol=0, atol=1e-11)


def test_truth_step_and_tolerance_refinement(experiment):
    cfg, data, _, _, integration = experiment
    refined, _ = integrate_truth(data.times, cfg['initial_pose'], cfg['command_v_omega'],
                                 max_step=.0125, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(data.truth, refined, rtol=0, atol=1e-10)
    assert integration['refinement']['passed']
    assert np.max(integration['refinement']['max_absolute_pose_difference']) < 1e-10


@pytest.mark.parametrize('times', [[.1, 1.], [0., 0.], [0., np.nan], [0., 5.1]])
def test_truth_rejects_invalid_time_grid(times):
    with pytest.raises(ValueError):
        integrate_truth(times, [0., 0., 0.], [1., .2])


def test_reproducibility_seed_changes_only_measurements(experiment):
    cfg, original, _, _, _ = experiment
    repeat, _ = simulate_smooth_landmarks(cfg)
    changed, _ = simulate_smooth_landmarks({**cfg, 'seed': 8})
    for a, b in zip(original, repeat):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(original.truth, changed.truth)
    np.testing.assert_array_equal(original.mask, changed.mask)
    assert not np.array_equal(original.measurements, changed.measurements)


def test_common_configuration_observation_policy_and_seeded_noise(experiment):
    cfg, data, _, _, _ = experiment
    previous = json.loads(Path('artifacts/milestone4b/results.json').read_text())['configuration']
    for key, value in previous.items():
        if key != 'true_disturbance_v_omega':
            assert cfg[key] == value
    with np.load('artifacts/milestone4b/dataset.npz') as old:
        np.testing.assert_array_equal(data.times, old['times'])
        np.testing.assert_array_equal(data.mask, old['mask'])
        np.testing.assert_array_equal(data.landmarks_at_times, old['landmarks'])
        np.testing.assert_array_equal(data.measurement_covariance, old['R'])
    assert data.mask.sum() == 300 and not data.mask[0].any()
    # Verify that each noisy record uses the same seed/order and independent
    # analytic range/bearing geometry; no estimator or truth spline is involved.
    rng = np.random.default_rng(7)
    for index in np.flatnonzero(data.mask[:, 0]):
        delta = data.landmark_map-data.truth[index, :2]
        clean = np.stack((np.linalg.norm(delta, axis=1),
                          np.arctan2(delta[:, 1], delta[:, 0])-data.truth[index, 2]), axis=-1)
        expected = clean+rng.multivariate_normal(np.zeros(2), data.measurement_covariance, size=6)
        expected[:, 1] = (expected[:, 1]+np.pi) % (2*np.pi)-np.pi
        np.testing.assert_allclose(data.measurements[index], expected, rtol=0, atol=2e-15)


def test_sampled_ls_is_optimal_and_not_exactly_representable(experiment):
    _, _, adapter, _, _ = experiment
    times = np.linspace(0., 5., 501)
    basis = adapter.evaluate(times, 5.)
    truth = smooth_disturbance(times)
    c, info = least_squares_approximation(basis, truth)
    residual = basis@c.reshape(2, 7).T-truth
    assert info['basis_rank'] == 7
    assert np.max(np.abs(basis.T@residual)) < 1e-11
    assert np.all(np.sqrt(np.mean(residual**2, axis=0)) > 1e-6)
    # Orthogonality gives the exact sampled least-squares Pythagorean identity.
    perturbation = np.random.default_rng(9).normal(0, .02, (7, 2))
    np.testing.assert_allclose(np.sum((residual+basis@perturbation)**2),
                               np.sum(residual**2)+np.sum((basis@perturbation)**2), rtol=2e-14)
    np.testing.assert_allclose(info['disturbance_metrics']['rmse_v_omega'],
                               np.sqrt(np.mean(residual**2, axis=0)), rtol=1e-12)


def test_ls_channel_order_time_scaling_and_cpp_endpoint(experiment):
    _, data, adapter, _, _ = experiment
    times = np.linspace(0., 5., 501)
    basis = adapter.evaluate(times, 5.)
    expected = np.r_[np.linspace(-.2, .2, 7), np.linspace(.1, -.3, 7)]
    values = basis@expected.reshape(2, 7).T
    actual, _ = least_squares_approximation(basis, values)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    np.testing.assert_allclose(disturbance_from_coefficients(basis, actual), values, rtol=0, atol=1e-15)
    from bspline_module import UniformBSpline
    cpp = UniformBSpline(3, 4)
    np.testing.assert_array_equal(basis[-1], cpp.basis_vector(4.))
    stages = rk4_stage_times(data.times)
    np.testing.assert_array_equal(adapter.evaluate(stages, 5.)[-1, -1], basis[-1])
    np.testing.assert_array_equal(adapter.evaluate(times/5., 1.), basis)
    # A unit linear ramp in physical seconds catches an alpha=M/L reversal.
    ramp_coefficients = np.r_[(np.arange(7)-1)*5/4, np.zeros(7)]
    np.testing.assert_allclose(np.asarray(disturbance_from_coefficients(basis, ramp_coefficients))[:, 0],
                               times, rtol=0, atol=2e-15)


def test_localization_metrics_wrap_and_use_position_norm():
    truth = np.zeros((3, 3))
    poses = np.array([[3., 4., 2*np.pi+.1], [0., 0., -.2], [0., 2., 2*np.pi+.3]])
    summary, error = localization_metrics(poses, truth)
    assert summary['position_rmse_m'] == pytest.approx(np.sqrt(29/3))
    assert summary['heading_rmse_rad'] == pytest.approx(np.sqrt(.14/3))
    assert summary['final_position_error_m'] == 2.
    assert summary['final_heading_error_rad'] == pytest.approx(.3)
    np.testing.assert_allclose(error[:, 2], [.1, -.2, .3], atol=1e-15)


def test_disturbance_metrics_time_varying_truth(experiment):
    _, _, adapter, _, _ = experiment
    times = np.linspace(0., 5., 501)
    basis = adapter.evaluate(times, 5.)
    truth = smooth_disturbance(times)
    c = np.linspace(-.1, .2, 14)
    metrics = disturbance_metrics(basis, c, truth)
    error = basis@c.reshape(2, 7).T-truth
    np.testing.assert_allclose(metrics['rmse_v_omega'], np.sqrt(np.mean(error**2, axis=0)), atol=1e-15)
    np.testing.assert_allclose(metrics['max_absolute_error_v_omega'], np.max(np.abs(error), axis=0), atol=1e-15)


def test_objective_independent_accumulation_and_all_coefficient_gradients(experiment):
    _, _, _, evaluator, _ = experiment
    evaluate = jax.jit(evaluator)
    value_gradient = jax.jit(jax.value_and_grad(lambda c: evaluator(c).innovation_quadratic))
    coefficients = np.linspace(-.05, .07, 14)
    result = checked_evaluation(evaluate, coefficients)
    residual = np.asarray(result.innovations)[np.asarray(result.valid_mask)]
    covariance = np.asarray(result.innovation_covariances)[np.asarray(result.valid_mask)]
    independent = sum(e@np.linalg.solve(S, e) for e, S in zip(residual, covariance))
    assert float(result.innovation_quadratic) == pytest.approx(independent, abs=2e-11)
    check = gradient_validation(evaluate, value_gradient, {'nonconstant': coefficients})
    assert check['passed']


def test_all_comparisons_share_arrival_data_and_fixed_observations(experiment):
    cfg, data, _, evaluator, _ = experiment
    evaluate = jax.jit(evaluator)
    zero = checked_evaluation(evaluate, np.zeros(14))
    expected_hash = hashlib.sha256(data.measurements.tobytes()).hexdigest()
    for c in (np.zeros(14), np.r_[np.full(7, .02), np.full(7, -.01)], np.linspace(-.1, .1, 14)):
        result = checked_evaluation(evaluate, c)
        np.testing.assert_array_equal(result.poses[0], cfg['initial_pose'])
        np.testing.assert_array_equal(result.covariances[0], cfg['arrival_covariance'])
        np.testing.assert_array_equal(result.valid_mask, data.mask)
        audit = covariance_audit(result, zero)
        assert audit['valid_count'] == audit['requested_count'] == 300
        assert audit['rejected_count'] == 0
    assert hashlib.sha256(data.measurements.tobytes()).hexdigest() == expected_hash


def test_constant_optimization_repeat_uses_zero_start_without_truth(experiment):
    _, _, _, evaluator, _ = experiment
    first = optimize_coefficients(evaluator, np.zeros(2), constant=True)
    second = optimize_coefficients(evaluator, np.zeros(2), constant=True)
    assert first['success'] and second['success']
    np.testing.assert_array_equal(first['initial_parameters'], [0., 0.])
    np.testing.assert_array_equal(first['coefficients'], second['coefficients'])
    assert first['objective_final'] == second['objective_final']
    assert not first['invalid_trials']


def test_smooth_invalid_trial_aborts_with_no_penalty_or_gradient(experiment, monkeypatch):
    import estimation.coefficient_optimization as optimizer
    _, _, _, evaluator, _ = experiment
    returned = []

    def invalid_trial(fun, initial, **kwargs):
        returned.append(fun(np.full(2, np.nan)))
        pytest.fail('Invalid trial must not return a finite penalty or gradient')

    monkeypatch.setattr(optimizer, 'minimize', invalid_trial)
    result = optimize_coefficients(evaluator, np.zeros(2), constant=True)
    assert not returned
    assert not result['success'] and result['status'] == 'stopped_invalid_trial'
    assert result['gradient_evaluations'] == 0
    assert result['invalid_trials'][0]['parameters'] == [None, None]
    np.testing.assert_array_equal(result['coefficients'], np.zeros(14))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('path,digest', [
    ('artifacts/milestone4b/results.json', '1de4435b60a2d43e744a75cf1c6e07e460f4253852acda6f5f168cac1c548237'),
    ('artifacts/milestone4b/dataset.npz', '3237b8918e1a1f0780e3706acf3493d743a456461735713de8cf01abf09c5abc'),
])
def test_validated_constant_experiment_preserved_byte_for_byte(path, digest):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_preserved_constant_objective_reproduces_saved_fit_values():
    from benchmarks.milestone4b import create_experiment
    _, _, _, evaluator, _ = create_experiment()
    evaluate = jax.jit(evaluator)
    saved = json.loads(Path('artifacts/milestone4b/results.json').read_text())
    for fit in saved['experiments'].values():
        actual = checked_evaluation(evaluate, np.asarray(fit['coefficients']))
        assert float(actual.innovation_quadratic) == pytest.approx(fit['objective_final'], rel=0, abs=2e-10)


def test_validation_gate_allows_a_negative_full_vs_constant_result():
    # Numerical validation and reconstructing better than zero do not require
    # the full spline to beat the restricted model on the sampled realization.
    payload = {
        'truth_integration': {'refinement': {'passed': True}},
        'experiments': {name: {'success': True, 'invalid_trials': [],
            'final_candidate': {'candidate_valid': True, 'numpy_consistency': {'passed': True}},
            'objective_final': 10., 'disturbance_metrics': {'rmse_v_omega': errors}}
            for name, errors in [('zero', [1., 1.]), ('constant_constrained', [.1, .1]), ('full_spline', [.2, .2])]},
        'covariance_audit': {name: {'identical_accepted_mask': True, 'rejected_count': 0,
                                  'valid_count': 300, 'quadratic_with_zero_candidate_covariances': 9.}
                             for name in ('zero', 'constant_constrained', 'full_spline')},
        'gradient_validation': {'passed': True},
        'least_squares_diagnostic': {'normal_equation_residual_max': 1e-15},
        'reproducibility': {name: {'coefficient_max_abs_difference': 0., 'objective_max_abs_difference': 0.,
                                  'repeat_success': True, 'accepted_objective_history_identical': True}
                            for name in ('constant_constrained', 'full_spline')},
        'current_test_record': {'exit_code': 0, 'scope': 'complete repository suite'},
    }
    assert all(validation_checks(payload).values())
    payload['experiments']['full_spline']['disturbance_metrics']['rmse_v_omega'] = [2., 2.]
    assert not validation_checks(payload)['full_spline_reconstructs_both_channels_better_than_zero']
    payload['covariance_audit']['full_spline']['quadratic_with_zero_candidate_covariances'] = 11.
    assert not validation_checks(payload)['full_spline_reduces_residual_cost_with_reference_covariances']
