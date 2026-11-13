"""Independent mathematical references and domain tests for Stage 4A."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.fixed_window_objective import (
    InvalidCandidateError, checked_evaluation, checked_value_and_grad,
    innovation_quadratic, make_fixed_window_objective,
)
from estimation.known_landmark_ekf import filter_known_landmarks
from estimation.propagation import rk4_stage_times
from simulation.continuous_unicycle import analytical_unicycle, simulate_known_landmarks
from splines import UniformBasis


@pytest.fixture(scope="module", autouse=True)
def precision():
    with jax.experimental.enable_x64():
        yield


def prepare(times, arrival, command, y, ell, mask, *, Qc=None, R=None, B=None):
    adapter = UniformBasis(3, 4)
    if B is None:
        B = adapter.precompute_jax(rk4_stage_times(times)-times[0], times[-1]-times[0])
    return make_fixed_window_objective(
        arrival, times, command, B, np.zeros((2, 2)) if Qc is None else Qc,
        y, ell, np.eye(2)*.1 if R is None else R, mask, coefficient_count=14)


def numpy_measurement(pose, landmark):
    delta = landmark-pose[:2]
    return np.array([np.hypot(*delta), np.arctan2(delta[1], delta[0])-pose[2]])


@pytest.mark.parametrize("channel", [0, 1], ids=["linear", "angular"])
@pytest.mark.parametrize("grid", [0, 1])
def test_nonconstant_analytic_trajectory_innovations_and_cost(channel, grid, record_property):
    origin, L, a = 11., 2.5, .13
    elapsed = (np.linspace(0., L, 11) if grid == 0 else
               np.unique(np.r_[np.linspace(0., L, 21), [.17, .93, 1.87]]))
    times = origin+elapsed
    r0 = np.array([1.2, -.7, .4])
    command_values = np.array([.7, 0.]) if channel == 0 else np.array([0., .2])
    # For this unclamped cubic basis the affine coefficient ramp reproduces
    # w(sigma)=a*sigma: at s=0 the active indices have mean 1, at s=M mean M+1.
    c = np.zeros(14)
    c[channel*7:(channel+1)*7] = a*(L/4)*(np.arange(7)-1)
    basis = UniformBasis(3, 4)
    expected_input = a*elapsed
    np.testing.assert_allclose(basis.evaluate(elapsed, L) @ c[channel*7:(channel+1)*7],
                               expected_input, rtol=0, atol=3e-16)
    truth = np.broadcast_to(r0, (len(times), 3)).copy()
    if channel == 0:
        distance = .7*elapsed+.5*a*elapsed**2
        truth[:, :2] += distance[:, None]*np.array([np.cos(r0[2]), np.sin(r0[2])])
    else:
        truth[:, 2] += .2*elapsed+.5*a*elapsed**2
    landmarks = np.broadcast_to([[5., 3.], [-3., 4.]], (len(times), 2, 2)).copy()
    predictions = np.array([[numpy_measurement(pose, landmark) for landmark in row]
                            for pose, row in zip(truth, landmarks)])
    offsets = np.stack((.03+.01*elapsed, -.02+.003*elapsed), axis=-1)
    y = predictions+offsets[:, None]
    y[..., 1] = np.arctan2(np.sin(y[..., 1]), np.cos(y[..., 1]))
    mask = np.zeros((len(times), 2), dtype=bool)
    mask[[0, len(times)//2, -1]] = True
    R = np.array([[.04, .002], [.002, .01]])
    command = lambda t: jnp.asarray(command_values)
    # Zero uncertainty keeps the EKF mean exactly on the analytic path despite
    # nonzero innovations. This isolates propagation and prior-cost semantics.
    evaluator = prepare(times, (r0, np.zeros((3, 3))), command, y, landmarks, mask, R=R)
    result = jax.jit(evaluator)(c)
    assert result.candidate_valid
    np.testing.assert_allclose(result.prior_poses, truth, rtol=0, atol=3e-14)
    np.testing.assert_allclose(result.poses, truth, rtol=0, atol=3e-14)
    expected_e = y-predictions
    expected_e[..., 1] = np.arctan2(np.sin(expected_e[..., 1]), np.cos(expected_e[..., 1]))
    expected_terms = [e @ np.linalg.solve(R, e) for e in expected_e[mask]]
    np.testing.assert_allclose(result.innovations[mask], expected_e[mask], rtol=0, atol=2e-14)
    np.testing.assert_allclose(result.innovation_covariances[mask], np.broadcast_to(R, (6, 2, 2)),
                               rtol=0, atol=1e-16)
    np.testing.assert_allclose(result.innovation_quadratic, sum(expected_terms), rtol=0, atol=2e-13)
    np.testing.assert_allclose(result.logdet_diagnostic, 6*np.linalg.slogdet(R)[1], rtol=0, atol=2e-13)
    # Explicit mutation check: wrongly halved physical-to-spline time must fail.
    wrong_B = basis.precompute_jax((rk4_stage_times(times)-origin)/2, L)
    wrong = prepare(times, (r0, np.zeros((3, 3))), command, y, landmarks, mask, R=R, B=wrong_B)(c)
    assert wrong.candidate_valid
    assert np.max(np.abs(np.asarray(wrong.prior_poses)-truth)) > .05
    assert abs(float(wrong.innovation_quadratic)-sum(expected_terms)) > .01
    record_property('state_max_abs_error', float(np.max(np.abs(np.asarray(result.prior_poses)-truth))))
    record_property('innovation_max_abs_error', float(np.max(np.abs(np.asarray(result.innovations)[mask]-expected_e[mask]))))
    record_property('objective_abs_error', abs(float(result.innovation_quadratic)-sum(expected_terms)))
    record_property('incorrect_time_scaling_objective_error', abs(float(wrong.innovation_quadratic)-sum(expected_terms)))


def mixed_geometry_window():
    times = np.array([3., 4.])
    y = np.array([[[1., .1], [2., -.2], [np.nan, np.nan]],
                  [[1., .1], [2., -.2], [np.nan, np.nan]]])
    ell = np.array([[[1., 0.], [3., 1.], [np.nan, np.nan]],
                    [[1., 0.], [3., 1.], [np.nan, np.nan]]])
    mask = np.array([[False, False, False], [True, True, False]])
    return prepare(times, (np.zeros(3), np.eye(3)*.01), lambda t: jnp.zeros(2), y, ell, mask)


def test_candidate_dependent_geometry_and_checked_interfaces():
    evaluator = mixed_geometry_window()
    good, collision = np.zeros(14), np.r_[np.ones(7), np.zeros(7)]
    assert checked_evaluation(evaluator, good).candidate_valid
    result = jax.jit(evaluator)(collision)
    assert not result.candidate_valid
    assert (int(result.requested_count), int(result.valid_count), int(result.masked_count),
            int(result.rejected_count), int(result.invalid_geometry_count)) == (2, 1, 4, 1, 1)
    np.testing.assert_array_equal(result.valid_mask[1], [False, True, False])
    for value in (result.innovation_quadratic, result.logdet_diagnostic, result.gaussian_diagnostic):
        assert np.isnan(value)
    with pytest.raises(InvalidCandidateError, match="geometry_rejected=1"):
        checked_evaluation(evaluator, collision)
    with pytest.raises(InvalidCandidateError):
        checked_value_and_grad(evaluator, collision)
    value, gradient = checked_value_and_grad(evaluator, good)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    scalar = lambda c: innovation_quadratic(evaluator, c)
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(scalar)))(np.stack((good, collision)))
    assert np.isfinite(values[0]) and np.isfinite(gradients[0]).all()
    assert np.isnan(values[1])  # Undefined costs never turn into zero-cost optima.


@pytest.mark.parametrize("problem", ["nonfinite", "asymmetric", "indefinite", "singular_S"])
def test_numerical_rejections_are_safe_under_jit_and_ad(problem):
    P = np.eye(3)*.01
    R = np.eye(2)*.1
    if problem == "nonfinite":
        P[0, 0] = np.nan
    elif problem == "asymmetric":
        P[0, 1] = 1e-5
    elif problem == "indefinite":
        P[0, 0] = -.001
    else:
        P[:] = 0.
        R[:] = 0.
    times = np.array([0., .1])
    evaluator = prepare(times, (np.zeros(3), P), lambda t: jnp.zeros(2),
                        np.ones((2, 1, 2)), np.full((2, 1, 2), 3.), np.ones((2, 1), bool), R=R)
    result = jax.jit(evaluator)(np.zeros(14))
    assert not result.candidate_valid
    value, gradient = jax.jit(jax.value_and_grad(lambda c: innovation_quadratic(evaluator, c)))(np.zeros(14))
    assert np.isnan(value)
    assert np.isfinite(gradient).all()  # Invalid solves/eigensystems are quarantined.
    with pytest.raises(InvalidCandidateError):
        checked_evaluation(evaluator, np.zeros(14))


def test_overflow_after_last_observation_is_rejected():
    times = np.array([0., 1.])
    evaluator = prepare(times, (np.zeros(3), np.eye(3)*.01), lambda t: jnp.zeros(2),
                        np.ones((2, 1, 2)), np.full((2, 1, 2), 3.), np.array([[True], [False]]))
    c = np.r_[np.full(7, 1e200), np.zeros(7)]
    result = jax.jit(evaluator)(c)
    assert not result.candidate_valid and not result.states_finite
    assert not np.isfinite(result.prior_covariances[-1]).all()
    assert np.isnan(result.innovation_quadratic)
    with pytest.raises(InvalidCandidateError, match="finite_states=False"):
        checked_value_and_grad(evaluator, c)


def test_covariance_tolerance_and_intermediate_posterior_validation(monkeypatch):
    import estimation.fixed_window_objective as module
    times = np.array([0., .1])
    ell = np.full((2, 2, 2), 3.)
    mask = np.array([[True, True], [False, False]])
    original = module.measurement_update

    def damaged_update(*args):
        (r, P), info = original(*args)
        return (r, P.at[0, 0].set(-1.)), info

    monkeypatch.setattr(module, "measurement_update", damaged_update)
    evaluator = prepare(times, (np.zeros(3), np.eye(3)*.01), lambda t: jnp.zeros(2),
                        np.ones((2, 2, 2)), ell, mask)
    result = jax.jit(evaluator)(np.zeros(14))
    assert not result.candidate_valid and not result.covariances_psd
    assert int(result.numerical_rejection_count) == 2
    monkeypatch.setattr(module, "measurement_update", original)
    # Numerical PSD tolerance accepts tiny negative roundoff, without clipping it.
    P = np.diag([-5e-11, .01, .01])
    evaluator = prepare(times, (np.zeros(3), P), lambda t: jnp.zeros(2),
                        np.ones((2, 2, 2)), ell, np.zeros((2, 2), bool))
    result = evaluator(np.zeros(14))
    assert result.candidate_valid
    assert float(result.final_covariance[0, 0]) == P[0, 0]


def decomposition_window():
    times = 7.+np.linspace(0., 1.5, 13)
    r0 = np.array([.3, -.2, .4])
    P0 = np.array([[.04, .003, .001], [.003, .03, -.002], [.001, -.002, .02]])
    ell = np.broadcast_to([[3., 2.], [-2., 4.]], (len(times), 2, 2)).copy()
    y = np.array([[numpy_measurement(r0+np.array([.8*(t-7), .15*(t-7), .1*(t-7)]), landmark)
                   for landmark in row] for t, row in zip(times, ell)])
    mask = np.zeros((len(times), 2), bool)
    mask[::3] = True
    command = lambda t: jnp.array([.8, .1])
    Qc = np.diag([.005, .001])
    R = np.diag([.03, .003])
    B = UniformBasis(3, 4).precompute_jax(rk4_stage_times(times)-7., 1.5)
    evaluator = prepare(times, (r0, P0), command, y, ell, mask, Qc=Qc, R=R, B=B)
    return evaluator, (r0, P0), (times, command, B, Qc, y, ell, R, mask)


def test_candidate_order_and_parity_with_preserved_ekf():
    evaluator, arrival, args = decomposition_window()
    a = np.linspace(-.04, .08, 14)
    b = np.random.default_rng(8).normal(scale=.04, size=14)
    prior = tuple(v.copy() for v in arrival)
    compiled = jax.jit(evaluator)
    first, other, again = compiled(a), compiled(b), compiled(a)
    assert first.candidate_valid and other.candidate_valid
    for x, y in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(again)):
        np.testing.assert_array_equal(x, y)
    for x, y in zip(prior, arrival):
        np.testing.assert_array_equal(x, y)
    times, command, B, Qc, y, ell, R, mask = args
    reference = filter_known_landmarks(arrival, times, command, B, a, Qc, y, ell, R, mask)
    for actual, expected in ((first.poses, reference.poses), (first.covariances, reference.covariances),
                             (first.innovations, reference.updates.innovation),
                             (first.innovation_covariances, reference.updates.innovation_covariance)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)


def test_innovation_and_covariance_gradient_decomposition_both_blocks(record_property):
    evaluator, _, _ = decomposition_window()
    candidates = [np.linspace(-.04, .08, 14), np.random.default_rng(9).normal(scale=.06, size=14)]

    def components(c):
        result = evaluator(c)
        return result.innovations, result.innovation_covariances

    jacobian = jax.jit(jax.jacrev(components))
    full_gradient = jax.jit(jax.grad(lambda c: innovation_quadratic(evaluator, c)))
    for index, c in enumerate(candidates):
        result = checked_evaluation(evaluator, c)
        de, dS = jax.tree_util.tree_map(np.asarray, jacobian(c))
        valid = np.asarray(result.valid_mask)
        e = np.asarray(result.innovations)[valid]
        S = np.asarray(result.innovation_covariances)[valid]
        solved = np.linalg.solve(S, e[..., None])[..., 0]
        from_e = 2*np.einsum('ki,kij->j', solved, de[valid])
        from_S = -np.einsum('ki,kijl, kj->l', solved, dS[valid], solved)
        gradient = np.asarray(full_gradient(c))
        np.testing.assert_allclose(from_e+from_S, gradient, rtol=3e-12, atol=3e-12)
        for contribution in (from_e, from_S):
            assert np.linalg.norm(contribution[:7]) > 1e-5
            assert np.linalg.norm(contribution[7:]) > 1e-5
        # Omitting dS must be detected, rather than hidden by a small covariance effect.
        assert np.max(np.abs(gradient-from_e)) > 1e-4
        record_property(f'candidate_{index}_max_abs_error', float(np.max(np.abs(from_e+from_S-gradient))))
        record_property(f'candidate_{index}_linear_covariance_gradient_norm', float(np.linalg.norm(from_S[:7])))
        record_property(f'candidate_{index}_angular_covariance_gradient_norm', float(np.linalg.norm(from_S[7:])))
        print('decomposition max absolute discrepancy:', np.max(np.abs(from_e+from_S-gradient)),
              'dS block norms:', np.linalg.norm(from_S[:7]), np.linalg.norm(from_S[7:]))


@pytest.mark.parametrize("omega", [0., 1e-16, -1e-16, 1e-14, -1e-14, 1e-10, -1e-10])
def test_analytical_truth_near_zero_turn_rate(omega):
    times = np.array([0., .1, 1., 5.])
    initial = np.array([.2, -.3, .7])
    # Independent Taylor expansion; omitted terms < 1e-28 at these rates/times.
    theta = initial[2]
    x = initial[0]+times*np.cos(theta)-.5*omega*times**2*np.sin(theta)
    y = initial[1]+times*np.sin(theta)+.5*omega*times**2*np.cos(theta)
    expected = np.stack((x, y, theta+omega*times), axis=-1)
    np.testing.assert_allclose(analytical_unicycle(times, initial, 1., omega), expected,
                               rtol=0, atol=2e-15)


def test_original_nonzero_truth_is_preserved_to_float64_precision():
    data = simulate_known_landmarks()
    theta = .15*data.times
    old_truth = np.stack((1.15/.15*np.sin(theta), 1.15/.15*(1-np.cos(theta)), theta), axis=-1)
    np.testing.assert_allclose(data.truth, old_truth, rtol=0, atol=3e-15)
