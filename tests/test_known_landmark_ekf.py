"""Stage 3B.1: known-landmark updates, localization, and full EKF gradients."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.known_landmark_ekf import (
    filter_known_landmarks, measurement_update, sequential_updates,
)
from estimation.propagation import propagate, rk4_stage_times
from models.range_bearing import innovation, range_bearing, wrap_angle
from splines import UniformBasis


@pytest.fixture(scope="module", autouse=True)
def enable_float64():
    with jax.experimental.enable_x64():
        yield


def reference_measurement(r, landmark):
    dx, dy = np.asarray(landmark) - np.asarray(r)[:2]
    return np.array([math.hypot(dx, dy), math.atan2(dy, dx) - r[2]])


def reference_update(state, y, landmark, Sigma):
    """Independent NumPy update for well-separated landmarks."""
    r, P = state
    dx, dy = landmark - r[:2]
    q = dx**2 + dy**2
    H = np.array([[-dx / math.sqrt(q), -dy / math.sqrt(q), 0.], [dy / q, -dx / q, -1.]])
    e = y - reference_measurement(r, landmark)
    e[1] = math.atan2(math.sin(e[1]), math.cos(e[1]))
    S = H @ P @ H.T + Sigma
    K = np.linalg.solve(S, (P @ H.T).T).T
    J = np.eye(3) - K @ H
    return (r + K @ e, J @ P @ J.T + K @ Sigma @ K.T), (e, S, K)


def circle(r0, v, omega, times):
    theta = r0[2] + omega * times
    return np.stack((r0[0] + v / omega * (np.sin(theta) - np.sin(r0[2])),
                     r0[1] + v / omega * (np.cos(r0[2]) - np.cos(theta)), theta), axis=-1)


def basis_for(times, start=0.):
    adapter = UniformBasis(3, 4)
    B = adapter.precompute_jax(rk4_stage_times(times) - start, times[-1] - start)
    return adapter, B


@pytest.mark.parametrize("r,landmark", [([1., -2., .4], [4., 2.]),
                                       ([-.2, 1., 4.], [-3., -.5]),
                                       ([0., 0., -5.], [0., -2.])])
def test_measurement_predictions_against_independent_geometry(r, landmark):
    predicted, H, valid = jax.jit(range_bearing)(np.array(r), np.array(landmark))
    assert valid
    assert predicted.dtype == jnp.float64 and H.dtype == jnp.float64
    np.testing.assert_allclose(predicted, reference_measurement(r, landmark), rtol=0, atol=2e-15)


def test_analytical_measurement_jacobian_against_autodiff_and_finite_differences():
    rng = np.random.default_rng(13)
    h = 1e-6
    max_ad, max_fd = 0., 0.
    for _ in range(16):
        r = rng.normal(size=3)
        # Keep perturbations well away from zero distance and atan2's cut.
        displacement = np.array([rng.uniform(.8, 4), rng.uniform(-3, 3)])
        landmark = r[:2] + displacement
        _, H, valid = range_bearing(r, landmark)
        assert valid
        prediction = lambda pose: range_bearing(pose, landmark)[0]
        ad = jax.jacfwd(prediction)(r)
        fd = np.column_stack([(np.asarray(prediction(r + e)) - np.asarray(prediction(r - e))) / (2 * h)
                              for e in h * np.eye(3)])
        max_ad = max(max_ad, float(np.max(np.abs(H - ad))))
        max_fd = max(max_fd, float(np.max(np.abs(H - fd))))
        np.testing.assert_allclose(H, ad, rtol=0, atol=2e-15)
        np.testing.assert_allclose(H, fd, rtol=2e-8, atol=1e-9)
    print(f"\nMeasurement H max error: autodiff={max_ad:.6e}, FD h={h}: {max_fd:.6e}")


@pytest.mark.parametrize("predicted,observed,expected", [
    (np.pi - 1e-5, -np.pi + 2e-5, 3e-5),
    (-np.pi + 1e-5, np.pi - 2e-5, -3e-5),
    (3 * np.pi - 1e-5, -np.pi + 2e-5, 3e-5),
])
def test_wrapped_innovations_across_pi(predicted, observed, expected):
    e = jax.jit(innovation)(jnp.array([2.2, observed]), jnp.array([2., predicted]))
    np.testing.assert_allclose(e, [.2, expected], rtol=0, atol=3e-15)
    derivative = jax.jacfwd(lambda y: innovation(y, jnp.array([2., predicted])))(jnp.array([2.2, observed]))
    np.testing.assert_array_equal(derivative, np.eye(2))
    assert float(wrap_angle(np.pi)) == -np.pi


@pytest.mark.parametrize("distance", [0., .5e-4, 1e-4])
def test_near_zero_distance_rejection_is_finite_and_jittable(distance):
    r = np.zeros(3)
    landmark = np.array([distance, 0.])
    prediction, H, valid = jax.jit(range_bearing)(r, landmark, min_range=1e-4)
    assert not valid
    np.testing.assert_array_equal(prediction, np.zeros(2))
    np.testing.assert_array_equal(H, np.zeros((2, 3)))
    initial = (r, np.eye(3) * .1)
    updated, info = jax.jit(measurement_update)(initial, np.array([.2, .4]), landmark,
                                              np.eye(2), min_range=1e-4)
    assert not info.valid
    for actual, expected in zip(updated, initial):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(info.innovation_covariance, np.eye(2))
    gradient = jax.jacfwd(lambda pose: measurement_update((pose, initial[1]),
                                                         np.array([.2, .4]), landmark,
                                                         np.eye(2), min_range=1e-4)[0][0])(r)
    np.testing.assert_array_equal(gradient, np.eye(3))


def test_nonfinite_landmark_rejected_and_above_threshold_accepted():
    assert not range_bearing(np.zeros(3), np.array([np.nan, 1.]))[2]
    assert range_bearing(np.zeros(3), np.array([2e-4, 0.]), min_range=1e-4)[2]


def test_single_update_matches_independent_numpy_joseph_reference():
    state = (np.array([.4, -.3, .7]),
             np.array([[.2, .03, .01], [.03, .15, -.02], [.01, -.02, .08]]))
    landmark = np.array([3., -1.])
    y = reference_measurement(np.array([.1, -.1, .5]), landmark) + [.03, -.01]
    Sigma = np.array([[.04, .002], [.002, .01]])
    expected, diagnostics = reference_update(state, y, landmark, Sigma)
    actual, info = jax.jit(measurement_update)(state, y, landmark, Sigma)
    assert info.valid
    for output, reference in zip(actual, expected):
        np.testing.assert_allclose(output, reference, rtol=0, atol=2e-15)
    for output, reference in zip((info.innovation, info.innovation_covariance, info.gain), diagnostics):
        np.testing.assert_allclose(output, reference, rtol=0, atol=2e-15)
    P = np.asarray(actual[1])
    assert np.max(np.abs(P - P.T)) < 2e-15
    assert np.min(np.linalg.eigvalsh(P)) > 0


def test_sequential_relinearization_multiple_landmarks_at_one_timestamp():
    initial = (np.array([.5, -.4, .3]), np.diag([.3, .25, .1]))
    truth = np.array([.1, .1, -.1])
    landmarks = np.array([[3., 1.], [-2., 3.], [1., -4.]])
    measurements = np.array([reference_measurement(truth, ell) for ell in landmarks])
    Sigma = np.diag([.04, .005])
    (r, P), info = jax.jit(sequential_updates)(initial, measurements, landmarks, Sigma)
    reference = initial
    for i in range(3):
        reference, diagnostics = reference_update(reference, measurements[i], landmarks[i], Sigma)
        np.testing.assert_allclose(info.innovation[i], diagnostics[0], rtol=0, atol=3e-15)
    np.testing.assert_allclose(r, reference[0], rtol=0, atol=3e-15)
    np.testing.assert_allclose(P, reference[1], rtol=0, atol=3e-15)
    assert not np.allclose(info.innovation[1], innovation(measurements[1], range_bearing(initial[0], landmarks[1])[0]))
    # Contrast with one stacked update linearized only at the original prior.
    H = np.concatenate([np.asarray(range_bearing(initial[0], ell)[1]) for ell in landmarks])
    e = np.concatenate([np.asarray(innovation(y, range_bearing(initial[0], ell)[0]))
                        for y, ell in zip(measurements, landmarks)])
    R = np.kron(np.eye(3), Sigma)
    S = H @ initial[1] @ H.T + R
    K = np.linalg.solve(S, (initial[1] @ H.T).T).T
    stacked_pose = initial[0] + K @ e
    difference = np.linalg.norm(r - stacked_pose)
    assert difference > 1e-4
    print(f"\nSequential vs stacked posterior pose difference={difference:.6e}")


def test_masks_empty_batches_and_initial_timestamp_updates():
    initial = (np.array([.2, .1, -.3]), np.eye(3) * .05)
    landmark = np.array([[3., 1.], [np.nan, np.nan]])
    measurements = np.array([reference_measurement(np.zeros(3), landmark[0]), [np.nan, np.nan]])
    Sigma = np.array([np.diag([.03, .01]), np.full((2, 2), np.nan)])
    actual, info = sequential_updates(initial, measurements, landmark, Sigma, mask=[True, False])
    reference, _ = measurement_update(initial, measurements[0], landmark[0], Sigma[0])
    for output, expected in zip(actual, reference):
        np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(info.valid, [True, False])
    empty, empty_info = sequential_updates(initial, np.empty((0, 2)), np.empty((0, 2)), np.eye(2))
    for output, expected in zip(empty, initial):
        np.testing.assert_array_equal(output, expected)
    assert empty_info.valid.shape == (0,)
    adapter = UniformBasis(3, 4)
    B = adapter.precompute_jax(np.empty((0, 4)), L=1.)
    result = jax.jit(lambda c: filter_known_landmarks(
        initial, np.array([.3]), lambda t: jnp.array([1., .2]), B, c, np.eye(2) * .01,
        measurements[None], landmark[None], Sigma[None], np.array([[True, False]]),
    ))(np.zeros(2 * adapter.num_coefficients))
    np.testing.assert_allclose(result.poses[0], reference[0], rtol=0, atol=2e-15)
    assert result.updates.valid.shape == (1, 2)


def test_multiple_observation_times_match_manual_propagation_and_updates():
    start = 5.
    times = start + np.array([0., .07, .23, .51, .9, 1.4])
    adapter, B = basis_for(times, start)
    c = np.random.default_rng(15).normal(scale=.1, size=2 * adapter.num_coefficients)
    initial = (np.array([.2, -.1, .4]), np.diag([.05, .05, .02]))
    Qc = np.diag([.02, .005])
    command = lambda t: jnp.stack((1 + .1 * jnp.cos(t), .2 * jnp.sin(t)))
    landmark_map = np.array([[3., 1.], [-2., 3.], [1., -4.]])
    landmarks = np.broadcast_to(landmark_map, (len(times), 3, 2))
    mask = np.array([[True, False, False], [False, False, False], [True, True, True],
                     [False, True, False], [False, False, False], [True, True, False]])
    measurements = np.array([[reference_measurement(np.array([t - start, .1, .3]), ell)
                              for ell in landmark_map] for t in times])
    Sigma = np.broadcast_to(np.diag([.04, .01]), (len(times), 3, 2, 2))
    result = jax.jit(lambda coefficients: filter_known_landmarks(
        initial, times, command, B, coefficients, Qc, measurements, landmarks, Sigma, mask,
    ))(c)
    current = initial
    for i in range(len(times)):
        if i:
            history = propagate(current, times[i-1:i+1], command, B[i-1:i], c, Qc)
            current = (history[0][-1], history[1][-1])
        for j in range(3):
            if mask[i, j]:
                current, _ = reference_update(tuple(np.asarray(x) for x in current),
                                               measurements[i, j], landmarks[i, j], Sigma[i, j])
        np.testing.assert_allclose(result.poses[i], current[0], rtol=0, atol=5e-14)
        np.testing.assert_allclose(result.covariances[i], current[1], rtol=0, atol=5e-14)
    np.testing.assert_array_equal(result.updates.valid, mask)


def test_known_landmark_localization_with_simulated_noisy_measurements():
    rng = np.random.default_rng(17)
    times = np.unique(np.r_[np.linspace(0, 6., 121), [.17, .63, 1.37, 2.23, 3.41, 4.57, 5.83]])
    adapter, B = basis_for(times)
    r0 = np.array([.2, -.3, .4])
    truth = circle(r0, 1., .2, times)
    initial = (r0 + [.5, -.4, .2], np.diag([.36, .36, .09]))
    landmark_map = np.array([[4., 1.], [-2., 3.], [1., -4.]])
    landmarks = np.broadcast_to(landmark_map, (len(times), 3, 2))
    mask = np.zeros((len(times), 3), dtype=bool)
    mask[::6] = True
    mask[-1] = True
    measurements = np.array([[reference_measurement(r, ell) for ell in landmark_map] for r in truth])
    measurements += rng.normal(size=measurements.shape) * [.05, .015]
    measurements[..., 1] = np.arctan2(np.sin(measurements[..., 1]), np.cos(measurements[..., 1]))
    command = lambda t: jnp.array([1., .2])
    c = np.zeros(2 * adapter.num_coefficients)
    Qc = np.diag([.005, .001])
    result = jax.jit(lambda coefficients: filter_known_landmarks(
        initial, times, command, B, coefficients, Qc, measurements, landmarks,
        np.diag([.05**2, .015**2]), mask,
    ))(c)
    open_loop = propagate(initial, times, command, B, c, Qc)[0]
    position_rmse = np.sqrt(np.mean(np.sum((np.asarray(result.poses)[:, :2] - truth[:, :2])**2, axis=-1)))
    heading_rmse = np.sqrt(np.mean(np.asarray(wrap_angle(result.poses[:, 2] - truth[:, 2]))**2))
    open_rmse = np.sqrt(np.mean(np.sum((np.asarray(open_loop)[:, :2] - truth[:, :2])**2, axis=-1)))
    assert position_rmse < .12
    assert heading_rmse < .04
    assert position_rmse < open_rmse
    np.testing.assert_array_equal(result.updates.valid, mask)
    P = np.asarray(result.covariances)
    asymmetry = np.max(np.abs(P - P.transpose(0, 2, 1)))
    min_eigenvalue = np.min(np.linalg.eigvalsh(P))
    assert asymmetry < 2e-15
    assert min_eigenvalue > 0
    print(f"\nSeed-17 known-landmark localization: position RMSE={position_rmse:.9e}, "
          f"heading RMSE={heading_rmse:.9e}, open-loop position RMSE={open_rmse:.9e}, "
          f"accepted observations={int(mask.sum())}")
    print(f"Joseph/full-filter covariance max asymmetry={asymmetry:.6e}, min eigenvalue={min_eigenvalue:.6e}")


def test_full_ekf_coefficient_gradients_innovations_covariances_gain_and_state():
    times = np.array([0., .07, .19, .31, .52, .73, 1., 1.17, 1.4])
    adapter, B = basis_for(times)
    n = adapter.num_coefficients
    truth = circle(np.array([.1, -.2, .3]), 1.1, .25, times)
    initial = (truth[0] + [.12, -.1, .07], np.diag([.05, .04, .02]))
    landmark_map = np.array([[3., 1.], [-2., 3.], [1., -4.]])
    landmarks = np.broadcast_to(landmark_map, (len(times), 3, 2))
    mask = np.array([[True, False, True], [False, False, False], [True, True, True],
                     [False, True, False], [True, True, False], [True, False, True],
                     [False, False, False], [True, True, True], [True, True, True]])
    measurements = np.array([[reference_measurement(r, ell) for ell in landmark_map] for r in truth])
    measurements += np.random.default_rng(19).normal(size=measurements.shape) * [.03, .01]
    c = np.random.default_rng(20).normal(scale=.08, size=2 * n)
    command = lambda t: jnp.stack((1.1 + .07 * jnp.sin(t), .25 + .04 * jnp.cos(2 * t)))

    def components(coefficients):
        result = filter_known_landmarks(initial, times, command, B, coefficients,
                                       np.diag([.015, .006]), measurements, landmarks,
                                       np.diag([.12**2, .04**2]), mask)
        info = result.updates
        e, S, K = info.innovation, info.innovation_covariance, info.gain
        solved = jnp.linalg.solve(S, e[..., None])[..., 0]
        innovation_cost = jnp.sum(e * solved)
        covariance_cost = .05 * jnp.sum(jnp.where(info.valid[..., None, None], S**2, 0.))
        gain_cost = .02 * jnp.sum(K**2)
        r, P = result.poses[-1], result.covariances[-1]
        final_cost = .5 * jnp.sum(r**2) + .6 * jnp.trace(P) + .2 * jnp.sum(P**2)
        return jnp.stack((innovation_cost, covariance_cost, gain_cost, final_cost))

    objective = jax.jit(lambda coefficients: jnp.sum(components(coefficients)))
    actual = np.asarray(jax.jit(jax.grad(objective))(c))
    component_gradients = np.asarray(jax.jit(jax.jacrev(components))(c))
    np.testing.assert_allclose(actual, component_gradients.sum(axis=0), rtol=3e-13, atol=3e-13)
    assert actual.dtype == np.float64
    # S, K, innovations, and final state/covariance all retain coefficient dependence.
    for gradient in component_gradients:
        assert np.linalg.norm(gradient[:n]) > 1e-7
        assert np.linalg.norm(gradient[n:]) > 1e-7
    for h in [1e-5, 3e-6]:
        fd = np.array([(float(objective(c + e)) - float(objective(c - e))) / (2 * h)
                       for e in h * np.eye(c.size)])
        np.testing.assert_allclose(actual, fd, rtol=3e-6, atol=3e-8)
        errors = np.abs(actual - fd)
        print(f"\nFull EKF gradient h={h}: max c_v error={np.max(errors[:n]):.6e}, "
              f"max c_omega error={np.max(errors[n:]):.6e}")
    print("Full EKF component gradient block norms [innovation, S, K, final]:",
          [(float(np.linalg.norm(row[:n])), float(np.linalg.norm(row[n:]))) for row in component_gradients])


def test_no_observations_reduces_to_validated_stage3a():
    times = np.array([0., .13, .7, 1.])
    adapter, B = basis_for(times)
    initial = (np.array([.1, .2, .3]), np.eye(3) * .03)
    c = np.ones(2 * adapter.num_coefficients) * .05
    command = lambda t: jnp.stack((1 + .1 * t, .2 * t))
    Qc = np.diag([.01, .005])
    result = filter_known_landmarks(initial, times, command, B, c, Qc,
                                    np.empty((len(times), 0, 2)), np.empty((len(times), 0, 2)), np.eye(2))
    reference = propagate(initial, times, command, B, c, Qc)
    np.testing.assert_array_equal(result.poses, reference[0])
    np.testing.assert_array_equal(result.covariances, reference[1])


def test_precision_and_shape_errors_are_explicit():
    with jax.experimental.disable_x64():
        with pytest.raises(RuntimeError, match="Enable JAX float64"):
            range_bearing(np.zeros(3), np.ones(2))
        with pytest.raises(RuntimeError, match="Enable JAX float64"):
            measurement_update((np.zeros(3), np.eye(3)), np.ones(2), np.ones(2), np.eye(2))
    with pytest.raises(ValueError, match="landmark"):
        range_bearing(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="Sigma"):
        sequential_updates((np.zeros(3), np.eye(3)), np.zeros((2, 2)), np.ones((2, 2)), np.ones(2))
    with pytest.raises(ValueError, match="mask"):
        sequential_updates((np.zeros(3), np.eye(3)), np.zeros((2, 2)), np.ones((2, 2)), np.eye(2), mask=[True])
