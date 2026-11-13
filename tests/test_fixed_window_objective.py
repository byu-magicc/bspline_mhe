"""Stage 4A fixed-window objective and complete-coefficient gradient checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.fixed_window_objective import (
    innovation_quadratic, make_fixed_window_objective,
)
from estimation.propagation import rk4_stage_times
from simulation.continuous_unicycle import simulate_known_landmarks
from splines import UniformBasis, disturbance_from_coefficients


@pytest.fixture(scope="module", autouse=True)
def enable_float64():
    with jax.experimental.enable_x64():
        yield


@pytest.fixture(scope="module")
def experiment():
    data = simulate_known_landmarks()
    adapter = UniformBasis(3, 4)
    B = adapter.precompute_jax(rk4_stage_times(data.times), L=5.)
    arrival = (jnp.zeros(3, dtype=jnp.float64),
               jnp.diag(jnp.array([.05**2, .05**2, np.deg2rad(2.)**2])))
    Qc = jnp.diag(jnp.array([.01, .0025], dtype=jnp.float64))
    command = lambda t: jnp.array([1., .2], dtype=jnp.float64)
    evaluator = make_fixed_window_objective(
        arrival, data.times, command, B, Qc, data.measurements,
        data.landmarks_at_times, data.measurement_covariance, data.mask,
        coefficient_count=2 * adapter.num_coefficients,
    )
    n = adapter.num_coefficients
    c_true = jnp.r_[jnp.full(n, .15), jnp.full(n, -.05)]
    return data, adapter, B, arrival, evaluator, c_true


def test_constant_disturbance_reconstruction_in_both_channels(experiment):
    _, adapter, _, _, _, c_true = experiment
    times = np.array([0., .13, 1.17, 2.5, 3.79, 5.])
    basis = adapter.precompute_jax(times, L=5.)
    values = disturbance_from_coefficients(basis, c_true)
    np.testing.assert_allclose(values, np.broadcast_to([.15, -.05], (len(times), 2)),
                               rtol=0, atol=2e-15)


def test_objective_sums_valid_innovations_and_matches_independent_numpy(experiment):
    data, _, _, _, evaluator, c_true = experiment
    result = jax.jit(evaluator)(c_true)
    assert bool(result.candidate_valid)
    assert int(result.requested_count) == 300
    assert int(result.valid_count) == 300
    assert int(result.invalid_geometry_count) == 0
    assert bool(result.positive_definite)
    valid = data.mask
    e = np.asarray(result.innovations)[valid]
    S = np.asarray(result.innovation_covariances)[valid]
    quadratic = sum(float(residual @ np.linalg.solve(covariance, residual))
                    for residual, covariance in zip(e, S))
    logdet = sum(float(np.linalg.slogdet(covariance)[1]) for covariance in S)
    np.testing.assert_allclose(result.innovation_quadratic, quadratic, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(result.logdet_diagnostic, logdet, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(result.gaussian_diagnostic, quadratic + logdet,
                               rtol=2e-13, atol=2e-12)


def test_masks_same_timestamp_reproducibility_and_fixed_arrival(experiment):
    data, adapter, B, arrival, _, c_true = experiment
    repeated_data = simulate_known_landmarks(seed=7)
    np.testing.assert_array_equal(data.truth, repeated_data.truth)
    np.testing.assert_array_equal(data.measurements, repeated_data.measurements)
    # Keep only four landmark slots at every scheduled timestamp; the remaining
    # padded values deliberately contain NaNs and must never be evaluated.
    mask = data.mask.copy()
    mask[:, -2:] = False
    measurements = data.measurements.copy()
    landmarks = data.landmarks_at_times.copy()
    measurements[:, -2:] = np.nan
    landmarks[:, -2:] = np.nan
    command = lambda t: jnp.array([1., .2], dtype=jnp.float64)
    evaluator = make_fixed_window_objective(
        arrival, data.times, command, B, jnp.diag(jnp.array([.01, .0025])),
        measurements, landmarks, data.measurement_covariance, mask,
        coefficient_count=2 * adapter.num_coefficients,
    )
    first, second = evaluator(c_true), evaluator(c_true)
    assert int(first.requested_count) == 200
    assert int(first.valid_count) == 200
    assert bool(first.candidate_valid)
    np.testing.assert_array_equal(first.final_pose, second.final_pose)
    np.testing.assert_array_equal(first.final_covariance, second.final_covariance)
    np.testing.assert_array_equal(arrival[0], np.zeros(3))
    # Different coefficients rerun from the same immutable prior (the captured
    # prior is unchanged, while the resulting terminal estimate changes).
    zero = evaluator(jnp.zeros_like(c_true))
    assert not np.allclose(first.final_pose, zero.final_pose)
    np.testing.assert_array_equal(arrival[0], np.zeros(3))


def test_simulation_rejects_measurement_grid_not_aligned_to_rk4_grid():
    with pytest.raises(ValueError, match="integer multiple"):
        simulate_known_landmarks(measurement_interval=.11)


def test_requested_invalid_geometry_is_reported_not_silently_counted(experiment):
    data, adapter, B, arrival, _, _ = experiment
    # Put one requested landmark at the arrival pose so its first-time geometry
    # is invalid. No observation at time zero in the default data, so construct
    # a one-node update explicitly to exercise the candidate status.
    times = np.array([0.])
    measurements = np.array([[[0., 0.]]])
    landmarks = np.array([[[0., 0.]]])
    evaluator = make_fixed_window_objective(
        arrival, times, lambda t: jnp.array([1., .2]),
        jnp.empty((0, 4, adapter.num_coefficients)), jnp.diag(jnp.array([.01, .0025])),
        measurements, landmarks, data.measurement_covariance, np.array([[True]]),
        coefficient_count=2 * adapter.num_coefficients,
    )
    result = evaluator(jnp.zeros(2 * adapter.num_coefficients))
    assert int(result.requested_count) == 1
    assert int(result.valid_count) == 0
    assert int(result.invalid_geometry_count) == 1
    assert not bool(result.candidate_valid)
    assert np.isnan(float(result.innovation_quadratic))


def _central_difference(evaluator, coefficients, step):
    scalar = lambda c: innovation_quadratic(evaluator, c)
    eye = jnp.eye(coefficients.size, dtype=jnp.float64)
    candidates = jnp.concatenate((coefficients[None] + step * eye,
                                  coefficients[None] - step * eye), axis=0)
    values = jax.jit(jax.vmap(scalar))(candidates)
    return np.asarray((values[:coefficients.size] - values[coefficients.size:]) / (2 * step))


def test_jax_gradients_all_coefficients_multiple_points_and_steps(experiment):
    _, adapter, _, _, evaluator, c_true = experiment
    n = adapter.num_coefficients
    nonconstant = jnp.linspace(-.08, .11, 2 * n, dtype=jnp.float64)
    candidates = [jnp.zeros(2 * n), c_true, nonconstant]
    scalar = lambda c: innovation_quadratic(evaluator, c)
    compiled_gradient = jax.jit(jax.grad(scalar))
    maximum_absolute = 0.
    maximum_relative = 0.
    for coefficients in candidates:
        result = evaluator(coefficients)
        assert bool(result.candidate_valid)
        gradient = np.asarray(compiled_gradient(coefficients))
        assert np.isfinite(float(scalar(coefficients))) and np.isfinite(gradient).all()
        for step in (1e-4, 1e-5):
            finite_difference = _central_difference(evaluator, coefficients, step)
            absolute = np.abs(gradient - finite_difference)
            relative = absolute / np.maximum(np.abs(finite_difference), 1e-5)
            maximum_absolute = max(maximum_absolute, float(absolute.max()))
            maximum_relative = max(maximum_relative, float(relative.max()))
            # Tolerances are checked across every coefficient in both blocks.
            np.testing.assert_allclose(gradient, finite_difference, rtol=2e-5, atol=2e-5)
    assert maximum_absolute < 2e-5
    print(f"\nStage 4A gradient max abs={maximum_absolute:.6e}; "
          f"max scaled relative={maximum_relative:.6e}; steps=1e-4,1e-5; "
          f"coefficients/channel={n}")


def test_jit_objective_gradient_and_covariance_dependence(experiment):
    _, _, _, _, evaluator, c_true = experiment
    objective = lambda c: innovation_quadratic(evaluator, c)
    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    value, gradient = value_and_gradient(c_true)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradient)).all()
    shifted = c_true.at[0].add(.01)
    first_S = np.asarray(evaluator(c_true).innovation_covariances)
    shifted_S = np.asarray(evaluator(shifted).innovation_covariances)
    assert not np.allclose(first_S, shifted_S)
