"""Differentiable sequential known-landmark EKF for the robot pose only.

All association assignments are supplied as fixed landmark coordinates. There
is no association search, landmark state, or coefficient optimization here.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from estimation.propagation import rk4_step
from models.range_bearing import innovation, range_bearing


class UpdateDiagnostics(NamedTuple):
    prediction: jax.Array
    jacobian: jax.Array
    innovation: jax.Array
    innovation_covariance: jax.Array
    gain: jax.Array
    valid: jax.Array


class FilterResult(NamedTuple):
    poses: jax.Array
    covariances: jax.Array
    updates: UpdateDiagnostics


def _state64(state):
    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX float64 with jax.config.update("jax_enable_x64", True)')
    r, P = (jnp.asarray(value, dtype=jnp.float64) for value in state)
    if r.shape != (3,) or P.shape != (3, 3):
        raise ValueError("state must be a (pose (3,), covariance (3,3)) tuple")
    return r, P


def _skipped():
    # Neutral finite placeholders: zero residual/gain and identity S. Use valid
    # to select actual diagnostics. In particular, do not count identity S as data.
    return UpdateDiagnostics(jnp.zeros(2, dtype=jnp.float64),
                             jnp.zeros((2, 3), dtype=jnp.float64),
                             jnp.zeros(2, dtype=jnp.float64), jnp.eye(2, dtype=jnp.float64),
                             jnp.zeros((3, 2), dtype=jnp.float64), jnp.asarray(False))


def measurement_update(state, y, landmark, Sigma, min_range=1e-8):
    """One Joseph-form update, returning ((r,P), UpdateDiagnostics).

    Sigma is a finite symmetric positive definite (2,2) measurement covariance.
    P must be finite, symmetric positive semidefinite. These value conditions
    and finite measurement values are caller preconditions; shapes/precision
    are checked during tracing. Near-zero/invalid geometry skips the update
    with valid=False. Angular innovations wrap to [-pi,pi); pose heading remains
    unwrapped as in Stage 3A. No stop_gradient or covariance projection is used.
    """
    state = _state64(state)
    y, Sigma = (jnp.asarray(value, dtype=jnp.float64) for value in (y, Sigma))
    if y.shape != (2,) or Sigma.shape != (2, 2):
        raise ValueError("y must have shape (2,) and Sigma must have shape (2,2)")
    r, P = state
    predicted, H, valid = range_bearing(r, landmark, min_range)

    def correct(_):
        e = innovation(y, predicted)
        PHt = P @ H.T
        S = H @ PHt + Sigma
        K = jnp.linalg.solve(S, PHt.T).T
        updated_r = r + K @ e
        residual_map = jnp.eye(3, dtype=jnp.float64) - K @ H
        updated_P = residual_map @ P @ residual_map.T + K @ Sigma @ K.T
        info = UpdateDiagnostics(predicted, H, e, S, K, valid)
        return (updated_r, updated_P), info

    return jax.lax.cond(valid, correct, lambda _: (state, _skipped()), operand=None)


def _observation_arrays(measurements, landmarks, Sigma, mask):
    measurements, landmarks = (jnp.asarray(value, dtype=jnp.float64)
                               for value in (measurements, landmarks))
    if measurements.ndim < 2 or measurements.shape[-1] != 2 or landmarks.shape != measurements.shape:
        raise ValueError("measurements and landmarks must have matching shapes (...,slots,2)")
    Sigma = jnp.asarray(Sigma, dtype=jnp.float64)
    if Sigma.ndim < 2 or Sigma.shape[-2:] != (2, 2):
        raise ValueError("Sigma must have trailing shape (2,2)")
    Sigma = jnp.broadcast_to(Sigma, measurements.shape[:-1] + (2, 2))
    mask = jnp.ones(measurements.shape[:-1], dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    if mask.shape != measurements.shape[:-1]:
        raise ValueError("mask must match the observation-slot axes")
    return measurements, landmarks, Sigma, mask


def sequential_updates(state, measurements, landmarks, Sigma, mask=None, min_range=1e-8):
    """Apply (slots,2) observations in supplied order, relinearizing each time.

    landmarks supplies the fixed known coordinate associated with each record.
    Sigma may be shared (2,2) or per slot (slots,2,2). False mask entries skip
    evaluation and return valid=False diagnostics. No automatic sorting occurs.
    Empty observation batches are supported. Sequential relinearization can
    differ from a stacked update and can depend on observation order.
    """
    state = _state64(state)
    measurements, landmarks, Sigma, mask = _observation_arrays(measurements, landmarks, Sigma, mask)
    if measurements.ndim != 2:
        raise ValueError("sequential measurements must have shape (slots,2)")

    def update(current, record):
        y, landmark, noise, requested = record
        return jax.lax.cond(
            requested, lambda _: measurement_update(current, y, landmark, noise, min_range),
            lambda _: (current, _skipped()), operand=None,
        )

    return jax.lax.scan(update, state, (measurements, landmarks, Sigma, mask))


def filter_known_landmarks(state, times, command, basis_stages, coefficients, Qc,
                           measurements, landmarks, Sigma, mask=None, min_range=1e-8):
    """Propagate and update at each grid time; return posterior histories.

    times: (N,) strictly increasing integration nodes, including every desired
      observation time. Prepare the grid with Stage 3A rk4_stage_times.
    basis_stages: (N-1,4,M+d), precomputed with the existing C++ adapter.
    measurements/landmarks: (N,slots,2), in deterministic slot order at each node.
    Sigma: shared (2,2) or per observation (N,slots,2,2).
    mask: (N,slots); use False at integration-only nodes or unused padded slots.

    Observations at times[0] update the supplied initial arrival tuple directly.
    At later nodes reuse the unmodified Stage 3A RK4 step, then sequentially
    update. Output states are posterior (N,3) / (N,3,3); diagnostic arrays have
    leading axes (N,slots). Capture command in a closure for jit. Time grids,
    associations, and masks are fixed during coefficient differentiation.
    """
    state = _state64(state)
    times, basis_stages = (jnp.asarray(value, dtype=jnp.float64) for value in (times, basis_stages))
    measurements, landmarks, Sigma, mask = _observation_arrays(measurements, landmarks, Sigma, mask)
    if times.ndim != 1 or times.shape[0] < 1:
        raise ValueError("times must have shape (N,) with N >= 1")
    if basis_stages.ndim != 3 or basis_stages.shape[:2] != (times.shape[0] - 1, 4):
        raise ValueError("basis_stages must have shape (N-1,4,M+d)")
    if measurements.ndim != 3 or measurements.shape[0] != times.shape[0]:
        raise ValueError("measurements must have shape (N,slots,2)")

    initial, initial_info = sequential_updates(state, measurements[0], landmarks[0], Sigma[0], mask[0], min_range)

    def advance(current, data):
        t, dt, basis, y, ell, noise, requested = data
        predicted = rk4_step(current, t, dt, command, basis, coefficients, Qc)
        updated, info = sequential_updates(predicted, y, ell, noise, requested, min_range)
        return updated, (updated, info)

    _, (history, update_history) = jax.lax.scan(
        advance, initial, (times[:-1], jnp.diff(times), basis_stages,
                           measurements[1:], landmarks[1:], Sigma[1:], mask[1:]),
    )
    poses, covariances = jax.tree_util.tree_map(
        lambda first, rest: jnp.concatenate((first[None], rest), axis=0), initial, history,
    )
    info = jax.tree_util.tree_map(
        lambda first, rest: jnp.concatenate((first[None], rest), axis=0), initial_info, update_history,
    )
    return FilterResult(poses, covariances, info)
