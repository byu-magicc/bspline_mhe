"""Fixed-capacity differentiable EKF-SLAM with supplied integer associations.

Inactive state/covariance storage is zero. Association lookup uses active IDs,
not IDs as indices; numeric sentinel values are not reserved. All numerical
work is JAX. Shape checks happen at trace time; covariance value conditions
(finite symmetric PSD P/Qc, SPD Sigma) are caller preconditions.
"""

from operator import index
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from estimation.known_landmark_ekf import UpdateDiagnostics
from models.range_bearing import innovation
from models.slam import augmented_dynamics, augmented_jacobians, augmented_range_bearing, inverse_range_bearing
from splines import disturbance_from_coefficients

MASKED, INITIALIZED, UPDATED, INVALID, OVERFLOW = range(5)


class SlamState(NamedTuple):
    x: jax.Array
    covariance: jax.Array
    active: jax.Array
    truth_ids: jax.Array


class ObservationResult(NamedTuple):
    diagnostics: UpdateDiagnostics
    status: jax.Array
    slot: jax.Array


class SlamResult(NamedTuple):
    states: SlamState
    observations: ObservationResult
    overflow: jax.Array


class LandmarkCapacityError(RuntimeError):
    pass


def _float64(value):
    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX float64 with jax.config.update("jax_enable_x64", True)')
    return jnp.asarray(value, dtype=jnp.float64)


def _ids(value):
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.signedinteger):
        raise ValueError("truth IDs must be signed integers (not floating-point IDs)")
    return value.astype(jnp.int64)


def neutralize_inactive(state):
    """Validate static shapes and remove all inactive storage from calculations."""
    x, P = _float64(state.x), _float64(state.covariance)
    active, ids = jnp.asarray(state.active, dtype=bool), _ids(state.truth_ids)
    if active.ndim != 1 or active.shape[0] < 1:
        raise ValueError("active must have shape (capacity,) with capacity >= 1")
    D = 3 + 2 * active.shape[0]
    if x.shape != (D,) or P.shape != (D, D) or ids.shape != active.shape:
        raise ValueError("SLAM state shapes do not match capacity")
    selected = jnp.concatenate((jnp.ones(3, dtype=bool), jnp.repeat(active, 2)))
    return SlamState(jnp.where(selected, x, 0.),
                     jnp.where(selected[:, None] & selected[None, :], P, 0.),
                     active, jnp.where(active, ids, 0))


def create_state(pose, covariance, capacity=16):
    capacity = index(capacity)
    if capacity < 1:
        raise ValueError("capacity must be positive")
    pose, covariance = _float64(pose), _float64(covariance)
    if pose.shape != (3,) or covariance.shape != (3, 3):
        raise ValueError("initial pose/covariance must have shapes (3,) / (3,3)")
    D = 3 + 2 * capacity
    return SlamState(jnp.zeros(D, dtype=jnp.float64).at[:3].set(pose),
                     jnp.zeros((D, D), dtype=jnp.float64).at[:3, :3].set(covariance),
                     jnp.zeros(capacity, dtype=bool), jnp.zeros(capacity, dtype=jnp.int64))


def initialize_landmark(state, slot, truth_id, y, Sigma):
    """Insert inverse-model mean/covariance into an inactive in-bounds slot.

    This low-level primitive assumes slot is inactive and y is valid. The
    observation dispatcher checks those conditions. Initialization consumes
    the observation once; it is not followed by correction with that same y.
    """
    state = neutralize_inactive(state)
    y, Sigma = _float64(y), _float64(Sigma)
    landmark, Jr, Jy = inverse_range_bearing(state.x[:3], y)
    # Jx has only three nonzero robot columns. An explicit reduction over them
    # also avoids an incompatible transposed DotThunk output layout observed
    # with the JAX 0.4.35 CPU backend when cross and cross.T feed scatters.
    cross = jnp.sum(state.covariance[:, :3, None] * Jr.T[None, :, :], axis=1)
    marginal = Jr @ cross[:3] + Jy @ Sigma @ Jy.T
    indices = 3 + 2 * slot + jnp.arange(2)
    P = state.covariance.at[:, indices].set(cross)
    P = P.at[indices, :].set(cross.T)
    P = P.at[indices[:, None], indices[None, :]].set(marginal)
    return SlamState(state.x.at[indices].set(landmark), P,
                     state.active.at[slot].set(True), state.truth_ids.at[slot].set(truth_id))


def rk4_slam_step(state, t, dt, command, basis_stages, coefficients, Qc):
    """Coupled JAX RK4, extending Stage 3A with stationary landmark blocks."""
    state = neutralize_inactive(state)
    t, dt, basis_stages, coefficients, Qc = map(_float64, (t, dt, basis_stages, coefficients, Qc))
    if t.shape != () or dt.shape != () or Qc.shape != (2, 2):
        raise ValueError("t/dt must be scalar; Qc must have shape (2,2)")
    if basis_stages.ndim != 2 or basis_stages.shape[0] != 4:
        raise ValueError("basis_stages must have shape (4,M+d)")
    initial = (state.x, state.covariance)

    def rhs(current, time, basis):
        x, P = current
        u = command(time)
        w = disturbance_from_coefficients(basis, coefficients)
        A, G = augmented_jacobians(x, u, w)
        return augmented_dynamics(x, u, w), A @ P + P @ A.T + G @ Qc @ G.T

    def intermediate(k, scale):
        return jax.tree_util.tree_map(lambda y, rate: y + scale * dt * rate, initial, k)

    k1 = rhs(initial, t, basis_stages[0])
    k2 = rhs(intermediate(k1, .5), t + dt / 2, basis_stages[1])
    k3 = rhs(intermediate(k2, .5), t + dt / 2, basis_stages[2])
    k4 = rhs(intermediate(k3, 1.), t + dt, basis_stages[3])
    x, P = jax.tree_util.tree_map(lambda y, a, b, c, d: y + dt * (a + 2*b + 2*c + d) / 6,
                                 initial, k1, k2, k3, k4)
    return neutralize_inactive(SlamState(x, P, state.active, state.truth_ids))


def _empty_info(D, status, slot=-1):
    diagnostics = UpdateDiagnostics(jnp.zeros(2, dtype=jnp.float64),
                                    jnp.zeros((2, D), dtype=jnp.float64),
                                    jnp.zeros(2, dtype=jnp.float64), jnp.eye(2, dtype=jnp.float64),
                                    jnp.zeros((D, 2), dtype=jnp.float64), jnp.asarray(False))
    return ObservationResult(diagnostics, jnp.asarray(status, dtype=jnp.int32),
                              jnp.asarray(slot, dtype=jnp.int32))


def observe(state, y, truth_id, Sigma, requested=True, min_range=1e-8):
    """Initialize/update one supplied ID; return explicit status and diagnostics.

    Arbitrary signed int64 IDs are matched only against active slots. Allocation
    uses the lowest free slot. Decisions depend on IDs, masks, and validity,
    not any optimized continuous association criterion. New observations need
    finite y and range > min_range. Existing updates also check geometry.
    """
    state = neutralize_inactive(state)
    y, Sigma = _float64(y), _float64(Sigma)
    truth_id = _ids(truth_id)
    min_range = _float64(min_range)
    if y.shape != (2,) or Sigma.shape != (2, 2) or truth_id.shape != () or min_range.shape != ():
        raise ValueError("expected y (2,), Sigma (2,2), scalar integer ID and scalar min_range")
    D = state.x.shape[0]

    def consume(_):
        matches = state.active & (state.truth_ids == truth_id)
        slot = jnp.argmax(matches)

        def correct(_):
            predicted, H, valid = augmented_range_bearing(state.x, slot, state.active, min_range)

            def valid_update(_):
                e = innovation(y, predicted)
                PHt = state.covariance @ H.T
                S = H @ PHt + Sigma
                K = jnp.linalg.solve(S, PHt.T).T
                x = state.x + K @ e
                J = jnp.eye(D, dtype=jnp.float64) - K @ H
                P = J @ state.covariance @ J.T + K @ Sigma @ K.T
                updated = neutralize_inactive(SlamState(x, P, state.active, state.truth_ids))
                info = ObservationResult(UpdateDiagnostics(predicted, H, e, S, K, valid),
                                          jnp.asarray(UPDATED, dtype=jnp.int32), slot.astype(jnp.int32))
                return updated, info

            return jax.lax.cond(valid, valid_update,
                                lambda _: (state, _empty_info(D, INVALID, slot)), operand=None)

        def new(_):
            free = ~state.active
            new_slot = jnp.argmax(free)

            def allocate(_):
                updated = initialize_landmark(state, new_slot, truth_id, y, Sigma)
                return updated, _empty_info(D, INITIALIZED, new_slot)

            return jax.lax.cond(jnp.any(free), allocate,
                                lambda _: (state, _empty_info(D, OVERFLOW)), operand=None)

        valid_y = jnp.all(jnp.isfinite(y)) & jnp.isfinite(min_range) & (min_range > 0) & (y[0] > min_range)
        return jax.lax.cond(valid_y,
                            lambda _: jax.lax.cond(jnp.any(matches), correct, new, operand=None),
                            lambda _: (state, _empty_info(D, INVALID)), operand=None)

    return jax.lax.cond(jnp.asarray(requested, dtype=bool), consume,
                        lambda _: (state, _empty_info(D, MASKED)), operand=None)


def sequential_observations(state, measurements, truth_ids, Sigma, mask=None, min_range=1e-8):
    state = neutralize_inactive(state)
    y, ids, noise, mask = _observation_arrays(measurements, truth_ids, Sigma, mask)
    if y.ndim != 2:
        raise ValueError("sequential measurements must have shape (slots,2)")

    def update(current, row):
        return observe(current, *row[:3], requested=row[3], min_range=min_range)

    return jax.lax.scan(update, state, (y, ids, noise, mask))


def _observation_arrays(measurements, truth_ids, Sigma, mask):
    y, ids, noise = _float64(measurements), _ids(truth_ids), _float64(Sigma)
    if y.ndim < 2 or y.shape[-1] != 2 or ids.shape != y.shape[:-1]:
        raise ValueError("measurement (...,slots,2) and ID shapes must match")
    if noise.ndim < 2 or noise.shape[-2:] != (2, 2):
        raise ValueError("Sigma must have trailing shape (2,2)")
    noise = jnp.broadcast_to(noise, y.shape[:-1] + (2, 2))
    mask = jnp.ones(ids.shape, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    if mask.shape != ids.shape:
        raise ValueError("mask and ID shapes must match")
    return y, ids, noise, mask


def filter_slam(state, times, command, basis_stages, coefficients, Qc,
                 measurements, truth_ids, Sigma, mask=None, min_range=1e-8):
    """Posterior histories at increasing grid times; padded observation slots.

    Qc may be shared (2,2) or per interval (N-1,2,2). State metadata is scanned
    with fixed shapes. Reuse Stage 3A rk4_stage_times for the host grid and the
    existing spline adapter. Initial-time observations consume the arrival
    prior once. Returned overflow is sticky across the entire history.
    """
    state = neutralize_inactive(state)
    times, B = _float64(times), _float64(basis_stages)
    y, ids, noise, mask = _observation_arrays(measurements, truth_ids, Sigma, mask)
    if times.ndim != 1 or times.shape[0] < 1:
        raise ValueError("times must have shape (N,) with N >= 1")
    N = times.shape[0]
    if B.ndim != 3 or B.shape[:2] != (N-1, 4) or y.ndim != 3 or y.shape[0] != N:
        raise ValueError("basis (N-1,4,M+d) and observations (N,slots,2) must match times")
    Qc = jnp.broadcast_to(_float64(Qc), (N-1, 2, 2))
    initial, first_info = sequential_observations(state, y[0], ids[0], noise[0], mask[0], min_range)

    def advance(current, row):
        t, dt, basis, intensity, obs, obs_ids, obs_noise, requested = row
        predicted = rk4_slam_step(current, t, dt, command, basis, coefficients, intensity)
        updated, info = sequential_observations(predicted, obs, obs_ids, obs_noise, requested, min_range)
        return updated, (updated, info)

    _, (history, records) = jax.lax.scan(advance, initial,
        (times[:-1], jnp.diff(times), B, Qc, y[1:], ids[1:], noise[1:], mask[1:]))
    states = jax.tree_util.tree_map(lambda first, rest: jnp.concatenate((first[None], rest)), initial, history)
    records = jax.tree_util.tree_map(lambda first, rest: jnp.concatenate((first[None], rest)), first_info, records)
    return SlamResult(states, records, jnp.any(records.status == OVERFLOW))


def raise_on_overflow(result):
    """Python boundary check, called after compiled execution and device sync."""
    if bool(np.asarray(jax.device_get(result.overflow))):
        positions = np.argwhere(np.asarray(jax.device_get(result.observations.status)) == OVERFLOW)
        raise LandmarkCapacityError(f"Landmark capacity {result.states.active.shape[-1]} exhausted; "
                                    f"overflow observations at grid/observation slots {positions.tolist()}. "
                                    "Increase capacity and rerun; existing landmarks were preserved.")
