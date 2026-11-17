"""Checked fixed-window orchestration of the preserved Stage 3B.2 EKF-SLAM.

The arrival posterior, data and discrete update plan are immutable constants.
Only spline coefficients vary. All EKF equations remain in estimation.slam.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from estimation.fixed_window_objective import COVARIANCE_TOLERANCE, _covariance_flags, _innovation_pd
from estimation.propagation import rk4_stage_times
from estimation.slam import (
    INITIALIZED, INVALID, MASKED, UPDATED, LandmarkCapacityError, SlamState,
    _empty_info, create_state, observe, rk4_slam_step,
)
from models.slam import augmented_range_bearing


class SlamWindowResult(NamedTuple):
    innovation_quadratic: object
    logdet_diagnostic: object
    gaussian_diagnostic: object
    requested_count: object
    valid_count: object
    masked_count: object
    rejected_count: object
    invalid_geometry_count: object
    numerical_rejection_count: object
    positive_definite: object
    candidate_valid: object
    numerical_valid: object
    states_finite: object
    covariances_symmetric: object
    covariances_psd: object
    inactive_storage_neutral: object
    innovations: object
    innovation_covariances: object
    valid_mask: object
    initialization_mask: object
    requested_correction_mask: object
    requested_initialization_mask: object
    initialization_count: object
    correction_count: object
    statuses: object
    slots: object
    states: SlamState
    prior_states: SlamState
    final_state: SlamState


def fixed_observation_plan(arrival, ids, mask):
    """Freeze ID lookup, lowest-free-slot births and correction decisions.

    The plan depends only on arrival metadata, integer IDs and requested masks;
    neither coefficients, measured values nor true landmark coordinates enter.
    """
    active = np.asarray(arrival.active, dtype=bool)
    birth_ids = np.asarray(arrival.truth_ids)
    ids, mask = np.asarray(ids), np.asarray(mask, dtype=bool)
    if ids.shape != mask.shape or ids.ndim != 2 or not np.issubdtype(ids.dtype, np.signedinteger):
        raise ValueError('IDs must be signed integers matching a (times,slots) mask')
    if not np.issubdtype(birth_ids.dtype, np.signedinteger):
        raise ValueError('arrival IDs must be signed integers')
    if birth_ids.shape != active.shape or len(set(birth_ids[active].tolist())) != int(active.sum()):
        raise ValueError('arrival active IDs must be unique with matching shapes')
    mapping = {int(truth_id): slot for slot, truth_id in enumerate(birth_ids) if active[slot]}
    statuses, slots = np.full(ids.shape, MASKED, dtype=np.int32), np.full(ids.shape, -1, dtype=np.int32)
    for position in np.ndindex(ids.shape):
        if not mask[position]:
            continue
        truth_id = int(ids[position])
        if truth_id not in mapping:
            free = [slot for slot in range(len(active)) if slot not in mapping.values()]
            if not free:
                raise LandmarkCapacityError(f'fixed observation plan exhausts capacity {len(active)} at {position}')
            mapping[truth_id] = free[0]
            statuses[position] = INITIALIZED
        else:
            statuses[position] = UPDATED
        slots[position] = mapping[truth_id]
    return statuses, slots


def _state_flags(state, tolerance):
    selected = jnp.r_[jnp.ones(3, dtype=bool), jnp.repeat(state.active, 2)]
    finite, symmetric, psd = _covariance_flags(state.covariance, tolerance)
    neutral = (jnp.all(jnp.where(selected, 0., state.x) == 0.)
               & jnp.all(jnp.where(selected[:, None] & selected[None, :], 0., state.covariance) == 0.)
               & jnp.all(jnp.where(state.active, 0, state.truth_ids) == 0))
    # PSD of the padded matrix is equivalent to PSD on its active subspace
    # when inactive storage is exactly zero. Zeros are not negative eigenvalues.
    return jnp.stack((finite & jnp.all(jnp.isfinite(state.x)), symmetric, psd, neutral))


def make_slam_window_objective(arrival, times, command, basis_stages, Qc,
                               measurements, truth_ids, Sigma, mask, *,
                               coefficient_count=14, min_range=1e-8,
                               covariance_tolerance=COVARIANCE_TOLERANCE):
    """Return a JIT/AD-compatible evaluator with Stage 4A checked-boundary fields.

    requested_count/valid_count include all consumed observations. valid_mask
    and the objective include corrections only. An invalid record makes the
    entire objective undefined (NaN), never a reduced sum. Numerical failures
    quarantine subsequent arithmetic. Invalid histories are not estimates.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('Stage 4C requires JAX float64')
    rk4_stage_times(times)
    if (not np.isfinite(covariance_tolerance) or covariance_tolerance < 0
            or not np.isfinite(min_range) or min_range <= 0):
        raise ValueError('invalid covariance tolerance or minimum range')
    planned, planned_slots = fixed_observation_plan(arrival, truth_ids, mask)
    arrival = SlamState(jnp.array(arrival.x,dtype=jnp.float64,copy=True),
        jnp.array(arrival.covariance,dtype=jnp.float64,copy=True),
        jnp.array(arrival.active,dtype=bool,copy=True),jnp.array(arrival.truth_ids,dtype=jnp.int64,copy=True))
    times, B, Qc, y, noise = (jnp.array(value, dtype=jnp.float64, copy=True)
                              for value in (times, basis_stages, Qc, measurements, Sigma))
    ids, mask = jnp.array(truth_ids, dtype=jnp.int64, copy=True), jnp.array(mask, dtype=bool, copy=True)
    expected, expected_slots = jnp.asarray(planned), jnp.asarray(planned_slots)
    capacity = len(arrival.active)
    dimension = 3+2*capacity
    if (arrival.x.shape != (dimension,) or arrival.covariance.shape != (dimension, dimension)
            or arrival.truth_ids.shape != (capacity,) or capacity < 1):
        raise ValueError('arrival state shapes must match a positive capacity')
    if (B.ndim != 3 or B.shape[:2] != (len(times)-1, 4) or coefficient_count != 2*B.shape[-1]
            or Qc.shape != (2, 2) or y.shape != mask.shape+(2,) or mask.shape[0] != len(times)):
        raise ValueError('inconsistent grid, basis, coefficient or observation shapes')
    noise = jnp.broadcast_to(noise, mask.shape+(2, 2))
    neutral_state = create_state(jnp.zeros(3), jnp.eye(3), capacity)

    def evaluate(coefficients):
        coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
        if coefficients.shape != (coefficient_count,):
            raise ValueError('coefficient shape does not match precomputed basis')
        arrival_flags = _state_flags(arrival, covariance_tolerance)
        initial_ok = (jnp.all(arrival_flags) & jnp.all(jnp.isfinite(coefficients))
                      & jnp.all(jnp.stack(_covariance_flags(Qc, covariance_tolerance)))
                      & jnp.all(jnp.isfinite(B)))
        storage = jax.lax.cond(initial_ok, lambda: arrival, lambda: neutral_state)

        def observation(carry, row):
            state, alive = carry
            measurement, truth_id, R, requested, status, slot = row

            def attempt():
                valid_y = jnp.all(jnp.isfinite(measurement)) & (measurement[0] > min_range)

                def correction_precheck():
                    prediction, H, geometry = augmented_range_bearing(state.x, slot, state.active, min_range)
                    S = H@state.covariance@H.T+R
                    return (geometry & jnp.all(jnp.isfinite(prediction)) & jnp.all(jnp.isfinite(H)),
                            _innovation_pd(S, covariance_tolerance))

                geometry, pd = jax.lax.cond(status == INITIALIZED,
                    lambda: (valid_y, jnp.asarray(True)), correction_precheck)
                ready = geometry & pd & _innovation_pd(R, covariance_tolerance) & valid_y
                updated, info = jax.lax.cond(ready,
                    lambda: observe(state, measurement, truth_id, R, min_range=min_range),
                    lambda: (state, _empty_info(dimension, INVALID, slot)))
                flags = _state_flags(updated, covariance_tolerance)
                accepted = ready & jnp.all(flags) & (info.status == status) & (info.slot == slot)
                safe = jax.lax.cond(accepted, lambda: updated, lambda: state)
                return ((safe, accepted), (info, accepted, ~geometry, geometry & ~accepted, flags, pd))

            def skip():
                return ((state, alive), (_empty_info(dimension, MASKED), jnp.asarray(False),
                        jnp.asarray(False), requested & ~alive, jnp.full((4,), alive), jnp.asarray(True)))

            return jax.lax.cond(requested & alive, attempt, skip)

        def node(state, alive, index):
            return jax.lax.scan(observation, (state, alive),
                (y[index], ids[index], noise[index], mask[index], expected[index], expected_slots[index]))

        (initial, alive), first_info = node(storage, initial_ok, 0)

        def advance(carry, index):
            state, alive = carry
            predicted = jax.lax.cond(alive,
                lambda: rk4_slam_step(state, times[index-1], times[index]-times[index-1],
                                     command, B[index-1], coefficients, Qc), lambda: state)
            flags = _state_flags(predicted, covariance_tolerance) & alive
            ready = jnp.all(flags)
            safe = jax.lax.cond(ready, lambda: predicted, lambda: state)
            (updated, alive), records = node(safe, ready, index)
            return (updated, alive), (predicted, flags, updated, records)

        (terminal, alive), history = jax.lax.scan(advance, (initial, alive), jnp.arange(1, len(times)))
        predicted, prior_flags, posterior, info_history = history
        prepend = lambda first, rest: jnp.concatenate((first[None], rest))
        info, accepted, geometric, numerical, posterior_flags, pd = jax.tree_util.tree_map(prepend, first_info, info_history)
        states = jax.tree_util.tree_map(prepend, initial, posterior)
        priors = jax.tree_util.tree_map(prepend, arrival, predicted)
        flags = jnp.all(prepend(arrival_flags, prior_flags), axis=0) & jnp.all(posterior_flags, axis=(0, 1))
        corrections = accepted & (expected == UPDATED)
        initializations = accepted & (expected == INITIALIZED)
        solve = jnp.linalg.solve(info.diagnostics.innovation_covariance, info.diagnostics.innovation[..., None])[..., 0]
        quadratic = jnp.sum(jnp.where(corrections, jnp.sum(info.diagnostics.innovation*solve, axis=-1), 0.))
        _, logdets = jnp.linalg.slogdet(info.diagnostics.innovation_covariance)
        logdet = jnp.sum(jnp.where(corrections, logdets, 0.))
        numerical_ok = initial_ok & alive & jnp.all(flags) & jnp.all(jnp.isfinite(jnp.stack((quadratic, logdet, quadratic+logdet))))
        valid = numerical_ok & jnp.all(accepted | ~mask)
        domain = lambda value: jax.lax.cond(valid, lambda: value, lambda: jnp.asarray(jnp.nan))
        requested_count, valid_count = jnp.sum(mask), jnp.sum(accepted)
        return SlamWindowResult(domain(quadratic), domain(logdet), domain(quadratic+logdet),
            requested_count, valid_count, mask.size-requested_count, requested_count-valid_count,
            jnp.sum(geometric), jnp.sum(numerical), jnp.all(pd), valid, numerical_ok,
            flags[0], flags[1], flags[2], flags[3], info.diagnostics.innovation,
            info.diagnostics.innovation_covariance, corrections, initializations,
            expected == UPDATED, expected == INITIALIZED, jnp.sum(initializations), jnp.sum(corrections),
            info.status, info.slot, states, priors, terminal)

    return evaluate
