"""Fixed-window innovation objective on a fixed requested measurement set.

The JAX evaluator reports invalid candidates explicitly. Its scalar is NaN
outside the valid domain; checked Python interfaces raise InvalidCandidateError.
All EKF mathematics comes from the unmodified Stage 3A/3B.1 public functions.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from estimation.known_landmark_ekf import UpdateDiagnostics, measurement_update
from estimation.propagation import rk4_stage_times, rk4_step
from models.range_bearing import range_bearing


COVARIANCE_TOLERANCE = 1e-10


class ObjectiveResult(NamedTuple):
    innovation_quadratic: jax.Array
    logdet_diagnostic: jax.Array
    gaussian_diagnostic: jax.Array
    requested_count: jax.Array
    valid_count: jax.Array
    invalid_geometry_count: jax.Array
    positive_definite: jax.Array
    candidate_valid: jax.Array
    innovations: jax.Array
    innovation_covariances: jax.Array
    final_pose: jax.Array
    final_covariance: jax.Array
    valid_mask: jax.Array
    masked_count: jax.Array
    rejected_count: jax.Array
    numerical_rejection_count: jax.Array
    numerical_valid: jax.Array
    states_finite: jax.Array
    covariances_symmetric: jax.Array
    covariances_psd: jax.Array
    prior_poses: jax.Array
    prior_covariances: jax.Array
    poses: jax.Array
    covariances: jax.Array


def _covariance_flags(P, tolerance):
    """Validate only; symmetrization/scaling below never changes EKF values."""
    finite = jnp.all(jnp.isfinite(P))
    safe = jnp.where(jnp.isfinite(P), P, 0.)
    scale = jnp.maximum(1., jnp.max(jnp.abs(safe)))
    scaled = safe / scale
    eigenvalues = jnp.linalg.eigvalsh((scaled + scaled.T) / 2)
    symmetric = finite & (jnp.max(jnp.abs(scaled - scaled.T)) <= tolerance / scale)
    psd = finite & (eigenvalues[0] >= -tolerance / scale)
    return finite, symmetric, psd


def _state_flags(state, tolerance):
    r, P = state
    finite, symmetric, psd = _covariance_flags(P, tolerance)
    return jnp.stack((finite & jnp.all(jnp.isfinite(r)), symmetric, psd))


def _innovation_pd(S, tolerance):
    finite, symmetric, _ = _covariance_flags(S, tolerance)
    safe = jnp.where(jnp.isfinite(S), S, 0.)
    scale = jnp.maximum(1., jnp.max(jnp.abs(safe)))
    scaled = safe / scale
    return finite & symmetric & (jnp.linalg.eigvalsh((scaled + scaled.T) / 2)[0] > 0)


def _empty_diagnostics():
    return UpdateDiagnostics(jnp.zeros(2), jnp.zeros((2, 3)), jnp.zeros(2),
                             jnp.eye(2), jnp.zeros((3, 2)), jnp.asarray(False))


def make_fixed_window_objective(
    arrival_state, times, command, basis_stages, Qc, measurements, landmarks,
    Sigma, mask, *, coefficient_count, min_range=1e-8,
    covariance_tolerance=COVARIANCE_TOLERANCE,
):
    """Capture a fixed prior and dataset and return ``evaluate(c)`` for JIT/AD.

    The valid domain requires finite coefficients/states, symmetric PSD state
    covariances (absolute symmetry/PSD tolerance 1e-10 by default), and every
    requested observation to have valid geometry and SPD innovation covariance.
    All integration-node priors and every sequential posterior are checked.
    Inputs/assignments/masks remain fixed; basis rows must match the physical
    grid and elapsed window times. No covariance repair, jitter or penalty is
    applied. Invalid candidates have NaN objective/diagnostics, never a reduced
    objective over the remaining observations. Gradients exist only within the
    valid domain, away from geometry and angular-wrap boundaries.

    On numerical failure, further propagation/updates are quarantined using
    finite internal placeholders and explicit rejection status. Such placeholders
    are not estimates and must not be consumed from an invalid result.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX float64 with jax.config.update("jax_enable_x64", True)')
    rk4_stage_times(times)  # Host validation, including finite increasing times.
    if not np.isfinite(covariance_tolerance) or covariance_tolerance < 0:
        raise ValueError("covariance_tolerance must be finite and nonnegative")
    if not np.isfinite(min_range) or min_range <= 0:
        raise ValueError("min_range must be finite and positive")
    arrival_state = tuple(jnp.array(v, dtype=jnp.float64, copy=True) for v in arrival_state)
    times, basis_stages, Qc, measurements, landmarks, Sigma = (
        jnp.array(v, dtype=jnp.float64, copy=True)
        for v in (times, basis_stages, Qc, measurements, landmarks, Sigma))
    mask = jnp.array(mask, dtype=bool, copy=True)
    if arrival_state[0].shape != (3,) or arrival_state[1].shape != (3, 3):
        raise ValueError("arrival_state must contain pose (3,) and covariance (3,3)")
    if (basis_stages.ndim != 3 or basis_stages.shape[:2] != (len(times)-1, 4)
            or coefficient_count != 2*basis_stages.shape[-1] or coefficient_count < 2):
        raise ValueError("basis_stages must be (N-1,4,n), coefficient_count=2*n")
    if (measurements.ndim != 3 or measurements.shape[0] != len(times)
            or measurements.shape[-1] != 2 or landmarks.shape != measurements.shape
            or mask.shape != measurements.shape[:-1] or Qc.shape != (2, 2)):
        raise ValueError("inconsistent measurements, landmarks, mask, or Qc shapes")
    Sigma = jnp.broadcast_to(Sigma, mask.shape + (2, 2))

    def evaluate(coefficients):
        coefficients = jnp.asarray(coefficients, dtype=jnp.float64)
        if coefficients.shape != (coefficient_count,):
            raise ValueError(f"coefficients must have shape ({coefficient_count},)")
        arrival_flags = _state_flags(arrival_state, covariance_tolerance)
        active = (jnp.all(arrival_flags) & jnp.all(jnp.isfinite(coefficients))
                  & jnp.all(jnp.stack(_covariance_flags(Qc, covariance_tolerance)))
                  & jnp.all(jnp.isfinite(basis_stages)))
        # These are quarantined storage, never a repaired covariance/estimate.
        storage = jax.lax.cond(active, lambda: arrival_state,
                               lambda: (jnp.zeros(3), jnp.eye(3)))

        def observe(carry, record):
            state, alive = carry
            y, ell, noise, requested = record

            def attempt():
                prediction, H, geometry = range_bearing(state[0], ell, min_range)
                S = H @ state[1] @ H.T + noise
                pd = _innovation_pd(S, covariance_tolerance)
                ready = (geometry & pd & _innovation_pd(noise, covariance_tolerance)
                         & jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(prediction))
                         & jnp.all(jnp.isfinite(H)))
                updated, info = jax.lax.cond(
                    ready, lambda: measurement_update(state, y, ell, noise, min_range),
                    lambda: (state, _empty_diagnostics()))
                flags = _state_flags(updated, covariance_tolerance)
                accepted = ready & jnp.all(flags)
                next_state = jax.lax.cond(accepted, lambda: updated, lambda: state)
                numerical_ok = (~geometry) | (ready & jnp.all(flags))
                return ((next_state, numerical_ok),
                        (info, accepted, ~geometry, geometry & ~accepted, flags, pd | ~geometry))

            def skip():
                return ((state, alive), (_empty_diagnostics(), jnp.asarray(False),
                        jnp.asarray(False), requested & ~alive,
                        jnp.full((3,), alive), jnp.asarray(True)))

            return jax.lax.cond(requested & alive, attempt, skip)

        def node(state, alive, y, ell, noise, requested):
            return jax.lax.scan(observe, (state, alive), (y, ell, noise, requested))

        (initial, alive), initial_updates = node(storage, active, measurements[0], landmarks[0], Sigma[0], mask[0])

        def advance(carry, data):
            state, alive = carry
            t, dt, basis, y, ell, noise, requested = data
            predicted = jax.lax.cond(
                alive, lambda: rk4_step(state, t, dt, command, basis, coefficients, Qc),
                lambda: state)
            flags = _state_flags(predicted, covariance_tolerance) & alive
            ready = jnp.all(flags)
            storage = jax.lax.cond(ready, lambda: predicted, lambda: state)
            (updated, alive), updates = node(storage, ready, y, ell, noise, requested)
            return (updated, alive), (predicted, flags, updated, updates)

        (terminal, alive), history = jax.lax.scan(
            advance, (initial, alive), (times[:-1], jnp.diff(times), basis_stages,
            measurements[1:], landmarks[1:], Sigma[1:], mask[1:]))
        predicted, prior_flags, posterior, update_history = history
        prepend = lambda first, rest: jnp.concatenate((first[None], rest), axis=0)
        updates = jax.tree_util.tree_map(prepend, initial_updates, update_history)
        info, valid, geometry_rejected, numerical_rejected, post_flags, innovation_pd = updates
        prior_poses, prior_covariances = jax.tree_util.tree_map(prepend, arrival_state, predicted)
        poses, covariances = jax.tree_util.tree_map(prepend, initial, posterior)
        prior_flags = prepend(arrival_flags, prior_flags)
        flags = jnp.all(prior_flags, axis=0) & jnp.all(post_flags, axis=(0, 1))

        solved = jnp.linalg.solve(info.innovation_covariance, info.innovation[..., None])[..., 0]
        terms = jnp.sum(info.innovation * solved, axis=-1)
        _, logdet = jnp.linalg.slogdet(info.innovation_covariance)
        quadratic = jnp.sum(jnp.where(valid, terms, 0.))
        logdet = jnp.sum(jnp.where(valid, logdet, 0.))
        finite_cost = jnp.all(jnp.isfinite(jnp.stack((quadratic, logdet, quadratic + logdet))))
        numerical_valid = active & alive & jnp.all(flags) & finite_cost
        candidate_valid = numerical_valid & jnp.all(valid | ~mask)
        # NaN denotes an undefined objective outside its domain, not a penalty.
        domain_value = lambda value: jax.lax.cond(candidate_valid, lambda: value, lambda: jnp.asarray(jnp.nan))
        requested_count = jnp.sum(mask, dtype=jnp.int32)
        valid_count = jnp.sum(valid, dtype=jnp.int32)
        return ObjectiveResult(
            domain_value(quadratic), domain_value(logdet), domain_value(quadratic+logdet),
            requested_count, valid_count, jnp.sum(geometry_rejected), jnp.all(innovation_pd),
            candidate_valid, info.innovation, info.innovation_covariance, terminal[0], terminal[1],
            valid, mask.size-requested_count, requested_count-valid_count, jnp.sum(numerical_rejected),
            numerical_valid, flags[0], flags[1], flags[2], prior_poses, prior_covariances, poses, covariances)

    return evaluate


class InvalidCandidateError(ValueError):
    """Candidate is outside the fixed-window objective's valid domain."""


def checked_evaluation(evaluator, coefficients):
    """Python boundary: raise before an invalid candidate can be used."""
    result = evaluator(coefficients)
    if not bool(result.candidate_valid):
        raise InvalidCandidateError(
            f"Invalid fixed-window candidate: requested={int(result.requested_count)}, "
            f"valid={int(result.valid_count)}, geometry_rejected={int(result.invalid_geometry_count)}, "
            f"numerical_rejected={int(result.numerical_rejection_count)}; "
            f"finite_states={bool(result.states_finite)}, covariance_symmetric={bool(result.covariances_symmetric)}, "
            f"covariance_psd={bool(result.covariances_psd)}, innovation_pd={bool(result.positive_definite)}, "
            f"numerical_valid={bool(result.numerical_valid)}")
    return result


def checked_value_and_grad(evaluator, coefficients):
    """Checked Python value/gradient boundary; never differentiate invalid data.

    Future SciPy callers must handle InvalidCandidateError by rejecting the
    trial/domain, rather than changing K or assigning an arbitrary finite cost.
    This function implements no coefficient search or optimization policy.
    """
    checked_evaluation(evaluator, coefficients)
    value, gradient = jax.value_and_grad(lambda c: innovation_quadratic(evaluator, c))(coefficients)
    if not np.isfinite(value) or not np.isfinite(np.asarray(gradient)).all():
        raise InvalidCandidateError("Nonfinite objective or gradient for fixed-window candidate")
    return value, gradient


def innovation_quadratic(evaluator, coefficients):
    """JAX scalar: J on the valid domain, NaN outside it; no finite penalties."""
    return evaluator(coefficients).innovation_quadratic
