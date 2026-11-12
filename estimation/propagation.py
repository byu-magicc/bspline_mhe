"""Coupled, manually written JAX RK4 for the robot pose and covariance.

Precompute basis rows with splines.UniformBasis at rk4_stage_times(times),
expressed as elapsed times from the spline window start. The differentiable
functions consume those constants and never evaluate the C++ binding.
"""

import jax
import jax.numpy as jnp
import numpy as np

from models.unicycle import dynamics, input_noise_jacobian, state_jacobian
from splines import disturbance_from_coefficients


def rk4_stage_times(times):
    """Host-side stage times (steps, 4) from finite, strictly increasing edges.

    Nonuniform intervals can end at arbitrary observation times. A single edge
    represents no propagation and returns an empty (0, 4) array. Convert these
    times to elapsed window times before passing to the existing basis adapter.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all():
        raise ValueError("times must be a nonempty finite one-dimensional array")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    midpoints = times[:-1] + (times[1:] - times[:-1]) / 2
    return np.stack((times[:-1], midpoints, midpoints, times[1:]), axis=-1)


def _state64(state):
    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX float64 with jax.config.update("jax_enable_x64", True)')
    r, P = (jnp.asarray(value, dtype=jnp.float64) for value in state)
    if r.shape != (3,) or P.shape != (3, 3):
        raise ValueError("state must be a (pose (3,), covariance (3,3)) tuple")
    return r, P


def rk4_step(state, t, dt, command, basis_stages, coefficients, Qc):
    """Advance (r,P) over [t,t+dt] using four precomputed basis rows.

    command(time) must be a pure JAX-compatible callable returning [v,omega].
    basis_stages has shape (4,n), in start/midpoint/midpoint/end order;
    coefficients has shape (2*n,) ordered [c_v;c_omega]. Qc is any symmetric
    positive semidefinite (2,2) input-noise spectral density, including correlated
    inputs. dt is a nonnegative scalar. Keep times and basis values consistent.

    Shapes and precision are checked during tracing; finite values, dt >= 0,
    and covariance/Qc symmetry and positive semidefiniteness are caller
    preconditions. No covariance projection or heading wrapping is applied.
    """
    state = _state64(state)
    t, dt, basis_stages, coefficients, Qc = (
        jnp.asarray(value, dtype=jnp.float64)
        for value in (t, dt, basis_stages, coefficients, Qc)
    )
    if t.shape != () or dt.shape != () or Qc.shape != (2, 2):
        raise ValueError("t and dt must be scalars and Qc must have shape (2,2)")
    if basis_stages.ndim != 2 or basis_stages.shape[0] != 4:
        raise ValueError("basis_stages must have shape (4,n)")

    def rhs(current, stage_time, basis_row):
        r, P = current
        u = command(stage_time)
        w = disturbance_from_coefficients(basis_row, coefficients)
        A = state_jacobian(r, u, w)
        G = input_noise_jacobian(r)
        return dynamics(r, u, w), A @ P + P @ A.T + G @ Qc @ G.T

    def intermediate(rate, scale):
        return jax.tree_util.tree_map(lambda y, k: y + scale * dt * k, state, rate)

    k1 = rhs(state, t, basis_stages[0])
    k2 = rhs(intermediate(k1, .5), t + dt / 2, basis_stages[1])
    k3 = rhs(intermediate(k2, .5), t + dt / 2, basis_stages[2])
    k4 = rhs(intermediate(k3, 1.), t + dt, basis_stages[3])
    return jax.tree_util.tree_map(
        lambda y, a, b, c, d: y + dt * (a + 2 * b + 2 * c + d) / 6,
        state, k1, k2, k3, k4,
    )


def propagate(state, times, command, basis_stages, coefficients, Qc):
    """Return (poses, covariances) at every time edge, including the initial one.

    Uses lax.scan over arbitrary positive step sizes. times has shape (N,),
    basis_stages (N-1,4,n); outputs have shapes (N,3) and (N,3,3). Prepare and
    validate the time grid with rk4_stage_times outside jit/grad. For a new grid,
    recompute the basis. Capture command in a closure when using jax.jit, or
    mark it as a static argument. A returned terminal tuple can start another
    propagation interval after a future measurement update.
    """
    state = _state64(state)
    times, basis_stages = (jnp.asarray(value, dtype=jnp.float64)
                           for value in (times, basis_stages))
    if times.ndim != 1 or times.shape[0] == 0:
        raise ValueError("times must have shape (N,) with N >= 1")
    if (basis_stages.ndim != 3 or basis_stages.shape[:2] != (times.shape[0] - 1, 4)):
        raise ValueError("basis_stages must have shape (N-1,4,n)")

    def advance(current, step):
        t, dt, basis = step
        updated = rk4_step(current, t, dt, command, basis, coefficients, Qc)
        return updated, updated

    _, history = jax.lax.scan(advance, state,
                              (times[:-1], jnp.diff(times), basis_stages))
    return jax.tree_util.tree_map(
        lambda initial, propagated: jnp.concatenate((initial[None], propagated), axis=0),
        state, history,
    )
