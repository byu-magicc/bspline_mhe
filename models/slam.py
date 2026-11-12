"""Continuous fixed-capacity SLAM and inverse range-bearing Jacobians."""

import jax
import jax.numpy as jnp

from models.unicycle import dynamics, state_jacobian, input_noise_jacobian
from models.range_bearing import range_bearing


def _array64(value):
    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX float64 with jax.config.update("jax_enable_x64", True)')
    return jnp.asarray(value, dtype=jnp.float64)


def augmented_dynamics(x, u, w):
    x = _array64(x)
    return jnp.concatenate((dynamics(x[:3], u, w), jnp.zeros_like(x[3:])))


def augmented_jacobians(x, u, w):
    """A and G with stationary, noise-free landmark rows/columns."""
    x = _array64(x)
    D = x.shape[0]
    A = jnp.zeros((D, D), dtype=jnp.float64).at[:3, :3].set(state_jacobian(x[:3], u, w))
    G = jnp.zeros((D, 2), dtype=jnp.float64).at[:3].set(input_noise_jacobian(x[:3]))
    return A, G


def inverse_range_bearing(r, y):
    """Return initialized landmark, J_r (2,3), J_y (2,2)."""
    r, y = _array64(r), _array64(y)
    if r.shape != (3,) or y.shape != (2,):
        raise ValueError("r and y must have shapes (3,) and (2,)")
    rho, beta = y
    phi = r[2] + beta
    co, si = jnp.cos(phi), jnp.sin(phi)
    landmark = r[:2] + rho * jnp.stack((co, si))
    Jr = jnp.array([[1., 0., -rho * si], [0., 1., rho * co]])
    Jy = jnp.array([[co, -rho * si], [si, rho * co]])
    return landmark, Jr, Jy


def augmented_range_bearing(x, slot, active, min_range=1e-8):
    """Return h, full H, validity; inactive/out-of-bounds slots are invalid."""
    x = _array64(x)
    active = jnp.asarray(active, dtype=bool)
    capacity = active.shape[0]
    if x.shape != (3 + 2 * capacity,) or capacity < 1:
        raise ValueError("x must match positive landmark capacity")
    slot = jnp.asarray(slot)
    if slot.shape != () or not jnp.issubdtype(slot.dtype, jnp.integer):
        raise ValueError("slot must be a scalar integer")
    safe_slot = jnp.clip(slot, 0, capacity - 1)
    selected = (slot >= 0) & (slot < capacity) & active[safe_slot]

    def evaluate(_):
        indices = 3 + 2 * safe_slot + jnp.arange(2)
        prediction, Hr, valid = range_bearing(x[:3], x[indices], min_range)
        H = jnp.zeros((2, x.shape[0]), dtype=jnp.float64).at[:, :3].set(Hr)
        H = H.at[:, indices].set(-Hr[:, :2])
        return prediction, H, valid

    return jax.lax.cond(selected, evaluate,
                        lambda _: (jnp.zeros(2, dtype=jnp.float64),
                                   jnp.zeros((2, x.shape[0]), dtype=jnp.float64), jnp.asarray(False)),
                        operand=None)
