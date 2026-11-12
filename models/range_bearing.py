"""Float64 range-bearing model for robot pose and a fixed known landmark."""

import jax
import jax.numpy as jnp


def _array64(value, shape, name):
    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX float64 with jax.config.update("jax_enable_x64", True)')
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return value


def wrap_angle(angle):
    """Wrap to [-pi, pi); differentiable away from the branch cut."""
    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX float64 with jax.config.update("jax_enable_x64", True)')
    angle = jnp.asarray(angle, dtype=jnp.float64)
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def range_bearing(r, landmark, min_range=1e-8):
    """Return (prediction (2,), analytical H (2,3), geometry_valid).

    The predicted bearing is atan2(dy,dx)-theta, without wrapping; wrap the
    bearing innovation. Valid geometry requires finite coordinates and pose,
    finite positive min_range, and distance strictly greater than min_range.
    Invalid geometry returns zeros and False without evaluating singular
    sqrt/atan2 derivatives. The validity boundary is a discrete decision.
    """
    r = _array64(r, (3,), "r")
    landmark = _array64(landmark, (2,), "landmark")
    min_range = _array64(min_range, (), "min_range")
    delta = landmark - r[:2]
    q = delta @ delta
    valid = (jnp.all(jnp.isfinite(r)) & jnp.all(jnp.isfinite(landmark))
             & jnp.isfinite(min_range) & (min_range > 0) & (q > min_range**2))

    def evaluate(_):
        dx, dy = delta
        distance = jnp.sqrt(q)
        prediction = jnp.stack((distance, jnp.arctan2(dy, dx) - r[2]))
        H = jnp.stack((jnp.array([-dx / distance, -dy / distance, 0.]),
                       jnp.array([dy / q, -dx / q, -1.])))
        return prediction, H

    prediction, H = jax.lax.cond(
        valid, evaluate,
        lambda _: (jnp.zeros(2, dtype=jnp.float64), jnp.zeros((2, 3), dtype=jnp.float64)),
        operand=None,
    )
    return prediction, H, valid


def innovation(y, prediction):
    """Range residual and consistently wrapped bearing residual."""
    y = _array64(y, (2,), "y")
    prediction = _array64(prediction, (2,), "prediction")
    residual = y - prediction
    return jnp.stack((residual[0], wrap_angle(residual[1])))
