"""Float64 robot-pose dynamics and analytical Jacobians from project.md."""

import jax
import jax.numpy as jnp


def _vector(value, length, name):
    if not jax.config.x64_enabled:
        raise RuntimeError('Enable JAX float64 with jax.config.update("jax_enable_x64", True)')
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return value


def dynamics(r, u, w):
    """f(r,u,w) for pose [px,py,theta], command [v,omega], and [wv,womega]."""
    r, u, w = _vector(r, 3, "r"), _vector(u, 2, "u"), _vector(w, 2, "w")
    speed = u[0] + w[0]
    return jnp.stack((speed * jnp.cos(r[2]), speed * jnp.sin(r[2]), u[1] + w[1]))


def state_jacobian(r, u, w):
    """Analytical df/dr, including the linear disturbance in effective speed."""
    r, u, w = _vector(r, 3, "r"), _vector(u, 2, "u"), _vector(w, 2, "w")
    speed = u[0] + w[0]
    z = jnp.zeros((), dtype=jnp.float64)
    return jnp.stack((jnp.stack((z, z, -speed * jnp.sin(r[2]))),
                      jnp.stack((z, z, speed * jnp.cos(r[2]))),
                      jnp.stack((z, z, z))))


def input_noise_jacobian(r):
    """G mapping linear/angular input noise into the robot pose derivative."""
    r = _vector(r, 3, "r")
    z, one = jnp.zeros((), dtype=jnp.float64), jnp.ones((), dtype=jnp.float64)
    return jnp.stack((jnp.stack((jnp.cos(r[2]), z)),
                      jnp.stack((jnp.sin(r[2]), z)),
                      jnp.stack((z, one))))
