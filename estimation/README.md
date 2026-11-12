# Stage 3A: differentiable robot propagation

`models/unicycle.py` contains JAX float64 `dynamics(r,u,w)`,
`state_jacobian(r,u,w)`, and `input_noise_jacobian(r)` for robot pose
`r=[px,py,theta]`. `estimation/propagation.py` integrates the coupled tuple
`(r,P)` using the input-noise model `P_dot=A@P+P@A.T+G@Qc@G.T`.
`Qc` is a configurable symmetric positive semidefinite 2-by-2 continuous-time
input-noise spectral density; correlated inputs are supported. It is distinct
from the original baseline's discrete process covariance. Test values have no
claimed calibration to that covariance.

## Interface

- `rk4_stage_times(times)` validates a finite, strictly increasing NumPy time
  grid and returns `(steps,4)` stage times, ordered start/midpoint/midpoint/end.
- Precompute basis values through the existing `splines.UniformBasis` adapter,
  using elapsed times relative to the window start and horizon `L`.
- `rk4_step(state,t,dt,command,basis_stages,coefficients,Qc)` returns a new `(r,P)`
  tuple. The basis shape is `(4,M+d)` and `c=[c_v;c_omega]` has length `2(M+d)`.
- `propagate(state,times,command,basis_stages,coefficients,Qc)` scans over a
  nonuniform grid. Basis shape is `(len(times)-1,4,M+d)`. It returns pose and
  covariance histories including the initial tuple, with shapes `(len(times),3)`
  and `(len(times),3,3)`. A terminal tuple can initialize a later interval.

At each RK4 stage the code evaluates `command(stage_time)`, reconstructs the
two disturbance channels from that stage's basis row, and evaluates `f`, `A`,
and `G` at that stage's intermediate state. The covariance is an RK4 state
component, not a post-propagation approximation. There are no NumPy operations
or C++ basis evaluations inside the differentiable step/scan.

Run from the repository root with the Milestone 2 extension on `PYTHONPATH`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from estimation.propagation import propagate, rk4_stage_times
from splines import UniformBasis

jax.config.update("jax_enable_x64", True)  # Required; no silent float32 fallback.
window_start, L = 10.0, 2.0
times = window_start + np.array([0., .1, .35, .6, 1.1, 2.])
adapter = UniformBasis(degree=3, M=4)
B = adapter.precompute_jax(rk4_stage_times(times) - window_start, L)
initial = (jnp.zeros(3), jnp.eye(3) * .01)
Qc = jnp.diag(jnp.array([.03, .01]))

def command(t):
    return jnp.stack((1. + .1*jnp.cos(t), .2 + .05*jnp.sin(t)))

def prediction(c):
    return propagate(initial, times, command, B, c, Qc)

c = jnp.zeros(2 * adapter.num_coefficients)  # Zero disturbance for Baseline B.
poses, covariances = jax.jit(prediction)(c)
gradient = jax.jit(jax.grad(lambda c: jnp.sum(prediction(c)[0][-1]**2)
                          + jnp.trace(prediction(c)[1][-1])))(c)
```

Capture the command callable in a closure when applying `jax.jit`, or mark it
static. The callable must use JAX-compatible operations and return shape `(2,)`.
Changing integration times requires new basis arrays. A single time edge
returns the initial tuple only; `rk4_step` also permits `dt=0`.

## Numerical policies and limits

Enable JAX float64 before calling the model or propagation functions. Numerical
inputs are then converted to float64, including float32 input arrays. Shapes
are checked during tracing; finite values, nonnegative step durations, and
symmetric positive semidefinite initial covariance and `Qc` are caller
preconditions. The host-side stage-time helper checks the integration grid.

Heading remains unwrapped to keep propagation smooth through the angle branch
cut. The code applies no covariance symmetrization, jitter, eigenvalue clipping,
or positive semidefinite projection. Explicit RK4 does not guarantee positive
semidefinite covariance for arbitrary large steps; check step refinement and
covariance eigenvalues for each application.

Fourth-order convergence assumes sufficiently smooth inputs on each step.
Place step boundaries at spline knots with insufficient smoothness and at
command discontinuities. For piecewise constant commands, call `rk4_step` (or
`propagate` over each constant-input interval) with an interval-local command
callable that returns the held input at all four stages, including the right
endpoint. A global right-continuous callable can otherwise introduce the next
interval's command into the preceding step's final stage. Degree-zero splines
retain the endpoint caveat documented in `splines/README.md` and have not been
validated for fourth-order propagation.

The interface accepts precomputed constants, so gradients here are with respect
to the coefficient vector, not integration times or basis evaluation times.
Measurement updates, landmark augmentation, and coefficient optimization are
later stages. Full Milestone 3 completion still requires Stage 3B and Baseline B
replay on the saved simulation data.

Numerical results and exact test commands are in
[`artifacts/milestone3a/report.md`](../artifacts/milestone3a/report.md).
