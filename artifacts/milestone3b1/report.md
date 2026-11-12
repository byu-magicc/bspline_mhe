# Stage 3B.1: differentiable EKF with known landmarks

Implemented a robot-pose continuous-discrete EKF with fixed, known landmark
coordinates and sequential range-bearing updates. The state remains
`r=[px,py,theta]` with a 3-by-3 covariance. Propagation calls the unchanged
Stage 3A `rk4_step`, including its coupled covariance dynamics
`P_dot=A P+P A.T+G Qc G.T`. All four RK4 stages use the existing precomputed
C++ uniform B-spline basis and coefficient ordering `c=[c_v;c_omega]`.

This is the known-landmark portion of Stage 3B. Full Baseline B SLAM and the
Milestone 3 replay completion gate remain future work.

## Files and preservation

Added files:

- `models/range_bearing.py`: predicted measurements, analytical robot-pose
  measurement Jacobian, geometry validity, and wrapped bearing innovations.
- `estimation/known_landmark_ekf.py`: one measurement update, sequential updates,
  and propagation/update scans over a supplied time grid.
- `tests/test_known_landmark_ekf.py`: numerical, localization, and gradient tests.
- `artifacts/milestone3b1/report.md`: this implementation and validation record.

All **243 preexisting files** matched a content-hash snapshot after the tests,
including Stage 3A code/results, baseline sources, diagnostic tools, earlier
tests and artifacts, the spline adapter, and project documents. The sibling
library remains clean on `uniform_development`, commit
`bb334d0f75cd0a244c73c0de3468d049e0185ebe`.

## Measurement and update policy

For `delta=landmark-r[:2]`, `q=delta.T@delta`, the prediction is
`[sqrt(q), atan2(delta_y,delta_x)-theta]`. The analytical Jacobian is exactly the
robot-pose Jacobian specified in the task. The prediction's bearing remains
unwrapped. Only the bearing innovation is wrapped, consistently to `[-pi,pi)`.
Pose heading remains unwrapped, matching Stage 3A and avoiding an added state
branch cut during propagation/update differentiation.

`min_range` is configurable and defaults to `1e-8` in position units. Geometry
is valid only when the pose and landmark coordinates are finite, the threshold
is finite and positive, and the distance is **strictly greater** than the
threshold. At or below the threshold the update is skipped, the input state
and covariance are retained exactly, and diagnostics report `valid=False`.
The singular range/bearing calculation is inside `jax.lax.cond`, so coincident
landmarks do not evaluate `sqrt`/`atan2` derivatives at zero. This is a numerical
geometry check, independent of any baseline association rule or threshold.

For valid observations, the code forms `e`, `H`, `S=H P H.T+Sigma`, and computes
the gain with `solve(S,(P H.T).T).T`. It applies `r+=K e` and the Joseph form
`P+=(I-KH) P (I-KH).T+K Sigma K.T`. There is no explicit inverse, covariance
projection, or `stop_gradient` on any EKF quantity.

Observations at a timestamp are processed in their supplied slot order. Every
observation relinearizes at the posterior state/covariance from the preceding
observation. Therefore order matters for nonlinear updates, and sequential
updates can differ from one stacked update linearized at the original prior.
Known associations are supplied directly as fixed landmark coordinates for
each observation; there is no association search or reassociation.

## Interfaces

Enable `jax.config.update("jax_enable_x64", True)` before numerical use. Like
Stage 3A, the new model and filter require float64 and raise an explicit error
when it is disabled. Inputs are converted to float64 after that check.

- `range_bearing(r,landmark,min_range=1e-8)` returns `(prediction,H,valid)` with
  shapes `(2,)`, `(2,3)`, and scalar boolean.
- `innovation(y,prediction)` returns `(2,)`, wrapping only its bearing entry.
- `measurement_update(state,y,landmark,Sigma,min_range=1e-8)` returns
  `((posterior_pose,posterior_covariance), UpdateDiagnostics)`.
- `sequential_updates(state,measurements,landmarks,Sigma,mask=None,min_range=1e-8)`
  accepts `(slots,2)` measurements and associated coordinates. Noise covariance
  is shared `(2,2)` or per-slot `(slots,2,2)`. A `(slots,)` mask skips unused
  records; an omitted mask requests all records. Empty batches are supported.
- `filter_known_landmarks(state,times,command,basis_stages,coefficients,Qc,
  measurements,landmarks,Sigma,mask=None,min_range=1e-8)` accepts an integration
  grid `times (N,)`, stage basis `(N-1,4,M+d)`, observations and associated
  coordinates `(N,slots,2)`, and optional mask `(N,slots)`. `Sigma` may be shared
  `(2,2)` or per observation `(N,slots,2,2)`.

The complete filter returns `FilterResult(poses,covariances,updates)` with
posterior shapes `(N,3)` and `(N,3,3)`. Diagnostic fields `prediction`, `jacobian`,
`innovation`, `innovation_covariance`, `gain`, and `valid` have leading axes
`(N,slots)` and trailing shapes `(2,)`, `(2,3)`, `(2,)`, `(2,2)`, `(3,2)`, and
`()`, respectively. At the initial timestamp, observations update the supplied
arrival tuple directly. At subsequent timestamps, propagation precedes updates.

Use a strictly increasing integration grid containing every observation time,
adding integration-only nodes as needed for RK4 accuracy. Multiple observations
at one time occupy multiple slots at one node, rather than duplicate grid times.
Set masks to false at nodes without measurements and in unused slots. The slot
axis is fixed per call for JIT/scan compatibility; the number of actual records
can vary with masks. Skipped diagnostics are finite neutral placeholders:
zero prediction/Jacobian/innovation/gain, identity innovation covariance, and
`valid=False`. Use `valid` when accumulating statistics or covariance-based costs;
identity placeholders are not observed innovation covariances.

Example setup to run from the repository root with the extension on
`PYTHONPATH`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from estimation.known_landmark_ekf import filter_known_landmarks
from estimation.propagation import rk4_stage_times
from splines import UniformBasis

jax.config.update("jax_enable_x64", True)
times = np.array([0., .07, .23, .51])
adapter = UniformBasis(degree=3, M=4)
B = adapter.precompute_jax(rk4_stage_times(times), L=times[-1])
initial = (jnp.zeros(3), jnp.diag(jnp.array([.1, .1, .03])))
known_landmarks = np.broadcast_to([[3., 1.], [-2., 3.]], (len(times), 2, 2))
observations = np.array([[[3.16, .32], [3.61, 2.16]],
                         [[0., 0.], [0., 0.]],
                         [[2.94, .30], [3.74, 2.15]],
                         [[2.68, .29], [0., 0.]]])
mask = np.array([[True, True], [False, False], [True, True], [True, False]])

def run(c):
    return filter_known_landmarks(
        initial, times, lambda t: jnp.array([1., .2]), B, c,
        jnp.diag(jnp.array([.01, .003])), observations, known_landmarks,
        jnp.diag(jnp.array([.05**2, .015**2])), mask,
    )

result = jax.jit(run)(jnp.zeros(2 * adapter.num_coefficients))
```

The command callable must use JAX-compatible operations and should be captured
in the JIT closure. Recompute basis arrays when the integration/window grid
changes. Stage 3A's interval-local command policy for command discontinuities
still applies; callers can compose `propagate` and `sequential_updates` over
individual held-command intervals.

## Test commands and outcomes

```sh
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q -s tests/test_known_landmark_ekf.py
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q \
  tests/test_estimator_a.py tests/test_diagnose_initializations.py \
  tests/test_visualize_estimator_a.py tests/test_uniform_basis.py \
  tests/test_robot_propagation.py
```

The new suite passed **19 tests in 11.03 s**. The regression command passed
**161 tests in 21.04 s**, covering Milestone 1, Milestone 2, and Stage 3A.
Both commands exited successfully on their first runs. Validation used Python
3.12.6, NumPy 2.0.2, JAX/JAXLIB 0.4.35, pytest 8.3.3, macOS, and JAX float64.
These are suite wall times including compilation, not steady-state EKF runtime
benchmarks.

| Check | Result |
| --- | ---: |
| Analytical `H` vs. `jax.jacfwd`, seed-13 cases | max absolute error `1.110223e-16` |
| Analytical `H` vs. central differences, step `1e-6` | max absolute error `3.043112e-10` |
| Measurement predictions vs. independent geometry | passed at absolute tolerance `2e-15` |
| One Joseph update vs. independent NumPy reference | passed at absolute tolerance `2e-15` |
| Wrapped innovations in both directions across pi | passed at absolute tolerance `3e-15` |
| Sequential vs. stacked posterior pose norm difference | `1.012036e-2` |

Sixteen Jacobian cases stay away from zero distance and the `atan2` branch cut.
Jacobian finite differences use `rtol=2e-8`, `atol=1e-9`; autodiff comparisons
use absolute tolerance `2e-15`. Tests also check unwrapped predictions, bearing
innovations separated by multiple revolutions, exact pi's wrapping convention,
the near-zero boundary, finite skipped updates/derivatives under JIT, masked NaN
padding, empty batches, and initial-timestamp updates. A nonuniform-grid test
compares the filter with direct Stage 3A propagation and independent NumPy
updates. With no observations, filter histories exactly match Stage 3A.

## Simulated known-landmark localization

The seed-17 fixture uses a six-second analytical circular truth trajectory,
speed `1`, angular velocity `.2`, and initial truth `[.2,-.3,.4]`. The grid has
128 nodes: 121 uniform nodes plus seven irregular observation/integration times.
Three fixed landmarks are `[4,1]`, `[-2,3]`, and `[1,-4]`. There are **69 supplied
observations, all accepted**, over 23 observation timestamps; other slots are
masked. Measurement standard deviations are `.05` for range and `.015` radians
for bearing. Measurements are generated with independent NumPy geometry and
fixed noise, with bearings wrapped before filtering.

The initial pose is biased by `[.5,-.4,.2]`, with covariance
`diag([.36,.36,.09])`. `Qc=diag([.005,.001])`; all spline coefficients are zero.
The analytical truth has no injected process noise; `Qc` is the configured
filter uncertainty. Truth and measurement noise are not generated by the filter. Errors are
evaluated over all posterior grid nodes, including the initial-time update.

| Metric | Value |
| --- | ---: |
| Position RMSE | `2.739662921e-2` position units |
| Wrapped heading RMSE | `9.331405256e-3` radians |
| Same initial pose and commands, propagation-only position RMSE | `3.498082583e-1` position units |
| Maximum covariance asymmetry | `2.114194e-18` |
| Minimum covariance eigenvalue across posterior histories | `4.233891e-5` |

The position and heading checks pass predetermined limits `.12` and `.04`.
Joseph updates and propagated posterior covariances remain symmetric and
positive definite in this fixture without projections or diagonal jitter.
This single-seed numerical localization test is not a SLAM benchmark, a Monte
Carlo comparison, or evidence of MHE benefit.

## Differentiation through the complete EKF

A nine-node nonuniform-grid fixture uses cubic splines (`degree=3`, `M=4`),
14 coefficients, three fixed landmarks, time-varying commands, and different
observation counts per node. Seed 19 fixes measurement perturbations; seed 20
fixes coefficients with scale `.08`. The initial covariance is
`diag([.05,.04,.02])`, `Qc=diag([.015,.006])`, and
`Sigma=diag([.12**2,.04**2])`. All masks and associations remain fixed.

The scalar validation objective contains four components:

1. Sum of innovation quadratics `e.T solve(S,e)`.
2. `.05*sum(S**2)` over valid observations.
3. `.02*sum(K**2)` over gains.
4. `.5*sum(r_final**2)+.6*trace(P_final)+.2*sum(P_final**2)`.

The extra terms exercise numerical derivatives; they do not define an MHE
optimization objective. The implementation adds no optimizer or log-determinant
term. JIT-compiled `jax.grad` of the scalar is compared with central differences
for every coefficient. Component gradients are also checked independently to
confirm that innovations, innovation covariance, gain, and terminal quantities
each retain dependence on both coefficient blocks.

| FD step | Max absolute `c_v` gradient error | Max absolute `c_omega` gradient error |
| ---: | ---: | ---: |
| `1e-5` | `9.108601e-10` | `6.201839e-10` |
| `3e-6` | `2.901814e-9` | `2.436798e-9` |

Checks use `rtol=3e-6`, `atol=3e-8`. These steps balance central-difference
truncation and float64 subtraction error for the fixture's coefficient scale;
the two step sizes produce consistent results. The maximum measured error is
well below the absolute tolerance. Gradient block norms are:

| Objective component | `c_v` norm | `c_omega` norm |
| --- | ---: | ---: |
| Innovation quadratics | `7.5399335` | `5.4284541` |
| Innovation covariance term | `1.4665507e-6` | `2.2510299e-6` |
| Kalman gain term | `1.1066374e-3` | `1.4570857e-3` |
| Final state/covariance term | `.16893224` | `.02739973` |

## Limits and next gate

Inputs must be finite for requested observations; `Sigma` must be symmetric
positive definite, and initial covariance and `Qc` symmetric positive
semidefinite. Shape checks are performed during tracing; these matrix value
conditions remain caller preconditions. The geometry check rejects singular
measurement configurations but does not implement a statistical innovation
gate. The validity threshold and angular branch cuts are discrete boundaries:
derivative checks apply away from those boundaries. Masked padding is never
evaluated by an update.

Stage 3A's step-size, process-noise calibration, smoothness, and degree-zero
endpoint limits remain applicable. Explicit RK4 does not guarantee covariance
positive semidefiniteness for arbitrary large steps. Slot padding requires a
static maximum observation count per compiled call. Sequential relinearization
is order-dependent; callers must retain a deterministic observation order.

There is no unknown-landmark initialization, state augmentation, nearest-neighbor
association, or MHE optimization. Stage 3B.2 and Milestone 4 have not begun. The
remaining Milestone 3 gate includes future landmark-state handling and Baseline B
validation on the saved SLAM replay datasets with documented noise assumptions.
