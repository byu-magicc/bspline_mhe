# Stage 3A validation: robot state and covariance propagation

Stage 3A implements robot-only continuous-time propagation. The mean dynamics,
analytical state Jacobian, and input-noise Jacobian follow `project.md`.
Covariance obeys `P_dot=A P+P A.T+G Qc G.T`. Tests use diagonal `Qc` initially
and also check zero and correlated input noise. No process-noise mapping from
Baseline A is assumed.

The state and covariance are integrated as a coupled JAX tuple with manually
written RK4. The command, disturbance, dynamics, and both Jacobians are
evaluated at every RK4 stage. Basis values are constants precomputed through
the existing C++ uniform B-spline adapter. No library or Milestone 1/2 source
or saved result is changed by this implementation.

## Commands and environment

From the repository root, using the already built Milestone 2 extension:

```sh
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q -s tests/test_robot_propagation.py
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q \
  tests/test_estimator_a.py tests/test_diagnose_initializations.py \
  tests/test_visualize_estimator_a.py tests/test_uniform_basis.py
```

The Stage 3A command passed **23 tests in 5.00 s**. This includes a JIT-compiled
gradient through the complete multi-step propagation. The environment is Python
3.12.6, NumPy 2.0.2, JAX/JAXLIB 0.4.35, pytest 8.3.3 on macOS. The tests explicitly
enable JAX float64. Timings are test-suite wall times, including compilation;
they are not estimator runtime benchmarks.

The Milestone 1/2 regression command passed **138 tests in 16.17 s**. Content
hashes for all **236 preexisting files** (including baseline sources, saved
results, spline integration, existing tests, and project documents) matched
the snapshot taken before implementation. The sibling spline checkout remained
clean on `uniform_development` at `bb334d0f75cd0a244c73c0de3468d049e0185ebe`.

Added files:

- `models/__init__.py`
- `models/unicycle.py`
- `estimation/__init__.py`
- `estimation/propagation.py`
- `estimation/README.md`
- `tests/test_robot_propagation.py`
- `artifacts/milestone3a/report.md`

An initial run had three test errors from subtracting a Python list of reference
matrices from a JAX array. Converting the reference list to a NumPy array fixed
those errors. No model, propagation formula, or numerical tolerance was changed.

## Jacobians and analytical motion

Twelve seed-7 random model cases compare `A` with `jax.jacfwd` and central finite
differences. `G` is checked against differentiation with respect to both commanded
and disturbance inputs.

| Check | Maximum absolute error |
| --- | ---: |
| Analytical `A` vs. `jax.jacfwd` | 0 |
| Analytical `A` vs. central differences, step `1e-6` | `2.305801e-10` |
| Analytical `G` vs. `jax.jacfwd` | 0 |
| Straight-motion pose, three headings, nonuniform steps | `8.881784e-16` |
| Circular-motion pose, omega `-0.8`, steps `0.01` | `2.307488e-12` |
| Circular-motion pose, omega `1.3`, steps `0.01` | `1.608225e-11` |

Straight motion uses effective speed `1.6` over `2.3 s`, including nonzero spline
coefficients in both channels whose angular contribution cancels the command.
The circular tests use effective speed `1.7`, duration `1.4 s`, and initial pose
`[0.2,-0.3,2.9]`. Heading is deliberately unwrapped, including passage through pi.

## RK4 convergence and covariance

For constant speed `1.4`, turn rate `1.1`, duration `1.6 s`, and initial pose
`[0.2,-0.4,0.3]`, compare the terminal pose with the exact circle solution. The
initial covariance and noise spectral density are

```text
P0 = [[.04, .003, .002], [.003, .03, -.001], [.002, -.001, .02]]
Qc = diag([.04, .015])
```

The covariance reference uses the exact state transition
`Phi(T,s)=I+integral_s^T A(t)dt` (here `A(t1)A(t2)=0`) and an independent 64-point
Gauss-Legendre noise integral. The reference does not call the RK4 code or its
Jacobian functions. State errors are Euclidean norms; covariance errors are
Frobenius norms.

| Steps | Step size | State error | Covariance error |
| ---: | ---: | ---: | ---: |
| 4 | .4 | `2.56802188e-5` | `6.40720279e-6` |
| 8 | .2 | `1.59807363e-6` | `3.82121870e-7` |
| 16 | .1 | `9.97716801e-8` | `2.34009054e-8` |
| 32 | .05 | `6.23404621e-9` | `1.45008266e-9` |

State refinement ratios are **16.0695, 16.0173, 16.0043**. Covariance ratios are
**16.7674, 16.3294, 16.1376**. Both approach the expected factor of 16 for global
fourth-order error. Across these histories, maximum covariance asymmetry is
`1.734723e-18` and minimum covariance eigenvalue is `1.513642e-2`.

For fixed heading and speed, `A^2=0` and `Q=G Qc G.T` is constant. The independent
analytical covariance check uses

```text
Phi(t) = I + A*t
P(t) = Phi(t) P0 Phi(t).T + Q*t
       + (A Q + Q A.T)*t^2/2 + A Q A.T*t^3/3
```

At heading `.7`, speed `1.2`, over a nonuniform grid ending at `1.8 s`, maximum
absolute covariance errors are `6.938894e-18` for zero noise, `1.387779e-17` for
`Qc=diag([.04,.015])`, and `1.387779e-17` for correlated noise
`Qc=[[.04,.009],[.009,.015]]`. These polynomial cases are exact under RK4 up to
roundoff and passed at unchanged absolute tolerance `2e-15`.

## Disturbance gradients and stage timing

A seed-11 cubic-disturbance test differentiates a scalar objective containing
both terminal state and terminal covariance, with time-varying commands and
nonuniform steps. For fixed symmetric weight matrix `W` specified in the test,
the objective is `.5*r.T@W@r + .6*sum(P*W) + .2*sum(P**2)`.

| Central-difference step | Max absolute `c_v` gradient error | Max absolute `c_omega` gradient error |
| ---: | ---: | ---: |
| `1e-5` | `3.407485e-11` | `4.039992e-11` |
| `3e-6` | `9.452211e-11` | `1.065781e-10` |

Both finite-difference checks passed with `rtol=2e-7`, `atol=2e-9`. The steps are
appropriate for order-one objective values, float64, and coefficients of scale
`.15`; the smaller step shows the expected increase in subtraction error.
The covariance-only objective has gradient block norms `3.802275e-2` for `c_v`
and `2.366168e-3` for `c_omega`, demonstrating sensitivity through covariance
propagation for both channels.

Separate tests integrate cubic commanded inputs and cubic spline disturbances
against independent Gauss-Legendre quadrature on intervals within spline
segments. Both disturbance channels are checked with an absolute time offset,
different midpoint/start basis values, repeated midpoint stages, and endpoint
evaluations. These tests pass at absolute tolerance `2e-14`. Nonuniform-grid
restart checks show that a terminal tuple can initialize another propagation
interval, including boundaries reserved for future observation events.

## Scope and next gate

These are numerical propagation tests, not SLAM comparison results. Explicit
RK4 has no unconditional positive semidefinite covariance guarantee for large
steps; application step sizes still require refinement and stability checks.
Fourth-order convergence requires sufficiently smooth inputs within a step.
For piecewise constant commands, use an interval-local callable so the final
stage uses the held command for that interval. Degree-zero spline endpoint and
smoothness limitations from Milestone 2 remain documented.

See `estimation/README.md` for the interface and caller preconditions. The next
gate is Stage 3B measurement updates, landmark handling, complete EKF gradients,
and Baseline B validation on the saved replay datasets. Stage 3A adds none of
those components and adds no MHE optimizer.
