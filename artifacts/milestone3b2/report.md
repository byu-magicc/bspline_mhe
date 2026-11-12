# Stage 3B.2 — Differentiable EKF-SLAM with unknown landmarks

**Milestone 3 completion gate: satisfied for the saved simulation dataset and
known-association scope.** The continuous-discrete zero-disturbance EKF-SLAM
(Baseline B) passes the required numerical tests and runs reproducibly on the
same saved data as Baseline A. This does not complete the later Monte Carlo,
nearest-neighbor, MHE, or Victoria Park validation gates.

Validation date: November 11, 2026. Python 3.12.6, NumPy 2.0.2, JAX/JAXLIB
0.4.35, pytest 8.3.3, macOS ARM64; JAX float64 is explicitly enabled.

## Resumed progress and files

The interrupted run left `models/slam.py` and `estimation/slam.py` as draft
implementations, plus a preservation snapshot. There were no Stage 3B.2 tests,
replay outputs, or report, and no Stage 3B.2 tests had run before interruption.
This work resumed those modules, added validation and replay code, and resolved
the compiled augmentation issue described below.

Added files relative to the start of the original Stage 3B.2 task:

- `models/slam.py`: augmented dynamics/Jacobians, inverse observation model and
  initialization Jacobians, and the full augmented measurement Jacobian.
- `estimation/slam.py`: fixed-capacity state, ID lookup, initialization,
  sequential Joseph updates, coupled RK4 propagation, scan-based filter, and
  explicit overflow reporting/checking.
- `benchmarks/estimator_b.py`: saved-data replay, command/noise conversion,
  active-subspace metrics, compilation/runtime measurements, and exact repeats.
- `tests/test_slam.py`: 14 numerical and architecture tests.
- `tests/test_estimator_b.py`: four replay-adapter/data-preservation tests.
- `artifacts/milestone3b2/report.md`: this report.
- `artifacts/milestone3b2/summary.json`: replay metrics, configurations,
  environment, dataset hash, and new source hashes.
- `artifacts/milestone3b2/original_known.npz` and `noisy_known.npz`: posterior
  state/covariance/mask/ID histories, observation decisions and diagnostics,
  replay timings, consumed measurements, and command/noise phase arrays.

All 247 previously validated files were checked against the snapshot taken
before Stage 3B.2. Earlier source, tests, project documents, and artifacts remain
unchanged. The sibling spline repository remains on `uniform_development` with
no source changes. Baseline A and its saved results were not rerun or replaced
by the new replay harness; its regression tests use their existing temporary
outputs.

## State, associations, and inactive storage

`SlamState(x,covariance,active,truth_ids)` is an immutable JAX PyTree. Capacity
defaults to 16 and is configurable when creating/compiling a filter. Array
shapes are `(3+2*N_max,)`, `(3+2*N_max,3+2*N_max)`, `(N_max,)`, and `(N_max,)`.
The active mask distinguishes live IDs from inactive metadata, so no ID value
is reserved as a sentinel. IDs are signed int64 integers; tests include negative
and large nonconsecutive values such as `-7` and `90000000000`.

Each requested observation searches active IDs. The lowest free slot is
allocated for a new ID. A repeated ID updates the existing slot, including
repeated IDs at the same timestamp. IDs are used only for association; landmark
positions are initialized from the measurements. A test changes the evaluation
truth map and verifies the complete filtered trajectory/covariance is unchanged.

`neutralize_inactive` zeros inactive state entries and covariance rows/columns
before numerical operations. Inactive IDs are also neutralized and excluded
from lookup. The augmented measurement Jacobian has zero columns for unrelated
and inactive slots. Tests inject NaNs and misleading IDs into inactive storage
and verify active estimates match clean storage exactly. Masked observation
padding is not evaluated.

Covariance checks select robot coordinates plus active landmark coordinates.
Inactive storage is checked separately for exact zero. Zero eigenvalues from
inactive padding are expected; no inversion of the padded covariance or
artificial diagonal uncertainty is introduced.

## Propagation, initialization, and updates

The augmented dynamics reuse Stage 3A's robot dynamics and analytical `A_r` and
`G_r`, with zero landmark dynamics and zero direct landmark noise. The new
`rk4_slam_step` applies the same manually written coupled-tuple RK4 algorithm
to `(x,P)` at the larger dimension. It evaluates commands, spline disturbances,
`A`, `G`, and dynamics at all four stages. Stage 3A's robot-specific shape
checks and its validated code remain unchanged. Tests compare the robot block
against Stage 3A under varying commands, nonzero coefficients, and nonuniform
steps, and compare full covariance transport with an analytical reference.

Initialization uses the specified inverse range-bearing model and `J_r,J_y`.
With `J_x=[J_r,0]`, it inserts `P_xell=P J_x.T` and
`P_ellell=J_x P J_x.T+J_y Sigma J_y.T`, preserving all existing active blocks.
Cross-covariances include the robot and all earlier landmarks. Observation noise
is independent of the prior and other observations. General SPD `Sigma`,
including correlated range/bearing noise, is supported. The first observation
is consumed by initialization once; it is not reused as a correction.

Existing landmarks use the Stage 3B.1 range-bearing model and wrapped innovation.
The full `H` places the robot Jacobian in the first three columns and the
negative position columns in the selected landmark block. Gain computation is
`solve(S,(P H.T).T).T`, followed by the full-state Joseph update. Each observation
relinearizes after the preceding observation. Internal heading stays unwrapped;
bearing innovations wrap to `[-pi,pi)`.

An initial new-test run passed five tests and failed nine with an XLA CPU
`DotThunk` output-layout error during compiled initialization. The direct sparse
`P @ J_x.T` product was being lowered into a transposed output incompatible with
that backend. The new augmentation code now reduces explicitly over the three
nonzero robot columns and computes the marginal as `J_r @ cross[:3]` plus the
measurement-noise term. This is algebraically identical, remains differentiable,
and passes the independent covariance reference. No validated source, model,
or numerical tolerance was changed to resolve this failure.

## Interfaces and overflow

- `create_state(pose,covariance,capacity=16)` creates a robot prior with neutral
  inactive storage. Enable JAX float64 first.
- `inverse_range_bearing(r,y)` returns the landmark, `J_r`, and `J_y`.
- `initialize_landmark(state,slot,truth_id,y,Sigma)` is a low-level primitive
  requiring an inactive in-bounds slot and valid observation. Use the dispatcher
  below for automatic allocation and overflow handling.
- `observe(state,y,truth_id,Sigma,requested=True,min_range=1e-8)` returns a new
  state and `ObservationResult(diagnostics,status,slot)`.
- `sequential_observations` accepts `(slots,2)` measurements, `(slots,)` IDs,
  shared/per-slot covariance, and masks, processing in supplied order.
- `rk4_slam_step` accepts the state, start time, duration, pure JAX command
  callable, four precomputed basis rows, coefficients, and `Qc`.
- `filter_slam` accepts increasing integration nodes, `(N-1,4,M+d)` basis,
  `(N,slots,2)` observations, `(N,slots)` IDs/masks, and shared or per-observation
  covariance. `Qc` may be shared or per integration interval. Initial-node
  observations update/initialize from the arrival prior directly.

`SlamResult` contains posterior `states`, observation results, and an aggregate
`overflow` flag. History fields have a leading grid-node axis. Diagnostics have
leading grid-node/observation-slot axes. Codes are `MASKED=0`, `INITIALIZED=1`,
`UPDATED=2`, `INVALID=3`, `OVERFLOW=4`. New/invalid/masked/overflow records use
neutral diagnostics (zero innovation/gain/Jacobian, identity `S`, and
`diagnostics.valid=False`); use status to count initialization and use diagnostic
validity to accumulate correction innovations.

Overflow preserves existing landmarks and records the failed allocation
explicitly. The compiled filter can continue updating existing landmarks and
returns a sticky aggregate flag. Call `raise_on_overflow(result)` at the Python
boundary after compiled execution: it raises `LandmarkCapacityError` with the
capacity and offending grid/observation slots. The replay harness checks each
compiled cycle's statuses and raises immediately at that boundary. It never
silently reports overflow as a successful replay.

New initialization rejects nonfinite observations and range at or below
`min_range`; updates also reject invalid predicted geometry. These numerical
checks are discrete. Derivative validation stays away from their boundaries and
angular branch cuts. IDs, masks, and initialization choices stay fixed during
coefficient differentiation. There is no `stop_gradient` on numerical EKF terms.
Inputs must provide valid finite active state/covariance, PSD `P,Qc`, and SPD
measurement covariance; matrix-value conditions are caller preconditions.

## Numerical tests

Commands from the repository root with the existing Milestone 2 extension:

```sh
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q -s tests/test_slam.py -x
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q -s tests/test_estimator_b.py
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q \
  tests/test_estimator_a.py tests/test_diagnose_initializations.py \
  tests/test_visualize_estimator_a.py tests/test_uniform_basis.py \
  tests/test_robot_propagation.py tests/test_known_landmark_ekf.py
```

After resolving the compiled layout issue, **14 SLAM tests passed in 14.51 s**.
All **four replay-adapter tests passed in 2.29 s** and **180 existing regression
tests passed in 30.18 s**. No requested tolerance was relaxed. Suite timings
include JAX compilation and are not steady-state runtime measurements.

| Check | Numerical result |
| --- | ---: |
| Inverse model/Jacobians vs. autodiff | passed at absolute tolerance `1e-12` |
| Initialization `J_r` vs. central/Richardson FD | max error `1.489919e-13` |
| Initialization `J_y` vs. central/Richardson FD | max error `2.796652e-13` |
| Covariance augmentation vs. independent NumPy, diagonal/correlated Sigma | max error `6.938894e-18` |
| Augmentation covariance asymmetry | at most `3.469447e-18` |
| Augmentation active minimum eigenvalue, diagonal/correlated cases | `.003374793` / `.002862555` |
| Full augmented measurement H vs. autodiff | max error `0` |
| Full augmented measurement H vs. central FD | max error `5.266321e-11` |
| Full covariance propagation vs. analytical constant-heading solution | max error `6.938894e-18` |
| RTR replay covariance vs. discrete linearized reference | max error `6.661338e-16` |

Initialization finite differences use Richardson extrapolation of central
differences at steps `.002` and `.001` to meet the `1e-12` absolute requirement
without second-order truncation error dominating. Augmented measurement finite
differences use step `1e-6`. Covariance augmentation uses tolerance `1e-11`;
active-subspace symmetry and minimum eigenvalues use `1e-10` and `-1e-10`.
Inactive storage is required to be exactly zero.

Tests also cover full cross-covariances with prior landmarks, same-timestamp
initialization and updates, arbitrary integer IDs, capacity exhaustion under
JIT, Python overflow exceptions, inactive-slot isolation, invalid initialization,
empty batches, JIT/scan compatibility, and robot-block equivalence to Stage 3A.

### Coefficient differentiation

A six-node nonuniform-grid test has no observations at the first node, then
initializes and updates three landmarks. Thus coefficients affect the arrival
state of new landmarks and all later state/covariance updates. It uses cubic
splines, four segments, 14 coefficients (seed 22, scale `.08`), time-varying
commands, `Qc=diag([.02,.006])`, and correlated
`Sigma=[[.02,.002],[.002,.004]]`.

The scalar numerical-validation objective combines innovation quadratics,
first-initialized-landmark mean/covariance terms, and terminal full-state and
covariance terms. It is a derivative test, not an MHE objective or optimizer.
JIT-compiled `jax.grad` is checked against central differences for every
coefficient; component derivatives are nonzero for both blocks.

| FD step | Max `c_v` gradient error | Max `c_omega` gradient error |
| ---: | ---: | ---: |
| `1e-5` | `2.680642e-9` | `3.770369e-9` |
| `3e-6` | `9.993311e-9` | `1.035167e-8` |

Both checks pass the required `rtol=1e-5`, `atol=1e-7`. The initialization-only
objective has gradient block norms `.0466627155` and `.00979888669`. Smaller FD
steps show greater subtraction error but remain below the absolute tolerance.
Observation decisions are unchanged under the tested coefficient perturbation.

## Baseline B saved-data replay and noise mapping

```sh
PYTHONPATH=build/bsplines_library python3 -B -m benchmarks.estimator_b --repeats 2
```

Both saved observation modes were replayed for seed 7, 500 steps, `dt=.1 s`,
with known associations and zero estimated disturbance. The initial robot prior,
measurement covariance `diag([10**2,(10*pi/180)**2])`, observations/order, truth,
and commands match the saved Milestone 1 dataset. The `original` mode consumes
the saved clean pair; `noisy` consumes the saved noisy pair. Each full replay is
repeated twice, and all numerical state/covariance/mask/ID/status/diagnostic
arrays match exactly, excluding timings. No data/noise is regenerated.

The saved command is `(a,d,b)` = rotation1, translation, rotation2 increments,
not speed/turn-rate. To avoid inventing a constant-twist endpoint path, represent
it as three held phases over durations `(dt/4,dt/2,dt/4)`, with controls
`(0,a/dt1)`, `(d/dtT,0)`, and `(0,b/dt2)`. This is an explicit intra-step timing
convention; only the endpoint is observed in this dataset. Phase-local callables
hold the command at every RK4 stage, including each right endpoint.

For Baseline A's independent increment covariance, derive

```text
m1 = alpha0*a^2 + alpha1*d^2
mT = alpha2*d^2 + alpha3*(a^2+b^2)
m2 = alpha0*b^2 + alpha1*d^2
Qc1 = diag([0, m1/dt1])
QcT = diag([mT/dtT, 0])
Qc2 = diag([0, m2/dt2])
```

This matches each integrated input variance to the corresponding increment
variance. For the RTR nominal motion, transporting the first rotation's
uncertainty through translation, adding longitudinal translation uncertainty,
and then adding the final rotation uncertainty reproduces
`R diag([m1,mT,m2]) R.T`, where

```text
R = [[-d*sin(theta+a), cos(theta+a), 0],
     [ d*cos(theta+a), sin(theta+a), 0],
     [1,              0,            1]]
```

The full prior covariance is transported by the same endpoint state transition,
including robot/map cross-covariances. Tests verify all three saved command
types against independent NumPy equations, with zero measured endpoint mean
error and covariance errors below `6.67e-16`. This is a first-order EKF covariance
mapping; it does not equate continuous noise values with discrete covariances
or claim equality of the full nonlinear stochastic distributions. The same
command/noise convention must be retained in future paired B/C replays.

### Replay results

Position/map errors use simulator coordinate units; heading errors are wrapped
and in radians. Map RMSE uses only active landmarks and their known IDs in the
same anchored initial frame, with no fitted alignment.

| Mode | Estimator | Position RMSE | Heading RMSE | Final map RMSE | Mapped landmarks |
| --- | --- | ---: | ---: | ---: | ---: |
| original | A, saved | `2.880460` | `.0308218` | `1.192448` | 8 |
| original | B | `2.888632` | `.0308463` | `1.177080` | 8 |
| noisy | A, saved | `9.446002` | `.0544445` | `15.484536` | 8 |
| noisy | B | `10.151191` | `.0575457` | `16.862017` | 8 |

Baseline B is slightly worse in position/heading in the clean case and worse
in all three errors in the noisy case; its clean map error is slightly lower.
These negative results are retained. A uses its original batch update, frozen
measurement linearizations, simplified covariance correction, and original
control-flow behavior. B uses sequential relinearization, Joseph updates, and
per-observation initialization. Matching prediction moments does not make these
different estimators identical. This A/B replay characterizes the implementation
change; B/C remains the principal comparison for isolating spline-MHE effects.

Each B mode has 1,000 supplied observations: eight initializations and 992
existing-landmark updates, with zero invalid records and zero overflows. Known
association correctness is 100%; existing-match coverage is 99.2%; the used
fraction including initialization is 100%. Active counts end at eight; inactive
state and covariance storage is exactly zero at every saved node.

| B diagnostic | original | noisy |
| --- | ---: | ---: |
| Max active covariance asymmetry | `1.563194e-12` | `7.673862e-13` |
| Min active covariance eigenvalue | `1.080760e-4` | `1.080759e-4` |
| Mean correction NIS | `.0392508` | `1.933966` |
| Mean robot NEES | `.480451` | `2.303024` |

Eigenvalues use the symmetric part of the active covariance for diagnostics;
the filter itself is not symmetrized or projected. NIS excludes initialization
and padding. NEES uses robot marginal covariance and wrapped heading error.
These empirical means from one trajectory are diagnostics, not Monte Carlo
consistency tests. The clean-observation mode intentionally retains the legacy
assumed measurement noise despite consuming clean measurements.

### Compilation and runtime

| B timing | original | noisy |
| --- | ---: | ---: |
| Lower/compile call (seconds) | `.570590` | `.000689` (cached) |
| Steady cycle mean (ms) | `.226111` | `.225438` |
| Steady cycle p95 (ms) | `.244484` | `.237794` |
| Total timed estimator seconds, repeat 1 | `.113542` | `.112849` |
| Total timed estimator seconds, repeat 2 | `.112569` | `.112589` |

The executable is warmed with a discarded cycle, then each replay starts from
the original prior. The noisy case reuses the same executable shapes and gets
a compilation-cache hit. Timed cycles include three RK4 phases, sequential
observations, and device synchronization. Dataset loading, command/noise and
basis precomputation, input slicing, host history copies, covariance checks,
metrics, and file writes are excluded. Cycle statistics pool both repeats.
These are reproducible accuracy runs with measured runtime, not a controlled
performance comparison against the historical A timings (which include their
own motion-noise construction and use different execution machinery).

## Completion and remaining limits

The requested Stage 3B.2 numerical checks and all earlier regressions pass.
Baseline B runs stably and exactly reproducibly on both saved known-association
simulation modes, with explicit noise mapping, errors, map counts, covariance
checks, and compilation/runtime reporting. Therefore the existing Milestone 3
completion criterion is met for this initial saved-data validation gate.

Capacity and observation-slot shapes are static per compiled call. Recompile
with a larger capacity after overflow; there is no landmark eviction. Active
IDs must be unique signed int64 values. Low-level initialization requires a
valid free slot; the observation dispatcher enforces allocation policy.
Initialization/update validity and angular branch cuts remain discrete
boundaries. Gradients are validated locally for fixed associations and decisions.

Explicit RK4 still requires suitable step sizes and sufficiently smooth inputs
inside each step; it has no unconditional PSD guarantee for arbitrary large
steps. The replay's phase convention matches observed endpoint moments at zero
disturbance but does not identify the physical continuous trajectory between
saved observations. The configured sensor/process assumptions and intra-step
command convention must be retained or explicitly reconsidered for future
paired experiments.

This stage implements no nearest-neighbor association, MHE optimizer, or
Milestone 4 functionality. The next gate is fixed-window input-spline estimation
with known landmark positions and injected disturbances, before moving-window
work and final three-estimator Monte Carlo comparisons. Victoria Park remains
deferred until the simulation MHE and comparison gates pass.
