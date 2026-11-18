# Stage 5B moving-horizon input-spline EKF-SLAM

This stage implements the approved independent-forward-EKF arrival policy and
componentwise clipping of projected initial guesses. It preserves the Stage 5A
integral projection/fallback, Stage 4C replay objective, checked optimizer,
all EKF equations and original baseline source. It performs known-ID association.

## Interfaces and information flow

`estimation/moving_horizon.py` adds:

- `ReplayData`: immutable integration times, range/bearing arrays, signed IDs,
  covariance, observation masks and an explicit scheduled-scan mask. This
  interface has no landmark-coordinate or truth-pose argument.
- `ForwardArrivals`: an independent zero-disturbance `filter_slam` replay. It
  retains full augmented states/covariances and persistent ID/slot metadata.
  `at(t)` returns the complete posterior at a saved node, or propagates its
  latest available posterior to an intermediate time without future observations.
- `window(end,L)`: a fixed-duration look-back replay with the frozen forward
  arrival, original measurement/ID rows, observation masks and fresh C++ basis
  evaluations for all four RK4 stages. Its first row is measurement-free.
- `initial_guess`: validated Stage 5A projection followed by componentwise
  clipping to the unchanged coefficient bounds. It records raw and feasible
  vectors, clipped component count, maximum magnitude and L1 magnitude.
- `MovingHorizonSLAM.run`: a generator reporting both projected and zero-start
  fits on each identical window. Each method uses its own previous coefficients
  and reported MHE map; neither feeds anything into the forward EKF.

The implementation is an offline replay of causal measurements. Precomputing
the forward history does not expose future observations to an arrival: its
posterior depends only on measurements at or before that time. Tests change
future measurements and check arrival equality. No optimized state, map,
covariance or disturbance is written to this history.

## Time and observation accounting

The horizon is `[a,t]=[t-L,t]` and replay consumes only `a<t_i<=t`. The independent
forward posterior includes observations at `a`; these rows are excluded from
the window. There is no double use of a birth observation as a correction.
The same historical measurement can appear in consecutive overlapping windows,
each of which restarts from its own independent arrival; this is the specified
receding-window policy, not repeated correction of one persistent MHE filter.

Full-length horizons start at the first scheduled scan at or after `t_initial+L`.
Before that time, only the independent forward EKF is available; no shorter
horizon, negative-time prior, or invented prehistory is used. Optimization is
performed at every eligible scheduled scan, including scans with zero detections.
Schedule times are not restricted to spline knots.

Boundary arithmetic can differ from a saved time by one or two floating-point
bits (e.g. `5.1-5` versus `0.1`). `canonical_time` equates times within eight ULPs
at their absolute time scale. This handles representation error only; genuinely
nonaligned boundaries remain nonaligned. The C++ basis grid pins the elapsed
window endpoints to exactly 0 and L, avoiding a spurious evaluation outside
the binding's strict domain. Commands still use absolute time at every RK4 stage.
Interior basis stage times are recomputed whenever the window grid changes.

An arrival between saved integration nodes is propagated from the preceding
forward node, which is itself a causal posterior. RK4 intervals remain at most
the configured integration step. Such a split can produce ordinary RK4
partition differences from an unsplit forward trajectory; zero-disturbance
equivalence is therefore checked against the appropriate replay on the same
arrival and time partition, not an assertion of bitwise partition invariance.

The replay interface requires a resolved integration grid and observation rows
at its nodes; it does not resample measurements. Gaps between eligible optimization
times exceeding L are not assigned an invented warm-start overlap: Stage 5A
rejects a projection shift outside [0,L]. The validated simulations have 0.1 s
scheduled scan intervals, regardless of detection count.

## Maps, updates and objective

`filter_slam`, `rk4_slam_step`, `observe` and `make_slam_window_objective` are reused
unchanged. Full robot-landmark and landmark-landmark cross-covariances travel in
the frozen arrival. Previously initialized landmarks remain active when invisible.
New IDs within a replay are initialized from measured range/bearing and the
candidate-dependent pose using the existing inverse model/covariance augmentation.
Fixed plans use only available IDs and masks, and remain fixed within each fit.
Neither identity nor initialization supplies ground-truth landmark coordinates.

Only the innovation quadratic is minimized. Innovation covariance is recomputed
for every correction and every coefficient candidate. The existing JAX objective
differentiates through state/covariance RK4 propagation, augmentation, sequential
measurement Jacobians, Kalman gains and Joseph updates. Births are consumed
exactly once without an innovation term; empty scans add no correction term.
No likelihood determinant, arrival penalty, coefficient penalty or smoothing
regularizer was added.

Each reported MHE state/map is the final posterior of that optimized window.
The covariance is **conditional EKF covariance; spline coefficients treated as
fixed**. It omits fitted-coefficient uncertainty. There is no separate live MHE
filter, no feedback into the arrival filter and no extrapolation used to report
measurement-time states.

## Initial guesses, failures and discontinuities

Both methods use deterministic zero initialization on the first full window.
Subsequent projected starts use the actual elapsed shift, the unmodified Stage 5A
integral projection and its documented continuation/fallback, then clip each
component to the existing `[-0.5,0.5]` bounds. Zero-start comparisons use zeros
at every update. Clipping changes only the initial guess. The objective, bound
values and every `DEFAULT_OPTIONS` value remain unchanged.

The preserved checked L-BFGS-B wrapper validates preflight candidates and every
trial. Invalid candidates stop estimation without a finite penalty, silent retry,
dropped data, covariance repair, bound change or alternate solver. Partial failure
records are retained; they do not become reported estimates. A valid finite
iterate with SciPy non-convergence remains explicitly labeled rather than being
called converged. Covariance symmetry/PSD and inactive padding use the existing
1e-10 policy without tolerance changes.

For each method, consecutive report records store raw pose/map changes and
overlap revisions. An overlap revision compares the new optimized replay at
the previous report time with the previous estimate at that same physical time.
Heading differences wrap to [-pi,pi); map comparisons use common active IDs.
New births are not counted as jumps of previously mapped landmarks.

## Validation and benchmark scope

The implementation gate runs the complete repository suite before deterministic
benchmarking; see `artifacts/milestone5b/implementation_gate.json`, pytest stdout
and XML for the actual command/counts. New tests cover boundary inclusion,
non-measurement/integration-aligned arrivals, future-observation independence,
births crossing windows, persistent IDs/cross-covariances, empty scans/windows,
zero replay equivalence, full-objective finite differences, deterministic repeated
optimization, clipping, invalid-candidate stopping, and simulation reproducibility.

The first run found a test comparing a saved canonical 0.3 s timestamp with the
slightly different float literal. The assertion now uses the canonical window
endpoint; no estimator formula, measurement gate or numerical tolerance changed.
Initial failure evidence remains in `implementation_initial*`.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone5b/pytest.xml -o junit_family=legacy
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5b --case regression_constant
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5b --case all
```

The initial suite has six paired seed-7 cases: zero/constant/smooth disturbances
and the Stage 4C regression/forward sensor assumptions. Disturbances start at t=0,
with no artificial initialization prefix. Default duration 5.2 s gives three
eligible full-window report times (5.0,5.1,5.2 s); all earlier scans construct
causal forward arrivals and the first window. These short runs expose startup
birth handling, warm starts, clipping and computation without claiming long-run
performance. Empty detections do not remove times from this schedule.

The independent simulator uses analytic constant-input truth or DOP853 sinusoidal
truth, checked against a tighter DOP853 solution. It applies true-pose visibility
before adding the same full-slot seeded noise stream; no truth process noise is
injected. Both starts receive one immutable dataset and arrival per window.

The [generated report](../artifacts/milestone5b/report.md) retains accuracy,
disturbance reconstruction, cost, convergence, clipping, iterations, gradient
timing, compilation, update-time mean/p95 and negative outcomes. Full arrivals,
replays, birth/correction masks, IDs, slots, coefficient histories, optimizer
evaluations and covariance histories are retained separately for audit.

The preserved objective captures each window as constants, so this first version
recompiles for new windows. Explicit lower/compile cost, cold checked-reference
auxiliary compilation and steady optimizer timings are separated. Fixed method
order can affect compilation cache timings; iteration counts/final costs and
warm execution timings should accompany total wall time. The forward EKF is
also profiled with dynamic scan inputs after compilation, with each result
checked against its saved causal history.

No real-time deadline, global optimum, Monte Carlo reliability or 95% empirical
coverage is claimed. Longer deterministic sequences and compilation reuse are
appropriate next engineering steps. A Monte Carlo trial count must be proposed
from measured runtime and a specified paired-effect precision target; performing
those trials, nearest-neighbor association and Victoria Park requires approval.
