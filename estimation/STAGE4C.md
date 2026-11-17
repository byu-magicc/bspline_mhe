# Stage 4C fixed-window input-spline MHE-SLAM

The new `slam_window_objective.py` orchestrates the unchanged Stage 3B.2
`rk4_slam_step` and `observe` primitives. It does not reimplement dynamics,
RK4, inverse range-bearing initialization, Jacobians, Kalman gains or Joseph
updates. It reuses Stage 4A's covariance/SPD domain checks and Python checked
interfaces, and Stage 4B's checked SciPy L-BFGS-B optimizer unchanged.

The coefficient vector is `[c_v;c_omega]` with seven coefficients per channel.
The C++ adapter evaluates RK4 stage times minus t0 with L=5, M=4, degree=3.
The arrival state, covariance, metadata, observation arrays and plan are copied
before candidate tracing. The plan uses only IDs, masks and arrival metadata,
with lowest-free-slot allocation. No true landmark coordinates enter filtering.
All discrete decisions must match the fixed plan; their values do not depend
on coefficients. Valid continuous operations remain differentiable throughout.

The result fields `requested_count` and `valid_count` count all consumed
observations, including births. `valid_mask` is specifically the correction
mask used by J, logdet and Gaussian diagnostics. Initialization masks and
counts are reported separately. A new landmark consumes its first observation
once, without an innovation term; a later observation at the same timestamp
corrects it normally. Only correction innovations contribute to
`J=sum(e.T solve(S,e))`. No likelihood or regularization term is added.

All requested births and updates must succeed. An invalid candidate returns
undefined costs and explicit status/counts; its history is not an estimate.
Subsequent arithmetic is quarantined after failure. The checked optimizer
aborts on the first invalid preflight/trial without returning a finite penalty,
dropping data, changing bounds or retrying. Covariance symmetry/PSD checks use
the original absolute tolerance 1e-10 and never modify a covariance. Padding
must stay exactly zero. PSD of neutral padded storage is equivalent to active
PSD, but diagnostic minimum eigenvalues are calculated only on active blocks.
Node priors and every sequential posterior are checked, not all internal RK4
stages. The integration step remains the previously validated 0.025 s.

Configuration A uses the full t=2 posterior after six births and 114 corrections
at 0.1 through 2.0 s. All six landmarks are already active. Its 300 optimization
observations occur strictly after t=2. Configuration B uses an observation-free
command-only propagation of the original t=0 robot prior and Qc to t=2, with
an empty map. It does not use A's posterior marginal or pre-window measurements.
Its six t=2.1 observations initialize the map, leaving 294 corrections. The
same seven-second dataset is used across A/B and each paired zero/fitted replay.
The t=0 prior anchors the gauge; metrics are not rigidly aligned after fitting.

The saved `configuration_a_gate*` records capture successful tests and fits
before Configuration B was implemented. `configuration_b_constant_gate*`
captures successful constant validation before smooth cases were added. The
final suite reruns both configurations and checks for regressions.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4c --test-record artifacts/milestone4c/test_record.json
```

The default command reproduces both configurations and both disturbance types,
with two zero-start fits each, numerical gradients at three vectors and a true
constant/independent LS diagnostic. The existing gate and preservation records
in the artifact directory retain the initial development-order evidence. The
optional test record describes an actual separately run complete suite.

Truth uses independent analytic unicycle solutions for constant-input phases
and independently integrated analytic sinusoids for the smooth phase. A split,
tighter DOP853 reference checks truth integration. Qc is an estimator
assumption; there is no stochastic truth process-noise injection.

Figures use chi-squared threshold 5.991 on actual 2x2 marginal covariance blocks.
They are labelled “Conditional EKF covariance; spline coefficients treated as
fixed.” They exclude fitted-coefficient uncertainty. Invalid active histories
or marginal covariances are rejected before plotting, without clipping or
repair. All histories, masks, ID maps, dense disturbances, trial/iteration
histories and PNG/PDF figures are saved under `artifacts/milestone4c/`.

This remains fixed-window known-association estimation. Moving-window arrival
updates, boundary-crossing map birth accounting and coefficient warm starts
require Milestone 5 research decisions. Nearest-neighbor association, final
paired Monte Carlo comparisons and Victoria Park remain deferred.
