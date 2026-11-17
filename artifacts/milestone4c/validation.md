# Stage 4C validation and interpretation

**The controlled Stage 4C completion criterion is satisfied for both
configurations, with constant and smooth disturbances.** All requested data
remain fixed, all primary fits converge reproducibly, numerical gradients are
validated, and the complete suite passes. This is a fixed-window,
known-association result; no moving-window implementation or statistical
performance guarantee is implied.

## Intermediate validation order

Configuration A was implemented and validated before Configuration B code was
added. [configuration_a_gate.json](configuration_a_gate.json) records the actual
12 passing focused tests, zero-filter equivalence, NumPy accumulation,
all-coefficient gradients and exact paired fit repeats. Its retained result/XML
files and source hashes capture that intermediate implementation.

Configuration B was then implemented with an empty arrival map. Its constant
experiment passed the combined 20 focused tests and repeated fitting before
smooth cases were added; see
[configuration_b_constant_gate.json](configuration_b_constant_gate.json) and
its XML/result copies. The final implementation subsequently reruns A to
check for regressions after adding the B path.

An initial A test fixture exposed floating-point cancellation: `2.2-2.0` is
slightly larger than literal `0.2`. The strict C++ adapter correctly rejected
the inconsistent endpoint. The fixture now uses `L=times[-1]-times[0]`; no
basis clipping, library changes or tolerance relaxation was introduced.
That correction preceded the successful A-only gate.

## Executed final verification

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone4c/pytest.xml -o junit_family=legacy
```

Outcome: **295 passed**, zero failures/errors/skips and no warnings;
exit code 0. Pytest reported 195.77 s; subprocess wall time was
200.357 s. This includes all 270 pre-existing tests and
25 Stage 4C tests. No required tests were left unrun. Actual stdout/XML and
source digests are retained in [test_record.json](test_record.json),
[pytest_stdout.txt](pytest_stdout.txt) and [pytest.xml](pytest.xml).

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4c --test-record artifacts/milestone4c/test_record.json
```

Outcome: exit code 0. All four comparisons completed, each with two independent
zero-start optimization runs on its fixed saved dataset, the zero reference,
true-constant/independent-LS diagnostic, three full-vector finite-difference
checks, and PNG/PDF figures. Coefficients and accepted objective histories
repeat exactly. Repeats share one noise realization and are not Monte Carlo.
The benchmark embeds the actual suite record and does not itself execute pytest.

Focused tests cover arrival/data copying, full map cross-correlations, strict
t0 boundary accounting, filter equivalence, birth/correction partitioning,
repeated IDs at one timestamp, fixed integer slot decisions, capacity errors,
NumPy objective/logdet references, JIT/scan compatibility, invalid covariance,
noise, geometry and SciPy trials, differentiable covariance augmentation,
metric serialization and covariance-ellipse/figure calculations.

## Mathematical preservation and measurement accounting

The objective orchestrates the unchanged augmented `rk4_slam_step` and `observe`
functions. It reuses the Stage 4A domain predicates and checked Python
interfaces and the unchanged Stage 4B SciPy wrapper. Dynamics, covariance
propagation, inverse range-bearing augmentation, sequential relinearization,
Kalman gains and Joseph updates are not reimplemented.

Configuration A freezes the complete posterior after 120 pre-window
observations (six births and 114 corrections), including t=2.0 s. Its saved
arrival contains all robot/map blocks, cross-correlations, active mask and IDs.
The optimization mask excludes t=2.0 and includes 300 correction observations
at 2.1 through 7.0 s. Every replay starts from the identical captured arrival.

Configuration B constructs a distinct arrival by propagating the original
robot prior/Qc with commands and zero disturbance to t=2, consuming no
observations and carrying no map. It is not A's posterior marginal. The six
records at t=2.1 initialize the map; 294 later records correct it. No true map
coordinate enters initialization or filtering. Births contribute no innovation
quadratic/logdet term; a later record of the same ID at the same time can
correct normally. The original pose prior anchors the frame. No rigid map or
pose alignment is used in reported metrics.

All 300 requested records are consumed with known-ID correctness and used
coverage 1.0 in every replay. Correction coverage is 1.0 for A and 0.98 for B;
the remaining 0.02 in B consists of valid initializations, not rejected data.
The saved masks, IDs and observation slots independently verify this accounting.
A and B have different arrival information and different correction sets,
so their raw objective values are not directly comparable as equal-data costs.
The controlled estimator comparison is Baseline B versus fitted MHE within
each configuration and disturbance dataset.

The fixed requested set/plan never changes during optimization. Every requested
birth/update must be valid. Node priors and each sequential posterior are
checked for finite state/covariance, symmetry and active PSD using the original
absolute tolerance 1e-10; inactive padding stays exactly zero. Measurement R
and update S must be SPD. Invalid candidates have undefined costs and checked
exceptions, not partial sums or finite penalties. Invalid trial fault-injection
tests confirm no value/gradient is returned to SciPy. All actual candidates
and deterministic box probes were valid; no retry or domain revision occurred.

Only J is minimized. Logdet and J+logdet use exactly the correction set as
separate diagnostics; the latter omits the Gaussian dimension constant.
The coefficient bounds and stopping options remain those of Stage 4B.

## Independent truth and numerical differentiation

The seven-second trajectory has zero injected disturbance during its first
two seconds. Constant phases use independent NumPy analytic unicycle
solutions. Smooth truth uses DOP853 on the analytic sine/cosine inputs, with
max step 0.05 s, rtol=1e-11 and atol=1e-13. A separate split DOP853 reference
uses max step 0.025 s, rtol=1e-12 and atol=1e-14. Every reported discrepancy
is below 1e-10 per component. The integration is split at t0 to handle the
input discontinuity. Tests independently check smooth heading analytically.
Neither estimator RK4 nor the C++ basis generates truth. Qc is an estimator
assumption; stochastic process noise is not injected into truth.

All 14 objective derivatives are checked at zero, a nonconstant candidate and
the fitted candidate, using float64 central differences at steps 1e-4 and 1e-5.
Every perturbed candidate passes the checked boundary. Absolute/relative
tolerances are 2e-5, with a separate maximum absolute discrepancy <2e-5.
The largest actual discrepancy across the four experiments is
1.25118e-06. Active-bound finite differences verify the raw objective
in its valid ambient domain; feasibility constraints are handled separately
by the projected-gradient diagnostic.

At the first Configuration B observation, focused tests differentiate new
landmark means, landmark covariance blocks and robot/landmark cross-covariances
with respect to both coefficient channels. These derivatives match central
differences (step 1e-5, rtol=2e-5, atol=2e-7) and are nonzero. Independent
Stage 3B.2 augmentation/Jacobian references remain in the complete regression
suite. Zero-coefficient histories also match the preserved filter within
2e-14, including the dynamic initialization path.

## Accuracy and negative results

| Configuration / truth | Baseline J | Fitted J | Position RMSE change | Heading RMSE change | Landmark RMSE change |
|---|---:|---:|---:|---:|---:|
| a_constant | 504.382548829 | 489.749966318 | -8.623% | -14.470% | -9.969% |
| b_constant | 496.060384434 | 480.740131038 | -25.981% | -30.389% | -46.372% |
| a_smooth | 497.822151176 | 489.563321765 | -3.596% | -7.343% | +30.527% |
| b_smooth | 489.547364382 | 480.717346235 | +16.365% | +17.364% | +21.550% |

Negative percentages mean reduced error. Both constant cases improve all
listed accuracy metrics in this realization. All four fits reconstruct both
disturbance channels better than zero. The A smooth fit improves robot errors
but worsens landmark RMSE; the B smooth fit worsens position, heading and map
RMSE despite lower J and better disturbance RMSE. These negative results are
part of the validation outcome and are not hidden by a required-winner test.

This behavior shows that minimizing the finite-data innovation quadratic does
not minimize every anchored pose/map error. Joint map/pose adjustment, finite
measurement noise, the assumed R/Qc and weak coefficient directions are
plausible contributors, not established unique causes. This task does not add
an unrequested regularizer, change the objective or tune noise/bounds to force
a positive outcome. Further paired-noise and sensitivity analysis should
investigate the smooth empty-map failure case.

The true constant coefficients and independent sampled smooth approximation
are diagnostic references only. They never initialize the primary fits. True
inputs can yield worse pose/map errors than a fitted noisy-data solution under
this uncertain prior/map and finite realization; lower cost is not a proof of
correct reconstruction, calibrated covariance or global identifiability.

## Constraints, convergence and covariance interpretation

All four fits report SciPy function-reduction convergence before the configured
gtol=1e-6. Projected infinity norms range from about 7.53e-5 to 2.82e-4.
Raw norms are larger because coefficient indices 0, 6 and 13 are at bounds
in each fit (first/last linear coefficients and last angular coefficient).
No strict-stationarity or unconstrained-optimum claim is made. Bounds can
particularly affect boundary disturbance reconstruction. Function, gradient,
iteration and probe records are saved in each result/history JSON.

Covariance figures say exactly:

> Conditional EKF covariance; spline coefficients treated as fixed.

They omit uncertainty in fitted spline coefficients. Position ellipses use
actual 2x2 robot/landmark marginals and threshold 5.991; full widths are twice
the square roots of threshold-scaled eigenvalues. Robot ellipses occur at six
regularly spaced window times. Error panels show wrapped heading error and
componentwise ±2 standard-deviation envelopes. Full active histories and
marginals are validated without projection or clipping. Inactive landmark
storage is neither plotted nor included in active minimum eigenvalues.
A uses triangle map markers for pre-window births; B uses circles for births
within the window. The t0 boundary and anchored world coordinates are explicit.
The saved active covariance eigenvalues are positive; zero eigenvalues of
inactive padded storage are expected and excluded from that statistic.

There are **40 figures** (20 PNG, 20 vector PDF), including separate figures
for both configurations and both truths. Figure smoke tests and visual
inspection check readable labels, equal map/trajectory axes and conditional
covariance labelling. These are offline fixed-window results, not real-time
latency or empirical 95% coverage evidence.

## Runtime, preservation and saved evidence

Report timing tables separate explicit evaluation/value-gradient compilation,
the checked cold reference (including possible auxiliary VJP compilation),
preparation, warm optimization and steady-state mean/p95 execution. Warm
optimization excludes preparation. Baseline diagnostic-gradient timing is
not a cost required to run the zero-disturbance filter. No cross-machine timing
or statistical-runtime claim is made.

All **163 pre-existing protected files** match their
pre-task SHA-256 digests, including Baselines A/B, old mathematical modules,
all earlier tests/artifacts, and tracked sibling spline sources. The sibling
remains on uniform_development and both its worktree and the baseline worktree
are clean. Original user .DS_Store changes remain untouched. All final source
files match the digests captured with the complete test run. Only project.md's
status/accounting record and the following new implementation/docs/tests were
added for Stage 4C:

- `estimation/slam_window_objective.py`
- `simulation/slam_window.py`
- `benchmarks/milestone4c.py`
- `benchmarks/milestone4c_reporting.py`
- `benchmarks/slam_window_plots.py`
- `estimation/STAGE4C.md`
- `tests/test_slam_window_a.py`
- `tests/test_slam_window_b.py`
- `tests/test_milestone4c_reporting.py`

The numerical artifacts are confined to artifacts/milestone4c/. See
[preservation.json](preservation.json), [validation.json](validation.json),
[configuration.json](configuration.json), [results.json](results.json) and the
per-experiment NPZ/JSON files. Post-serialization checks independently recompute
pose/map/disturbance metrics, J and logdet from saved arrays, verify arrivals,
compare paired observations and all masks/slots, inspect figure file formats,
and confirm source/preservation hashes.

## Completion and issues before Milestone 5

```json
{
  "required_experiments_present": true,
  "configuration_a_validated_before_b": true,
  "constant_b_validated_before_smooth": true,
  "all_numerical_experiment_checks": true,
  "arrival_and_update_accounting": true,
  "identical_datasets_across_configurations": true,
  "independent_truth_validated": true,
  "all_previous_protected_files_preserved": true,
  "current_complete_test_suite": true,
  "all_figures_generated": true
}
```

Stage 4C's controlled completion criterion is met. Before implementing
Milestone 5, choose and document a sliding-arrival information policy that
avoids reusing measurements, map-initialization accounting as records cross
window boundaries, and coefficient shifting/warm starts. Investigate the
negative smooth empty-map result and the effects of noise, active bounds and
stopping criteria; preserve it as an accuracy/runtime limitation. Conditional
covariances do not resolve fitted-input uncertainty automatically.

No moving-window shifting, arrival-cost update, nearest-neighbor association
or Victoria Park integration was implemented. Monte Carlo across noise/seeds,
other disturbance/noise/sparsity scenarios and final three-estimator comparisons
remain later validation gates, not claims established by deterministic repeats.
