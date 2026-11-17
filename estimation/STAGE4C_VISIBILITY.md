# Stage 4C visibility audit and combined SLAM figures

This extension keeps the fixed [2,7] s window, original dynamics and numerical
integration, landmark initialization equations, innovation quadratic and
coefficient optimizer. Configuration A/B below refers to arrival information;
**Baseline B** is the zero-disturbance estimator in either configuration.

## What the original simulation did

- `simulation/slam_window.py:simulate_slam_window` inherited Stage 4B's explicit
  `observation_policy`: all six landmarks at positive 0.1 s multiples. It had no
  range gate, no angular field-of-view gate, no occlusion and no sensor offset.
  The effective sensor was unlimited-range and 360°. All six landmarks produced
  a noisy measurement at each of the 70 scans. True robot pose determined
  measurement values, but never selected which landmarks were measured.
- This is a valid idealized sensor assumption, and matches the documented
  controlled Stage 4B/4C experiment. It was insufficiently explained in the
  figures. The circular arrangement of the six fixed coordinates is not an
  implementation defect. The older class-project `generator.markersInFov`
  separately uses wrapped true-pose bearings, a default pi-radian FOV, no range
  limit and at most two observations. Stage 4C does not call that generator.
- `benchmarks/milestone4c.py:prepare_a` starts with only the robot prior at t=0.
  The preserved continuous-discrete `filter_slam` consumes observations at
  0.1–2.0 s: six births at 0.1 s and 114 corrections. The **complete** t=2
  posterior, including robot/map uncertainty, cross-correlations, active mask
  and IDs, becomes the arrival. No prior map is supplied.
- `estimation/slam.py:observe` matches IDs only against active IDs; a new ID
  calls `initialize_landmark`. The inverse range/bearing mean and Jacobians
  propagate robot and measurement uncertainty and all existing cross-covariances
  into the new landmark. A birth consumes its measurement once and contributes
  no innovation cost. Later observations correct it through Joseph updates.
  Known association supplies identity; no true landmark coordinate enters this
  filter. The full truth map is retained solely for simulation, metrics and plots.
- Configuration B uses `prepare_b`: command/Qc-only propagation of the original
  robot prior to 2 s, with no measurements and an empty map. In the regression
  scenario all six first post-window-start records at 2.1 s initialize it.
- The original landmark figure contains **both** black ground-truth crosses and
  colored active filter estimates with final marginal covariances. Its global
  pre-window label was correct for original Configuration A because all six
  were actually mapped, and its in-window label was correct for B. Such a global
  label is insufficient with mixed birth times; marker shapes now follow each ID.

## Explicit sensors and saved observations

`simulation/slam_window.py:sensor_settings` defines two assumptions:

| Scenario | Minimum range | Maximum range | Angular FOV | Extrinsics |
|---|---:|---:|---:|---|
| `regression` | 0 m | unlimited (`null` in JSON) | 2 pi rad (360°) | colocated, aligned |
| `forward` | 0.2 m | 6 m | pi rad (180° forward) | colocated, aligned |

The forward sensor is an illustration chosen before fitting, not a sensor
specification or optimized experimental design. Landmarks, commands, noise,
prior, Qc, knots, degree, bounds, optimizer options and seeds remain shared.
The forward FOV follows the older simulation's forward-hemisphere convention;
its finite range is an additional explicit assumption. There is no observation
cap, occlusion, probabilistic missed detection or estimated-pose visibility test.

The `sensor` config dictionary accepts `range_min`, `range_max` and
`field_of_view_rad`. Visibility uses true robot pose and landmark geometry:
`range=hypot(dx,dy)` and `bearing=wrap(atan2(dy,dx)-theta)` in [-pi,pi).
Both range bounds and `abs(bearing)<=FOV/2` are inclusive. It is fixed before
noise and before optimization. The existing measurement/inverse model assumes
zero extrinsics; nonzero sensor offsets/yaw are explicitly rejected to prevent
inconsistent generated measurements. Supporting nonzero extrinsics would
require a separately authorized measurement-model change.

At each scheduled scan, full-slot Gaussian noise is drawn in the original order,
then noise is added only to visible rows. This preserves bitwise regression
data and identical noise for shared visible IDs across scenarios. Non-observed
slots contain neutral zeros and have false masks. Actual IDs are given by
`observation_ids[time,mask[time]]` in saved NPZ files and explicitly listed for
every scheduled scan, including empty ones, in `observations_*.json`.

## Information and initialization accounting

Configuration A now allows a partial or empty arrival map: its only map
information is valid pre-window observations through t=2. The old assertion
requiring all six landmarks to be initialized was removed. Window replay
continues to exclude the t=2 records already consumed by the arrival.
Configuration B retains its distinct observation-free empty-map arrival and
the same strictly-after-2 measurement stream. Both estimators within each
configuration consume the same immutable observations, masks and IDs.

Fixed-capacity arrays and false masks already supported changing scan counts
and empty scans in `filter_slam` and the differentiable objective. Their
mathematical implementations required no changes. The fixed plan decides
births and corrections using available IDs/masks, without coordinates or
coefficient-dependent visibility. A newly visible ID initializes exactly when
its first available observation is consumed. Landmarks never observed remain
inactive and have no estimated marker or ellipse. Their true crosses may still
appear as labeled references. No map prior, invented birth, or post-fit frame
alignment is introduced. Initialization times are read from actual birth statuses.

## Figures and reproduction

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4c_visibility
```

The command runs regression and forward sensors, both arrival configurations,
and constant/smooth truth. Existing Stage 4C gates already cover their original
development order. The new command retains two zero-start fits, zero-replay
equivalence, full-vector gradients, independent truth checks and saved histories
for each new case. Results go only to `artifacts/milestone4c_visibility/`, under
`regression/` and `forward/`. Original `artifacts/milestone4c/` results are read
for regression checks and remain unchanged. See the generated
[report](../artifacts/milestone4c_visibility/report.md) for actual counts,
birth times, errors, reconstruction outcomes and figure links.

Each `*_slam_map.png` and vector PDF combines trajectories and final maps on
world x (m), y (m) axes with equal aspect. Black denotes truth, blue Baseline B,
orange input-spline estimates, and green the window start. A trajectory inset
makes robot differences visible on the full landmark scale. ID labels accompany
true references; colored estimated map markers use triangles for pre-window
births and circles for in-window births, separately for each ID.

The selected scan is the scheduled time closest to the window midpoint
(t=4.5 s). A gray true-pose sensor footprint shows range/FOV, faint lines plot
the actual noisy range/bearing records, black squares mark recorded IDs, and
gray rings mark IDs already mapped before that scan. For regression the
footprint covers the displayed map and the title states unlimited range. It
does not imply an artificial finite cutoff. Ground-truth crosses and gray ID
references are not extra map estimates. The footprint is a truth diagnostic.

Robot ellipses show 95% 2D position marginals at t=2,3,4,5,6,7 s. Landmark
ellipses show final marginals at t=7 s, using chi-square threshold 5.991.
**Conditional EKF covariance; spline coefficients treated as fixed.** The
ellipses omit fitted-coefficient uncertainty. Invalid covariance histories and
marginals are rejected without clipping or repairs.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q tests/test_slam_visibility.py tests/test_slam_window_a.py tests/test_slam_window_b.py tests/test_milestone4c_reporting.py
```

Focused tests cover inclusive range/FOV boundaries, heading/bearing wrapping,
pose-dependent visibility, invalid settings/extrinsics, geometry before noise,
bitwise saved regression data, no unknown coordinates in arrival construction,
partial-map arrivals, delayed births, masked and all-empty scans, zero-replay
equivalence, gradients of an empty objective and figure generation. Existing
Stage 4C tests cover objective gradients and map augmentation derivatives.
Paired deterministic repetitions do not establish statistical identifiability
or outperforming EKF-SLAM in general. Reduced visibility can worsen recovery;
the new report retains such outcomes. No Milestone 5 work is implemented.
