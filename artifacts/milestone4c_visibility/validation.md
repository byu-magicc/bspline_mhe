# Visibility extension validation

The audit, assumptions and source trace are in
[STAGE4C_VISIBILITY.md](../../estimation/STAGE4C_VISIBILITY.md). Actual counts,
birth times, accuracy and disturbance metrics, including negative outcomes,
are in [report.md](report.md). All experiments remain fixed-window,
known-association, paired seed-7 cases; deterministic repeats are not Monte Carlo.

Executed commands:

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q tests/test_slam_visibility.py tests/test_slam_window_a.py tests/test_slam_window_b.py tests/test_milestone4c_reporting.py --junitxml=artifacts/milestone4c_visibility/pytest.xml
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone4c_visibility/pytest.xml
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4c_visibility
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4c_visibility --report-only
```

- Focused checks: **36 passed**, 92.90 s, exit 0. The full suite subsequently
  replaced the focused XML at the same path.
- Complete suite: **306 passed**, no failures/errors/skips, 228.40 s pytest
  time, 233.229 s subprocess time, exit 0. Five warnings come from pre-existing
  `record_property` use with pytest's default `xunit2` XML format. They concern
  report metadata. `-o junit_family=legacy` avoids those warnings on future runs.
  Actual stdout, XML, command and source hashes are retained in
  [test_record.json](test_record.json), [pytest_stdout.txt](pytest_stdout.txt)
  and [pytest.xml](pytest.xml). No required test was left unrun.
- All eight numerical cases completed, exit 0, with two independent zero-start
  fits each. All 16 fits converged, with exactly repeated coefficients and
  accepted histories; all requested observations remained valid.
  [run_stdout.txt](run_stdout.txt) retains actual optimizer progress.
- Every zero-coefficient replay matched the original Baseline B filter at
  absolute tolerance 2e-14, including partial/empty maps and delayed births.
  Every case also passed the original float64 all-14-coefficient gradient checks
  at zero, a nonconstant vector, and the fitted vector, with central steps
  1e-4 and 1e-5 and the original 2e-5 tolerances.
- All seven-second regression dataset fields match their original Stage 4C
  counterparts **bitwise**, and the four zero/fitted objective pairs reproduce
  the original saved results. Shared forward/regression observations have the
  same full-slot seeded measurement-noise realization.
- Both forward cases initialize ID 0 before the window only for Configuration
  A. ID 2 initializes at its first available in-window observation: 4.7 s for
  constant truth, 4.0 s for smooth truth. B initializes ID 0 at 2.1 s. IDs 1,
  3, 4, 5 never enter the estimated map. The two scenarios have no empty
  scheduled scans in this realization; focused tests additionally force empty
  scans, long unobserved intervals, and a completely empty window, checking
  J=0, zero gradient, no births and exact zero-filter equivalence where applicable.
- There are 96 new PNG/PDF figures, including eight combined maps in both
  formats. Combined maps were refreshed from saved numerical histories after
  final layout changes; no fitting is needed for that plotting operation.
  Visual inspection covered regression and finite-FOV maps, mixed birth-marker
  shapes, the footprint, recorded rays, readable IDs and external legends/zoom.
- After the complete suite, only the report generator was edited to explicitly
  list metrics worsened by fitting. The final `--report-only` execution passed
  all six [saved-data validation checks](validation.json). Estimator, simulator,
  plotting and test sources match their complete-suite source hashes.
- [preservation.json](preservation.json) checks 328 pre-task files. Only three
  existing files changed: the Stage 4C simulator, its benchmark configuration/
  arrival adapter, and its plotting module. All original Stage 4C artifacts,
  original baseline source, mathematical estimator/objective modules, spline
  implementations and prior tests are unchanged. The sibling spline repository
  remains clean on `uniform_development`.

Reduced visibility worsens fitted **linear disturbance** RMSE in all four
paired cases relative to the unlimited sensor, while angular RMSE decreases
slightly. All forward fitted disturbance RMSE values are still below assuming
zero disturbance in this realization. Position RMSE improves over Baseline B
in all four forward fits, but both smooth fits worsen heading and landmark
RMSE despite lower J. Different map coverage also prevents interpreting map
RMSE across sensors as a matched-landmark comparison. No sensor, noise, spline,
bound or optimizer setting was tuned to hide these outcomes. Neither lower
cost nor these deterministic comparisons proves identifiability or calibrated
uncertainty. Covariances condition on fixed fitted coefficients.

Milestone 5 arrival-information/boundary-birth/warm-start research decisions,
final paired Monte Carlo comparisons, nearest-neighbor association and
Victoria Park remain outside this extension.
