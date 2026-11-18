# Stage 5B executed validation and limitations

Stage 5B's **controlled deterministic implementation is validated**. The scope
is six seed-7 cases, each with three consecutive full L=5 s windows ending at
5.0, 5.1 and 5.2 s, optimized from projected/clipped and zero initial guesses.
This is 18 windows and 36 fits, not Monte Carlo or long-run reliability evidence.
See [report.md](report.md), [results.json](results.json), and
[implementation details](../../estimation/STAGE5B.md).

## Executed commands and order

The initial implementation-only run had **12 passes and one failure**: a test
compared a saved canonical 0.3 s time to the different binary representation of
literal `0.3`. Its assertion now uses the canonical window endpoint; measurement
inclusion, estimator equations and tolerances were not changed. The corrected
test passed, followed by the complete repository gate:

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone5b/pytest.xml -o junit_family=legacy
```

Outcome: **381 passed**, no failures/errors/skips/warnings, pytest 370.17 s,
subprocess 378.134 s, exit 0. This includes 16 new Stage 5B cases and all 365
existing tests. The source hashes in [implementation_gate.json](implementation_gate.json)
match the final estimator, simulator, spline and test files. No required test
was left unrun. Actual stdout/XML and initial failure evidence are retained.

Only after this gate passed, the three-update constant-regression pilot ran:

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5b --case regression_constant
```

It completed, exit 0, with six converged fits. Its saved schedule, arrivals,
birth/update accounting, ID maps and clipping were audited before continuing:

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5b --case all
```

This reused the completed pilot and ran the other five short cases, exit 0.
All **36 fits converged**, with no invalid numerical candidates or retries.
First-window fits start from zero in both methods and match exactly. The
three-update unit-test sequence is independently run twice with exact repeated
coefficients/costs. The noisy six-case suite is one realization, not two trials.

Post-serialization gradient and report commands also completed with exit 0:

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5b_gradients
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5b --report-only
```

Three full-size saved windows cover the empty initial map, a boundary birth
entering the arrival, and a finite-sensor window ending in an empty scan.
All 14 coefficient derivatives are checked at zero, a nonconstant vector and
the optimized vector with float64 central steps 1e-4 and 1e-5. The original
rtol/atol 2e-5 and separate max absolute discrepancy <2e-5 are retained.
The largest measured discrepancy was **1.659552e-6**. No tolerance was relaxed.
Bound-active coefficients are perturbed in the valid ambient objective domain,
as in the existing Stage 4C gradient checks. See [gradient_checks.json](gradient_checks.json).

All ten [saved-data checks](validation.json) pass. These check complete schedules,
boundary posteriors/masks, exactly-once births, persistent forward/replay IDs,
raw projection and clipping, identical first fits, the full suite and full-size
gradients. The three finite-sensor constant updates have **zero current detections**
and all six corresponding fits still run; historical window innovations remain.

## Results and runtime interpretation

Projected starts used **1,482 iterations**, versus **1,660** for zero starts,
across all six cases. Warm optimization totaled **178.63 s** versus **192.89 s**.
Recorded update wall time totaled **414.01 s** versus **405.83 s**; fewer iterations
did not produce lower total wall time in this startup-heavy sample. Explicit
evaluation/value-gradient compilation totals were about **101.49/101.30 s**,
with additional cold-reference/preflight preparation reported separately.
Fixed method order and compilation caches affect timing; no statistically
significant runtime gain is claimed from three updates per case.

Recorded update timing excludes the host initial-guess operation and artifact
I/O. A separate warm-cache remeasurement found mean projection/clipping time
about **16.5 microseconds** and zero-start construction about **12.3 microseconds**;
these are not retroactively added to the measured timings. The implementation
reuses the validated Stage 4C constant-capture factory and recompiles new windows.
It is an offline estimator, not a real-time implementation. The independent
forward EKF has separately compiled dynamic scan inputs; each profiled result
matches its saved forward history.

Clipping occurred in **9 of 18** projected-start updates, equivalently **9 of 12**
updates after the deterministic zero-initialized first windows. There were **15**
clipped components in total; the largest absolute adjustment was **0.312062**.
The original coefficient bounds, objective and optimizer tolerances are unchanged.
Raw projections, feasible guesses and clipping magnitudes remain auditable.

Warm/zero starts generally converge to very similar costs and accuracy but have
mixed iteration/runtime outcomes. MHE worsens position, heading and map RMSE in
both zero-disturbance cases and in regression smooth truth. It improves those
metrics in the constant cases and forward smooth truth. Finite-sensor constant
endpoint angular-disturbance RMSE is about 0.05838 rad/s, worse than the baseline's
0.05 rad/s despite better pose/map errors. These outcomes are retained without
changing bounds, noise, objective or stopping tolerances to favor MHE.

Endpoint and pooled-window disturbance metrics differ: the former evaluates the
reported input at the right edge, while the latter pools overlapping 501-point
window grids. They are not independent statistical samples. Errors stay in the
original prior-anchored frame with no rigid alignment; map coverage accompanies
landmark RMSE. Conditional EKF covariance omits fitted-coefficient uncertainty.
Lower innovation cost and SciPy convergence do not imply correct input recovery,
calibrated covariance, gradient stationarity or a global optimum.

## Preservation, files and next steps

All **556 pre-task file hashes match**, including approved Stage 5A, Stage 4C and
visibility artifacts, original baseline, spline implementation, prior tests and
project guidance. The sibling spline repository remains on `uniform_development`.
Only these new source/documentation paths and this artifact directory were added:

- `estimation/moving_horizon.py`
- `simulation/moving_horizon.py`
- `tests/test_moving_horizon.py`
- `benchmarks/milestone5b.py`
- `benchmarks/milestone5b_gradients.py`
- `estimation/STAGE5B.md`
- `artifacts/milestone5b/`

Per-case NPZs retain the data, forward states/covariances/statuses, exact arrivals,
window basis/data/IDs/masks, full optimized histories, births/corrections and
covariances. JSON retains all optimizer trials, gradient timings, clipping and
pose/map jumps. Each case has PNG/PDF endpoint-estimate and accuracy/runtime
figures (24 figures total). Scalar and array audits use the serialized evidence.

Recommend longer deterministic sequences, inspecting overlap revisions and
endpoint disturbance errors, and reusable compilation before further benchmarking.
The report proposes an approval-dependent **15-seed paired variance pilot**, then
a final trial count selected from measured paired standard deviation and a
specified confidence-interval width. Current short-case timing projects about
3.44 hours for that pilot; it is not a performed Monte Carlo experiment.

**Work stops here.** No Monte Carlo runs, new nearest-neighbor association or
Victoria Park integration were performed. Those require additional approval.
