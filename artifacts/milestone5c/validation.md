# Stage 5C validation and execution record

Completed six seed-7 scenarios, 11 consecutive complete horizons each, with independent projected and zero starts: **66 windows / 132 fits**. All commands exited successfully. The mathematical estimator and previous artifacts remain unchanged.

| Validation | Result |
|---|---|
| Initial diagnostic/cache tests | 8 passed, 36.31 s |
| Complete repository regressions | 389 passed, 388.41 s; no failures/errors/skips |
| Original Stage 4C/5A/5B files and sources | All 689 protected SHA-256 values unchanged |
| Tested core/test sources | All 36 still match the full-test run |
| Feasibility experiment | All returned values/gradients bitwise equal on 3 saved windows; full prior fit exactly reproduced |
| Global-time truth and error profiles | Independent truth difference exactly zero; all serialized profiles reproduced |
| Birth/empty-scan and boundary masks | All 66 windows audited; optimization includes empty endpoint scans |
| Zero-coefficient replay | Every window agrees with independent forward EKF at atol=2e-14, rtol=0 |
| Full whitening sensitivity | All 14 columns, 3 selected full-size windows: maximum FD absolute error 2.204042e-08; h=1e-5, rtol=2e-5, atol=2e-6 |
| q.T q = J | Maximum saved difference 2.899014e-12 |
| 2 D.T q = grad J | Maximum absolute difference 3.343496e-11; checked rtol=2e-9, atol=2e-8 |
| Deterministic noninitial solve repeat | Final regression/smooth zero-start coefficients and cost bitwise equal |
| Feasible terminal-coefficient probes | Final forward/constant j6 changes alter pose; q and J exactly unchanged |
| Direct host remeasurement | 198 complete field comparisons bitwise equal; basis work measured directly |
| Covariance | All histories finite and symmetric within 1e-10; active covariance checks passed without repair |
| Published figures | 34 PNG/PDF pairs, representative figures visually inspected |
| Sibling spline repository | uniform_development branch, clean worktree |

All 132 fits report success using the unchanged function-reduction criterion; **none meets projected-gradient gtol=1e-6**. No invalid optimizer trials occurred. Rank deficiency is reported, not repaired.

## Reproduction commands and retained output

**compilation feasibility:** passed; bitwise equivalence on 3 prior windows and one complete solve. Output: [feasibility_stdout.txt](feasibility_stdout.txt).

~~~sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.stage5c_feasibility
~~~

**initial focused tests:** 8 passed in 36.31s. Output: [diagnostic_initial_stdout.txt](diagnostic_initial_stdout.txt).

~~~sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q tests/test_stage5c_diagnostics.py --junitxml=artifacts/milestone5c/diagnostic_initial.xml -o junit_family=legacy
~~~

**complete regressions:** 389 passed in 388.41s; no failures, errors or skips. Output: [pytest_stdout.txt](pytest_stdout.txt).

~~~sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone5c/pytest.xml -o junit_family=legacy
~~~

**extended deterministic suite:** 66 windows; 132 fits; exit 0. Output: [extended_stdout.txt](extended_stdout.txt).

~~~sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c
~~~

**selected full-size gradients and exact repeat:** passed; exit 0. Output: [saved_validation_stdout.txt](saved_validation_stdout.txt).

~~~sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c --validate
~~~

**direct host remeasurement:** all 66 windows bitwise equal on 3 repeats each; exit 0. Output: [host_profile_stdout.txt](host_profile_stdout.txt).

~~~sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.stage5c_host_profile
~~~

**final report and serialized audits:** all audits passed; 34 PNG/PDF pairs; exit 0. Output: [report_stdout.txt](report_stdout.txt).

~~~sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c --report-only
~~~

Commands run from the repository root. Existing completed cases are reused by report generation; no saved fit is silently reoptimized. The selected --validate command explicitly repeats one solve to test reproducibility.

## Profiling limitation

The completed C++ basis wrapper frame was absent from cProfile in this environment. Its original raw zero is not treated as a measurement; the buffer remainder includes that work. A separate direct timer remeasures all windows three times, with bitwise result checks. Online update timings are preserved. See host_profile_remeasurement.json and the detailed timing table in report.md.

## Findings and limits

Right-endpoint RMSE exceeds the prespecified [1.25,3.75] s interior in all six cases for each channel. The left endpoint can be worse than the right. Empty endpoint scans are not necessary for boundary degradation. The final two forward/constant windows have local rank 12/14; the final window has no correction after sigma=3.6 s while j6 begins at sigma=3.75 s. Direct feasible probes demonstrate unconstrained terminal inputs for that particular objective/data pair.

Pose/map degradation remains in regression/zero, regression/smooth and forward/zero. First-window arrival mean errors are zero; these cases retain full local sensitivity rank and observed endpoints. Lower costs and repeatable solver results do not establish accurate disturbance reconstruction or a unique causal explanation.

Projected starts use 4812 total iterations versus 5618 from zero, but require more iterations in 16/66 windows. Projected starts clip in 55/66 windows; the clipping policy and bounds remain unchanged.

The benchmark-only reusable JAX program preserves the original computation and is pinned to JAX 0.4.35. It removes repeated compilation; it does not make a full differentiated EKF replay inexpensive. No production refactor is included.

[Main report](report.md), [machine-readable summary](diagnostics.json), [execution/environment/source record](execution_record.json), [cache feasibility](compilation_feasibility.md), [selected checks](saved_validation.json), [preservation](preservation.json).

Stage 5C complete. No Monte Carlo, nearest-neighbor association, Victoria Park, tuning, regularization, objective change or major refactor performed.
