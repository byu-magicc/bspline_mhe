# Stage 5C diagnostic methods and reproducibility

Stage 5C characterizes the approved Stage 5B estimator. All estimator, optimizer,
spline, simulation and prior benchmark source files remain unchanged. New code
lives under `benchmarks/`, with tests and separate `artifacts/milestone5c/` output.
No objective, covariance, prior, sensor, bounds, tolerance or association policy
is tuned. No Monte Carlo, nearest-neighbor or Victoria Park work is performed.

## Scope fixed before extended results

Reference parameters remain d=3, M=4, L=5 s, RK4 0.025 s and scheduled scans
every 0.1 s. Zero, constant and smooth disturbances start at t=0 as in Stage 5B.
Both original sensors are retained: unlimited/360°, and 0.2–6 m/180° forward.
Seed 7, all noise assumptions and the robot prior are unchanged.

The feasibility/profile assessment selects duration **6.0 s**, giving **11 full
consecutive windows** ending at 5.0–6.0 s in each of six scenarios, with two
initializations (132 fits total). This extends the three-window Stage 5B run
without treating overlapping windows as independent trials. The choice and
expected compute benefits were recorded before extended diagnostics in
`artifacts/milestone5c/compilation_feasibility.md`.

## Isolated compilation experiment

`benchmarks/stage5c_cache.py` uses `jax.make_jaxpr` on the unchanged fixed-window
factory and lifts captured arrays into runtime arguments of that same program.
It caches compiled evaluation/value-gradient functions by graph/result-tree
fingerprint. Shape/structure changes get separate entries; no padding scheme,
observation order, slot policy or time grid is changed to force reuse.

This is an isolated benchmark adapter, pinned to the installed JAX 0.4.35
`jax.core.eval_jaxpr` interpreter. It is not a production API migration or EKF
architecture refactor. A JAX upgrade requires revalidation. The optimizer
orchestration is a static copy of the original function with only bridge
construction injected. An AST test requires exact agreement after that
substitution. Candidate checks, checked AD references, all preflight probes,
callbacks, bound handling, options and failure behavior are inherited/preserved.

Before extension, three saved windows had bitwise-equal results/gradients at
zero/nonconstant vectors, reused one structural graph, and a complete cached
fit exactly reproduced the saved coefficients, cost and 133 iterations.
`benchmarks/stage5c_feasibility.py` records this evidence. The original Stage 5B
API still uses its original compilation behavior.

One structural replay graph has separate evaluator, value/gradient, sensitivity
and checked-AD auxiliary executables. Program reuse does not mean one single
XLA compilation for every operation. Compilation and lowering are measured
separately from tracing and execution. Full EKF replay still occurs for every
candidate; caching never reuses candidate states or invalidates data dependence.

## Error profiles and time coordinates

The fixed 501-point sigma grid covers [0,5] s and includes every knot and both
endpoints. For each optimized vector C_k, estimates use the existing C++ basis
on sigma, while true inputs use **global time t_k-L+sigma**. Independent closed
forms are compared with the simulation's existing truth-input function.

For each channel, E(sigma)=sqrt(mean_k(error_k(sigma)^2)). The prespecified
primary interior is [1.25,3.75] s; its scalar comparison is
sqrt(mean_sigma(E(sigma)^2)) over that interval. The open interior excluding
only endpoints is also recorded, so the endpoint conclusion can be inspected
under a second definition. Comparisons are deterministic inequalities, not
independent-sample tests. No confidence intervals or p-values are constructed.

Separate linear/angular figures compare both starts and the zero-input forward
reference for each scenario and sensor. Shading identifies the interior only.
Representative first/middle/last windows show true/estimated inputs, absolute
errors, both boundaries, knots and actual measurement times/IDs. Births,
corrections and scheduled empty scans are distinguished.

## Basis and exact whitening sensitivity

Every basis evaluation uses `UniformBasis` and its C++ binding. Support follows
the existing index convention, intersected with [0,L]. Integrated squared
basis magnitudes are the diagonal of the validated Stage 5A zero-shift Gram
matrix, evaluated with piecewise Gauss–Legendre quadrature. Independent adaptive
integration tests check those values. The unclamped right endpoint uses
coefficients 4,5,6 with weights [1/6,4/6,1/6], not only the last coefficient.

For correction records, q=L(c)^-1 r(c), S(c)=L(c)L(c).T, with lower-Cholesky
whitening in the original range/bearing and observation order. Exact JAX forward
mode computes D=dq/dc at every fitted vector, including derivatives of S and
Cholesky whitening. Masked and birth rows are zero in the padded diagnostic
buffer and removed before norms/SVD; they never become correction records.

Every fit checks q.T q=J and 2 D.T q=grad J. Numerical rank uses
max(actual correction rows,14)*eps64*s_max; a separate 1e-3 relative practical
rank is retained. Condition numbers are null for numerical deficiency. Columns
retain their physical units, and covariance-dependent whitening makes spectra
convention dependent. These quantities do not establish formal nonlinear
observability. All column norms and singular values are retained per window.

Selected full-size windows use central differences of every q column at step
1e-5, rtol 2e-5 and atol 2e-6. These are checked at the actual fitted vector,
including ambient perturbations of bound-active coefficients. A synthetic test
with coefficient-dependent covariance detects an accidentally frozen whitening.

## Information, arrivals and degradation

Every window records scan counts, observed/empty scans, corrections, births,
mapped counts, relative observation times/IDs and age of the last valid
correction. If no correction lies in the window, that age can refer to the last
causal forward correction before its left boundary. Births are not corrections.

Errors at the report time and at the arrival boundary are kept separate.
Arrival/full-history covariance diagnostics preserve the complete augmented
SLAM state, cross-covariances and unchanged 1e-10 symmetry/PSD criteria. Counts
and matched IDs accompany map errors. All metrics retain the original anchored
frame with no post-hoc alignment. Conditional EKF covariance treats fitted
spline coefficients as fixed.

Both starts remain independent; neither solution is selected on behalf of the
other. Initial/final costs, calls, iterations, termination reasons, gradient
norms, active/near bounds (1e-10/1e-3 distance), clipping and overlap revisions
are retained. A baseline-S quadratic of optimized residuals is a diagnostic
cost decomposition only and never enters optimization.

Descriptive correlations/group RMSEs are not causal evidence. In particular,
an all-empty-endpoint scenario lacks an internal observed/empty contrast.
Symmetric basis energy cannot alone explain asymmetric errors, numerical full
rank does not guarantee accurate inputs, and a lower J or SciPy function-reduction
termination is not a truth-error or global-optimality criterion.

## Timing interpretation

Window preparation is profiled using cProfile, with synchronized direct timing
of arrival retrieval. In the local Python 3.12/pybind environment cProfile omitted
the completed `UniformBasis.evaluate` frame. The raw zero `basis_seconds` is
therefore unresolved, not a measured zero; raw `observation_buffer_seconds`
includes basis work and is relabeled as a buffer/basis remainder in summaries.
`stage5c_host_profile.py` directly remeasures all 66 saved windows three times,
separating basis/arrival/remaining host work and checking every returned field
bitwise against saved windows. These are explicitly post-run timings, not
replacements for online fit timings. The host
birth/observation plan is timed inside the unchanged objective factory.
Projection and clipping have separate direct timers and bitwise policy tests.
Tracing, lowering, explicit compilation, checked objective evaluation,
value/gradient execution, SciPy optimization and final replay are separate.

Some times are nested: preparation includes retrieval/buffers/basis; factory
time includes the birth plan; SciPy time includes objective/gradient trials.
They must not be summed twice. Value/gradient execution includes the value
required by the existing SciPy pair. Update timing includes host initialization
but excludes sensitivity, other diagnostics and artifact publication. Those
operations are measured separately. NPZ writes use run-time timers; JSON
serialization is remeasured after estimation. Plot construction has an overall
timer, and figure publication has individual layout/render/write timers.

Diagnostics retain mean, median and p95 for each category and distinguish zero
compile cost on hits from cold compilation events. Profile instrumentation,
fixed method order and OS effects limit fine timing comparisons. No real-time
claim is made.

## Commands and artifacts

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.stage5c_feasibility
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone5c/pytest.xml -o junit_family=legacy
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c --validate
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.stage5c_host_profile
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c --report-only
```

The initial eight new diagnostic/cache tests and full **389-test** regression
suite passed before extended runs. Selected full-size FD/repeat checks and
post-serialization audits are reported with the final artifacts. Existing case
files are reused for report generation, not silently reoptimized. Stage 4C,
5A and 5B artifacts are read-only inputs protected by a pre-task SHA-256 snapshot.

The completed suite contains 66 windows/132 successful function-reduction
terminations. No fit meets projected-gradient gtol=1e-6. Exact full-whitening
sensitivity checks have maximum central-FD error 2.21e-8; a selected noninitial
solve reproduces cost and coefficients bitwise. All right-endpoint profiles
exceed the defined interior in both channels, but the right endpoint is not
always worse than the left. In the last forward/constant window, the final
coefficient in each channel has no correction support: feasible changes alter
endpoint pose while leaving q and J bitwise unchanged. These findings motivate
review, not silent regularization or optimizer changes.

See `artifacts/milestone5c/report.md`, `diagnostics.json`, `cache_events.json`,
`saved_validation.json`, per-case results/NPZs, stdout/XML, and preservation
records for actual findings and limitations. Work stops at Stage 5C; major
refactoring, objective/noise changes, regularization and later benchmarks require
separate approval.
