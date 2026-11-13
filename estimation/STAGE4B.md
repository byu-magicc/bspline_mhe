# Stage 4B coefficient optimization

`estimation/coefficient_optimization.py` adds SciPy L-BFGS-B around the unchanged
Stage 4A evaluator. Enable JAX float64 and construct the evaluator with the fixed
arrival prior, commands, known landmarks, requested observations and RK4 basis
constants. The only minimized scalar is `result.innovation_quadratic`.

`optimize_coefficients(evaluator, initial, constant=False, ...)` fits all
`2*(M+d)` coefficients. With `constant=True`, it fits `[a,b]` and differentiates
through `constant_to_spline(z,n) = [a*ones(n); b*ones(n)]`. The default initial
parameters for the benchmark are zeros, with exploratory bounds [-0.5,0.5] per
parameter. Bounds are constraints, not an added objective penalty.

Before SciPy runs, the wrapper checks the initial point, compares the compiled
value/gradient with `checked_value_and_grad`, and checks deterministic feasibility
probes: the box center, axis endpoints, four sign patterns and eight seed-7
interior samples. Duplicate probes are removed. Passing these samples does not
certify the entire box.

Every SciPy objective request first calls the checked Stage 4A evaluation at the
Python boundary. The scalar and gradient come from the compiled composed JAX
objective, including covariance dependence. SciPy receives a `(value,gradient)`
pair with `jac=True`. Invalid candidate status or a nonfinite gradient raises
`InvalidCandidateError`; no partial objective, NaN gradient or finite penalty is
returned to SciPy. The wrapper catches the exception, records parameters,
diagnostic status and phase, and stops with `success=False`. An invalid trial
requires an explicit revision of the region or strategy. There is no automatic
retry with changed bounds, measurement set or objective. After a failed trial,
the returned point is the last accepted iterate, labelled as an aborted run.
Nonfinite rejected parameters are recorded as JSON null; no substitute value
or gradient is returned to SciPy.

Default stopping options are ftol=1e-12, gtol=1e-6, maxiter=500, maxfun=2000,
maxls=40 and maxcor=10. A successful SciPy status can reflect relative objective
reduction before the projected-gradient target is achieved. Both raw gradient
norms and the projected infinity norm are therefore reported with the actual
termination message. Convergence is not a proof of a global optimum or correct
disturbance reconstruction.

Explicit evaluation and value/gradient `lower(...).compile()` times exclude
execution. The Stage 4A checked-reference first call can incur auxiliary VJP
compilation and is labelled separately. Preparation includes these calls,
warmup and probes; the subsequent optimization runtime excludes preparation.
Per-request host validity checks and compiled value/gradient execution times
are also recorded with mean/p95. Call counts distinguish requested functions
from gradients actually returned when a trial is rejected.

The benchmark replays the Stage 4A seed-7 constant-disturbance experiment in two
parameterizations, repeats each twice, and saves the exact truth/observations.
Additional noiseless observations on the same truth/geometry and a tighter-ftol
full fit diagnose noise fitting and stopping tolerance. These are separate
controls, not replacements for the paired primary experiments.

Reconstruction RMSE and maximum absolute error are sampled at 501 uniformly
spaced elapsed times including endpoints, using the C++ spline adapter. They
measure the disturbance function rather than coefficient differences. The
reported maximum is a dense-grid maximum, not an analytic continuous supremum.

`innovation_sensitivity` returns a stacked prior-innovation Jacobian with respect
to the full coefficient vector. For 300 two-component updates it is 600x14.
Each block is whitened with `solve(cholesky(S_k), D_k)` at that evaluation point.
This diagnostic freezes S for whitening; optimization still differentiates S.
Numerical rank uses `max(rows,cols)*eps64*sigma_max`; a separate practical rank
uses `1e-3*sigma_max`. Neither rank is a proof of global identifiability, and
inverse singular values are not calibrated confidence intervals for this
experiment. Noise model calibration, conditioning and parameterization matter.

Run the benchmark from the repository root:

```sh
PYTHONPATH=build/bsplines_library python3 -B -m benchmarks.milestone4b
```

`--test-record path.json` embeds a separately executed current test record.
The benchmark never executes pytest or invents test counts. Results are in
`artifacts/milestone4b/`; completion is assessed from convergence, validity,
reconstruction, repeatability, sensitivities, noiseless controls and actual
current test evidence. This validation concerns controlled known-landmark
constant disturbances. Smooth disturbances, other seeds, Monte Carlo,
fixed-window SLAM and moving-horizon estimation require later validation.
