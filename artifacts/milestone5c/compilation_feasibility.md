# Compilation reuse feasibility, assessed before extended simulation

The preserved Stage 5B code captured fresh arrival, time, basis, measurement,
ID/mask and fixed birth-plan arrays in each objective closure. Those arrays
have fixed shapes in the reference configuration, while their values change.

The isolated `benchmarks/stage5c_cache.py` adapter traces the **unchanged**
objective using `jax.make_jaxpr`, passes its captured arrays as runtime arguments,
and interprets the same operation graph using the installed JAX 0.4.35 API.
It caches compiled evaluation and value/gradient functions by exact printed
graph plus result-tree fingerprint. A new structure or shape receives a new
entry. It does not reshape, reorder, remove or invent data or landmark slots.
The code is pinned to that JAX version and requires revalidation after an upgrade.

Before extension, three saved Stage 5B windows were tested: initial empty-map
regression constant, its next shifted window, and finite-sensor constant ending
in an empty scan. One compiled program served all three. At zero and a
nonconstant coefficient vector, **every returned value and coefficient gradient
was bitwise equal** to the reference path. A complete first-window solve also
reproduced the saved final cost, all coefficients and 133 iterations exactly.
See feasibility.json and feasibility_stdout.txt.

The first evaluation/value-gradient compile took about 0.95/2.15 s, with
0.24/2.05 s lowering. Subsequent windows needed no such compilation. Tracing
each new objective took about 0.08–0.14 s. Steady objective and gradient execution
remained similar to the original path; this is compilation reuse, not a cheaper
estimator or objective. The original checked AD reference, all preflight probes,
every per-trial validity check, bounds, options, callbacks and failure policy
remain. The optimizer orchestration is a static copy with only bridge
construction injected; an AST equality test enforces that limited difference.
All original estimator and optimizer source files remain untouched.

This is a benchmark-only adapter, not a new production EKF architecture.
It requires no changes to RK4, EKF updates, the innovation cost, differentiation,
arrivals, masks, associations, births or warm-start projection. It is therefore
within the request's permission for isolated, numerically equivalent caching.
A broader runtime-parameter API/refactor is deferred and would require approval.

Expected benefit is removal of repeated lower/compile and auxiliary-AD startup
costs after the first graph, while objective/gradient and SciPy work remain.
Based on Stage 5B's 18–29 s update costs and the measured cache behavior, choose
**6.0 s** for the initial extended deterministic suite: **11 complete consecutive
windows per scenario** at 5.0–6.0 s, every 0.1 s, versus three before. Six scenarios
and two starts give 132 fits. The projected budget is tens of minutes rather
than a multi-hour suite. This duration is chosen for computational practicality,
before inspecting extended error profiles. No accuracy tuning is performed.

Eight initial diagnostic/cache tests passed before the complete regression
suite. Extended simulations wait for that suite to pass. All Stage 4C/5A/5B
artifacts remain read-only and are protected by a pre-task hash snapshot.
