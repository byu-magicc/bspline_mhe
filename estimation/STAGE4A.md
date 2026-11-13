# Stage 4A fixed-window interface

The fixed requested set `K` is the true entries of the supplied observation
mask, fixed with timestamps, associations, landmark coordinates, input command,
arrival mean/covariance, and C++ spline basis arrays before candidate evaluation.
`c = [c_v; c_omega]` has `2*(M+d)` coefficients. Prepare the basis at
`rk4_stage_times(times) - times[0]`, with `L = times[-1] - times[0]`; changing
this grid requires recomputing the basis through `UniformBasis`.

`make_fixed_window_objective(...)` captures immutable array copies and returns
a JAX-compatible evaluator. It orchestrates the original public `rk4_step` and
`measurement_update` functions without modifying their equations. Observations
at the arrival time update the supplied prior directly; that prior must not
already include those observations.

For candidates in the valid domain:

```
J(c) = sum_{k in K} e_k(c).T @ solve(S_k(c), e_k(c))
```

There is no coefficient penalty or log-determinant term in J. Log determinant
and J plus log determinant are diagnostics. All three use exactly K. Bearing
innovations wrap through the original measurement model. Valid computations
retain differentiation through states, covariances, innovations, gains and
Joseph updates; no `stop_gradient` is applied to these quantities.

The valid domain requires finite coefficients, finite node states/covariances,
symmetric PSD prior/posterior covariances, finite costs, and valid geometry with
SPD innovation covariance for every requested record. All integration-node
priors and all sequential posteriors are checked, including nodes beyond the
last measurement. The absolute symmetry/PSD tolerance is 1e-10, matching the
validated Stage 3B.2 policy; it is a diagnostic allowance for roundoff, not a
matrix modification. Strict positive definiteness is required for S. No
projection, symmetrization of the EKF covariance, clipping or jitter is applied.

An invalid candidate returns `candidate_valid=False`, explicit counts and an
actual accepted-record mask. Its costs are NaN (undefined), rather than a
partial sum over fewer observations. Numerical failure quarantines subsequent
arithmetic with internal storage placeholders; histories/final states from
invalid results must not be used as estimates. Invalid and masked records never
enter statistics; benchmark success statistics are suppressed for invalid
candidates.

Use `checked_evaluation(evaluator, c)` at the Python boundary. It raises
`InvalidCandidateError` with geometry/numerical status and observation counts.
`checked_value_and_grad(evaluator, c)` rejects invalid candidates before
requesting a gradient, then also checks the gradient for finite values. The
JAX scalar `innovation_quadratic(evaluator, c)` supports `jax.grad` and `jax.jit`
within the valid domain; it returns NaN outside it.

A future SciPy integration must use the checked boundary and explicitly reject
out-of-domain trial steps, or restrict its search to a validated domain. It
must keep K fixed and must not replace failures with arbitrary finite penalties.
No objective gradient is defined for an invalid candidate. Angular wrap and
geometry boundaries are nonsmooth. Valid gradients must be checked at candidate
and finite-difference perturbation points inside the domain. No coefficient
optimizer or trial-step rejection algorithm is implemented by Stage 4A.

Checks cover integration nodes and sequential posteriors, not the entire
continuous trajectory or every internal RK4 stage. Step refinement is still
needed to establish integration accuracy. An empty K has the mathematical cost
zero but supplies no disturbance-identification information.
