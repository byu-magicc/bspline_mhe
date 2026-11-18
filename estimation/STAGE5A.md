# Stage 5A: integral projection for spline warm starts

Stage 5A is a standalone, host-side initialization operation. The user requires
an explicit approval gate before Stage 5B. It implements no moving-window
estimator, arrival distribution, measurement scheduler or optimizer integration.
The existing Stage 4C EKF, objective, optimizer and saved results are unchanged.

## Definition and coefficient convention

Let n=M+d, h=L/M and b(sigma)=b_M^d(M sigma/L). The existing flat ordering is
`c=[c_v;c_omega]`; `C=c.reshape(2,n)` and `w(sigma)=C @ b(sigma)`.
For 0 <= delta <= L, the overlap objective is

```text
min_C integral_0^(L-delta) || C b(sigma) - C_old b(sigma+delta) ||^2 d sigma.
G = integral b(sigma) b(sigma).T d sigma
B = integral b(sigma) b(sigma+delta).T d sigma
C_new G = C_old B.T
```

Every basis value comes from the existing C++ `UniformBSpline` through
`UniformBasis.evaluate`. No recurrence or alternative spline basis is added.
The projection API is in `splines/warm_start.py`:

```python
from splines.warm_start import make_warm_start_projection

p = make_warm_start_projection(degree=3, M=4, L=5.0, delta=0.1)
c_initial = p.apply(c_previous)  # flat (14,) or matrix (2,7), same output shape
G, B = p.gram, p.cross
diagnostics = dict(p.diagnostics)
```

Inputs are finite float64 coefficients. The projection does not mutate them
or modify the MHE innovation objective. Constants are separated before applying
the coefficient map, using partition of unity to preserve constant channels
without ill-conditioned amplification of floating-point row-sum errors.

## Quadrature and solution

Integrate on normalized s in [0,M-delta*M/L] with physical weight L/M.
Split at both new knots j and shifted old knots j-delta*M/L; endpoint-only
knots do not create extra intervals. Gauss–Legendre order d+1 integrates products
of degree 2d exactly on these intervals, up to floating-point evaluation error.
An explicit alternate order supports quadrature convergence studies.

With quadrature weights W, define A=sqrt(W) b and D=sqrt(W) b_shifted.
Then G=A.T A, B=A.T D. A coefficient map T satisfying A T approximately D
also solves G T=B in the least-squares sense. Therefore C_new=C_old T.T
satisfies the requested normal equation. The full-rank implementation uses
reduced QR and a triangular-system solve; it does not invert G or square the
condition number during solution. G and B remain available for independent
normal-equation residual checks.

The cache uses exact `(d,M,L,delta,quadrature_order)` keys, with a maximum of
128 entries. Repeated keys return the same object. Cached matrices have immutable
backing storage and diagnostics are read-only. Changing geometry or quadrature
requires a distinct entry. `projection_cache_info()` and
`clear_projection_cache()` expose deterministic cache inspection/reset.

## Rank, conditioning and continuation policy

Check actual Gram symmetry and eigenvalues; do not symmetrize or repair G.
Squared singular values of A supply a more stable Gram rank/condition diagnostic.
The numerical rank counts lambda_i > n*eps64*lambda_max. A materially negative
Gram eigenvalue or asymmetric Gram raises an error. The full condition number
is `(s_max/s_min)^2`, reported even when the matrix is numerically rank deficient;
it is `None` (infinity) for structural singularity. Diagnostic eigenvalues and
the absolute rank threshold are retained in results.

For delta<h the exact Gram is positive definite, but near h the last basis
has very little support in the overlap. For delta>=h there are at least
floor(delta/h) exactly unobserved terminal basis coefficients. Thus a direct
inverse or unconditional full-rank solve is inappropriate.

The explicit reference continuation is

```text
q = floor(delta/h)
C_ref[:,j] = C_old[:,min(j+q,n-1)].
```

It translates complete knot intervals and repeats the final old coefficient
in new terminal slots. At an exact integer-knot shift this is used directly,
and reconstructs the overlap exactly. For delta=0 use the exact identity.
For delta=L there is no overlap; repeat the old final coefficient in all slots.
The degree-zero basis retains the dependency's half-open right endpoint at L;
pointwise equality at that isolated endpoint is excluded, without changing any
integral or modifying the library.

For a noninteger shift with deficient **numerical** Gram rank, solve the resolved
weighted least-squares correction around C_ref using SVD, and retain C_ref in
unresolved directions. The singular-value cutoff is the square root of the
reported Gram threshold. This defines a reproducible minimum-change continuation
for the resolved problem. It is a numerical approximation to the full integral
minimum when mathematically nonzero directions have been discarded. No diagonal
regularizer, smoothing penalty, or extra MHE objective term is introduced.

Fractional shifts generally have nonzero projection error because shifted
internal knots no longer coincide with new knots. They can still be exact on
an overlap containing just one polynomial span in both bases. The validation
tests distinguish these cases rather than assuming a universal zero or nonzero
projection error.

The continuation is a coefficient policy, not a claim of physically correct
future input. Repeating the last coefficient does not imply that an unclamped
spline immediately becomes a constant disturbance after the overlap. Plot
shading explicitly identifies that unconstrained interval.

## Bounds and scope limits

The integral projection is unconstrained. In the preserved Stage 4C coefficient
checks, a 0.1 s shift produces maximum magnitudes about 0.64–0.67 from prior
coefficients bounded by 0.5. Stage 5A deliberately does not clip these coefficients
or replace the projection with a bounded fit. Stage 5B must explicitly account
for feasible optimizer initialization while preserving the specified bounds.
Projection error measures initialization overlap agreement, not subsequent MHE
convergence, disturbance accuracy, uncertainty or runtime performance.

## Validation and gate

The [numerical report](../artifacts/milestone5a/report.md) and
[machine-readable evidence](../artifacts/milestone5a/results.json) retain
normal-equation errors, independent adaptive integrated errors, Gram ranks and
condition numbers, quadrature convergence, preserved Stage 4C coefficient checks,
the explicit fallback and the initial failed tests and their diagnoses.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q tests/test_warm_start_projection.py tests/test_uniform_basis.py tests/test_slam_window_a.py tests/test_slam_window_b.py tests/test_milestone4c_reporting.py tests/test_slam_visibility.py --junitxml=artifacts/milestone5a/pytest.xml -o junit_family=legacy
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5a
```

Outcome: **188 passed**, no errors/failures/skips/warnings; exit 0. This includes
all 59 new projection tests, all 93 existing basis tests, and all 36 Stage 4C/
visibility tests. Other earlier-estimator tests were not rerun for this isolated
module addition. All 534 captured pre-task files, including their tests and
saved Stage 4C/visibility artifacts, remain unchanged.

Only new files were added: `splines/warm_start.py`,
`tests/test_warm_start_projection.py`, `benchmarks/milestone5a.py`, this document,
and `artifacts/milestone5a/`. No original baseline, existing spline adapter,
sibling dependency, EKF implementation or mathematical objective was changed.

**Stop at Stage 5A. Stage 5B requires explicit user approval.** The next step
after that approval is integration under the user-specified independent forward
EKF arrival policy, with posterior boundary accounting, fixed known associations,
and measurement-time optimization. No such integration is present in this stage.
