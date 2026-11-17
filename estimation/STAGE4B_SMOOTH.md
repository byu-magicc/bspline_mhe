# Stage 4B smooth-disturbance extension

This extends the controlled known-landmark Stage 4B experiment. The Stage 4A
objective, checked Stage 4B optimization wrapper, original constant results,
SLAM baselines and sibling spline library are preserved.

`simulation/smooth_unicycle.py` integrates analytic disturbances
`[0.15*sin(2*pi*sigma/5), 0.05*cos(2*pi*sigma/5)]` with SciPy DOP853, independently
of JAX, the estimator RK4 and the spline basis. The primary maximum internal
step is 0.05 s with rtol=1e-11 and atol=1e-13. A separate 0.025 s integration
with tenfold tighter tolerances must agree within 1e-10 per pose component.
Tests also compare position against independent adaptive quadrature using the
analytical integrated heading. There is no stochastic process-noise injection
into truth; Qc remains the original estimator assumption.

`benchmarks/milestone4b_smooth.py` copies all shared configuration from the
saved constant Stage 4B experiment. It constructs one immutable evaluator for
the new, saved seed-7 dataset. Comparisons are a fixed zero reference, a
two-parameter constant fit, and a full 14-coefficient fit. Both fits start at
zero and use the original bounds/options/checked invalid-candidate policy.
Two repeats check deterministic reproducibility, not statistical precision.
An invalid trial stops the benchmark, saves failure.json and writes a failure
report; it does not trigger a penalty, changed bounds or retry.

All disturbance metrics use the same 501 uniform samples including both
endpoints. The independent least-squares diagnostic uses this C++ basis grid
and minimizes sampled squared error directly. Its coefficients are neither an
estimator nor an initialization. Pose metrics use 201 posterior integration
nodes, including arrival. Position RMSE is sqrt(mean(dx²+dy²)); heading errors
wrap to [-pi,pi]. Maximum disturbance errors are sampled maxima.

Raw and covariance-whitened innovation sensitivities follow Stage 4B's rank
rules. Coefficient-error projections use the independent LS coefficients as
their reference because there is no exact true spline coefficient vector.
The weakest four directions are reconstructed as functions and inspected at
the first/last 0.5 s and in the interior. Local full rank does not prove global
identifiability. A separate noiseless full-spline diagnostic helps assess
measurement-noise fitting and representation error.

The covariance audit retains identical requested/accepted masks, valid counts,
S diagonals and trace ratios, unweighted residual sums, and each residual
quadratic with the zero candidate's S held fixed. The actual optimizer still
differentiates S and minimizes only J. Log determinant and J+logdet remain
diagnostics.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4b_smooth --test-record artifacts/milestone4b_smooth/test_record.json
```

The optional test-record JSON must describe an actual separately executed
suite; the benchmark never runs pytest or invents test evidence. Results,
configuration, dataset, histories, dense CSV/NPZ reconstructions, localization,
sensitivity arrays, vector PDF/300-dpi PNG figures and the report are in
`artifacts/milestone4b_smooth/`. A controlled validation verdict includes
successful checked convergence, gradient checks, integration accuracy,
reproducibility, reconstruction better than zero, and residual reduction with
reference covariances. It never requires full-spline RMSE to beat the constant
fit. Performance conclusions must follow the measured metrics.

This task does not implement Stage 4C. Stage 4C still needs its documented
arrival/landmark-initialization measurement accounting decision. Moving
windows, reassociation, nearest-neighbor association, Victoria Park and final
Monte Carlo comparisons remain later work.
