# Stage 4B smooth extension validation

**Validated for the controlled seed-7, five-second, known-landmark experiment.**
All numerical checks and the complete repository suite pass. Stage 4B is ready
to transition to Stage 4C within this scope. Stage 4C has not been implemented;
its arrival/landmark-initialization measurement accounting remains a required
research decision before implementation.

## Executed checks

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone4b_smooth/pytest.xml -o junit_family=legacy
```

Outcome: **270 passed**, no failures, errors or skips, exit code 0;
pytest reported 137.01 s. The measured subprocess wall time was
140.820 s. This includes all 248 existing tests and
22 extension tests. Nothing was skipped or left unrun. The initial focused
run passed 21 tests; one validation-gate test was added before the complete run.
The final complete run has no warnings.

New checks cover analytic sinusoidal inputs, estimator/basis independence,
independent quadrature and analytic-heading references, integration refinement,
seeded-noise reproducibility, shared configuration/data/arrival conditions,
C++ endpoint/time scaling and coefficient order, sampled least-squares
optimality, JAX/central-difference objective gradients in both channels,
localization/disturbance metric definitions, invalid-trial aborts and strict
JSON diagnostics, and preservation/re-evaluation of the constant experiment.
No test requires full-spline reconstruction to beat the constant constraint.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4b_smooth --test-record artifacts/milestone4b_smooth/test_record.json
```

Outcome: exit code 0. Both primary models converged twice from zero with
identical coefficients, objective values and accepted objective histories.
The separately labelled noiseless full-spline control also converged.
All requested records remain valid for every actual trial; no penalties,
retries, changed bounds, measurement dropping or covariance repairs occurred.

The unchanged default ftol=1e-12 and gtol=1e-6 apply. Full-spline projected
gradient infinity norm is 0.000164046337, above gtol;
its termination is function reduction, not verified strict stationarity.
The constant fit projected norm is 1.14405724e-09.

The first linear spline coefficient (zero-based index 0) is exactly at the
lower bound -0.5; all other coefficients are interior. This active constraint
explains part of the difference between raw and projected gradient norms.
Boundary recovery should therefore be interpreted with both the box constraint
and weak/noisy directions in mind. No unconstrained-optimum claim is made and
the prescribed bounds were not changed.

## Quantitative assessment

Full-spline linear/angular disturbance RMSE is
[0.008806580295679722, 0.009198141181432017], versus
[0.10601204620803681, 0.03543762161742007] for the constant constraint:
reductions of 91.6928% and 74.0441%.
The further innovation-objective reduction is
7.92649392.
Position RMSE improves modestly and heading RMSE worsens slightly; the
experiment does not establish improvement in every metric.

The independent sampled LS errors are
[0.0012090725254914737, 0.00026732627550645773].
The noiseless full-spline errors are
[0.0012203199495266098, 0.00028031782640704596],
close to that representation limit. Together with the boundary and weak-mode
analysis, this supports noise and weak sensitivity as substantial contributors
to the larger noisy-fit reconstruction errors. The inputs are analytic
sinusoids outside the finite cubic spline family, not spline-generated truth.

Raw and whitened innovation Jacobians are 600 by 14 with numerical and practical
rank 14 at both reported points. Conditioning and weak right singular vectors
are compared with the original constant experiment in report.md/results.json.
Full local numerical rank is not proof of global identifiability. Projection
errors reference the independently fitted spline, since exact true coefficients
do not exist for this scenario.

Covariance audit and fixed-reference-S residual costs show that lower J does
not result from dropping observations or broad covariance inflation. The mean
full-fit S diagonal ratios to zero are
[1.0001836195964382, 0.9989800805571358].
With S held fixed at the zero reference, the full-fit residual cost is
465.463209189,
versus 473.363950495 at zero.

Float64 gradient differences at zero and the fitted full vector, for all 14
coefficients and steps 1e-4/1e-5, are below 1.1e-7 (acceptance maximum 2e-5).
Every perturbation is checked for validity. Truth integration refinement
agrees below 1e-10 per pose component; independent quadrature is tested too.

## Preservation and saved evidence

All **135 pre-existing protected files** match the pre-implementation
SHA-256 snapshot. This includes all existing artifacts, tests, baselines,
models, estimation code, simulation and spline adapter, plus tracked sibling
library files. The sibling remains on uniform_development and its worktree
and the original baseline worktree are clean. Existing user .DS_Store changes
were left untouched. project.md receives only the controlled extension status
record; the new implementation/docs/tests use separate files.

[preservation.json](preservation.json) stores every protected digest;
[validation.json](validation.json) stores current implementation digests,
commands and verdict. [test_record.json](test_record.json),
[pytest.xml](pytest.xml) and [pytest_stdout.txt](pytest_stdout.txt) retain actual
suite execution evidence. The final benchmark embeds that same test record.
The benchmark itself does not execute pytest. Saved dataset fingerprints and
reconstruction/localization RMSEs were independently checked after serialization.

Results are confined to artifacts/milestone4b_smooth/. Original constant-stage
results have not been regenerated or overwritten. New plots were visually
inspected and are available as 300-dpi PNG and vector PDF.

## Measured validation gate

```json
{
  "independent_truth_refinement": true,
  "all_primary_candidates_valid": true,
  "fits_converged_without_invalid_trials": true,
  "same_requested_observations": true,
  "objective_numpy_consistency": true,
  "gradients": true,
  "sampled_ls_normal_equations": true,
  "full_spline_reconstructs_both_channels_better_than_zero": true,
  "full_spline_reduces_residual_cost_with_reference_covariances": true,
  "paired_repeat_reproducibility": true,
  "current_complete_test_suite": true
}
```

This is not Monte Carlo validation. General reliability across seeds, abrupt
inputs, sparse observations and increased noise is unvalidated. Final
comparisons still require paired Monte Carlo trials. Fixed-window SLAM, moving
windows, nearest-neighbor association and Victoria Park remain later work.
