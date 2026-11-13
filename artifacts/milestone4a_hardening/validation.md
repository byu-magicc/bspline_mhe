# Stage 4A hardening validation record

All six independent-review findings have been addressed. This record covers the hardening test execution and numerical results, alongside the reproducible benchmark report in `report.md`. Stage 4B remains planned.

## Review findings and resolution

| Reviewed issue | Resolution and validation |
|---|---|
| Scalar drops invalid observations | The requested set K is fixed. Invalid candidates return undefined (NaN) objective/diagnostics. Checked Python evaluation and value/gradient interfaces raise `InvalidCandidateError`; mixed geometry failures after propagation are tested under JIT and batched AD. |
| Incomplete numerical validity | Finite states/covariances, covariance symmetry/PSD, and SPD innovation covariance are checked at node priors and every sequential posterior. Tests cover NaN, asymmetry, indefiniteness, singular S, an injected failing intermediate posterior, and overflow beyond the final observation. Invalid solves are gated; numerical failure quarantines later arithmetic without repairing a covariance. |
| Self-consistent tests miss wrong scaling | Independent analytic linear/angular tests use a nonzero window origin and two integration grids. Each test explicitly demonstrates that halved basis times disagree with the expected trajectory and objective. Candidate-order invariance uses a nonzero mean and correlated covariance, with parity against the preserved EKF. |
| Invalid placeholders contaminate statistics | The evaluator exposes the actual accepted mask and requested/valid/masked/rejected counts. Benchmark statistics use only accepted records and are suppressed for invalid candidates. Tests combine valid, invalid and masked observations at one timestamp. |
| Reports assert success unconditionally | Candidate status, independent NumPy calculations, basis reconstruction and gradient vectors determine the numerical verdict. Failure-path report tests cover invalid candidates, excessive/nonfinite gradient errors, invalid FD perturbations and failed consistency checks. Current test records are separate from historical counts. |
| Near-zero-turn truth cancellation | Midpoint/sinc truth is tested at zero and positive/negative rates 1e-16, 1e-14, 1e-10 against an independent Taylor expansion. The old nonzero-turn experiment is preserved within float64 accuracy. |

## Independent analytic cases

For elapsed sigma = t - 11 on a 2.5 s window, cubic degree 3 / M=4 coefficients `c_j = a*(L/M)*(j-1)`, a=0.13, reproduce w=a*sigma through the existing C++ adapter. The linear test keeps heading fixed and uses displacement v*sigma + a*sigma²/2. The angular test keeps position fixed and uses heading theta0 + omega*sigma + a*sigma²/2. Expected measurements use independent NumPy geometry; wrapped innovations and quadratic terms use independent trigonometric wrapping and NumPy solves. No evaluator propagation or measurement calculation supplies these references.

P0=0 and Qc=0 in these analytic isolation tests, so the nonzero residuals do not perturb the exact analytic mean; S is the independently specified measurement covariance. Nonzero correlated P0 is covered separately by candidate-order/parity tests and covariance gradient decomposition. The grids are uniform 0.25 s and a refined grid with extra nonuniform nodes.

| Case | Max state error | Max innovation error | Objective error | Objective error with wrong half-scaled basis times |
|---|---:|---:|---:|---:|
| test_nonconstant_analytic_trajectory_innovations_and_cost[0-linear] | 4.441e-16 | 2.238e-16 | 2.776e-16 | 0.89204 |
| test_nonconstant_analytic_trajectory_innovations_and_cost[0-angular] | 2.220e-16 | 3.192e-16 | 3.386e-15 | 10.4849 |
| test_nonconstant_analytic_trajectory_innovations_and_cost[1-linear] | 8.882e-16 | 8.882e-16 | 2.220e-16 | 0.89204 |
| test_nonconstant_analytic_trajectory_innovations_and_cost[1-angular] | 4.441e-16 | 2.220e-16 | 2.609e-15 | 10.4849 |

Assertions use absolute state tolerance 3e-14, innovation tolerance 2e-14 and objective tolerance 2e-13. The deliberately incorrect scaling must cause state error >0.05 and objective error >0.01. These checks now fail for the time-scaling mutation that passed the original seven tests.

## Gradient decomposition

The two nonconstant 14-vectors are `linspace(-0.04,0.08,14)` and `default_rng(9).normal(scale=0.06,size=14)`. JAX Jacobians of e and S are contracted separately using NumPy to form `2*(de)^T*solve(S,e)` and `-solve(S,e)^T*(dS)*solve(S,e)`. Their sum is compared against the independently differentiated scalar objective for every coefficient. Both terms must have nonzero norms in both coefficient blocks; omitting dS must cause a discrepancy >1e-4. No numerical quantity uses stop_gradient.

| Vector | Max decomposition discrepancy | Linear covariance-gradient norm | Angular covariance-gradient norm |
|---|---:|---:|---:|
| 0 | 1.421e-14 | 0.391937 | 0.0219827 |
| 1 | 3.580e-15 | 0.0621057 | 0.039222 |

Decomposition tolerances are atol=rtol=3e-12 in float64. Full-objective central finite differences cover zero, true-constant and nonconstant vectors, every coefficient, and h=1e-4 and 1e-5. Base and perturbed candidates are all checked for validity.

The fresh benchmark maximum absolute FD discrepancy is 3.034860e-07, and maximum relative discrepancy is 4.014497e-06 with denominator max(|FD|,1e-5). The FD assertions use atol=rtol=2e-5 plus maximum absolute error <2e-5. The tighter analytic and decomposition assertions test different properties without finite-difference subtraction error.

## Original experiment preservation

The 5 s seed-7 experiment remains valid for both candidates: 300 requested / 300 accepted / 906 masked integration slots / 0 rejected. J_zero=488.627026380014; J_true=470.362078441903. Maximum change in J from the original saved results is 2.672e-12. Current covariance, likelihood diagnostics and final errors are in results.json; state/covariance preservation differences are recorded in validation.json.

Every benchmark candidate satisfies finite-state, covariance symmetry/PSD and innovation SPD checks. Covariance tolerance is absolute 1e-10, matching the validated SLAM numerical policy and permitting roundoff without modifying matrices. Exact symmetry/PSD are not enforced by projection or jitter.

## Actual current test execution

```sh
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q --junitxml=/tmp/stage4a_hardening_pytest.xml -o junit_family=legacy
```

Outcome: **233 passed**, exit code 0. Pytest reported 115.73 s. The 35 focused Stage 4A/hardening/report tests also passed before the complete run. Original recorded counts of 7 focused / 205 total are historical; they are not represented as tests executed by the benchmark command.

```sh
PYTHONPATH=build/bsplines_library python3 -B -m benchmarks.milestone4a --output artifacts/milestone4a_hardening --test-record /tmp/stage4a_hardening_test_record.json
```

Outcome: completed, with all four numerical benchmark validation checks true. The current test record is embedded in results.json and validation.json; regenerating a benchmark without --test-record leaves the current test record null. The benchmark never claims to execute pytest.

## Preservation and remaining limitations

SHA-256 comparisons matched all 31 protected tracked files captured before hardening. Preserved files include the validated known-landmark EKF, RK4, SLAM/models/spline adapter and prior tracked milestone artifacts. Baseline A's submodule and the sibling C++ spline repository are clean. The original Stage 4A artifacts were read as reference and not overwritten.

The checked interface is a suitable foundation for Stage 4B on its documented valid domain. A future SciPy integration must explicitly reject out-of-domain trial steps or constrain the search; this pass implements no optimizer, line search, bounds, regularization, or rejection algorithm. Invalid-domain gradients are undefined and must not be consumed.

Checks cover integration-node priors and sequential posteriors, not every internal RK4 stage or continuous-time covariance. Time-grid refinement remains necessary for integration accuracy. Absolute covariance tolerances are documented experiment-scale assumptions and should be reconsidered if state units/scales change. The analytic isolation tests use zero covariance; stochastic consistency/calibration, identifiability, and disturbance reconstruction under optimization remain future validation gates.

## Original-result comparison details

```json
{
  "zero": {
    "innovation_quadratic": 2.6716406864579767e-12,
    "logdet_diagnostic": 0.0,
    "gaussian_diagnostic": 2.7284841053187847e-12,
    "final_pose": 8.881784197001252e-16,
    "final_covariance": 2.1006417091906648e-19,
    "final_pose_error_x_y_heading": 1.7763568394002505e-15
  },
  "true": {
    "innovation_quadratic": 1.4779288903810084e-12,
    "logdet_diagnostic": 0.0,
    "gaussian_diagnostic": 1.3642420526593924e-12,
    "final_pose": 4.440892098500626e-16,
    "final_covariance": 7.047314121155779e-19,
    "final_pose_error_x_y_heading": 8.881784197001252e-16
  }
}
```
