# Stage 4A hardening report

Numerical benchmark validation: **PASSED**. This verdict uses candidate status, independent NumPy accumulation, constant reconstruction, and the recorded gradient vectors.

## Files changed

- `estimation/fixed_window_objective.py`: guarded orchestration using the preserved RK4 and Joseph update; fixed requested-set domain, node/posterior checks, observation masks/counts, checked Python interfaces.
- `simulation/continuous_unicycle.py`: stable midpoint/sinc truth and stricter time-grid checks.
- `benchmarks/milestone4a.py`: validity-aware statistics and evidence-based reporting.
- `tests/test_fixed_window_objective.py`, `tests/test_stage4a_hardening.py`, `tests/test_milestone4a_reporting.py`: domain, independent analytic, decomposition, simulation, reporting, and preservation checks.
- `estimation/STAGE4A.md`, `project.md`: domain requirements and hardening record.
- `artifacts/milestone4a_hardening/report.md`, `results.json`, `validation.md`, `validation.json`: configuration, numerical results, current test record, independent checks and limitations.

## Objective and valid domain

`J(c) = sum_{k in K} e_k.T @ solve(S_k, e_k)` uses the requested set K fixed before evaluation. Every requested observation must be accepted for J to be defined. No regularization, log determinant, finite invalid-candidate penalty, covariance projection or jitter is added. Log determinant and J + log determinant remain diagnostics only.

The JAX evaluator returns explicit status and NaN costs outside the domain. Checked Python evaluation/value-gradient functions raise `InvalidCandidateError` before an invalid trial can be used. RK4 propagation and the original Joseph updates remain unmodified. Numerical failures quarantine subsequent arithmetic; internal placeholders from invalid results are not estimates.

Every integration-node prior and sequential posterior is checked for finite pose/covariance, absolute covariance asymmetry <= 1e-10 and minimum eigenvalue >= -1e-10. Innovation covariances must be symmetric and strictly positive definite. The absolute tolerance matches the validated Stage 3B.2 numerical policy, admits float64 roundoff, and does not repair a matrix. Eigenvalue scaling is diagnostic only.

## Reproducible configuration

```json
{
  "window_length_seconds": 5.0,
  "integration_step_seconds": 0.025,
  "measurement_interval_seconds": 0.1,
  "degree": 3,
  "segments": 4,
  "coefficients_per_channel": 7,
  "command_v_omega": [
    1.0,
    0.2
  ],
  "true_disturbance_v_omega": [
    0.15,
    -0.05
  ],
  "seed": 7,
  "initial_pose": [
    0.0,
    0.0,
    0.0
  ],
  "arrival_covariance": [
    [
      0.0025000000000000005,
      0.0,
      0.0
    ],
    [
      0.0,
      0.0025000000000000005,
      0.0
    ],
    [
      0.0,
      0.0,
      0.0012184696791468343
    ]
  ],
  "measurement_covariance": [
    [
      0.0025000000000000005,
      0.0
    ],
    [
      0.0,
      0.00030461741978670857
    ]
  ],
  "continuous_input_noise_spectral_density_Qc": [
    [
      0.01,
      0.0
    ],
    [
      0.0,
      0.0025
    ]
  ],
  "landmark_coordinates": [
    [
      5.0,
      2.0
    ],
    [
      7.0,
      -4.0
    ],
    [
      1.0,
      7.0
    ],
    [
      -4.0,
      4.0
    ],
    [
      -5.0,
      -3.0
    ],
    [
      3.0,
      -7.0
    ]
  ],
  "covariance_tolerance": 1e-10,
  "observation_policy": "all six landmarks at positive 0.1 s multiples; no arrival-time observation",
  "jax_enable_x64": true
}
```

The truth is a deterministic analytic mean trajectory. Qc is the estimator’s configured uncertainty assumption; it does not inject stochastic noise into the truth. Measurement noise is seeded independently.

## Candidate results

| Candidate | Status | Requested / valid / masked / rejected | J | Log determinant | Gaussian diagnostic |
|---|---|---|---:|---:|---:|
| zero | valid | 300 / 300 / 906 / 0 | 488.62702638 | -4052.24324748 | -3563.6162211 |
| true | valid | 300 / 300 / 906 / 0 | 470.362078442 | -4051.88470222 | -3581.52262377 |

Invalid and masked slots never enter innovation statistics. Statistics are suppressed entirely for invalid candidates. Final states/covariances, wrapped errors and valid-only innovation statistics are stored in `results.json`.

Constant reconstruction maximum error: `5.551115e-17`.

## Gradient checks

| Vector | FD step | Max absolute error | Max relative error* | Status |
|---|---:|---:|---:|---|
| zero | 0.0001 | 2.216147e-08 | 7.039788e-09 | passed |
| zero | 1e-05 | 2.130552e-07 | 5.446058e-08 | passed |
| true | 0.0001 | 2.463781e-08 | 8.173456e-07 | passed |
| true | 1e-05 | 1.061621e-07 | 4.014497e-06 | passed |
| nonconstant | 0.0001 | 2.651376e-08 | 6.131017e-09 | passed |
| nonconstant | 1e-05 | 3.034860e-07 | 5.934621e-08 | passed |

*Relative denominator is max(|FD|, 1e-5). Checks cover every coefficient in both channels with float64, steps 1e-4/1e-5, absolute/relative tolerances 2e-5 and maximum absolute discrepancy < 2e-5. All base and perturbed candidates must be valid.

Actual validation checks: `{'candidates': True, 'numpy_consistency': True, 'constant_reconstruction': True, 'gradients': True}`.

## Preservation of the original experiment

Differences from the original saved Stage 4A results (new minus original):

```json
{
  "zero": {
    "innovation_quadratic": 2.6716406864579767e-12,
    "logdet_diagnostic": 0.0,
    "gaussian_diagnostic": 2.7284841053187847e-12,
    "final_position_error_norm": 1.7850304567801345e-15
  },
  "true": {
    "innovation_quadratic": 1.4779288903810084e-12,
    "logdet_diagnostic": 0.0,
    "gaussian_diagnostic": 1.3642420526593924e-12,
    "final_position_error_norm": 7.537373503119227e-16
  }
}
```

## Test execution provenance

This benchmark command does not execute pytest. A benchmark numerical verdict is not evidence that the test suite ran.

Original Stage 4A run: 7 focused tests and 205 total tests passed; see artifacts/milestone4a/report.md.

Current run test record supplied separately:

```json
{
  "command": "PYTHONPATH=build/bsplines_library python3 -B -m pytest -q --junitxml=/tmp/stage4a_hardening_pytest.xml -o junit_family=legacy",
  "exit_code": 0,
  "passed": 233,
  "elapsed_seconds": 119.5753432920028,
  "recorded_at_utc": "2026-11-12T23:08:30.645111+00:00",
  "scope": "complete repository suite",
  "stdout": "........................................................................ [ 30%]\n........................................................................ [ 61%]\n........................................................................ [ 92%]\n.................                                                        [100%]\n------------ generated xml file: /tmp/stage4a_hardening_pytest.xml -------------\n233 passed in 115.73s (0:01:55)\n"
}
```

## Remaining limitations

The checked interfaces are the Stage 4B boundary. A future optimizer must reject invalid trials or stay in the valid domain, without changing K or inventing a finite penalty. Angular wrap and geometry boundaries remain nonsmooth. Invalid candidates have no defined objective gradient; checked evaluation rejects them before differentiation. Checks cover RK4 integration nodes and sequential posteriors, not every internal RK4 stage or continuous time between nodes. Step refinement remains necessary for accuracy.

No optimization, moving window, association search, or SLAM augmentation is implemented here. Stage 4B remains planned. See `validation.md` for the hardening test outcomes and review-issue resolution.
