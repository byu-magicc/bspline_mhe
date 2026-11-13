# Stage 4B validation and interpretation

**The Stage 4B completion criterion is satisfied for the controlled constant-disturbance known-landmark experiment.** Both parameterizations converge reproducibly, reconstruct both channels more accurately than assuming zero disturbance, and have checked local sensitivity diagnostics. This conclusion is limited to the specified seed-7 noisy experiment and the separate noiseless controls; it does not establish performance across noise realizations or smooth disturbances.

## Requested experiments

The constant-constrained fit estimates a=0.156539508683 m/s and b=-0.0495044936131 rad/s. Signed errors relative to [0.15,-0.05] are [0.006539508683271716, 0.000495506386921335]. It uses the JAX-composed mapping [a*ones(7); b*ones(7)] with a fixed prior and requested set.

| Fit | Objective reduction | Linear RMSE (m/s) | Angular RMSE (rad/s) | Linear max error | Angular max error | Final position error (m) | Wrapped heading error (rad) |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant_constrained | 18.2862365069 | 0.00653950868 | 0.000495506387 | 0.00653950868 | 0.000495506387 | 0.00602702779 | 0.00204229195 |
| full_spline | 18.4051617022 | 0.00994619178 | 0.00769959766 | 0.0190563216 | 0.0287232816 | 0.00614900744 | 0.0020678684 |

RMSE and maximum errors use 501 uniform elapsed-time samples, including endpoints, and the C++ basis adapter. Maximum errors are sampled maxima, not analytic continuous-time suprema. Zero-disturbance RMSE is [0.15,0.05]. Optimized coefficient vectors and all accepted-iterate/trial histories are in results.json. The raw saved dataset, config and SHA-256 fingerprint allow replay without regenerating noise.

## Objective reduction versus noise fitting

The full spline lowers J by a further 0.118925195 compared with the constant fit, while increasing linear RMSE by a factor 1.52094 and angular RMSE by a factor 15.5388. Thus lower innovation cost does not imply more accurate disturbance recovery. The constant parameterization supplies correct structural knowledge for this constant-truth experiment.

The additional noiseless controls retain truth, geometry, commands, measurement times, assumed R/Qc, prior, bounds and zero starts; only sampled observation noise is removed. They are separate diagnostic experiments and do not replace either requested noisy fit.

| Noiseless control | Final J | Linear RMSE | Angular RMSE | Iterations |
|---|---:|---:|---:|---:|
| constant_constrained | 9.233431e-22 | 1.240231e-12 | 1.915413e-13 | 6 |
| full_spline | 7.135060e-12 | 1.169190e-07 | 3.804521e-08 | 210 |

The sharp improvement without observation noise supports a noise-fitting explanation for the extra full-spline variation. It does not prove that noise is the only source of bias. Neither the objective nor Qc was changed, and no regularization or likelihood term was introduced.

## Local identifiability and conditioning

The stacked innovation derivative has 600 rows (300 sequential updates, two components each) and 14 columns. Both raw and Cholesky-whitened matrices have numerical rank 14 at zero and at the fitted coefficients. Rank uses max(rows,columns)*epsilon64*sigma_max. The separately reported 1e-3 relative threshold is only a practical conditioning diagnostic.

| Point | Raw condition number | Whitened condition number | Whitened smallest singular value | Coefficient-error energy in weakest four whitened directions |
|---|---:|---:|---:|---:|
| zero | 77.256557 | 94.870784 | 0.22000222 | 98.6445% |
| optimized | 84.738925 | 94.949093 | 0.21946303 | 98.6718% |

The fitted coefficient error is concentrated in the weakest local directions, consistent with overlapping spline support and weakly constrained coefficient combinations. Full rank excludes an infinitesimal null space at the evaluated points, not large statistical uncertainty or other nonlinear solutions. A coefficient can differ substantially from its constant truth while the reconstructed disturbance remains close because neighboring coefficients compensate.

Whitening uses solve(cholesky(S_k), D_k) at the current candidate. This is a mean-innovation sensitivity diagnostic, not the full Hessian of J; optimization still differentiates the coefficient dependence of S. Inverse singular values stored in JSON are local unit-noise scales, not calibrated confidence intervals. The assumed stochastic Qc is nonzero while simulated truth is deterministic, so statistical likelihood calibration is not established.

## Stopping criteria and tolerance control

Primary options are {'ftol': 1e-12, 'gtol': 1e-06, 'maxiter': 500, 'maxfun': 2000, 'maxls': 40, 'maxcor': 10}. Both fits stopped by SciPy's relative-function-reduction criterion. Their projected gradient infinity norms are 4.731563e-06 and 1.172955e-04, above gtol=1e-6. The report therefore does not claim that the gradient target was achieved. Neither selected fit has an active coefficient bound.

A separate zero-start full-spline fit with ftol=1e-15 ran 244 iterations, gave J=470.221864673395 (change -4.439130e-09), and projected-gradient norm 7.702924e-06. It also stopped on function reduction. Its disturbance RMSE is [0.009945855884482947, 0.007699858049967546], nearly unchanged. The noise-fitting tradeoff is therefore not explained by the default stopping threshold alone. No global optimality or strict stationarity claim is made.

## Runtime and reproducibility

All times below are first-run seconds on the recorded JAX CPU environment with float64. Explicit lower/compile times exclude execution. The checked-reference first call can include auxiliary VJP compilation and is reported separately; preparation includes all of these costs plus warmup/probes. Optimization timing starts after preparation, and includes a host validity check on every trial.

| Fit | Evaluation compile | Value/gradient compile | Auxiliary checked-reference first call | Total preparation | Warm optimization | Value/gradient mean / p95 | Validity-check mean / p95 |
|---|---:|---:|---:|---:|---:|---|---|
| constant_constrained | 0.715062 | 2.39425 | 2.40657 | 5.64129 | 0.124302 | 0.0132808 / 0.0147292 | 0.00402396 / 0.0042036 |
| full_spline | 0.601358 | 2.34888 | 2.56586 | 5.72199 | 3.28897 | 0.0136018 / 0.0143 | 0.00423554 / 0.0045925 |

Both primary fits were repeated twice. Coefficients, final objective and accepted objective history repeated exactly (maximum reported coefficient/objective differences zero). These repeats test computational reproducibility on one shared realization; they are not independent Monte Carlo trials and give no confidence interval over noise. Individual repetition timings and sensitivity compilation/execution times are retained in JSON. These are offline fixed-window timings, not moving-horizon update latency.

## Invalid candidate handling

The constant and full fits passed 17 and 41 deterministic box probes respectively. All actual trials remained valid; no trial was excluded or assigned a penalty. Probes cannot certify the box. Invalid-trial fault injection in the tests deliberately requests an interior landmark collision after propagation with a simultaneous valid observation; the adapter stops cleanly, records the diagnostic status and returns no value/gradient to SciPy. It retains the last accepted iterate as an aborted-run record. Invalid initial points, invalid probes and nonfinite gradients are separately tested.

## Test execution and preservation

```sh
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q --junitxml=/tmp/stage4b_pytest.xml -o junit_family=legacy
```

Outcome: **248 passed**, exit code 0; pytest reported 114.24 s. This includes all 233 prior tests and 15 Stage 4B tests. Tests cover constant mapping, real-EKF chain-rule/bridge consistency, SciPy/JAX value-gradient agreement, paired optimization reproducibility, invalid-domain/gradient stops, objective reduction, sampled reconstruction metrics, real and synthetic 600x14 sensitivity dimensions/ranks, correct covariance whitening, projected gradients, and refusal to claim completion from convergence/cost alone.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4b --test-record /tmp/stage4b_test_record.json
```

Outcome: both primary repeats, sensitivity evaluations, noiseless controls, tighter-ftol control, plots and saved results completed. Current test evidence is embedded separately; the benchmark does not execute pytest.

All 117 pre-existing protected files matched their pre-implementation SHA-256 digests. This covers the Stage 4A objective/interfaces/tests/artifacts, prior EKF code and baselines, simulation, spline adapter and sibling C++ source. Baseline A's submodule and the sibling spline repository remain clean. Only new Stage 4B files/artifacts and the authoritative project status were changed.

## Completion assessment and limitations

Actual evidence checks:

```json
{
  "required_experiments_and_diagnostics_present": true,
  "primary_fits_converged": true,
  "valid_trials_and_final_candidates": true,
  "objective_reduction": true,
  "both_channels_reconstructed_better_than_zero": true,
  "paired_repeat_reproducibility": true,
  "local_sensitivity_dimensions_and_rank": true,
  "noiseless_controls": true,
  "current_complete_test_suite": true
}
```

The controlled constant-disturbance completion gate is satisfied. Stronger claims remain unsupported: one seed, one geometry and one in-family constant truth do not establish reliability across noise, smooth/outside-family disturbances, sparse measurements or other scenarios. Smooth-disturbance extensions and Monte Carlo validation remain future work. No coefficient identifiability or uncertainty claims should be extrapolated from local numerical rank alone.

Stage 4C, moving-window estimation, nearest-neighbor association and Victoria Park integration remain deferred. Milestone 4 as a whole is not complete. No spline basis, EKF equation, innovation objective, regularization or invalid-candidate penalty was changed.

## Final serialization validation

After the recorded numerical experiments, a serialization-only patch made rejected nonfinite parameters JSON-safe. Valid-candidate computations are unchanged. The final code passed all 248 tests; the recorded numerical experiments were retained. The additional targeted test verified strict JSON export after an invalid nonfinite SciPy request. The separately retained earlier test record contains 247 passes; the current record and test_record.json contain 248 passes.
