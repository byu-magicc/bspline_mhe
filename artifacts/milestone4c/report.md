# Stage 4C — fixed-window input-spline MHE-SLAM

Controlled Stage 4C completion: **SATISFIED**.

The validated Stage 4B checked L-BFGS-B optimizer now reruns the preserved augmented Stage 3B.2 EKF-SLAM. Only 14 input-spline coefficients are optimized; there is no state-trajectory optimization. These are paired single-seed known-association experiments, not Monte Carlo or moving-window benchmarks.

## Reproduce

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4c --test-record artifacts/milestone4c/test_record.json
```

The default run repeats both configurations on constant and smooth truth. Saved intermediate gate records document the initial A-first validation and the B-constant gate before smooth experiments. The supplied test record describes a separately executed suite; the benchmark does not execute pytest.

## Configuration and independent truth

All shared constants follow Stage 4B: six landmarks, original robot prior, R and Qc, v=1 m/s, omega=0.2 rad/s, seed 7, degree 3, M=4, seven coefficients per disturbance channel. Estimator RK4 step is 0.025 s; every landmark is observed in fixed order every 0.1 s. The simulated duration is seven seconds. Disturbances are zero during the first two seconds. The optimization window is [2,7] s, with sigma=time-2 and spline scale M/L=0.8. Stage times are evaluated through the unchanged C++ adapter, including the exact right endpoint. All commanded inputs are evaluated by the original RK4 at every stage.

Constant post-arrival truth is [0.15,-0.05]. Smooth post-arrival truth is [0.15 sin(2 pi sigma/5), 0.05 cos(2 pi sigma/5)], outside the exact cubic spline family. Truth uses independent NumPy analytic trajectories for constant-input phases and independent SciPy DOP853 for the smooth phase (max step .05 s, rtol=1e-11, atol=1e-13). A split DOP853 reference uses max step .025 s, rtol=1e-12 and atol=1e-14, with an acceptance bound of 1e-10 per pose component. The disturbance discontinuity is an integration boundary. No EKF, JAX RK4 or spline generates truth. Qc remains an estimator uncertainty assumption; truth has deterministic inputs and seeded measurement noise.

## Arrival and observation accounting

**Configuration A:** the preserved zero-disturbance EKF-SLAM consumes observations at 0.1 through 2.0 s. Six initializations and 114 corrections construct the uncertain map. The complete posterior at 2.0 s, including every covariance cross-correlation and ID/slot mapping, is frozen. All six landmarks are active. Only observations strictly after 2.0 s enter the optimization replay: 300 corrections.

**Configuration B:** the original t=0 robot prior and Qc are propagated to 2.0 s using commanded inputs and zero disturbance, consuming no observations. This observation-free robot prior has an empty map; it is not A’s posterior marginal. The same observations strictly after 2.0 s are used. The six observations at 2.1 s initialize the map; the remaining 294 correct it. The original pose prior anchors the reference frame; errors are reported without post-hoc rigid alignment.

Every candidate restarts the identical captured arrival, covariance, active mask and integer ID map. An offline plan fixes birth/update/slot decisions from arrival metadata, IDs and requested masks. Truth coordinates never initialize the estimator. Repeated IDs at one timestamp initialize once and then correct sequentially. Births consume their measurement once, contribute no innovation cost, and retain differentiation through their inverse-model mean, covariance and cross-correlations.

## Objective and invalid-candidate policy

J=sum(e.T solve(S,e)) includes exactly the fixed requested correction set. Log determinant and J+logdet are separate diagnostics (the latter omits the Gaussian dimension constant). No regularization, logdet optimization, finite failure penalties, covariance clipping, jitter or repairs are introduced. All 14 coefficients start at zero with bounds [-0.5,0.5] and unchanged Stage 4B options. Every requested birth and correction must be valid. Node priors and every sequential posterior must be finite with symmetric PSD active covariance (absolute tolerance 1e-10); inactive storage stays exactly neutral. Measurement noise and correction S must be SPD. Invalid candidates have undefined cost and the checked optimizer stops on the first invalid preflight/trial without a retry or changed data.

## Baseline B versus fitted MHE-SLAM

| Configuration / truth | Estimator | J | w_v RMSE (m/s) | w_omega RMSE (rad/s) | Position RMSE (m) | Heading RMSE (rad) | Final position (m) | Final signed heading (rad) | Landmark RMSE (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a_constant | baseline_b | 504.3825488 | 0.15 | 0.05 | 0.061160398 | 0.013612804 | 0.087814682 | 0.029218102 | 0.080181855 |
| a_constant | optimized_mhe | 489.7499663 | 0.024162535 | 0.01795886 | 0.05588674 | 0.01164298 | 0.068000775 | 0.0076484734 | 0.072188848 |
| b_constant | baseline_b | 496.0603844 | 0.15 | 0.05 | 0.15596987 | 0.03693996 | 0.27443352 | 0.056728608 | 0.25696987 |
| b_constant | optimized_mhe | 480.740131 | 0.014308864 | 0.019184611 | 0.11544704 | 0.025714277 | 0.16236959 | 0.038069625 | 0.13780808 |
| a_smooth | baseline_b | 497.8221512 | 0.10596011 | 0.035390606 | 0.058987777 | 0.013106357 | 0.050598137 | 0.0074928855 | 0.061233973 |
| a_smooth | optimized_mhe | 489.5633218 | 0.021105103 | 0.017628266 | 0.056866637 | 0.012143936 | 0.067913184 | 0.006504728 | 0.07992713 |
| b_smooth | baseline_b | 489.5473644 | 0.10596011 | 0.035390606 | 0.086586175 | 0.020673754 | 0.11386141 | 0.031323789 | 0.10501468 |
| b_smooth | optimized_mhe | 480.7173462 | 0.017925881 | 0.019090222 | 0.10075605 | 0.02426359 | 0.13242066 | 0.036362067 | 0.12764502 |

Pose RMSE uses all 201 posterior integration nodes including arrival; heading errors wrap. Disturbance RMSE and maximum errors use the same 501 uniform elapsed samples including endpoints. Landmark RMSE uses active known IDs in the anchored frame. Full maximum errors and all diagnostic poses/covariances are stored in results.json and per-experiment files.

**a_constant:** objective reduction 14.632583; position/heading RMSE changes `{'position_rmse_m': -8.622667882605972, 'heading_rmse_rad': -14.470369923554538}` percent. Negative percentages indicate improvement.

**b_constant:** objective reduction 15.320253; position/heading RMSE changes `{'position_rmse_m': -25.981195819975067, 'heading_rmse_rad': -30.388995466611547}` percent. Negative percentages indicate improvement.

**a_smooth:** objective reduction 8.2588294; position/heading RMSE changes `{'position_rmse_m': -3.59589909867718, 'heading_rmse_rad': -7.34316874316222}` percent. Negative percentages indicate improvement.

**b_smooth:** objective reduction 8.8300181; position/heading RMSE changes `{'position_rmse_m': 16.36505162670545, 'heading_rmse_rad': 17.364215132196215}` percent. Negative percentages indicate improvement.

No test requires MHE-SLAM to beat Baseline B in every metric. A fit can lower innovation cost while fitting measurement/map noise or exploiting weak coefficient directions. The true constant coefficients need not minimize a finite noisy-data objective or the anchored map/pose error.

| Experiment | Reference coefficients | Reference J | Initial J | Optimized J | logdet (baseline / fitted) | J+logdet (baseline / fitted) |
|---|---|---:|---:|---:|---|---|
| a_constant | true constant | 490.6193389 | 504.3825488 | 489.7499663 | -4044.475555 / -4046.226194 | -3540.093007 / -3556.476228 |
| b_constant | true constant | 481.4130471 | 496.0603844 | 480.740131 | -3935.950739 / -3937.607861 | -3439.890355 / -3456.867729 |
| a_smooth | independent sampled LS (diagnostic only) | 490.3524308 | 497.8221512 | 489.5633218 | -4047.564533 / -4048.009057 | -3549.742382 / -3558.445735 |
| b_smooth | independent sampled LS (diagnostic only) | 481.4230147 | 489.5473644 | 480.7173462 | -3938.936991 / -3939.379751 | -3449.389627 / -3458.662405 |

Smooth reference coefficients fit the known analytic inputs with the C++ basis on the dense grid; they never initialize optimization or supply inputs to the EKF.

## Observation counts and covariance checks

| Experiment | Active at arrival / final | New landmarks during window | Corrections | Requested / valid / invalid | Baseline min active eigenvalue / asymmetry | Fitted min active eigenvalue / asymmetry |
|---|---|---:|---:|---|---|---|
| a_constant | 6 / 6 | 0 | 300 | 300 / 300 / 0 | 1.3447e-05 / 8.26e-16 | 1.3662e-05 / 1.17e-15 |
| b_constant | 0 / 6 | 6 | 294 | 300 / 300 / 0 | 1.4813e-05 / 6.83e-16 | 1.491e-05 / 4.09e-16 |
| a_smooth | 6 / 6 | 0 | 300 | 300 / 300 / 0 | 1.7453e-05 / 4.72e-16 | 1.6524e-05 / 6.14e-16 |
| b_smooth | 0 / 6 | 6 | 294 | 300 / 300 / 0 | 1.8506e-05 / 9.14e-16 | 1.722e-05 / 5.69e-16 |

Reported minima/asymmetries are over posterior integration nodes in the active subspace. The objective additionally checks node priors and every sequential posterior. Inactive storage is exactly zero and excluded from active eigenvalue metrics.

## Optimization, gradients and timing

| Experiment | Status / iterations | Function / gradient calls | Raw gradient L2 / inf | Projected gradient inf | Active coefficient bounds (zero-based) |
|---|---|---|---|---:|---|
| a_constant | converged / 66 | 77 / 77 | 0.149717 / 0.101223 | 0.000161917 | [0, 6, 13] |
| b_constant | converged / 57 | 61 / 61 | 0.195539 / 0.168905 | 7.5323e-05 | [0, 6, 13] |
| a_smooth | converged / 60 | 65 / 65 | 0.182599 / 0.144565 | 0.000281685 | [0, 6, 13] |
| b_smooth | converged / 71 | 78 / 78 | 0.177558 / 0.145673 | 0.000116867 | [0, 6, 13] |

a_constant termination: `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`.


b_constant termination: `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`.


a_smooth termination: `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`.


b_smooth termination: `CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH`.

The stopping criteria are ftol=1e-12, gtol=1e-6, maxiter=500, maxfun=2000, maxls=40 and maxcor=10. Function-reduction convergence can precede the gradient target; active constraints change projected gradients. Neither convergence nor a lower J proves global optimality, identifiability or uncertainty calibration. Raw/projected zero-reference gradient norms are diagnostic in results.json; Baseline B has zero optimizer iterations and function/gradient optimization calls.

| Experiment / evaluator | Evaluation compile (s) | Value/gradient compile (s) | Cold checked reference incl. auxiliary compilation (s) | Steady value/gradient mean / p95 (s) | Steady checked replay mean / p95 (s) | Warm optimization (s) | Preparation + optimization (s) |
|---|---:|---:|---:|---|---|---:|---:|
| a_constant / baseline_b | 1.1085 | 4.35326 | 4.42695 | 0.0861681 / 0.0864746 | 0.0365377 / 0.036694 | 0 | not optimized |
| a_constant / optimized_mhe | 1.05197 | 4.32841 | 4.23866 | 0.0853948 / 0.086644 | 0.0365408 / 0.0368498 | 9.44577 | 20.8144 |
| b_constant / baseline_b | 1.06315 | 4.32114 | 4.39334 | 0.0851943 / 0.0855694 | 0.0363196 / 0.0364159 | 0 | not optimized |
| b_constant / optimized_mhe | 1.05034 | 4.38515 | 4.10437 | 0.0855589 / 0.0861993 | 0.0362116 / 0.0364201 | 7.45475 | 18.731 |
| a_smooth / baseline_b | 1.06925 | 4.07403 | 4.68935 | 0.0855388 / 0.0859593 | 0.0364485 / 0.0365451 | 0 | not optimized |
| a_smooth / optimized_mhe | 1.04942 | 4.03282 | 4.79666 | 0.086147 / 0.086664 | 0.0367015 / 0.0369056 | 8.01549 | 19.6501 |
| b_smooth / baseline_b | 1.07331 | 4.15019 | 4.16496 | 0.0857936 / 0.0862848 | 0.0362391 / 0.0364264 | 0 | not optimized |
| b_smooth / optimized_mhe | 1.06994 | 5.04654 | 4.10439 | 0.0858459 / 0.0864693 | 0.0363985 / 0.0365833 | 9.56651 | 21.5496 |

Compilation is explicit lower/compile time without execution; the cold checked reference can include auxiliary VJP compilation. Warm optimization excludes preparation but checks every trial. The zero-reference objective/gradient timings use ten repeated evaluations; these diagnostic gradients are not part of Baseline B estimation. These are fixed-window offline costs, not online update latency.

| Experiment | Largest absolute FD discrepancy | Precision / FD steps | Repeated coefficients / accepted history |
|---|---:|---|---|
| a_constant | 1.2512e-06 | float64 / 1e-4, 1e-5 | {'coefficient_max_absolute_difference': 0.0, 'objective_absolute_difference': 0.0, 'accepted_objective_history_identical': True} |
| b_constant | 8.6244e-07 | float64 / 1e-4, 1e-5 | {'coefficient_max_absolute_difference': 0.0, 'objective_absolute_difference': 0.0, 'accepted_objective_history_identical': True} |
| a_smooth | 1.2138e-06 | float64 / 1e-4, 1e-5 | {'coefficient_max_absolute_difference': 0.0, 'objective_absolute_difference': 0.0, 'accepted_objective_history_identical': True} |
| b_smooth | 6.1975e-07 | float64 / 1e-4, 1e-5 | {'coefficient_max_absolute_difference': 0.0, 'objective_absolute_difference': 0.0, 'accepted_objective_history_identical': True} |

Every coefficient is checked at zero, a nonconstant vector and the fitted vector. Absolute/relative tolerances are 2e-5 with a separate maximum absolute discrepancy <2e-5; every perturbation must be valid. Focused tests separately differentiate initialized landmark means, marginal covariances and cross-correlations through their first observation. Deterministic repeats are not Monte Carlo trials.

## Covariance figures and interpretation

**Conditional EKF covariance; spline coefficients treated as fixed.** The optimized EKF covariance does not include uncertainty in the fitted spline coefficients. The 95% position ellipses use the 2x2 marginal blocks and chi-squared threshold 5.991. Error envelopes are componentwise ±2 standard deviations, not simultaneous confidence regions. Every plotted history/marginal is checked; invalid covariances are rejected without projection or clipping. The t0 prior anchors the frame.

### a_constant

![a_constant trajectory_covariance](a_constant_trajectory_covariance.png)

![a_constant landmark_covariance](a_constant_landmark_covariance.png)

![a_constant errors_uncertainty](a_constant_errors_uncertainty.png)

![a_constant disturbances](a_constant_disturbances.png)

![a_constant objective_history](a_constant_objective_history.png)

### b_constant

![b_constant trajectory_covariance](b_constant_trajectory_covariance.png)

![b_constant landmark_covariance](b_constant_landmark_covariance.png)

![b_constant errors_uncertainty](b_constant_errors_uncertainty.png)

![b_constant disturbances](b_constant_disturbances.png)

![b_constant objective_history](b_constant_objective_history.png)

### a_smooth

![a_smooth trajectory_covariance](a_smooth_trajectory_covariance.png)

![a_smooth landmark_covariance](a_smooth_landmark_covariance.png)

![a_smooth errors_uncertainty](a_smooth_errors_uncertainty.png)

![a_smooth disturbances](a_smooth_disturbances.png)

![a_smooth objective_history](a_smooth_objective_history.png)

### b_smooth

![b_smooth trajectory_covariance](b_smooth_trajectory_covariance.png)

![b_smooth landmark_covariance](b_smooth_landmark_covariance.png)

![b_smooth errors_uncertainty](b_smooth_errors_uncertainty.png)

![b_smooth disturbances](b_smooth_disturbances.png)

![b_smooth objective_history](b_smooth_objective_history.png)

Each figure also has a vector PDF. Triangle landmarks were initialized before the window; circle landmarks were initialized during it. Robot ellipses appear at six regularly spaced node times.

## Saved evidence, tests and completion gate

configuration.json, results.json and per-experiment JSON retain full numerical results and timings. dataset_constant.npz/dataset_smooth.npz save seven-second truth and observations. Per-experiment *_histories.npz files save times, pose/map/covariance/active/ID histories, observation masks and slots, innovations, dense spline values and accepted optimization costs. Separate *_optimization_history.json files retain accepted and trial evaluations. Initialization histories retain the pre-window map posterior. Intermediate gate JSON/XML files prove the requested development order.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone4c/pytest.xml -o junit_family=legacy
```

Actual complete-suite outcome: **295 passed**, exit code 0. See [test_record.json](test_record.json), pytest.xml and pytest_stdout.txt.

All 163 pre-existing protected files match their original SHA-256 hashes; see [preservation.json](preservation.json). Existing mathematical implementations and earlier artifacts are unchanged. Implementation choices and limitations are documented in [validation.md](validation.md).

```json
{
  "required_experiments_present": true,
  "configuration_a_validated_before_b": true,
  "constant_b_validated_before_smooth": true,
  "all_numerical_experiment_checks": true,
  "arrival_and_update_accounting": true,
  "identical_datasets_across_configurations": true,
  "independent_truth_validated": true,
  "all_previous_protected_files_preserved": true,
  "current_complete_test_suite": true,
  "all_figures_generated": true
}
```

Stage 4C’s controlled completion criterion is satisfied. Before Milestone 5, decide how a sliding-window arrival posterior is formed without reusing measurements, how map births crossing the boundary are accounted for, and how coefficients are shifted/warm-started. These decisions are not implemented here. Broader disturbance/noise/sparsity tests, statistical calibration and paired Monte Carlo comparisons remain future validation. Nearest-neighbor association and Victoria Park remain deferred.

## Final serialization audit and negative-case note

All saved arrivals, masks, ID/slot histories, paired data, objective/logdet values and reconstruction/localization/map metrics passed independent post-serialization checks. Known-ID correctness and consumed-observation coverage are 1.0; B correction coverage is 0.98 because six records initialize landmarks. See [validation.json](validation.json) and [validation.md](validation.md).

The smooth Configuration B fit lowers J and improves both disturbance RMSEs while worsening robot position, heading and landmark RMSE. Configuration A smooth also worsens landmark RMSE. These are controlled negative results; finite-noise fitting, joint map/pose adjustment and weak/active-bound directions warrant further investigation. Cross-configuration raw costs have different correction counts and priors; compare fitted versus zero within each configuration.
