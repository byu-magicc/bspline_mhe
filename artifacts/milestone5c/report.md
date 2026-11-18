# Stage 5C — moving-horizon diagnostics and computational profiling

## Executive summary

Six seed-7 scenarios are retained; available completed cases: 6. The selected duration is 6.0 s, with 11 complete windows per case at 5.0–6.0 s every 0.1 s, compared with three in Stage 5B. Both projected/clipped and zero starts are optimized independently. All original d=3, M=4, L=5 s, noise, prior, bounds, optimizer options, sensors and arrival rules remain. No estimator source file is changed.

Sensors are colocated/aligned: regression range [0,infinity], 360 degrees; forward range [0.2,6] m, 180 degrees. RK4 uses 0.025 s steps. Commands are [1 m/s,0.2 rad/s]. Independent zero-disturbance forward-EKF posteriors supply arrivals, including full SLAM covariance; reported MHE maps never feed back into that filter. Measurement noise is diag(0.05^2 m^2,(1 degree)^2); continuous input-noise intensity is diag(0.01,0.0025), with the original units/convention. The pose prior is diag(0.05^2,0.05^2,(2 degrees)^2). Full retained configurations appear in each case's results.json.

The compilation feasibility test reproduced every result/gradient bitwise on three saved Stage 5B windows and reproduced a full solve exactly. A benchmark-only adapter lifts captured arrays into runtime arguments of the same traced JAX program. It does not change data shapes, masks, slots, birth plans or EKF equations. See [compilation feasibility](compilation_feasibility.md) and feasibility.json.

These are deterministic diagnostics, not independent trials or confidence estimates. All windows overlap heavily. Shaded plot regions define the interior and never represent confidence intervals.

### Findings at a glance

- Completed 66 windows and 132 fits; 132 fits report convergence. No solver option or noise parameter was tuned.
- The right endpoint exceeds the defined interior in 6/6 linear and 6/6 angular warm-start profiles. The case-by-case table below tests consistency rather than presuming it.
- Warm-started position RMSE is worse than the forward EKF in: forward_zero, regression_smooth, regression_zero. Lower innovation cost does not establish improved truth accuracy.
- 1 structural replay program(s) served 264 preparation calls. Remaining warm objective/gradient/SciPy work still dominates many updates; this is not real-time performance.
- All 132 fits terminate by the unchanged function-reduction criterion; none meets projected-gradient gtol=1e-6. Solver success is therefore reported separately from first-order stationarity.
- The two boundary basis energies are symmetric and small relative to interior functions. Sensitivity, measurement timing, arrival information and fitted bounds must be examined together; basis support alone cannot explain a right/left error asymmetry.

## Position-dependent disturbance reconstruction

The 501-point grid includes sigma=0,5 and all knots 0,1.25,2.5,3.75,5. E(sigma)=sqrt(mean_windows(error(sigma)^2)). The primary interior is [1.25,3.75] s; its scalar RMS is sqrt(mean_sigma(E(sigma)^2)). The secondary open-interior RMS excludes only the two endpoints and is also retained in JSON. Endpoints and every window are included without discarding empty scans.

| Case | Start | Left / interior / right w_v RMSE (m/s) | Right > interior? | Left / interior / right w_omega RMSE (rad/s) | Right > interior? |
|---|---|---|---|---|---|
| forward_constant | projected | 0.0626704 / 0.0161845 / 0.178554 | True | 0.0467718 / 0.00775325 / 0.0514767 | True |
| forward_constant | zero | 0.0626703 / 0.0161844 / 0.168091 | True | 0.0467717 / 0.00775326 / 0.0771048 | True |
| forward_smooth | projected | 0.0761397 / 0.0214045 / 0.113687 | True | 0.0533414 / 0.0107891 / 0.056359 | True |
| forward_smooth | zero | 0.0761398 / 0.0214046 / 0.113687 | True | 0.0533414 / 0.0107891 / 0.0563589 | True |
| forward_zero | projected | 0.0591662 / 0.0230869 / 0.113185 | True | 0.0509959 / 0.0108429 / 0.0545436 | True |
| forward_zero | zero | 0.0591665 / 0.0230869 / 0.113185 | True | 0.0509959 / 0.010843 / 0.0545418 | True |
| regression_constant | projected | 0.0657668 / 0.0248808 / 0.076076 | True | 0.0271179 / 0.00633269 / 0.0414141 | True |
| regression_constant | zero | 0.0623179 / 0.0258989 / 0.0760947 | True | 0.0271175 / 0.0063899 / 0.0414519 | True |
| regression_smooth | projected | 0.0657996 / 0.0178154 / 0.057358 | True | 0.0279014 / 0.00703888 / 0.0408958 | True |
| regression_smooth | zero | 0.0657991 / 0.0178152 / 0.0573581 | True | 0.0279022 / 0.00703893 / 0.0408953 | True |
| regression_zero | projected | 0.0491318 / 0.0182638 / 0.0564032 | True | 0.0272875 / 0.00700702 / 0.0396191 | True |
| regression_zero | zero | 0.049132 / 0.0182635 / 0.0564031 | True | 0.0272859 / 0.00700736 / 0.0396202 | True |

Warm-started linear: right exceeds the defined interior in **6/6** completed scenarios. This is a deterministic comparison; it is not a statistical test of an independent population.

Warm-started angular: right exceeds the defined interior in **6/6** completed scenarios. This is a deterministic comparison; it is not a statistical test of an independent population.

![Linear profiles](linear_error_profiles.png)

![Angular profiles](angular_error_profiles.png)

## Basis support and exact whitened coefficient sensitivity

| j | Support inside window (s) | Integral of b_j squared (s) | Right endpoint weight |
|---:|---|---:|---:|
| 0 | [0.0, 1.25] | 0.00496031746 | 0 |
| 1 | [0.0, 2.5] | 0.299603175 | 0 |
| 2 | [0.0, 3.75] | 0.594246032 | 0 |
| 3 | [0.0, 5.0] | 0.599206349 | 0 |
| 4 | [1.25, 5.0] | 0.594246032 | 0.1666667 |
| 5 | [2.5, 5.0] | 0.299603175 | 0.6666667 |
| 6 | [3.75, 5.0] | 0.00496031746 | 0.1666667 |

![Basis and support](basis_support.png)

Basis energies are symmetric at the two boundaries. That representation fact alone cannot explain a right/left asymmetry in estimation error. The right endpoint uses coefficients 4,5,6 with weights [1/6,4/6,1/6]; focusing solely on the last coefficient would miss the larger weight of coefficient 5.

For every fit, q=L(c)^-1 r(c), where S(c)=L(c)L(c).T is lower Cholesky in the existing range/bearing and observation order. D=dq/dc differentiates both residuals **and covariance whitening**. This differs from the earlier fixed-S sensitivity diagnostic. Only correction rows enter q; padded/birth rows have zero diagnostic residual and are removed before SVD. q.T q=J and 2 D.T q=grad J are checked for every optimized window. Numerical rank uses max(rows,14)*eps64*s_max; practical rank at 1e-3 relative threshold is separately reported. Condition number is null for numerical deficiency. Columns retain physical coefficient units; cross-channel spectra are parameterization dependent. When S depends on coefficients, alternative valid whitening conventions can introduce coefficient-dependent rotations, so D spectra are also whitening-convention dependent. Lower forward-sensor column norms partly reflect fewer correction rows; no per-observation normalization is applied. This local analysis is not formal nonlinear observability.

| Case | Start | Rank range | Finite condition min–max / deficient windows | Median rightmost v / omega column norm | gtol met / windows | Max projected gradient |
|---|---|---|---|---|---|---:|
| forward_constant | projected | 12–14 | 405.3–2.203e+07 / 2 | 0.006563553 / 0.006201293 | 0/11 | 0.0001433 |
| forward_constant | zero | 12–14 | 405.3–2.203e+07 / 2 | 0.006563549 / 0.006201288 | 0/11 | 0.000151291 |
| forward_smooth | projected | 14–14 | 106–147.3 / 0 | 0.3966546 / 1.149883 | 0/11 | 8.29526e-05 |
| forward_smooth | zero | 14–14 | 106–147.3 / 0 | 0.3966546 / 1.149883 | 0/11 | 8.29526e-05 |
| forward_zero | projected | 14–14 | 106.1–150.6 / 0 | 0.3914512 / 1.148462 | 0/11 | 8.78629e-05 |
| forward_zero | zero | 14–14 | 106.1–150.6 / 0 | 0.3914512 / 1.148462 | 0/11 | 0.000120577 |
| regression_constant | projected | 14–14 | 87.01–116.3 / 0 | 0.6382843 / 1.342004 | 0/11 | 0.000257576 |
| regression_constant | zero | 14–14 | 87.01–116.3 / 0 | 0.6382843 / 1.342004 | 0/11 | 0.000454591 |
| regression_smooth | projected | 14–14 | 86.9–116.6 / 0 | 0.656305 / 1.340125 | 0/11 | 0.000369729 |
| regression_smooth | zero | 14–14 | 86.9–116.6 / 0 | 0.6563049 / 1.340125 | 0/11 | 0.000878247 |
| regression_zero | projected | 14–14 | 87.08–116.6 / 0 | 0.6535109 / 1.340626 | 0/11 | 0.000376438 |
| regression_zero | zero | 14–14 | 87.08–116.6 / 0 | 0.6535108 / 1.340626 | 0/11 | 0.000886229 |

| Case | Warm channel | Median column norms: left j0 / middle j2–4 / right j6 | Right weaker than left? |
|---|---|---|---|
| forward_constant | v | 0.576669 / 7.59949 / 0.00656355 | True |
| forward_constant | omega | 1.2357 / 15.3016 / 0.00620129 | True |
| forward_smooth | v | 0.578938 / 7.60618 / 0.396655 | True |
| forward_smooth | omega | 1.23642 / 15.3021 / 1.14988 | True |
| forward_zero | v | 0.578081 / 7.60536 / 0.391451 | True |
| forward_zero | omega | 1.23592 / 15.3053 / 1.14846 | True |
| regression_constant | v | 0.644688 / 7.92771 / 0.638284 | True |
| regression_constant | omega | 1.35136 / 15.3874 / 1.342 | True |
| regression_smooth | v | 0.645787 / 7.76791 / 0.656305 | False |
| regression_smooth | omega | 1.35137 / 15.408 / 1.34012 | True |
| regression_zero | v | 0.647077 / 7.74959 / 0.653511 | False |
| regression_zero | omega | 1.35132 / 15.4078 / 1.34063 | True |

## Visibility, last-correction age and endpoint errors

Every update retains scheduled scans, observed/empty scans, corrections, births, mapped counts, window-relative observation times/IDs and time since the latest valid correction. The latter includes a preceding forward correction if the window itself has none. Births do not count as corrections.

| Case | Scheduled scans | Observed scans | Empty scans | Corrections | Births | Active map | Latest-correction age (s) |
|---|---|---|---|---|---|---|---|
| forward_constant | 50–50 | 36–46 | 4–14 | 41–50 | 1–2 | 2–2 | 0.4–1.4 |
| forward_smooth | 50–50 | 50–50 | 0–0 | 54–56 | 1–2 | 2–2 | 0–0 |
| forward_zero | 50–50 | 50–50 | 0–0 | 57–59 | 1–2 | 2–2 | 0–0 |
| regression_constant | 50–50 | 50–50 | 0–0 | 294–300 | 0–6 | 6–6 | 0–0 |
| regression_smooth | 50–50 | 50–50 | 0–0 | 294–300 | 0–6 | 6–6 | 0–0 |
| regression_zero | 50–50 | 50–50 | 0–0 | 294–300 | 0–6 | 6–6 | 0–0 |

| Case | Start | Observed endpoints N / right RMSE v,omega | Empty endpoints N / right RMSE v,omega |
|---|---|---|---|
| forward_constant | projected | 0 / None | 11 / [0.1785538164297436, 0.051476747319925555] |
| forward_constant | zero | 0 / None | 11 / [0.16809068223921944, 0.07710477914987733] |
| forward_smooth | projected | 11 / [0.11368651708060788, 0.056358986238866704] | 0 / None |
| forward_smooth | zero | 11 / [0.11368655280077865, 0.05635888643688342] | 0 / None |
| forward_zero | projected | 11 / [0.11318530008520626, 0.0545435583711944] | 0 / None |
| forward_zero | zero | 11 / [0.11318482509661071, 0.054541762614969555] | 0 / None |
| regression_constant | projected | 11 / [0.07607604014358793, 0.041414116722028264] | 0 / None |
| regression_constant | zero | 11 / [0.07609471251656921, 0.04145185902066997] | 0 / None |
| regression_smooth | projected | 11 / [0.057358017037224165, 0.04089581602144453] | 0 / None |
| regression_smooth | zero | 11 / [0.05735812333557606, 0.04089531968539392] | 0 / None |
| regression_zero | projected | 11 / [0.05640318515251653, 0.0396191366766157] | 0 / None |
| regression_zero | zero | 11 / [0.05640308558349729, 0.039620248402426665] | 0 / None |

![Latest correction age and endpoint errors](correction_age_endpoint_error.png)

Group counts matter: a scenario with only empty endpoints has no within-scenario observed/empty contrast. Cross-scenario differences also change motion, map coverage and residual information. Correlations in diagnostics.json are descriptive only; no p-values, confidence intervals or causal claims are inferred from overlapping windows.

## Pose/map degradation, arrivals and numerical optimization

| Case | Method | Position RMSE m | Heading RMSE rad | Landmark RMSE m | Mapped count range |
|---|---|---:|---:|---:|---|
| forward_constant | forward | 0.1922813 | 0.05817357 | 0.1245054 | 2–2 |
| forward_constant | projected | 0.1377569 | 0.03088203 | 0.05940742 | 2–2 |
| forward_constant | zero | 0.1358632 | 0.04278539 | 0.05940741 | 2–2 |
| forward_smooth | forward | 0.06110909 | 0.03836644 | 0.1223483 | 2–2 |
| forward_smooth | projected | 0.05841341 | 0.02931994 | 0.1007204 | 2–2 |
| forward_smooth | zero | 0.05841338 | 0.02931996 | 0.1007204 | 2–2 |
| forward_zero | forward | 0.02470103 | 0.02136265 | 0.07044102 | 2–2 |
| forward_zero | projected | 0.04811589 | 0.02664192 | 0.08559658 | 2–2 |
| forward_zero | zero | 0.04811562 | 0.02664194 | 0.08559662 | 2–2 |
| regression_constant | forward | 0.231032 | 0.03667361 | 0.2408697 | 6–6 |
| regression_constant | projected | 0.09670745 | 0.01889089 | 0.1225223 | 6–6 |
| regression_constant | zero | 0.09310638 | 0.01959901 | 0.1223 | 6–6 |
| regression_smooth | forward | 0.0418177 | 0.01240453 | 0.06453723 | 6–6 |
| regression_smooth | projected | 0.1230673 | 0.02673395 | 0.172997 | 6–6 |
| regression_smooth | zero | 0.1230683 | 0.02673415 | 0.1729982 | 6–6 |
| regression_zero | forward | 0.06230254 | 0.01578956 | 0.09485303 | 6–6 |
| regression_zero | projected | 0.1062491 | 0.02326895 | 0.1485351 | 6–6 |
| regression_zero | zero | 0.1062494 | 0.02326895 | 0.1485351 | 6–6 |

Each update stores arrival pose/map errors and full active covariance diagnostics, all reported errors, active/near-bound coefficients (1e-10/1e-3 distance thresholds), projected-gradient norms, termination reasons, and consecutive endpoint/overlap revisions. Arrival error is evaluated at t-L, not t. Maps use known IDs and the original anchored frame with no post-hoc alignment. **Conditional EKF covariance; spline coefficients treated as fixed.**

An innovation-cost reduction is not a reduction in truth error. In addition to J, each update stores logdet diagnostics, innovation variances and the optimized residual quadratic using baseline S. These are diagnostic decompositions, not alternate optimized objectives. A lower cost can coexist with fitting a particular noise realization, weak input directions, representation error, or pose/map coupling. Nonzero arrival error may contribute, but the first window has zero arrival mean error at t=0; degradation there cannot require arrival mean bias. These observations do not establish a unique causal explanation. SciPy convergence often uses function reduction and need not meet gtol.

The zero-disturbance cases have exactly zero true disturbance and an exact zero-input forward model. Any fitted nonzero input is therefore reconstruction error for this realization. Low-cost nonzero solutions demonstrate that minimizing the innovation quadratic is not an input-accuracy criterion. The smooth input is outside the exact piecewise-cubic spline family. None of these observations justifies changing the objective during this diagnostic stage.

### Focused degradation evidence

**regression_smooth:** forward/warm position RMSE 0.0418177/0.1230673 m; arrival position error ranges 0–0.02921655 m. The first arrival has position/heading error 0 m / 0 rad. Local D rank ranges 14–14; active-bound counts range 0–3; gtol is met in 0/11 fits. Warm/zero report-position-error difference is at most 7.5e-06 m. Mean cost reduction decomposes algebraically into 7.12294 using optimized residuals with baseline S, plus -0.469083 from changing S at those residuals. This decomposition is conditional and order dependent; it does not identify a causal mechanism.

**regression_zero:** forward/warm position RMSE 0.06230254/0.1062491 m; arrival position error ranges 0–0.03405138 m. The first arrival has position/heading error 0 m / 0 rad. Local D rank ranges 14–14; active-bound counts range 0–3; gtol is met in 0/11 fits. Warm/zero report-position-error difference is at most 5.6019e-06 m. Mean cost reduction decomposes algebraically into 0.328982 using optimized residuals with baseline S, plus 0.194459 from changing S at those residuals. This decomposition is conditional and order dependent; it does not identify a causal mechanism.

**forward_zero:** forward/warm position RMSE 0.02470103/0.04811589 m; arrival position error ranges 0–0.03754375 m. The first arrival has position/heading error 0 m / 0 rad. Local D rank ranges 14–14; active-bound counts range 3–4; gtol is met in 0/11 fits. Warm/zero report-position-error difference is at most 1.5225e-06 m. Mean cost reduction decomposes algebraically into 1.17285 using optimized residuals with baseline S, plus 0.00070442 from changing S at those residuals. This decomposition is conditional and order dependent; it does not identify a causal mechanism.

| Case | Start | Maximum overlap pose revision m | Maximum matched-landmark overlap revision m |
|---|---|---:|---:|
| forward_constant | projected | 0.04665973 | 0.05037454 |
| forward_constant | zero | 0.04665964 | 0.05037418 |
| forward_smooth | projected | 0.07867958 | 0.01668784 |
| forward_smooth | zero | 0.07867918 | 0.01668778 |
| forward_zero | projected | 0.0800461 | 0.0197126 |
| forward_zero | zero | 0.08004437 | 0.0197117 |
| regression_constant | projected | 0.1922864 | 0.2338464 |
| regression_constant | zero | 0.07542337 | 0.1020935 |
| regression_smooth | projected | 0.08413046 | 0.1210557 |
| regression_smooth | zero | 0.0841292 | 0.1210538 |
| regression_zero | projected | 0.1376781 | 0.1954636 |
| regression_zero | zero | 0.1376859 | 0.1954749 |

Overlap revisions compare two window solutions at the same physical time and matched landmark IDs. They differ from ordinary motion between report times. All report-to-report pose/map deltas are also retained in updates.json. The three focused cases degrade despite observed endpoint scans and numerical full rank. Their near-identical warm/zero errors provide little evidence that the warm-start policy causes that degradation. Nonzero fitted inputs in zero-truth cases, weak boundary columns, active bounds and function-reduction termination are supported observations; noise fitting and pose/map/input coupling remain plausible explanations, rather than proven isolated causes.

## Warm starts and computational profile

| Case | Start | Iterations | Function / gradient calls | Converged | Clipped updates | Update mean / median / p95 s | SciPy optimization mean s |
|---|---|---:|---|---:|---:|---|---:|
| forward_constant | projected | 743 | 828 / 828 | 11 | 10 | 9.43239 / 9.49802 / 11.6893 | 6.46425 |
| forward_constant | zero | 1202 | 1311 / 1311 | 11 | 0 | 13.0706 / 13.1782 / 16.9389 | 10.2447 |
| forward_smooth | projected | 598 | 664 / 664 | 11 | 9 | 8.19375 / 7.52681 / 10.2574 | 5.26036 |
| forward_smooth | zero | 735 | 807 / 807 | 11 | 0 | 9.24224 / 8.84117 / 11.2244 | 6.35285 |
| forward_zero | projected | 603 | 673 / 673 | 11 | 9 | 8.23666 / 7.95758 / 10.02 | 5.30134 |
| forward_zero | zero | 696 | 763 / 763 | 11 | 0 | 8.82776 / 8.64666 / 10.5218 | 6.02065 |
| regression_constant | projected | 906 | 1000 / 1000 | 11 | 9 | 15.0334 / 15.2414 / 19.9004 | 11.3349 |
| regression_constant | zero | 875 | 968 / 968 | 11 | 0 | 14.5347 / 13.4648 / 20.1104 | 10.9652 |
| regression_smooth | projected | 925 | 1020 / 1020 | 11 | 9 | 15.2431 / 13.9963 / 21.6397 | 11.6248 |
| regression_smooth | zero | 1039 | 1130 / 1130 | 11 | 0 | 16.345 / 15.1674 / 22.9338 | 12.7692 |
| regression_zero | projected | 1037 | 1127 / 1127 | 11 | 9 | 16.9541 / 14.626 / 29.5547 | 12.7095 |
| regression_zero | zero | 1071 | 1160 / 1160 | 11 | 0 | 16.6414 / 15.1124 / 26.2741 | 13.0561 |

| Case | Start | Mean initial / final J | Dense-grid w_v / w_omega RMSE | Maximum clipping magnitude |
|---|---|---|---|---:|
| forward_constant | projected | 60.94552 / 59.47049 | 0.05969521 / 0.01888883 | 0.2096974 |
| forward_constant | zero | 73.53022 / 59.47049 | 0.05792602 / 0.02384045 | 0 |
| forward_smooth | projected | 62.7557 / 61.5706 | 0.03505431 / 0.01950889 | 0.3120616 |
| forward_smooth | zero | 68.52192 / 61.5706 | 0.03505428 / 0.01950887 | 0 |
| forward_zero | projected | 63.96194 / 63.2988 | 0.03556114 / 0.01889984 | 0.2928258 |
| forward_zero | zero | 64.47235 / 63.2988 | 0.03556102 / 0.01889973 | 0 |
| regression_constant | projected | 471.7463 / 469.6812 | 0.03086215 / 0.009765247 | 0.2020384 |
| regression_constant | zero | 485.45 / 469.6812 | 0.03092403 / 0.009784706 | 0 |
| regression_smooth | projected | 469.5353 / 468.3764 | 0.02516763 / 0.01047896 | 0.2111075 |
| regression_smooth | zero | 475.0303 / 468.3764 | 0.02516762 / 0.01047905 | 0 |
| regression_zero | projected | 468.9147 / 468.3352 | 0.02593232 / 0.01029131 | 0.2113294 |
| regression_zero | zero | 468.8586 / 468.3352 | 0.02593232 / 0.01029146 | 0 |

Every per-window initial/final cost, call count, termination message, gradient norm, projected gradient norm, clipping count/magnitude and measured runtime remains in each case's updates.json. Totals and means above summarize the same paired windows; they do not select either solution.

forward_constant: projected starts use more iterations in 0/11 windows (indices []) and higher measured update time in 1/11. The two controlled solutions are retained; neither is automatically selected.

forward_smooth: projected starts use more iterations in 1/11 windows (indices [6]) and higher measured update time in 1/11. The two controlled solutions are retained; neither is automatically selected.

forward_zero: projected starts use more iterations in 2/11 windows (indices [4, 6]) and higher measured update time in 4/11. The two controlled solutions are retained; neither is automatically selected.

regression_constant: projected starts use more iterations in 7/11 windows (indices [1, 2, 3, 4, 7, 9, 10]) and higher measured update time in 8/11. The two controlled solutions are retained; neither is automatically selected.

regression_smooth: projected starts use more iterations in 2/11 windows (indices [3, 4]) and higher measured update time in 3/11. The two controlled solutions are retained; neither is automatically selected.

regression_zero: projected starts use more iterations in 4/11 windows (indices [2, 7, 8, 9]) and higher measured update time in 5/11. The two controlled solutions are retained; neither is automatically selected.

The nested timing table in diagnostics.json records mean, median, p95 and totals for window preparation, synchronized arrival retrieval, buffer/basis remainder, host birth-plan preparation, warm projection, clipping, tracing, lowering, compilation, checked objective evaluation, joint value/gradient execution, SciPy optimization, final replay, diagnostics and artifacts. Some categories are nested: SciPy time includes trial objective/gradient work; window preparation includes retrieval/buffers/basis; factory time includes birth plans. They must not be summed twice. Value/gradient timing includes the value, as required by the unchanged SciPy pair. Update timing includes host initialization and final replay, excludes sensitivity/diagnostic/IO work. Preparation profiling uses cProfile and has instrumentation overhead; plotting is timed separately.

**Host-profiler limitation and correction:** cProfile omitted the completed UniformBasis.evaluate frame in this Python 3.12/pybind environment. Its raw basis_seconds=0 is unresolved and excluded from timing summaries; the original observation_buffer_seconds is explicitly relabeled buffer_plus_basis_remainder_seconds. Separate direct post-run timers in host_profile_remeasurement.json measure basis work and the remaining host buffer work on all 66 saved windows, with three repetitions each and bitwise comparison of every window/arrival field. These are labeled remeasurements and never replace online fit timings.

Plot generation wall time for this report: 19.95 s. Cache events and detailed lower/compile measurements remain in cache_events.json. The same operation graph is reused only on an exact structural fingerprint match. Any new shape/structure compiles separately; no shape is forced. The adapter is pinned to JAX 0.4.35 and is not a production API/refactor. A major runtime-parameter redesign remains deferred pending approval.

| Timing category | N | Mean s | Median s | p95 s | Total s |
|---|---:|---:|---:|---:|---:|
| scenario_truth_generation_seconds | 6 | 0.06687241 | 0.05447402 | 0.1068901 | 0.4012345 |
| scenario_forward_history_seconds_including_compile | 6 | 0.9072608 | 0.850339 | 1.134376 | 5.443565 |
| window_total_seconds | 66 | 0.00325048 | 0.002031563 | 0.00262249 | 0.2145317 |
| arrival_retrieval_seconds | 66 | 0.002135198 | 0.0009356665 | 0.00120875 | 0.1409231 |
| buffer_plus_basis_remainder_seconds | 66 | 0.001115282 | 0.001041709 | 0.001548354 | 0.07360858 |
| objective_factory_seconds | 66 | 0.004809268 | 0.003330396 | 0.004380885 | 0.3174117 |
| landmark_observation_plan_seconds | 66 | 0.0004824577 | 0.0004925205 | 0.0005329375 | 0.03184221 |
| window_NPZ_artifact_seconds | 66 | 0.001523597 | 0.001434541 | 0.002073865 | 0.1005574 |
| projection_seconds | 132 | 1.857639e-05 | 2.41652e-06 | 3.19352e-05 | 0.002452083 |
| clipping_seconds | 132 | 7.925494e-06 | 6.937509e-06 | 1.276062e-05 | 0.001046165 |
| additional_trace_seconds | 132 | 0.0417907 | 0.03637658 | 0.07692241 | 5.516373 |
| final_replay_seconds | 132 | 0.02996694 | 0.03689254 | 0.03810757 | 3.955635 |
| update_seconds | 132 | 12.64626 | 11.88704 | 21.77707 | 1669.307 |
| optimizer_program_trace_seconds | 132 | 0.08811701 | 0.07472619 | 0.1695274 | 11.63145 |
| optimizer_evaluation_lower_seconds | 132 | 0.00125494 | 0 | 0 | 0.1656521 |
| optimizer_gradient_lower_seconds | 132 | 0.01206296 | 0 | 0 | 1.592311 |
| optimizer_evaluation_compile_seconds | 132 | 0.004808759 | 0 | 0 | 0.6347562 |
| optimizer_value_gradient_compile_seconds | 132 | 0.0133645 | 0 | 0 | 1.764114 |
| optimizer_checked_reference_first_call_seconds_including_auxiliary_compilation | 132 | 0.1542116 | 0.1421228 | 0.1582972 | 20.35593 |
| optimizer_compiled_value_gradient_warmup_seconds | 132 | 0.07854112 | 0.08619915 | 0.09304094 | 10.36743 |
| optimizer_preparation_seconds | 132 | 2.447313 | 2.557274 | 2.861203 | 323.0453 |
| optimizer_optimization_seconds | 132 | 9.341987 | 8.597372 | 18.02915 | 1233.142 |
| optimizer_validity_check_seconds_mean | 132 | 0.02912701 | 0.02934767 | 0.03743697 | 3.844766 |
| optimizer_validity_check_seconds_p95 | 132 | 0.02946439 | 0.0309428 | 0.03793417 | 3.889299 |
| optimizer_value_gradient_seconds_mean | 132 | 0.07558055 | 0.07732305 | 0.08767537 | 9.976633 |
| optimizer_value_gradient_seconds_p95 | 132 | 0.07710586 | 0.08572707 | 0.09077714 | 10.17797 |
| trial_checked_objective_seconds | 11451 | 0.0300275 | 0.03651262 | 0.037318 | 343.8449 |
| trial_value_gradient_seconds | 11451 | 0.07681658 | 0.08549842 | 0.08706044 | 879.6267 |
| diagnostic_seconds | 132 | 0.3997154 | 0.3943287 | 0.4678263 | 52.76243 |
| artifact_seconds | 132 | 0.01438038 | 0.01461988 | 0.0168692 | 1.89821 |
| sensitivity_lower_seconds | 132 | 0.004928221 | 0 | 0 | 0.6505252 |
| sensitivity_compile_seconds | 132 | 0.007022052 | 0 | 0 | 0.9269108 |
| sensitivity_execution_seconds | 132 | 0.2807283 | 0.2932065 | 0.3274478 | 37.05613 |
| figure_publication_seconds | 34 | 0.5581225 | 0.5381551 | 0.756883 | 18.97617 |
| host_remeasurement_window_seconds | 198 | 0.001115202 | 0.000909 | 0.001075138 | 0.22081 |
| host_remeasurement_arrival_seconds | 198 | 0.0002907654 | 0.000284167 | 0.000344642 | 0.05757155 |
| host_remeasurement_basis_seconds | 198 | 0.0007476718 | 0.000554833 | 0.00059905 | 0.148039 |
| host_remeasurement_buffer_other_seconds | 198 | 7.676511e-05 | 7.0668e-05 | 0.0001074662 | 0.01519949 |
| JSON_serialization_remeasurement_seconds | 6 | 0.04278151 | 0.04466596 | 0.05094111 | 0.256689 |

NPZ artifact timings are from the fit run; JSON serialization timings are separate post-run remeasurements. Figure-publication samples include layout/render/write/close for PNG and PDF; the overall plot timer also includes figure construction. Cold compilation appears as nonzero events once per new structural program, with zeros for cache hits; medians can therefore be zero. One structural replay graph still has separate evaluator, value/gradient, sensitivity and checked-AD auxiliary executables. It does not mean a single XLA compilation for all operations.

## Representative window and time-series figures

### forward_constant

![forward_constant arrival and estimation](forward_constant/pose_map_arrival.png)

![forward_constant sensitivity](forward_constant/sensitivity.png)

![forward_constant window 0](forward_constant/window_000_disturbances_observations.png)

![forward_constant window 5](forward_constant/window_005_disturbances_observations.png)

![forward_constant window 10](forward_constant/window_010_disturbances_observations.png)

### forward_smooth

![forward_smooth arrival and estimation](forward_smooth/pose_map_arrival.png)

![forward_smooth sensitivity](forward_smooth/sensitivity.png)

![forward_smooth window 0](forward_smooth/window_000_disturbances_observations.png)

![forward_smooth window 5](forward_smooth/window_005_disturbances_observations.png)

![forward_smooth window 10](forward_smooth/window_010_disturbances_observations.png)

### forward_zero

![forward_zero arrival and estimation](forward_zero/pose_map_arrival.png)

![forward_zero sensitivity](forward_zero/sensitivity.png)

![forward_zero window 0](forward_zero/window_000_disturbances_observations.png)

![forward_zero window 5](forward_zero/window_005_disturbances_observations.png)

![forward_zero window 10](forward_zero/window_010_disturbances_observations.png)

### regression_constant

![regression_constant arrival and estimation](regression_constant/pose_map_arrival.png)

![regression_constant sensitivity](regression_constant/sensitivity.png)

![regression_constant window 0](regression_constant/window_000_disturbances_observations.png)

![regression_constant window 5](regression_constant/window_005_disturbances_observations.png)

![regression_constant window 10](regression_constant/window_010_disturbances_observations.png)

### regression_smooth

![regression_smooth arrival and estimation](regression_smooth/pose_map_arrival.png)

![regression_smooth sensitivity](regression_smooth/sensitivity.png)

![regression_smooth window 0](regression_smooth/window_000_disturbances_observations.png)

![regression_smooth window 5](regression_smooth/window_005_disturbances_observations.png)

![regression_smooth window 10](regression_smooth/window_010_disturbances_observations.png)

### regression_zero

![regression_zero arrival and estimation](regression_zero/pose_map_arrival.png)

![regression_zero sensitivity](regression_zero/sensitivity.png)

![regression_zero window 0](regression_zero/window_000_disturbances_observations.png)

![regression_zero window 5](regression_zero/window_005_disturbances_observations.png)

![regression_zero window 10](regression_zero/window_010_disturbances_observations.png)


## Feasible terminal-coefficient perturbation diagnostic

These fixed-model evaluations neither reoptimize nor replace any reported solution. They directly compare cost and reported endpoint state under feasible coefficient changes.

| Coefficient | Delta | Absolute cost change | Max q change | Endpoint pose change (x,y,theta) |
|---:|---:|---:|---:|---|
| 6 | 0.1 | 0 | 0 | [0.0031720315023511247, 0.004124044968787377, 0.0] |
| 13 | 0.1 | 0 | 0 | [-0.0014041106286670768, 0.0010551793402111542, 0.005208333333333037] |

The final forward/constant window is [1,6] s. Its latest correction is at sigma=3.6 s, while coefficient j=6 has support beginning at sigma=3.75 s. Both channel-j6 columns of D are exactly zero and the numerical rank is 12. The feasible probes above leave all correction residuals and J unchanged while changing the endpoint pose. Thus these terminal input components are unconstrained by the recorded innovation objective in this particular window; warm/zero differences in that interval need not signal a failed optimizer. This finding is specific to the saved observation schedule. Other cases have larger endpoint errors with full local rank and observed endpoint scans, so this mechanism cannot explain all boundary error.

## Numerical validation, preservation and reproduction

Complete repository test result: 389 passed, exit 0. Initial diagnostic tests and full regression stdout/XML are retained. The tests include Stage 5A projection, Stage 5B windows/births/empty scans, Stage 4C regression behavior, independent profile truth/time checks, covariance, exact compiled outputs/gradients, whitening finite differences and optimizer-orchestration identity. Selected full-size FD and repeat-fit checks are in saved_validation.json.

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q --junitxml=artifacts/milestone5c/pytest.xml -o junit_family=legacy
```

All 689 pre-task protected files preserved: True. Changed protected files: []. Stage 4C, 5A, 5B artifacts and all original mathematical implementations are unchanged.

```json
{
  "all_six_cases": true,
  "all_eleven_windows": true,
  "independent_truth_and_coordinates": true,
  "births_masks_empty_scans": true,
  "whitened_cost_identity": true,
  "covariance_finite_symmetric": true,
  "first_window_reproducible": true,
  "maximum_independent_truth_difference": 0.0,
  "maximum_whitened_cost_difference": 2.8990143619012088e-12,
  "all_previous_files_preserved": true,
  "regression_tests_passed": true,
  "selected_sensitivity_FD_and_reproducibility": true,
  "direct_host_profile_saved_windows_equal": true
}
```

```sh
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.stage5c_feasibility
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c --validate
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.stage5c_host_profile
PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c --report-only
```

## Recommended next steps and stopping condition

Review the unconstrained terminal-input directions, noise-fit evidence and function-reduction termination before changing estimator design. The saved feasible probes provide a reproducible test case for any future proposal concerning poorly informed terminal inputs. Review diagnostic parameter scaling and profile a longer deterministic duration before selecting a future paired seed/Monte Carlo budget; such runs require approval and are not performed here. If production compilation reuse is desired, approve a separate runtime-argument API change rather than adopting the pinned internal-JAX benchmark adapter without review. No Monte Carlo, nearest-neighbor association, Victoria Park, covariance tuning, regularization or objective change was started. Work stops after this diagnostic stage.
