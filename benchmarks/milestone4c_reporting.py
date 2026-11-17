"""Evidence-based reporting of the staged fixed-window SLAM experiments."""

import json

import numpy as np


def completion_checks(payload):
    experiments=payload['experiments']
    required={'a_constant','b_constant','a_smooth','b_smooth'}
    return {
        'required_experiments_present':required <= experiments.keys(),
        'configuration_a_validated_before_b':payload['configuration_a_gate']['validated_before_configuration_b_implementation']
            and payload['configuration_a_gate']['test_exit_code']==0,
        'constant_b_validated_before_smooth':payload['configuration_b_constant_gate']['validated_before_smooth_extension']
            and payload['configuration_b_constant_gate']['test_exit_code']==0,
        'all_numerical_experiment_checks':bool(experiments) and all(all(r['validation_checks'].values()) for r in experiments.values()),
        'arrival_and_update_accounting':required <= experiments.keys() and all(
            r['baseline_b']['requested_observations']==300 and r['optimized_mhe']['valid_observations']==300
            and r['optimized_mhe']['initialization_observations']==(0 if name.startswith('a_') else 6)
            and r['optimized_mhe']['correction_observations']==(300 if name.startswith('a_') else 294)
            for name,r in experiments.items()),
        'identical_datasets_across_configurations':required <= experiments.keys() and all(
            experiments['a_'+kind]['dataset_sha256']==experiments['b_'+kind]['dataset_sha256'] for kind in ('constant','smooth')),
        'independent_truth_validated':bool(experiments) and all(r['truth_validation']['passed'] for r in experiments.values()),
        'all_previous_protected_files_preserved':payload['preservation']['all_preserved'],
        'current_complete_test_suite':payload['current_test_record'] is not None
            and payload['current_test_record'].get('exit_code')==0
            and payload['current_test_record'].get('scope')=='complete repository suite',
        'all_figures_generated':bool(experiments) and all(len(r.get('figures',[]))==10 for r in experiments.values()),
    }


def write_report(output,payload):
    experiments=payload['experiments']
    checks=completion_checks(payload)
    verdict=all(checks.values())
    lines=['# Stage 4C — fixed-window input-spline MHE-SLAM','',
        f"Controlled Stage 4C completion: **{'SATISFIED' if verdict else 'NOT YET SATISFIED'}**.",'',
        'The validated Stage 4B checked L-BFGS-B optimizer now reruns the preserved augmented Stage 3B.2 EKF-SLAM. '
        'Only 14 input-spline coefficients are optimized; there is no state-trajectory optimization. '
        'These are paired single-seed known-association experiments, not Monte Carlo or moving-window benchmarks.','',
        '## Reproduce','', '```sh',
        'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4c --test-record artifacts/milestone4c/test_record.json',
        '```','',
        'The default run repeats both configurations on constant and smooth truth. Saved intermediate gate records '
        'document the initial A-first validation and the B-constant gate before smooth experiments. '
        'The supplied test record describes a separately executed suite; the benchmark does not execute pytest.','',
        '## Configuration and independent truth','',
        'All shared constants follow Stage 4B: six landmarks, original robot prior, R and Qc, '
        'v=1 m/s, omega=0.2 rad/s, seed 7, degree 3, M=4, seven coefficients per disturbance channel. '
        'Estimator RK4 step is 0.025 s; every landmark is observed in fixed order every 0.1 s. '
        'The simulated duration is seven seconds. Disturbances are zero during the first two seconds. '
        'The optimization window is [2,7] s, with sigma=time-2 and spline scale M/L=0.8. '
        'Stage times are evaluated through the unchanged C++ adapter, including the exact right endpoint. '
        'All commanded inputs are evaluated by the original RK4 at every stage.','',
        'Constant post-arrival truth is [0.15,-0.05]. Smooth post-arrival truth is '
        '[0.15 sin(2 pi sigma/5), 0.05 cos(2 pi sigma/5)], outside the exact cubic spline family. '
        'Truth uses independent NumPy analytic trajectories for constant-input phases and independent SciPy DOP853 '
        'for the smooth phase (max step .05 s, rtol=1e-11, atol=1e-13). A split DOP853 reference uses '
        'max step .025 s, rtol=1e-12 and atol=1e-14, with an acceptance bound of 1e-10 per pose component. '
        'The disturbance discontinuity is an integration boundary. No EKF, JAX RK4 or spline generates truth. '
        'Qc remains an estimator uncertainty assumption; truth has deterministic inputs and seeded measurement noise.','',
        '## Arrival and observation accounting','',
        '**Configuration A:** the preserved zero-disturbance EKF-SLAM consumes observations at 0.1 through 2.0 s. '
        'Six initializations and 114 corrections construct the uncertain map. The complete posterior at 2.0 s, '
        'including every covariance cross-correlation and ID/slot mapping, is frozen. All six landmarks are active. '
        'Only observations strictly after 2.0 s enter the optimization replay: 300 corrections.','',
        '**Configuration B:** the original t=0 robot prior and Qc are propagated to 2.0 s using commanded inputs '
        'and zero disturbance, consuming no observations. This observation-free robot prior has an empty map; '
        'it is not A’s posterior marginal. The same observations strictly after 2.0 s are used. '
        'The six observations at 2.1 s initialize the map; the remaining 294 correct it. '
        'The original pose prior anchors the reference frame; errors are reported without post-hoc rigid alignment.','',
        'Every candidate restarts the identical captured arrival, covariance, active mask and integer ID map. '
        'An offline plan fixes birth/update/slot decisions from arrival metadata, IDs and requested masks. '
        'Truth coordinates never initialize the estimator. Repeated IDs at one timestamp initialize once and '
        'then correct sequentially. Births consume their measurement once, contribute no innovation cost, and '
        'retain differentiation through their inverse-model mean, covariance and cross-correlations.','',
        '## Objective and invalid-candidate policy','',
        'J=sum(e.T solve(S,e)) includes exactly the fixed requested correction set. Log determinant and '
        'J+logdet are separate diagnostics (the latter omits the Gaussian dimension constant). '
        'No regularization, logdet optimization, finite failure penalties, covariance clipping, jitter or repairs '
        'are introduced. All 14 coefficients start at zero with bounds [-0.5,0.5] and unchanged Stage 4B options. '
        'Every requested birth and correction must be valid. Node priors and every sequential posterior '
        'must be finite with symmetric PSD active covariance (absolute tolerance 1e-10); inactive storage stays '
        'exactly neutral. Measurement noise and correction S must be SPD. Invalid candidates have undefined '
        'cost and the checked optimizer stops on the first invalid preflight/trial without a retry or changed data.','',
        '## Baseline B versus fitted MHE-SLAM','',
        '| Configuration / truth | Estimator | J | w_v RMSE (m/s) | w_omega RMSE (rad/s) | Position RMSE (m) | Heading RMSE (rad) | Final position (m) | Final signed heading (rad) | Landmark RMSE (m) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,run in experiments.items():
        for estimator in ('baseline_b','optimized_mhe'):
            r=run[estimator];loc=r['localization'];d=r['disturbance']['rmse_v_omega']
            lines.append(f"| {name} | {estimator} | {r['objective']:.10g} | {d[0]:.8g} | {d[1]:.8g} | "
                f"{loc['position_rmse_m']:.8g} | {loc['heading_rmse_rad']:.8g} | {loc['final_position_error_m']:.8g} | "
                f"{loc['final_heading_error_rad']:.8g} | {r['final_landmark_rmse_m']:.8g} |")
    lines+=['','Pose RMSE uses all 201 posterior integration nodes including arrival; heading errors wrap. '
        'Disturbance RMSE and maximum errors use the same 501 uniform elapsed samples including endpoints. '
        'Landmark RMSE uses active known IDs in the anchored frame. Full maximum errors and all diagnostic '
        'poses/covariances are stored in results.json and per-experiment files.','']
    for name,run in experiments.items():
        before,after=run['baseline_b'],run['optimized_mhe']
        change={metric:100*(after['localization'][metric]/before['localization'][metric]-1)
                for metric in ('position_rmse_m','heading_rmse_rad')}
        lines += [f"**{name}:** objective reduction {before['objective']-after['objective']:.8g}; "
            f"position/heading RMSE changes `{change}` percent. Negative percentages indicate improvement.",'']
    lines += ['No test requires MHE-SLAM to beat Baseline B in every metric. A fit can lower innovation cost while '
        'fitting measurement/map noise or exploiting weak coefficient directions. The true constant coefficients '
        'need not minimize a finite noisy-data objective or the anchored map/pose error.','',
        '| Experiment | Reference coefficients | Reference J | Initial J | Optimized J | logdet (baseline / fitted) | J+logdet (baseline / fitted) |',
        '|---|---|---:|---:|---:|---|---|']
    for name,run in experiments.items():
        kind='true constant' if name.endswith('constant') else 'independent sampled LS (diagnostic only)'
        key='true_coefficient_diagnostic' if name.endswith('constant') else 'least_squares_coefficient_diagnostic'
        b,m=run['baseline_b'],run['optimized_mhe']
        lines.append(f"| {name} | {kind} | {run[key]['objective']:.10g} | {b['objective']:.10g} | {m['objective']:.10g} | "
            f"{b['logdet_diagnostic']:.10g} / {m['logdet_diagnostic']:.10g} | {b['gaussian_diagnostic']:.10g} / {m['gaussian_diagnostic']:.10g} |")
    lines += ['','Smooth reference coefficients fit the known analytic inputs with the C++ basis on the dense '
        'grid; they never initialize optimization or supply inputs to the EKF.','',
        '## Observation counts and covariance checks','',
        '| Experiment | Active at arrival / final | New landmarks during window | Corrections | Requested / valid / invalid | Baseline min active eigenvalue / asymmetry | Fitted min active eigenvalue / asymmetry |',
        '|---|---|---:|---:|---|---|---|']
    for name,run in experiments.items():
        b,m=run['baseline_b'],run['optimized_mhe'];bc,mc=b['covariance'],m['covariance']
        lines.append(f"| {name} | {m['active_landmarks_at_arrival']} / {m['active_landmarks_final']} | {m['initialization_observations']} | "
            f"{m['correction_observations']} | {m['requested_observations']} / {m['valid_observations']} / {m['invalid_observations']} | "
            f"{bc['minimum_active_subspace_eigenvalue']:.5g} / {bc['maximum_covariance_asymmetry']:.3g} | "
            f"{mc['minimum_active_subspace_eigenvalue']:.5g} / {mc['maximum_covariance_asymmetry']:.3g} |")
    lines+=['','Reported minima/asymmetries are over posterior integration nodes in the active subspace. '
        'The objective additionally checks node priors and every sequential posterior. Inactive storage is '
        'exactly zero and excluded from active eigenvalue metrics.','',
        '## Optimization, gradients and timing','',
        '| Experiment | Status / iterations | Function / gradient calls | Raw gradient L2 / inf | Projected gradient inf | Active coefficient bounds (zero-based) |',
        '|---|---|---|---|---:|---|']
    for name,run in experiments.items():
        opt=run['optimization']
        active=[i for i,c in enumerate(opt['coefficients']) if abs(abs(c)-.5)<1e-10]
        lines.append(f"| {name} | {opt['status']} / {opt['iterations']} | {opt['function_evaluations']} / {opt['gradient_evaluations']} | "
            f"{opt['gradient_l2']:.6g} / {opt['gradient_inf']:.6g} | {opt['projected_gradient_inf']:.6g} | {active} |")
    for name,run in experiments.items():
        opt=run['optimization']
        lines += ['',f"{name} termination: `{opt['message']}`.",'']
    lines+=['The stopping criteria are ftol=1e-12, gtol=1e-6, maxiter=500, maxfun=2000, maxls=40 and maxcor=10. '
        'Function-reduction convergence can precede the gradient target; active constraints change projected '
        'gradients. Neither convergence nor a lower J proves global optimality, identifiability or uncertainty '
        'calibration. Raw/projected zero-reference gradient norms are diagnostic in results.json; Baseline B '
        'has zero optimizer iterations and function/gradient optimization calls.','',
        '| Experiment / evaluator | Evaluation compile (s) | Value/gradient compile (s) | Cold checked reference incl. auxiliary compilation (s) | Steady value/gradient mean / p95 (s) | Steady checked replay mean / p95 (s) | Warm optimization (s) | Preparation + optimization (s) |',
        '|---|---:|---:|---:|---|---|---:|---:|']
    for name,run in experiments.items():
        for evaluator in ('baseline_b','optimized_mhe'):
            t=run['baseline_timing'] if evaluator=='baseline_b' else run['optimization']['timing']
            lines.append(f"| {name} / {evaluator} | {t['evaluation_compile_seconds']:.6g} | {t['value_gradient_compile_seconds']:.6g} | "
                f"{t['checked_reference_first_call_seconds_including_auxiliary_compilation']:.6g} | {t['value_gradient_seconds_mean']:.6g} / {t['value_gradient_seconds_p95']:.6g} | "
                f"{t['validity_check_seconds_mean']:.6g} / {t['validity_check_seconds_p95']:.6g} | "
                f"{0. if evaluator=='baseline_b' else t['optimization_seconds']:.6g} | "
                f"{'not optimized' if evaluator=='baseline_b' else format(run['total_fitting_seconds_including_preparation'],'.6g')} |")
    lines+=['','Compilation is explicit lower/compile time without execution; the cold checked reference can '
        'include auxiliary VJP compilation. Warm optimization excludes preparation but checks every trial. '
        'The zero-reference objective/gradient timings use ten repeated evaluations; these diagnostic gradients '
        'are not part of Baseline B estimation. These are fixed-window offline costs, not online update latency.','',
        '| Experiment | Largest absolute FD discrepancy | Precision / FD steps | Repeated coefficients / accepted history |',
        '|---|---:|---|---|']
    for name,run in experiments.items():
        maximum=max(row['max_absolute_error'] for point in run['gradient_validation']['results'].values() for row in point.values())
        lines.append(f"| {name} | {maximum:.5g} | float64 / 1e-4, 1e-5 | {run['reproducibility']} |")
    lines+=['','Every coefficient is checked at zero, a nonconstant vector and the fitted vector. Absolute/relative '
        'tolerances are 2e-5 with a separate maximum absolute discrepancy <2e-5; every perturbation must be valid. '
        'Focused tests separately differentiate initialized landmark means, marginal covariances and cross-correlations '
        'through their first observation. Deterministic repeats are not Monte Carlo trials.','',
        '## Covariance figures and interpretation','',
        '**Conditional EKF covariance; spline coefficients treated as fixed.** The optimized EKF covariance '
        'does not include uncertainty in the fitted spline coefficients. The 95% position ellipses use '
        'the 2x2 marginal blocks and chi-squared threshold 5.991. Error envelopes are componentwise ±2 standard '
        'deviations, not simultaneous confidence regions. Every plotted history/marginal is checked; invalid '
        'covariances are rejected without projection or clipping. The t0 prior anchors the frame.','']
    for name in experiments:
        lines += [f'### {name}','']
        for suffix in ('trajectory_covariance','landmark_covariance','errors_uncertainty','disturbances','objective_history'):
            lines += [f'![{name} {suffix}]({name}_{suffix}.png)','']
    lines += ['Each figure also has a vector PDF. Triangle landmarks were initialized before the window; '
        'circle landmarks were initialized during it. Robot ellipses appear at six regularly spaced node times.','',
        '## Saved evidence, tests and completion gate','',
        'configuration.json, results.json and per-experiment JSON retain full numerical results and timings. '
        'dataset_constant.npz/dataset_smooth.npz save seven-second truth and observations. Per-experiment '
        '*_histories.npz files save times, pose/map/covariance/active/ID histories, observation masks and slots, '
        'innovations, dense spline values and accepted optimization costs. Separate *_optimization_history.json '
        'files retain accepted and trial evaluations. Initialization histories retain the pre-window map posterior. '
        'Intermediate gate JSON/XML files prove the requested development order.','']
    test=payload['current_test_record']
    if test:
        lines+=['```sh',test['command'],'```','',f"Actual complete-suite outcome: **{test['passed']} passed**, exit code {test['exit_code']}. "
            'See [test_record.json](test_record.json), pytest.xml and pytest_stdout.txt.','']
    else:
        lines+=['No current complete-suite record was supplied; completion cannot yet be claimed.','']
    lines += [f"All {payload['preservation']['checked_count']} pre-existing protected files match their original SHA-256 hashes; "
        'see [preservation.json](preservation.json). Existing mathematical implementations and earlier artifacts are unchanged. '
        'Implementation choices and limitations are documented in [validation.md](validation.md).','',
        '```json',json.dumps(checks,indent=2),'```','',
        ('Stage 4C’s controlled completion criterion is satisfied. ' if verdict else 'Stage 4C still has outstanding evidence checks. ')+
        'Before Milestone 5, decide how a sliding-window arrival posterior is formed without reusing measurements, '
        'how map births crossing the boundary are accounted for, and how coefficients are shifted/warm-started. '
        'These decisions are not implemented here. Broader disturbance/noise/sparsity tests, statistical calibration '
        'and paired Monte Carlo comparisons remain future validation. Nearest-neighbor association and Victoria Park '
        'remain deferred.','']
    (output/'report.md').write_text('\n'.join(lines))
