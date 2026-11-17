"""Audit and compare explicit Stage 4C sensors without overwriting old results."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from benchmarks.milestone4b_smooth import write_json
from benchmarks.milestone4c import configuration, prepare_a, prepare_b, fit_configuration
from benchmarks.slam_window_plots import plot_experiment, plot_slam_map
from estimation.slam import INITIALIZED
from simulation.slam_window import sensor_settings, simulate_slam_window


def initialization_times(times, statuses, ids):
    """Read actual birth records; never infer births from the truth map size."""
    return {str(int(ids[node,slot])):float(times[node])
            for node,slot in np.argwhere(statuses == INITIALIZED)}


def capture_preservation(output):
    """Snapshot protected inputs once, including original Stage 4C artifacts."""
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'before_sha256.json'
    if not manifest.exists():
        paths=[]
        for root in ('baseline','splines','estimation','simulation','benchmarks','tests','artifacts/milestone4c'):
            paths.extend(p for p in Path(root).rglob('*') if p.is_file()
                         and '__pycache__' not in p.parts and p.name!='.DS_Store')
        paths.extend(p for p in Path('../bsplines_library').rglob('*') if p.is_file()
                     and p.suffix in ('.cpp','.hpp','.h') and not any(x in p.parts for x in ('.git','build')))
        write_json(manifest,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def run_scenario(output, scenario, kind):
    output.mkdir(parents=True, exist_ok=True)
    cfg = configuration(kind)
    cfg['sensor'] = sensor_settings(scenario)
    cfg['observation_policy'] = 'true-pose inclusive range/FOV gate before noise; fixed ID order; no occlusion or detection dropout'
    data, truth_check = simulate_slam_window(cfg)
    arrival_a, window, ids, adapter, basis, prefix = prepare_a(cfg,data)
    stride = round(cfg['measurement_interval_seconds']/cfg['integration_step_seconds'])
    scheduled = np.arange(stride,len(data.times),stride)
    observed_ids = np.broadcast_to(np.arange(len(data.landmark_map)),data.mask.shape)
    split = len(prefix.states.x)-1
    prefix_births = initialization_times(data.times[:split+1],np.asarray(prefix.observations.status),observed_ids[:split+1])
    observation_audit = {
        'sensor':cfg['sensor'], 'scheduled_scan_count':len(scheduled),
        'total_observations':int(data.mask.sum()),
        'pre_window_observations':int(data.mask[:split+1].sum()),
        'window_observations':int(window.mask.sum()),
        'empty_scheduled_scans':int(np.sum(~data.mask[scheduled].any(axis=1))),
        'pre_window_initialization_times':prefix_births,
        'scans':[{'time_seconds':float(data.times[i]),'observed_ids':observed_ids[i,data.mask[i]].tolist()}
                 for i in scheduled]}
    write_json(output/f'observations_{kind}.json',observation_audit)
    np.savez_compressed(output/f'dataset_{kind}.npz',**data._asdict(),observation_ids=observed_ids)
    np.savez_compressed(output/f'initialization_{kind}_history.npz',
        **{key:np.asarray(value) for key,value in prefix.states._asdict().items()},
        observation_status=np.asarray(prefix.observations.status),times=data.times[:split+1])
    results={}
    for config,arrival in [('a',arrival_a),('b',prepare_b(cfg))]:
        label=config+'_'+kind
        record=fit_configuration(cfg,arrival,window,ids,adapter,basis,output,label)
        with np.load(output/f'{label}_histories.npz') as saved:
            arrays={key:saved[key] for key in saved.files}
        arrays['scan_scheduled']=(np.arange(split,len(data.times)) % stride == 0)
        arrays['scan_scheduled'][0]=False
        np.savez_compressed(output/f'{label}_histories.npz',**arrays)
        births=initialization_times(window.times,arrays['baseline_statuses'],ids)
        record['initialization_times_seconds']={**(prefix_births if config=='a' else {}),**births}
        record['unmapped_landmark_ids']=sorted(set(range(len(data.landmark_map)))-set(map(int,record['initialization_times_seconds'])))
        record['observations']=observation_audit
        record['truth_validation']=truth_check
        record['figures']=plot_experiment(output,label,arrays,data.truth,data.times)
        record['figures']+=plot_slam_map(output,label,arrays,cfg['sensor'])
        record['regression_reference']=None
        if scenario=='regression':
            with np.load(Path('artifacts/milestone4c')/f'dataset_{kind}.npz') as old:
                for field,value in data._asdict().items():
                    np.testing.assert_array_equal(value,old[field])
            old=json.loads((Path('artifacts/milestone4c')/f'{label}_results.json').read_text())
            differences={}
            for estimator in ('baseline_b','optimized_mhe'):
                differences[estimator]=abs(record[estimator]['objective']-old[estimator]['objective'])
                np.testing.assert_allclose(record[estimator]['objective'],old[estimator]['objective'],rtol=1e-10,atol=1e-8)
            record['regression_reference']={'dataset_bitwise_identical':True,'objective_absolute_differences':differences}
        write_json(output/f'{label}_results.json',record)
        results[label]=record
    return results


def summarize(output):
    records={}
    for scenario in ('regression','forward'):
        for path in sorted((output/scenario).glob('*_results.json')):
            records[scenario+'/'+path.stem.removesuffix('_results')]=json.loads(path.read_text())
    checks={
        'eight_cases_present':len(records)==8,
        'all_case_numerical_checks':bool(records) and all(all(r['validation_checks'].values()) for r in records.values()),
        'paired_datasets_across_arrivals':all(
            records[s+'/a_'+k]['dataset_sha256']==records[s+'/b_'+k]['dataset_sha256']
            for s in ('regression','forward') for k in ('constant','smooth')
            if s+'/a_'+k in records and s+'/b_'+k in records),
        'regression_datasets_bitwise_identical':all(r['regression_reference']['dataset_bitwise_identical']
            for name,r in records.items() if name.startswith('regression/')),
        'record_accounting':all(r['baseline_b']['requested_observations']==r['observations']['window_observations']
            and r['optimized_mhe']['valid_observations']==r['observations']['window_observations']
            and r['optimized_mhe']['initialization_observations']+r['optimized_mhe']['correction_observations']==r['observations']['window_observations']
            and r['baseline_b']['active_landmarks_final']==len(r['initialization_times_seconds']) for r in records.values()),
        'all_figures_present':all(len(r['figures'])==12 and all((output/name.split('/')[0]/p).is_file()
                                  for p in r['figures']) for name,r in records.items())}
    write_json(output/'validation.json',checks)
    lines=['# Stage 4C visibility audit and SLAM maps','',
        'These are paired seed-7 fixed-window checks, with two identical zero-start fits per case. '
        'They are not Monte Carlo trials or a Milestone 5 result. Full audit and assumptions: '
        '[STAGE4C_VISIBILITY.md](../../estimation/STAGE4C_VISIBILITY.md).','',
        'Regression sensor: unlimited range, 360° FOV. Illustrative forward sensor: '
        '0.2–6 m inclusive range, 180° FOV centered on robot heading. Both have zero extrinsics, '
        'no occlusion and no random detection dropout. Visibility uses noiseless true geometry. '
        'The same full-slot seeded noise stream pairs shared observations across scenarios.','',
        'Configuration A uses only valid measurements at 0.1–2 s to construct its full arrival posterior. '
        'Configuration B propagates the original robot prior to 2 s without observations and has an empty map. '
        'Both replay exactly the same records strictly after 2 s. No map prior is supplied.','',
        '| Case | Pre/window records | Arrival/final mapped | Birth times (ID: seconds) | Unmapped IDs |',
        '|---|---:|---:|---|---|']
    for name,r in records.items():
        b=r['baseline_b'];o=r['observations']
        lines.append(f"| {name} | {o['pre_window_observations']}/{o['window_observations']} | "
                     f"{b['active_landmarks_at_arrival']}/{b['active_landmarks_final']} | {r['initialization_times_seconds']} | {r['unmapped_landmark_ids']} |")
    lines+=['','| Case | Birth/correction records in window | Empty scheduled scans / total | Used-observation coverage |',
            '|---|---:|---:|---:|']
    for name,r in records.items():
        b=r['baseline_b'];o=r['observations']
        coverage=b['valid_observations']/b['requested_observations'] if b['requested_observations'] else 0.
        lines.append(f"| {name} | {b['initialization_observations']}/{b['correction_observations']} | "
                     f"{o['empty_scheduled_scans']}/{o['scheduled_scan_count']} | {coverage:g} |")
    lines+=['','Birth times come from recorded INITIALIZED statuses, not supplied map coordinates. '
            'Observation JSON files list actual IDs at every scheduled scan, including empty scans. '
            'Covariance ellipses and map markers only include active estimated landmarks.','',
            '| Case | Estimator | Position RMSE (m) | Heading RMSE (rad) | Map RMSE (m) | w_v RMSE (m/s) | w_omega RMSE (rad/s) | J |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,r in records.items():
        for estimator in ('baseline_b','optimized_mhe'):
            m=r[estimator];d=m['disturbance']['rmse_v_omega'];loc=m['localization']
            lines.append(f"| {name} | {estimator} | {loc['position_rmse_m']:.7g} | {loc['heading_rmse_rad']:.7g} | "
                         f"{m['final_landmark_rmse_m']:.7g} | {d[0]:.7g} | {d[1]:.7g} | {m['objective']:.9g} |")
    lines+=['','Errors use the original anchored frame with no post-fit alignment. Robot RMSE uses all 201 window nodes; '
            'map RMSE uses only mapped IDs at 7 s, so differing map coverage must accompany comparisons. '
            'Disturbance RMSE uses 501 elapsed samples. J sums correction innovations only; births contribute no term.','',
            '## Effect of reduced visibility','']
    for config in ('a','b'):
        for kind in ('constant','smooth'):
            key=config+'_'+kind
            if 'regression/'+key in records and 'forward/'+key in records:
                old=records['regression/'+key]['optimized_mhe']['disturbance']['rmse_v_omega']
                new=records['forward/'+key]['optimized_mhe']['disturbance']['rmse_v_omega']
                lines += [f"- {key}: fitted linear/angular disturbance RMSE changes from {old} to {new}."]
    lines+=['','Reduced visibility removes correction constraints and changes map coverage. '
            'Lower innovation cost across different sensors is not an accuracy comparison. '
            'Weak constraints and measurement noise can harm disturbance recovery or pose/map accuracy. '
            'No noise, bounds, spline parameters or optimizer tolerances were tuned for the forward scenario.','']
    for name,r in records.items():
        b,m=r['baseline_b'],r['optimized_mhe']
        worse=[key for key in ('position_rmse_m','heading_rmse_rad') if m['localization'][key]>b['localization'][key]]
        if m['final_landmark_rmse_m']>b['final_landmark_rmse_m']:
            worse.append('final_landmark_rmse_m')
        if worse:
            lines += [f"- {name}: fitting lowers J but worsens {', '.join(worse)} relative to Baseline B."]
    lines+=['',
            '## Figures and validation','',
            'Black crosses are all true landmark references. Blue and orange markers/ellipses are active final estimates; '
            'triangles identify pre-window births and circles in-window births. Green marks the window start. '
            'The gray footprint and sight lines use the labeled true sensor pose and actual recorded noisy range/bearing; '
            'squares identify observed IDs, gray circles IDs mapped before that scan. '
            'Robot 95% marginal ellipses occur at 2, 3, 4, 5, 6, 7 s; landmark ellipses are at 7 s. '
            '**Conditional EKF covariance; spline coefficients treated as fixed.**','']
    for name,r in records.items():
        lines += [f"![{name} combined SLAM map]({name}_slam_map.png)",'']
    lines+=['All per-case JSON files retain zero-replay equivalence to Baseline B (absolute tolerance 2e-14), '
            'all-coefficient central-difference checks (float64; steps 1e-4 and 1e-5), optimizer status/timing, '
            'observation/birth/correction counts, and exact repeated-fit checks. '
            'Regression dataset fields are compared bitwise to the original Stage 4C datasets. '
            'Original artifacts and preserved implementation hashes are checked in preservation.json.','',
            '```sh',
            'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone4c_visibility',
            'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m pytest -q tests/test_slam_visibility.py tests/test_slam_window_a.py tests/test_slam_window_b.py tests/test_milestone4c_reporting.py',
            '```','', 'Next validation gate: resolve the already documented Milestone 5 arrival/birth/warm-start research decisions.']
    test_path=output/'test_record.json'
    if test_path.exists():
        test=json.loads(test_path.read_text())
        lines+=['',f"Test outcome: {test['passed']} passed, exit code {test['exit_code']}; "
                f"see [test_record.json](test_record.json) and [pytest_stdout.txt](pytest_stdout.txt).",'',
                '```sh',test['command'],'```']
    lines+=['','```json',json.dumps(checks,indent=2),'```']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'results.json',records)
    manifest=json.loads((output/'before_sha256.json').read_text())
    changed=[p for p,digest in manifest.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest]
    allowed={'simulation/slam_window.py','benchmarks/milestone4c.py','benchmarks/slam_window_plots.py'}
    write_json(output/'preservation.json',{'checked_count':len(manifest),'changed':changed,
        'allowed_changes':sorted(allowed),'all_protected_preserved':set(changed)<=allowed})
    assert set(changed)<=allowed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/milestone4c_visibility'))
    parser.add_argument('--scenario',choices=('regression','forward','all'),default='all')
    parser.add_argument('--disturbance',choices=('constant','smooth','all'),default='all')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    capture_preservation(args.output)
    jax.config.update('jax_enable_x64',True)
    if not args.report_only:
        for scenario in (('regression','forward') if args.scenario=='all' else (args.scenario,)):
            for kind in (('constant','smooth') if args.disturbance=='all' else (args.disturbance,)):
                run_scenario(args.output/scenario,scenario,kind)
                jax.clear_caches()
    summarize(args.output)


if __name__=='__main__':
    main()
