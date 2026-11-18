"""Small paired seed-7 moving-horizon validation runs and runtime profiling."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.milestone4c import configuration,covariance_summary
from estimation.fixed_window_objective import InvalidCandidateError
from estimation.moving_horizon import ReplayData,ForwardArrivals,MovingHorizonSLAM,initial_guess
from estimation.slam import create_state,filter_slam,SlamState
from simulation.moving_horizon import simulate_moving,true_disturbance
from simulation.slam_window import sensor_settings
from splines import UniformBasis
from splines.warm_start import make_warm_start_projection


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def errors(states,truth,landmarks):
    pose=np.asarray(states.x)[:,:3]-truth
    pose[:,2]=(pose[:,2]+np.pi)%(2*np.pi)-np.pi
    landmark_errors=[];counts=[]
    for x,active,ids in zip(*(np.asarray(getattr(states,k)) for k in ('x','active','truth_ids'))):
        e=x[3:].reshape(-1,2)[active]-landmarks[ids[active]]
        landmark_errors.extend(np.sum(e**2,axis=1).tolist());counts.append(int(active.sum()))
    return {'position_rmse_m':float(np.sqrt(np.mean(np.sum(pose[:,:2]**2,axis=1)))),
        'heading_rmse_rad':float(np.sqrt(np.mean(pose[:,2]**2))),
        'landmark_rmse_m':float(np.sqrt(np.mean(landmark_errors))) if landmark_errors else None,
        'mapped_landmark_counts':counts}


def stats(values):
    a=np.asarray(values)
    return {'mean':float(np.mean(a)),'p95':float(np.percentile(a,95)),'total':float(a.sum())}


def profile_forward(forward,L):
    """Dynamic scan arguments separate baseline compile from warm execution."""
    data=forward.data;n=forward.degree+forward.M
    run=jax.jit(lambda state,times,y,ids,mask:filter_slam(state,times,forward.command,
        jnp.zeros((len(times)-1,4,n)),jnp.zeros(2*n),forward.Qc,y,ids,data.covariance,mask))
    compiled={};samples=[];reported=[];compile_seconds=0.;previous=0
    for i in np.flatnonzero(data.scheduled):
        grid=data.times[previous:i+1]
        mask=data.mask[previous:i+1].copy();mask[0]=False
        initial=forward.at(data.times[previous])
        args=(initial,grid,data.measurements[previous:i+1],data.ids[previous:i+1],mask)
        if len(grid) not in compiled:
            start=time.perf_counter();compiled[len(grid)]=run.lower(*args).compile()
            compile_seconds+=time.perf_counter()-start
            jax.block_until_ready(compiled[len(grid)](*args))
        start=time.perf_counter();result=compiled[len(grid)](*args);jax.block_until_ready(result)
        seconds=time.perf_counter()-start;samples.append(seconds)
        if data.times[i]>=L-1e-12:reported.append(seconds)
        expected=forward.at(data.times[i])
        for a,b in zip(result.states,expected):np.testing.assert_allclose(a[-1],b,rtol=0,atol=2e-14)
        previous=i
    return {'explicit_compile_seconds':compile_seconds,'all_scan_seconds':stats(samples),
            'reported_scan_seconds':stats(reported),'matches_saved_forward':True}


def run_case(output,name,duration=5.2):
    sensor,kind=name.split('_')
    cfg=configuration('smooth' if kind=='smooth' else 'constant')
    cfg.update(sensor=sensor_settings(sensor),disturbance_kind=kind,total_duration_seconds=duration,
               initialization_seconds=0.,disturbance_start_seconds=0.,
               arrival_policy='independent zero-disturbance forward EKF posterior at boundary',
               first_full_horizon_seconds=5.,optimization_schedule='every scheduled scan t>=L, including empty scans')
    cfg.pop('arrival_a',None)
    cfg['optimization_observations']='a_k < t_i <= t_k; boundary measurements belong to the forward arrival'
    if kind=='zero':cfg['true_disturbance_v_omega']=[0.,0.]
    if kind=='smooth':cfg['true_disturbance_functions']=['0.15*sin(2*pi*t/5)','0.05*cos(2*pi*t/5)']
    output.mkdir(parents=True,exist_ok=True)
    data,scheduled,truth_check=simulate_moving(cfg)
    ids=np.broadcast_to(np.arange(len(data.landmark_map),dtype=np.int64),data.mask.shape)
    replay=ReplayData(data.times,data.measurements,ids,data.measurement_covariance,data.mask,scheduled)
    np.savez_compressed(output/'dataset.npz',**data._asdict(),ids=ids,scheduled=scheduled,
                        disturbance_truth=true_disturbance(data.times,kind))
    initial=create_state(cfg['initial_pose'],cfg['arrival_covariance'],cfg['capacity'])
    begin=time.perf_counter()
    forward=ForwardArrivals(replay,initial,lambda t:jnp.asarray(cfg['command_v_omega']),
        cfg['continuous_input_noise_spectral_density_Qc'])
    forward_seconds=time.perf_counter()-begin
    forward_profile=profile_forward(forward,cfg['window_length_seconds'])
    covariance_summary(forward.result.states)
    np.savez_compressed(output/'forward_history.npz',**{k:np.asarray(v) for k,v in forward.result.states._asdict().items()},
                       statuses=np.asarray(forward.result.observations.status),times=data.times)
    engine=MovingHorizonSLAM(forward,cfg['window_length_seconds'])
    saved=[];states={mode:[] for mode in ('projected','zero')};endpoint_w={mode:[] for mode in states}
    dense_errors={mode:[] for mode in states};baseline=[]
    try:
        for index,row in enumerate(engine.run()):
            w=row['window'];source=int(np.searchsorted(data.times,w.end))
            baseline.append(forward.at(w.end))
            record={key:row[key] for key in ('time','window_preparation_seconds')}
            record.update(start=w.start,observed_ids_at_end=ids[source,data.mask[source]].tolist(),
                          requested_count=int(w.mask.sum()),fits={})
            arrays={k:np.asarray(getattr(w,k)) for k in ('times','basis','measurements','ids','mask','source_indices')}
            arrays.update({'arrival_'+k:np.asarray(v) for k,v in w.arrival._asdict().items()})
            np.savez_compressed(output/f'window_{index:03d}.npz',**arrays)
            for mode,entry in row['fits'].items():
                result=entry['replay'];states[mode].append(result.final_state)
                record['fits'][mode]={k:entry[k] for k in entry if k!='replay'}
                record['fits'][mode].update(initializations=int(result.initialization_count),
                    corrections=int(result.correction_count),valid_count=int(result.valid_count),
                    covariance=covariance_summary(result.states))
                c=np.asarray(entry['optimization']['coefficients'])
                adapter=UniformBasis(cfg['degree'],cfg['segments'])
                endpoint_w[mode].append(c.reshape(2,7)@adapter.evaluate(5.,5.))
                elapsed=np.linspace(0.,5.,501)
                dw=adapter.evaluate(elapsed,5.)@c.reshape(2,7).T-true_disturbance(w.start+elapsed,kind)
                dense_errors[mode].append(dw)
                np.savez_compressed(output/f'{mode}_{index:03d}_replay.npz',
                    **{k:np.asarray(v) for k,v in result.states._asdict().items()},
                    statuses=np.asarray(result.statuses),slots=np.asarray(result.slots),
                    innovations=np.asarray(result.innovations),innovation_covariances=np.asarray(result.innovation_covariances),
                    initialization_mask=np.asarray(result.initialization_mask),correction_mask=np.asarray(result.valid_mask),
                    coefficients=c,endpoint_disturbance=endpoint_w[mode][-1])
            saved.append(record)
            write_json(output/'updates.json',saved)
            print(json.dumps({'case':name,'time':row['time'],'empty_scan':not bool(data.mask[source].any()),
                'fits':{m:{k:e['optimization'][k] for k in ('status','iterations','objective_final')} for m,e in row['fits'].items()}}),flush=True)
    except InvalidCandidateError as error:
        failures=[{'time':r['time'],'fits':{m:{'optimization':e['optimization'],'clipping':e['clipping']} for m,e in r['fits'].items()}}
                  for r in engine.records]
        write_json(output/'failure.json',{'error':str(error),'records':failures})
        raise
    if not saved:raise ValueError('no complete moving windows')
    times=np.array([r['time'] for r in saved]);indices=np.searchsorted(data.times,times)
    stack=lambda sequence:jax.tree_util.tree_map(lambda *v:np.stack(v),*sequence)
    baseline=stack(baseline)
    summary={'configuration':cfg,'truth_validation':truth_check,'updates':saved,
        'dataset_sha256':hashlib.sha256(data.truth.tobytes()+data.measurements.tobytes()+data.mask.tobytes()).hexdigest(),
        'forward_history_seconds_including_compile':forward_seconds,
        'forward_profile':forward_profile,
        'baseline_b':errors(baseline,data.truth[indices],data.landmark_map),
        'report_count':len(times),'empty_report_scans':int(np.sum(~data.mask[indices].any(axis=1))),
        'scheduled_times':times.tolist(),'methods':{}}
    summary['baseline_b']['endpoint_disturbance_rmse_v_omega']=np.sqrt(np.mean(true_disturbance(times,kind)**2,axis=0)).tolist()
    report_arrays={'times':times,'truth':data.truth[indices],'landmark_truth':data.landmark_map,
                   'baseline_x':baseline.x,'baseline_covariance':baseline.covariance,
                   'baseline_active':baseline.active,'baseline_truth_ids':baseline.truth_ids,
                   'disturbance_truth':true_disturbance(times,kind)}
    for mode in states:
        series=stack(states[mode]);endpoint=np.asarray(endpoint_w[mode])
        fits=[r['fits'][mode] for r in saved]
        m=errors(series,data.truth[indices],data.landmark_map)
        m.update(endpoint_disturbance_rmse_v_omega=np.sqrt(np.mean((endpoint-true_disturbance(times,kind))**2,axis=0)).tolist(),
            window_disturbance_rmse_v_omega=np.sqrt(np.mean(np.asarray(dense_errors[mode])**2,axis=(0,1))).tolist(),
            iterations=[e['optimization']['iterations'] for e in fits],
            final_costs=[e['optimization']['objective_final'] for e in fits],
            convergence_status=[e['optimization']['status'] for e in fits],
            runtime_seconds=stats([e['total_update_seconds'] for e in fits]),
            warm_optimization_seconds=stats([e['optimization']['timing']['optimization_seconds'] for e in fits]),
            explicit_compile_seconds=stats([sum(e['optimization']['timing'][k] for k in ('evaluation_compile_seconds','value_gradient_compile_seconds')) for e in fits]),
            gradient_seconds_mean=[e['optimization']['timing']['value_gradient_seconds_mean'] for e in fits],
            gradient_seconds_p95=[e['optimization']['timing']['value_gradient_seconds_p95'] for e in fits],
            clipped_updates=sum(e['clipping']['clipped_count']>0 for e in fits),
            clipped_coefficients=sum(e['clipping']['clipped_count'] for e in fits),
            clipping_max_absolute=max(e['clipping']['clipping_max_absolute'] for e in fits),
            clipping_l1_total=sum(e['clipping']['clipping_l1'] for e in fits),
            covariance=covariance_summary(series))
        summary['methods'][mode]=m
        m['clipping_update_fraction']=m['clipped_updates']/len(fits)
        m['clipping_coefficient_fraction']=m['clipped_coefficients']/(14*len(fits))
        gradients=[r['value_gradient_seconds'] for e in fits for r in e['optimization']['evaluation_history'] if r['phase']=='optimization']
        m['gradient_evaluation_seconds']=stats(gradients)
        jumps=[e['jumps'] for e in fits if 'jumps' in e]
        m['maximum_overlap_position_revision_m']=max((float(np.linalg.norm(j['overlap_pose_revision'][:2])) for j in jumps),default=0.)
        m['maximum_overlap_heading_revision_rad']=max((abs(j['overlap_pose_revision'][2]) for j in jumps),default=0.)
        m['maximum_reported_landmark_jump_m']=max((float(np.linalg.norm(p)) for j in jumps for p in j['reported_map_delta']['position_differences']),default=0.)
        report_arrays.update({mode+'_'+key:np.asarray(value) for key,value in series._asdict().items()})
        report_arrays[mode+'_disturbance']=endpoint
    np.savez_compressed(output/'reported_estimates.npz',**report_arrays)
    plot_case(output,name,report_arrays)
    write_json(output/'results.json',summary)
    return summary


def plot_case(output,name,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'baseline':'#0072B2','projected':'#E69F00','zero':'#CC79A7'}
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].plot(a['truth'][:,0],a['truth'][:,1],color='black',label='Truth')
    axes[0].scatter(*a['landmark_truth'].T,color='black',marker='x')
    for i,p in enumerate(a['landmark_truth']):axes[0].annotate(str(i),p)
    for mode,color in colors.items():
        x=a[mode+'_x'];active=a[mode+'_active'][-1]
        axes[0].plot(x[:,0],x[:,1],color=color,marker='.',label='Forward EKF' if mode=='baseline' else mode+' MHE')
        axes[0].scatter(*x[-1,3:].reshape(-1,2)[active].T,facecolors='none',edgecolors=color)
    axes[0].set(xlabel='x (m)',ylabel='y (m)',aspect='equal',title=name+' (reported endpoints)')
    axes[0].legend(fontsize=7)
    for channel in range(2):
        axis=axes[channel+1];axis.plot(a['times'],a['disturbance_truth'][:,channel],color='black',label='Truth')
        axis.axhline(0,color=colors['baseline'],ls='--',label='Forward zero disturbance')
        for mode in ('projected','zero'):
            axis.plot(a['times'],a[mode+'_disturbance'][:,channel],marker='.',color=colors[mode],label=mode+' MHE')
        axis.set(xlabel='Report time (s)',ylabel='w_v (m/s)' if channel==0 else 'w_omega (rad/s)')
        axis.legend(fontsize=7)
    for axis in axes:axis.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'reported_estimates.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    updates=json.loads((output/'updates.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for mode,color in colors.items():
        error=a[mode+'_x'][:,:3]-a['truth']
        error[:,2]=(error[:,2]+np.pi)%(2*np.pi)-np.pi
        axes[0,0].plot(a['times'],np.linalg.norm(error[:,:2],axis=1),marker='.',color=color,label=mode)
        axes[0,1].plot(a['times'],error[:,2],marker='.',color=color,label=mode)
        if mode!='baseline':
            axes[1,0].plot(a['times'],[r['fits'][mode]['optimization']['iterations'] for r in updates],marker='.',color=color,label=mode)
            axes[1,1].plot(a['times'],[r['fits'][mode]['total_update_seconds'] for r in updates],marker='.',color=color,label=mode)
    for axis,title,ylabel in zip(axes.ravel(),['Position error','Wrapped heading error','Optimizer work','Update wall time'],['m','rad','iterations','s']):
        axis.set(title=title,ylabel=ylabel);axis.legend(fontsize=7);axis.grid(alpha=.2)
    for axis in axes[1]:axis.set_xlabel('Report time (s)')
    fig.suptitle(name);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'accuracy_runtime.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def validate_saved(output,cases):
    checks={'all_schedules_complete':True,'boundary_arrivals_and_masks':True,'birth_accounting':True,
            'same_map_ids_as_forward':True,'projection_and_clipping':True,'first_fits_identical':True}
    for name,r in cases.items():
        root=output/name
        with np.load(root/'dataset.npz') as saved:
            times,scheduled=saved['times'],saved['scheduled']
        eligible=times[scheduled & (times>=5.-1e-12)]
        np.testing.assert_array_equal(r['scheduled_times'],eligible)
        with np.load(root/'forward_history.npz') as saved:
            forward={k:saved[k] for k in ('x','covariance','active','truth_ids')}
        previous=None
        for i,row in enumerate(r['updates']):
            with np.load(root/f'window_{i:03d}.npz') as saved:w={k:saved[k] for k in saved.files}
            arrival_index=int(np.argmin(np.abs(times-row['start'])))
            assert abs(times[arrival_index]-row['start'])<1e-12
            assert not w['mask'][0].any()
            assert np.all(times[w['source_indices']]>row['start'])
            end_index=int(np.argmin(np.abs(times-row['time'])))
            for key in forward:np.testing.assert_array_equal(w['arrival_'+key],forward[key][arrival_index])
            for mode,e in row['fits'].items():
                with np.load(root/f'{mode}_{i:03d}_replay.npz') as saved:result={k:saved[k] for k in saved.files}
                for key in forward:np.testing.assert_array_equal(result[key][0],w['arrival_'+key])
                for key in ('active','truth_ids'):np.testing.assert_array_equal(result[key][-1],forward[key][end_index])
                assert int(np.sum(result['initialization_mask']))==e['initializations']
                assert int(np.sum(result['correction_mask']))==e['corrections']
                assert e['initializations']+e['corrections']==e['valid_count']==int(w['mask'].sum())
                assert not np.any(result['initialization_mask'] & result['correction_mask'])
                raw=np.zeros(14) if previous is None or mode=='zero' else make_warm_start_projection(
                    3,4,5.,row['time']-previous['time']).apply(previous['fits'][mode]['optimization']['coefficients'])
                np.testing.assert_array_equal(raw,e['clipping']['raw_projection'])
                np.testing.assert_array_equal(np.clip(raw,-.5,.5),e['optimization']['initial_parameters'])
            if previous is None:
                np.testing.assert_array_equal(row['fits']['projected']['optimization']['coefficients'],row['fits']['zero']['optimization']['coefficients'])
            previous=row
    return checks


def summarize(output):
    cases={p.parent.name:json.loads(p.read_text()) for p in sorted(output.glob('*/results.json'))}
    snapshot=json.loads((output/'before_sha256.json').read_text())
    changed=[p for p,h in snapshot.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    write_json(output/'preservation.json',{'checked_count':len(snapshot),'changed':changed,'all_preserved':not changed})
    validation=validate_saved(output,cases)
    validation.update(all_six_cases_present=len(cases)==6,all_original_files_preserved=not changed,
                      implementation_gate_passed=json.loads((output/'implementation_gate.json').read_text())['passed'])
    gradient_path=output/'gradient_checks.json'
    gradients=json.loads(gradient_path.read_text()) if gradient_path.exists() else None
    validation['saved_full_window_gradients_passed']=gradients is not None and gradients['passed']
    write_json(output/'validation.json',validation)
    # Profile the excluded host-only initial-guess operation separately. These
    # are remeasurement samples, not retroactively added to recorded latencies.
    initialization_samples={mode:[] for mode in ('projected','zero')}
    for r in cases.values():
        previous=None
        for row in r['updates']:
            for mode in initialization_samples:
                old=None if previous is None else previous['fits'][mode]['optimization']['coefficients']
                delta=0. if previous is None else row['time']-previous['time']
                initial_guess(old,delta,mode=mode)  # warm the cache in this report process
                for _ in range(30):
                    start=time.perf_counter();initial_guess(old,delta,mode=mode)
                    initialization_samples[mode].append(time.perf_counter()-start)
            previous=row
    init_profile={mode:stats(samples) for mode,samples in initialization_samples.items() if samples}
    write_json(output/'initialization_microprofile.json',init_profile)
    lines=['# Stage 5B moving-horizon input-spline MHE','',
        'Deterministic seed-7 paired validation. Every scheduled scan with a complete L=5 s horizon is optimized, '
        'including empty scans. Before t=5 s the independent forward EKF runs alone. '
        'These short startup runs profile feasibility and computation; they are not Monte Carlo validation.','',
        'All signals start at t=0: zero; constant [0.15,-0.05]; or smooth [0.15 sin(2 pi t/5),0.05 cos(2 pi t/5)]. '
        'Sensor assumptions match Stage 4C: regression unlimited/360°, forward 0.2–6 m/180°. '
        'There is no truth process-noise injection. The original R/Qc, robot prior, d=3, M=4, coefficient bounds '
        '[-0.5,0.5] and all optimizer tolerances are retained. Each reported window is optimized from both '
        'projected/clipped and zero initial guesses on identical data/arrival/ID plans. First guesses are zero.','',
        '| Case | Method | Position RMSE (m) | Heading RMSE (rad) | Map RMSE (m) | Mapped counts | Endpoint w_v / w_omega RMSE |',
        '|---|---|---:|---:|---:|---|---|']
    for name,r in cases.items():
        for method,m in [('baseline_b',r['baseline_b']),*r['methods'].items()]:
            lines.append(f"| {name} | {method} | {m['position_rmse_m']:.7g} | {m['heading_rmse_rad']:.7g} | "
                         f"{m['landmark_rmse_m']} | {m['mapped_landmark_counts']} | {m['endpoint_disturbance_rmse_v_omega']} |")
    lines+=['','| Case | Forward zero-replay J by window | Projected-start final J | Zero-start final J |',
            '|---|---|---|---|']
    for name,r in cases.items():
        baseline_costs=[u['fits']['zero']['optimization']['objective_initial'] for u in r['updates']]
        lines.append(f"| {name} | {baseline_costs} | {r['methods']['projected']['final_costs']} | {r['methods']['zero']['final_costs']} |")
    lines+=['','Robot metrics use the reported times only, with wrapped heading and no rigid alignment. '
            'Map RMSE pools errors over active known IDs at those times; mapped counts must accompany it. '
            'Endpoint disturbance metrics measure the reported right-endpoint input; window reconstruction '
            'RMSE is stored separately and pools overlapping 501-point window grids, not independent samples.','',
            '| Case | Method | Pooled window w_v / w_omega RMSE |',
            '|---|---|---|']
    for name,r in cases.items():
        for method,m in r['methods'].items():
            lines.append(f"| {name} | {method} | {m['window_disturbance_rmse_v_omega']} |")
    lines+=['',
            '| Case | Method | Iterations | Final J | Status | Update mean / p95 (s) | Warm optimization mean / p95 (s) | Explicit compilation total (s) |',
            '|---|---|---|---|---|---|---|---:|']
    for name,r in cases.items():
        for method,m in r['methods'].items():
            timing=m['runtime_seconds'];warm=m['warm_optimization_seconds']
            lines.append(f"| {name} | {method} | {m['iterations']} | {m['final_costs']} | {m['convergence_status']} | "
                         f"{timing['mean']:.5g} / {timing['p95']:.5g} | {warm['mean']:.5g} / {warm['p95']:.5g} | {m['explicit_compile_seconds']['total']:.5g} |")
    lines+=['','Per-update wall time includes window preparation, checked optimizer preparation/compilation, '
            'feasibility probes, warm optimization and final replay. It excludes artifact serialization/plotting. '
            'The preserved Stage 4C factory captures each window and compiles it separately; these measurements '
            'do not claim a reusable compilation cache or real-time operation. Explicit lower/compile time and '
            'cold checked-reference auxiliary compilation are recorded separately in each optimizer timing. '
            'Gradient execution mean/p95 and all raw evaluations are retained in updates.json.','',
            '| Case | Method | Gradient execution mean / p95 (s) | Max overlap position / heading revision | Max landmark jump (m) |',
            '|---|---|---|---|---:|']
    for name,r in cases.items():
        for method,m in r['methods'].items():
            g=m['gradient_evaluation_seconds']
            lines.append(f"| {name} | {method} | {g['mean']:.6g} / {g['p95']:.6g} | {m['maximum_overlap_position_revision_m']:.7g} m / {m['maximum_overlap_heading_revision_rad']:.7g} rad | {m['maximum_reported_landmark_jump_m']:.7g} |")
    lines+=['','| Case | Forward warm scan mean / p95 (s), all scans | Forward explicit compile (s) |',
            '|---|---|---:|']
    for name,r in cases.items():
        p=r['forward_profile'];t=p['all_scan_seconds']
        lines.append(f"| {name} | {t['mean']:.6g} / {t['p95']:.6g} | {p['explicit_compile_seconds']:.6g} |")
    lines+=['',
            '| Case | Reports / empty endpoint scans | Clipped warm-start updates / coefficients | Max clipping / total L1 |',
            '|---|---|---|---|']
    for name,r in cases.items():
        m=r['methods']['projected']
        lines.append(f"| {name} | {r['report_count']} / {r['empty_report_scans']} | {m['clipped_updates']} / {m['clipped_coefficients']} | {m['clipping_max_absolute']:.7g} / {m['clipping_l1_total']:.7g} |")
    lines+=['','Only the initial guess is clipped. Raw projection, feasible vector, affected component count, '
            'maximum absolute change and L1 change are saved for every update. The Stage 5A projection and its '
            'fallback are unchanged. A clipped start does not constrain the final optimizer beyond the unchanged bounds.','',
            '## Maps, jumps and numerical limitations','',
            'The forward EKF never receives optimized states, maps, covariances or disturbances. Its full posterior '
            'at the boundary includes boundary observations; replay uses only a<t_i<=t. Nonaligned boundaries '
            'propagate the last available forward posterior without future observations. Landmark births use '
            'inverse measurements once and retain all cross-covariances; invisible landmarks persist. '
            'Reported MHE maps are independent candidate histories, and known IDs never supply coordinates. '
            '**Conditional EKF covariance; spline coefficients treated as fixed.**','',
            'Per-update jumps include raw consecutive endpoint pose/map differences and overlap revisions: '
            'the newly fitted replay at the previous report time minus the previous reported estimate, with '
            'wrapped heading and common landmark IDs. New births are excluded from common-ID jump comparisons. '
            'All complete reported states, covariances, masks, slots, arrivals and optimization records are saved. '
            'A numerically invalid candidate aborts the run; a valid unconverged final iterate is explicitly labeled.','']
    lines+=['Convergence follows the original SciPy stopping rule, often function reduction. It does not '
            'guarantee the projected gradient meets gtol or establish a global optimum; raw/projected gradient '
            'norms and the exact termination messages remain in each update. Recorded update times exclude '
            'the host projection/clipping call itself as well as artifact I/O, so they are not complete end-to-end latency.','']
    if init_profile:
        lines+=['The excluded initial-guess operation is remeasured separately after estimation, with warm '
                'projection caches and 30 repetitions per update. These samples are not added retroactively '
                'to the original update timings: '+str(init_profile)+'. See initialization_microprofile.json.','']
    for name,r in cases.items():
        for method,m in r['methods'].items():
            worse=[k for k in ('position_rmse_m','heading_rmse_rad','landmark_rmse_m') if m[k] is not None and r['baseline_b'][k] is not None and m[k]>r['baseline_b'][k]]
            if worse:lines.append(f"- {name}/{method}: lower-cost fitting does not guarantee accuracy; worse than forward EKF in {', '.join(worse)}.")
            channels=[label for i,label in enumerate(('w_v','w_omega')) if m['endpoint_disturbance_rmse_v_omega'][i]>r['baseline_b']['endpoint_disturbance_rmse_v_omega'][i]]
            if channels:lines.append(f"- {name}/{method}: endpoint disturbance RMSE is worse than the zero-disturbance baseline for {', '.join(channels)}.")
        lines+=['',f'![{name} endpoint estimates]({name}/reported_estimates.png)','']
        lines+=[f'![{name} accuracy and computation]({name}/accuracy_runtime.png)','']
    gate=json.loads((output/'implementation_gate.json').read_text())
    statuses=[e['optimization']['status'] for r in cases.values() for u in r['updates'] for e in u['fits'].values()]
    lines+=['## Executed validation','',f"The complete implementation gate passed **{gate['passed_tests']} tests**, "
            f"exit {gate['exit_code']}, subprocess wall time {gate['wall_seconds']:.3f} s, before benchmarks.",'',
            '```sh',gate['command'],'```','',
            f"Saved cases contain {len(statuses)} optimization runs; {statuses.count('converged')} report convergence. "
            'See implementation_gate.json, pytest_stdout.txt, pytest.xml, pilot_stdout.txt and suite_stdout.txt for actual evidence. '
            'The three-update implementation test is independently rerun twice and checks exact coefficients/costs; '
            'the full noisy six-case suite is one seed, not multiple independent trials.','',
            f"All {len(snapshot)} captured pre-task files remain unchanged: {not changed}. "
            'Stage 5A, Stage 4C, the original baseline and all earlier artifacts are protected by SHA-256. '
            'Only new Stage 5B modules, tests, documentation and artifact files were added.','',
            '```json',json.dumps(validation,indent=2),'```','']
    if gradients:
        maximum=max(step['max_absolute_error'] for w in gradients['windows'].values()
                    for point in w['results'].values() for step in point.values())
        lines+=['Saved full-size windows additionally check every coefficient at zero, a nonconstant vector '
                'and the fitted vector, using float64 central differences at steps 1e-4 and 1e-5. '
                f"All checks passed: {gradients['passed']}; largest absolute discrepancy {maximum:.7g}. "
                'Original rtol/atol 2e-5 and max-absolute-error <2e-5 are unchanged. Bound-active coefficients '
                'are checked in the valid ambient objective domain, not by differentiating clipping. '
                'See gradient_checks.json and gradient_stdout.txt.','',
                '```sh',gradients['command'],'```','']
    if len(cases)==6:
        phase_seconds=sum(r['forward_history_seconds_including_compile']+sum(m['runtime_seconds']['total'] for m in r['methods'].values()) for r in cases.values())
        lines+=['## Proposed Monte Carlo plan — approval required','',
            f"The six-case one-seed run sums to {phase_seconds:.1f} s in recorded forward/MHE estimation phases "
            f"({phase_seconds/60:.2f} min), excluding tests, artifact I/O, plots and extra profiling. "
            f"At that measured cost, 15 paired variance-pilot seeds would require approximately {15*phase_seconds/3600:.2f} h; "
            f"a 97-trial final run would require roughly {97*phase_seconds/3600:.2f} h for the same short cases.",'',
            'Recommend extending deterministic trajectories and reducing repeated compilation first. Then, with approval, '
            'use 15 paired pilot seeds to estimate the across-seed standard deviation of the paired position-RMSE difference '
            '(approximate relative standard error of the SD is 1/sqrt(2*(15-1)) = 19%). '
            'For a 95% paired-mean confidence-interval half-width of 0.01 m, estimate N≈(1.96*s_delta/0.01)^2, '
            'then use Student-t critical values and the pilot uncertainty to revise N upward. '
            'For illustration, s_delta=0.02 m suggests about 16 trials; s_delta=0.05 m suggests 97. '
            'These SDs are assumptions, not measured across-seed estimates. A single seed cannot determine a statistically '
            'justified final trial count. Seeds, not overlapping windows, are the independent experimental units. '
            'Update this runtime budget after selecting longer deterministic trajectories. No Monte Carlo trial has been run.','']
    lines+=['## Reproduce and next steps','',
            '```sh','PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5b --case all --rerun','```','',
            'Without --rerun, existing completed cases are reused. A different duration requires --rerun or a fresh output '
            'with its own implementation gate and preservation snapshot. See estimation/STAGE5B.md for validation commands and implementation details. '
            'Inspect clipping and overlap revisions, profile longer deterministic runs and reduce repeated compilation '
            'before drawing broad performance conclusions. Monte Carlo, nearest-neighbor association and Victoria Park '
            'require additional approval.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(output/'results.json',cases)
    if changed:raise RuntimeError(f'Protected files changed: {changed}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choices=[s+'_'+k for s in ('regression','forward') for k in ('zero','constant','smooth')]
    parser.add_argument('--case',choices=choices+['all'],default='regression_constant')
    parser.add_argument('--duration',type=float,default=5.2)
    parser.add_argument('--output',type=Path,default=Path('artifacts/milestone5b'))
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--rerun',action='store_true',help='recompute completed Stage 5B cases')
    args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    if not args.report_only:
        gate=args.output/'implementation_gate.json'
        if not gate.exists() or not json.loads(gate.read_text())['passed']:
            parser.error('Run and record Stage 5B implementation validation before benchmarks')
        for name in choices if args.case=='all' else [args.case]:
            path=args.output/name/'results.json'
            if path.exists() and not args.rerun:
                if json.loads(path.read_text())['configuration']['total_duration_seconds']!=args.duration:
                    parser.error('saved duration differs; specify --rerun or a fresh output directory')
                continue
            run_case(args.output/name,name,args.duration)
            jax.clear_caches()
    summarize(args.output)


if __name__=='__main__':main()
