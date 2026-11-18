"""Scientific figures, serialized audits and evidence-based Stage 5C report."""

import hashlib,json,time
from pathlib import Path
import numpy as np

from benchmarks.stage5c_diagnostics import basis_diagnostics,grid,independent_truth,error_profile
from simulation.moving_horizon import true_disturbance
from splines import UniformBasis

FIGURE_TIMINGS=[]


def stats(values):
    a=np.asarray(values,dtype=float)
    return {'count':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p95':float(np.percentile(a,95)),'total':float(a.sum())} if len(a) else None


def correlation(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None


def collect(cases):
    result={}
    for name,r in cases.items():
        rows=r['updates'];summary={'window_count':len(rows),'methods':{}}
        for mode in ('projected','zero'):
            entries=[row['methods'][mode] for row in rows]
            e=[x['error'] for x in entries];fits=[x['optimization'] for x in entries]
            ages=[x['visibility']['seconds_since_latest_correction'] for x in entries]
            with np.load(Path('artifacts/milestone5c')/name/'profiles.npz') as saved:
                errors=saved[mode+'_errors']
            groups={}
            for empty in (False,True):
                selected=np.array([x['visibility']['empty_endpoint']==empty for x in entries])
                groups['empty' if empty else 'observed']={'count':int(selected.sum()),
                    'right_rmse':np.sqrt(np.mean(errors[selected,-1]**2,axis=0)).tolist() if selected.any() else None}
            timing={('buffer_plus_basis_remainder_seconds' if key=='observation_buffer_seconds' else key):
                    stats([x['preparation'][key] for x in rows]) for key in rows[0]['preparation'] if key!='basis_seconds'}
            for key in entries[0]['timing']:timing[key]=stats([x['timing'][key] for x in entries])
            for key in ('program_trace_seconds','evaluation_lower_seconds','gradient_lower_seconds',
                        'evaluation_compile_seconds','value_gradient_compile_seconds','preparation_seconds',
                        'checked_reference_first_call_seconds_including_auxiliary_compilation','optimization_seconds'):
                timing[key]=stats([f['timing'][key] for f in fits])
            for field,label in [('validity_check_seconds','objective_evaluation_seconds'),('value_gradient_seconds','value_gradient_seconds')]:
                timing[label]=stats([v[field] for f in fits for v in f['evaluation_history'] if v['phase']=='optimization'])
            timing['sensitivity_compile_seconds']=stats([x['sensitivity_timing']['compile_seconds'] for x in entries])
            timing['sensitivity_execution_seconds']=stats([x['sensitivity_timing']['execution_seconds'] for x in entries])
            timing['diagnostic_seconds']=stats([x['diagnostic_seconds'] for x in entries])
            timing['artifact_seconds']=stats([x['artifact_seconds'] for x in entries])
            table={
                'position_rmse_m':float(np.sqrt(np.mean([x['position_error_m']**2 for x in e]))),
                'heading_rmse_rad':float(np.sqrt(np.mean([x['heading_error_rad']**2 for x in e]))),
                'landmark_rmse_m':float(np.sqrt(np.mean([v*v for x in e for v in x['landmark_errors_m']]))),
                'mapped_counts':[x['mapped_landmarks'] for x in e],
                'iterations':sum(f['iterations'] for f in fits),
                'function_evaluations':sum(f['function_evaluations'] for f in fits),
                'gradient_evaluations':sum(f['gradient_evaluations'] for f in fits),
                'converged_count':sum(f['success'] for f in fits),
                'gtol_met_count':sum(f['projected_gradient_inf']<=1e-6 for f in fits),
                'max_projected_gradient':max(f['projected_gradient_inf'] for f in fits),
                'initial_cost':stats([f['objective_initial'] for f in fits]),
                'final_cost':stats([f['objective_final'] for f in fits]),
                'disturbance_rmse_v_omega':np.sqrt(np.mean(errors**2,axis=(0,1))).tolist(),
                'clipped_updates':sum(x['clipping']['clipped_count']>0 for x in entries),
                'max_clipping':max(x['clipping']['clipping_max_absolute'] for x in entries),
                'empty_endpoint_count':sum(x['visibility']['empty_endpoint'] for x in entries),
                'profile':r['profiles'][mode],'endpoint_groups':groups,'timings':timing,
                'numerical_ranks':[x['sensitivity']['numerical_rank'] for x in entries],
                'condition_numbers':[x['sensitivity']['condition_number'] for x in entries],
                'finite_condition_stats':stats([x['sensitivity']['condition_number'] for x in entries if x['sensitivity']['condition_number'] is not None]),
                'median_column_norms':np.median([x['sensitivity']['column_norms'] for x in entries],axis=0).tolist(),
                'active_bound_counts':[len(x['active_bound_indices']) for x in entries],
                'near_bound_counts':[len(x['near_bound_indices_1e3']) for x in entries],
                'max_overlap_pose_revision_m':max((float(np.linalg.norm(x['jumps']['pose_revision'][:2])) for x in entries if 'jumps' in x),default=0.),
                'max_map_revision_m':max((float(np.linalg.norm(d)) for x in entries if 'jumps' in x for d in x['jumps']['map_revision']['position_differences']),default=0.),
                'descriptive_correlations':{
                    'arrival_position_vs_report_position':correlation([x['arrival_error']['position_error_m'] for x in rows],[x['position_error_m'] for x in e]),
                    'last_correction_age_vs_right_abs_error_v':correlation(ages,np.abs(errors[:,-1,0])) if all(x is not None for x in ages) else None,
                    'last_correction_age_vs_right_abs_error_omega':correlation(ages,np.abs(errors[:,-1,1])) if all(x is not None for x in ages) else None}}
            summary['methods'][mode]=table
        b=[row['forward_error'] for row in rows]
        summary['forward']={'position_rmse_m':float(np.sqrt(np.mean([x['position_error_m']**2 for x in b]))),
            'heading_rmse_rad':float(np.sqrt(np.mean([x['heading_error_rad']**2 for x in b]))),
            'landmark_rmse_m':float(np.sqrt(np.mean([v*v for x in b for v in x['landmark_errors_m']]))),
            'mapped_counts':[x['mapped_landmarks'] for x in b]}
        result[name]=summary
    return result


def save(fig,out,name):
    import matplotlib.pyplot as plt
    begin=time.perf_counter()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    FIGURE_TIMINGS.append({'figure':str(out/name),'publication_seconds':time.perf_counter()-begin})


def figures(root,cases,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURE_TIMINGS.clear()
    begin=time.perf_counter();sigma=grid();knots=np.linspace(0,5,5)
    colors={'projected':'#E69F00','zero':'#CC79A7'}
    for ch,label,unit in [(0,'linear','m/s'),(1,'angular','rad/s')]:
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
        for row,sensor in enumerate(('regression','forward')):
            for col,kind in enumerate(('zero','constant','smooth')):
                name=sensor+'_'+kind;ax=axes[row,col]
                if name not in cases:continue
                run=cases[name]
                truth=np.stack([independent_truth(w['start']+sigma,kind) for w in run['updates']])
                ax.plot(sigma,np.sqrt(np.mean(truth[:,:,ch]**2,axis=0)),'--',color='#0072B2',label='Forward zero input')
                for mode,color in colors.items():
                    E=np.asarray(run['profiles'][mode]['rmse'])[:,ch]
                    ax.plot(sigma,E,color=color,ls='--' if mode=='zero' else '-',label=mode+' MHE')
                    ax.scatter([0.,5.],[E[0],E[-1]],color=color,s=18)
                ax.axvspan(1.25,3.75,color='.7',alpha=.16,label='Defined interior')
                for k in knots:ax.axvline(k,color='.5',ls=':',lw=.7)
                ax.set(title=name,ylabel=f'{label} RMSE ({unit})',xlim=(0.,5.));ax.grid(alpha=.15)
                if row==1:ax.set_xlabel('Elapsed window time sigma (s): left 0, right 5')
        axes[0,0].legend(fontsize=7)
        fig.suptitle('Seed-7 deterministic RMSE across overlapping windows; shading is interior, not confidence')
        save(fig,root,f'{label}_error_profiles')
    b=UniformBasis(3,4).evaluate(sigma,5.);info=basis_diagnostics()
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j in range(7):
        axes[0].plot(sigma,b[:,j],label=f'b{j}')
        a,z=info['support_in_window'][j];axes[1].plot([a,z],[j,j],lw=5)
    for ax in axes:
        for k in knots:ax.axvline(k,color='.5',ls=':',lw=.6)
        ax.set_xlabel('Elapsed window time sigma (s)');ax.grid(alpha=.15)
    axes[0].set(ylabel='Basis magnitude',title='C++ uniform unclamped cubic basis');axes[0].legend(ncol=2,fontsize=8)
    axes[1].set(ylabel='Coefficient index',title='Support intersected with [0,5] s',yticks=range(7))
    save(fig,root,'basis_support')
    for name,run in cases.items():
        out=root/name;updates=run['updates'];times=np.array([x['time'] for x in updates]);kind=name.split('_')[1]
        # Representative windows are preselected first, middle and last.
        for index in sorted(set((0,len(updates)//2,len(updates)-1))):
            r=updates[index];fig,axes=plt.subplots(5,1,figsize=(10,10),sharex=True)
            truth=independent_truth(r['start']+sigma,kind)
            axes[0].plot(sigma,truth[:,0],color='black',label='Truth');axes[2].plot(sigma,truth[:,1],color='black',label='Truth')
            for mode,color in colors.items():
                with np.load(out/f'{mode}_{index:03d}.npz') as saved:
                    est=saved['disturbance_estimate'];error=saved['disturbance_error']
                style={'color':color,'ls':'--' if mode=='zero' else '-','label':mode}
                axes[0].plot(sigma,est[:,0],**style)
                axes[1].plot(sigma,np.abs(error[:,0]),**style)
                axes[2].plot(sigma,est[:,1],**style)
                axes[3].plot(sigma,np.abs(error[:,1]),**style)
            v=r['methods']['projected']['visibility']
            correction_t=[];correction_id=[];birth_t=[];birth_id=[];empty_t=[]
            for scan in v['observation_rows']:
                t=scan['sigma']
                correction_t.extend([t]*len(scan['correction_ids']));correction_id.extend(scan['correction_ids'])
                birth_t.extend([t]*len(scan['birth_ids']));birth_id.extend(scan['birth_ids'])
                if not scan['observed_ids']:empty_t.append(t)
            axes[4].scatter(correction_t,correction_id,color='#0072B2',s=10,label='Corrections')
            axes[4].scatter(birth_t,birth_id,color='#009E73',marker='x',label='Births')
            axes[4].scatter(empty_t,[-1]*len(empty_t),color='.5',marker='|',label='Empty scheduled scans')
            for ax,y in zip(axes,['w_v (m/s)','|error| (m/s)','w_omega (rad/s)','|error| (rad/s)','Landmark ID']):
                ax.set_ylabel(y);ax.grid(alpha=.15);ax.set_xlim(0,5)
                for k in knots:ax.axvline(k,color='.5',ls=':',lw=.6)
                ax.legend(fontsize=7,loc='upper left')
            axes[-1].set_xlabel('Elapsed window time sigma (s)')
            age=v['seconds_since_latest_correction']
            age_label='none' if age is None else f'{age:g} s'
            fig.suptitle(f'{name}: window [{r["start"]:g},{r["time"]:g}] s; latest correction age {age_label}')
            save(fig,out,f'window_{index:03d}_disturbances_observations')
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for mode,color in colors.items():
            c=np.asarray(summary[name]['methods'][mode]['median_column_norms'])
            axes[0].plot(range(7),c[:7],color=color,label=mode+' v')
            axes[0].plot(range(7),c[7:],ls='--',color=color,label=mode+' omega')
        for i in sorted(set((0,len(updates)//2,len(updates)-1))):
            s=np.asarray(updates[i]['methods']['projected']['sensitivity']['singular_values'])
            axes[1].semilogy(np.arange(len(s)),np.where(s>0,s,np.nan),label=f't={times[i]:g}')
        for j,label in [(6,'v'),(13,'omega')]:
            axes[2].plot(times,[x['methods']['projected']['sensitivity']['column_norms'][j] for x in updates],label='rightmost '+label)
        axes[0].set(xlabel='Coefficient j',ylabel='Median column norm of D',title='Channel units retained; no nondimensional scaling')
        axes[1].set(xlabel='Singular index',ylabel='Singular value',title='Full whitening derivative; zeros omitted on log scale')
        axes[2].set(xlabel='Report time (s)',ylabel='Rightmost column norm',title='Right-support coefficients c_v6, c_omega6')
        for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
        fig.suptitle(name+' coefficient sensitivity (local diagnostic)');save(fig,out,'sensitivity')
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for mode,color in [('forward','#0072B2'),*colors.items()]:
            style={'color':color,'ls':'--' if mode=='zero' else '-','label':mode}
            es=[r['forward_error'] if mode=='forward' else r['methods'][mode]['error'] for r in updates]
            axes[0,0].plot(times,[e['position_error_m'] for e in es],**style)
            axes[0,1].plot(times,[e['heading_error_rad'] for e in es],**style)
            axes[1,0].plot(times,[e['landmark_rmse_m'] for e in es],**style)
            if mode!='forward':
                axes[2,0].plot(times,[r['methods'][mode]['optimization']['projected_gradient_inf'] for r in updates],**style)
                axes[2,1].plot(times,[len(r['methods'][mode]['active_bound_indices']) for r in updates],**style)
        axes[1,1].plot(times,[r['arrival_error']['position_error_m'] for r in updates],label='Arrival position error')
        axes[1,1].plot(times,[np.sqrt(sum(r['arrival_covariance']['robot_variances'][:2])) for r in updates],label='sqrt(Pxx+Pyy) at arrival')
        axes[2,0].axhline(1e-6,color='black',ls=':',label='gtol')
        for ax,title,unit in zip(axes.ravel(),['Position error','Wrapped heading error','Map RMSE','Arrival at t-L','Projected gradient','Coefficients at bounds'],['m','rad','m','m','infinity norm','count']):
            ax.set(title=title,ylabel=unit);ax.legend(fontsize=7);ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('Report time (s)')
        fig.suptitle(name+' pose/map/arrival/optimizer diagnostics');save(fig,out,'pose_map_arrival')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,run in cases.items():
        with np.load(root/name/'profiles.npz') as saved:e=saved['projected_errors']
        ages=[r['methods']['projected']['visibility']['seconds_since_latest_correction'] for r in run['updates']]
        for ch in range(2):axes[ch].scatter(ages,np.abs(e[:,-1,ch]),s=18,label=name)
    for ch,ax in enumerate(axes):
        ax.set(xlabel='Time since latest correction (s)',ylabel='Right-endpoint absolute error '+('m/s' if ch==0 else 'rad/s'))
        ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('Seed-7 descriptive association; overlapping windows are not independent')
    save(fig,root,'correction_age_endpoint_error')
    return time.perf_counter()-begin


def audit(root,cases):
    checks={'all_six_cases':len(cases)==6,'all_eleven_windows':all(r['window_count']==11 for r in cases.values()),
            'independent_truth_and_coordinates':True,'births_masks_empty_scans':True,
            'whitened_cost_identity':True,'covariance_finite_symmetric':True,
            'first_window_reproducible':True}
    max_truth=0.;max_cost=0.
    for name,r in cases.items():
        out=root/name;kind=name.split('_')[1];sigma=np.asarray(r['sigma'])
        with np.load(out/'dataset.npz') as saved:
            times=saved['times'];schedule=saved['scheduled']
        np.testing.assert_array_equal([row['time'] for row in r['updates']],times[schedule & (times>=5.-1e-12)])
        with np.load(out/'profiles.npz') as saved:profiles={k:saved[k] for k in saved.files}
        for i,row in enumerate(r['updates']):
            with np.load(out/f'window_{i:03d}.npz') as saved:w={k:saved[k] for k in saved.files}
            assert not w['mask'][0].any()
            assert abs(w['times'][-1]-w['times'][0]-5.)<1e-12
            for mode,entry in row['methods'].items():
                with np.load(out/f'{mode}_{i:03d}.npz') as saved:
                    truth=true_disturbance(row['start']+sigma,kind)
                    max_truth=max(max_truth,float(np.max(np.abs(truth-saved['disturbance_truth']))))
                    np.testing.assert_allclose(saved['disturbance_truth'],truth,rtol=2e-14,atol=2e-16)
                    estimate=UniformBasis(3,4).evaluate(sigma,5.)@saved['coefficients'].reshape(2,7).T
                    np.testing.assert_allclose(saved['disturbance_error'],estimate-truth,rtol=2e-14,atol=2e-16)
                    np.testing.assert_array_equal(profiles[mode+'_errors'][i],saved['disturbance_error'])
                    assert int(saved['correction_mask'].sum())+int(saved['initialization_mask'].sum())==int(w['mask'].sum())
                    assert np.array_equal(saved['correction_mask'] | saved['initialization_mask'],w['mask'])
                    q=saved['whitened_residual'];delta=abs(float(q@q)-entry['optimization']['objective_final'])
                    max_cost=max(max_cost,delta);assert delta<2e-9
                    for key in ('x','covariance'):assert np.isfinite(saved[key]).all()
                    asymmetry=np.max(np.abs(saved['covariance']-saved['covariance'].transpose(0,2,1)))
                    assert asymmetry<=1e-10
            if i==0:
                np.testing.assert_array_equal(row['methods']['projected']['optimization']['coefficients'],row['methods']['zero']['optimization']['coefficients'])
        for mode in ('projected','zero'):
            p=error_profile(profiles[mode+'_errors'],sigma)
            np.testing.assert_array_equal(p['rmse'],r['profiles'][mode]['rmse'])
    checks['maximum_independent_truth_difference']=max_truth
    checks['maximum_whitened_cost_difference']=max_cost
    return checks


def write_report(root):
    cases={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/results.json'))}
    if not cases:return
    summary=collect(cases);basis=basis_diagnostics();checks=audit(root,cases)
    snapshot=json.loads((root/'before_sha256.json').read_text())
    changed=[p for p,h in snapshot.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    checks['all_previous_files_preserved']=not changed
    tests=json.loads((root/'test_record.json').read_text());checks['regression_tests_passed']=tests['passed']
    selected=json.loads((root/'saved_validation.json').read_text()) if (root/'saved_validation.json').exists() else None
    checks['selected_sensitivity_FD_and_reproducibility']=selected is not None and selected['passed']
    plotting=figures(root,cases,summary)
    phases={}
    def add(label,value):phases.setdefault(label,[]).append(value)
    for r in cases.values():
        add('scenario_truth_generation_seconds',r['generation_seconds'])
        add('scenario_forward_history_seconds_including_compile',r['forward_history_seconds_including_compile'])
        for row in r['updates']:
            for key,value in row['preparation'].items():
                if key=='basis_seconds':continue  # cProfile omitted this frame: unresolved, not zero work.
                add('buffer_plus_basis_remainder_seconds' if key=='observation_buffer_seconds' else key,value)
            add('window_NPZ_artifact_seconds',row['window_artifact_seconds'])
            for entry in row['methods'].values():
                for key,value in entry['timing'].items():add(key,value)
                for key,value in entry['optimization']['timing'].items():
                    if key!='program_cache_hit' and isinstance(value,(int,float)) and value is not None:add('optimizer_'+key,value)
                for ev in entry['optimization']['evaluation_history']:
                    if ev['phase']=='optimization':
                        add('trial_checked_objective_seconds',ev['validity_check_seconds'])
                        add('trial_value_gradient_seconds',ev['value_gradient_seconds'])
                for key in ('diagnostic_seconds','artifact_seconds'):add(key,entry[key])
                for key,value in entry['sensitivity_timing'].items():add('sensitivity_'+key,value)
    for event in FIGURE_TIMINGS:add('figure_publication_seconds',event['publication_seconds'])
    host_path=root/'host_profile_remeasurement.json'
    host=json.loads(host_path.read_text()) if host_path.exists() else None
    if host:
        for row in host['records']:
            for key in ('window_seconds','arrival_seconds','basis_seconds','buffer_other_seconds'):
                add('host_remeasurement_'+key,row[key])
        checks['direct_host_profile_saved_windows_equal']=host['passed'] and host['all_saved_window_fields_bitwise_equal']
    # Independent remeasurement, not retroactively added to online-fit timings.
    for r in cases.values():
        begin=time.perf_counter();json.dumps(r,allow_nan=False);add('JSON_serialization_remeasurement_seconds',time.perf_counter()-begin)
    runtime={k:stats(v) for k,v in phases.items()}
    (root/'diagnostics.json').write_text(json.dumps({'cases':summary,'basis':basis,'validation':checks,'plotting_seconds':plotting,
        'runtime_phases':runtime,'figure_publication':FIGURE_TIMINGS},indent=2,allow_nan=False)+'\n')
    (root/'preservation.json').write_text(json.dumps({'checked_count':len(snapshot),'changed':changed,'all_preserved':not changed},indent=2)+'\n')
    lines=['# Stage 5C — moving-horizon diagnostics and computational profiling','',
        '## Executive summary','',
        f"Six seed-7 scenarios are retained; available completed cases: {len(cases)}. The selected duration is 6.0 s, "
        'with 11 complete windows per case at 5.0–6.0 s every 0.1 s, compared with three in Stage 5B. '
        'Both projected/clipped and zero starts are optimized independently. All original d=3, M=4, L=5 s, '
        'noise, prior, bounds, optimizer options, sensors and arrival rules remain. No estimator source file is changed.','',
        'Sensors are colocated/aligned: regression range [0,infinity], 360 degrees; forward range [0.2,6] m, '
        '180 degrees. RK4 uses 0.025 s steps. Commands are [1 m/s,0.2 rad/s]. Independent zero-disturbance '
        'forward-EKF posteriors supply arrivals, including full SLAM covariance; reported MHE maps never feed '
        'back into that filter. Measurement noise is diag(0.05^2 m^2,(1 degree)^2); continuous input-noise '
        'intensity is diag(0.01,0.0025), with the original units/convention. The pose prior is '
        'diag(0.05^2,0.05^2,(2 degrees)^2). Full retained configurations appear in each case\'s results.json.','',
        'The compilation feasibility test reproduced every result/gradient bitwise on three saved Stage 5B windows '
        'and reproduced a full solve exactly. A benchmark-only adapter lifts captured arrays into runtime arguments '
        'of the same traced JAX program. It does not change data shapes, masks, slots, birth plans or EKF equations. '
        'See [compilation feasibility](compilation_feasibility.md) and feasibility.json.','',
        'These are deterministic diagnostics, not independent trials or confidence estimates. All windows overlap '
        'heavily. Shaded plot regions define the interior and never represent confidence intervals.','',
        '### Findings at a glance','']
    fits=[entry['optimization'] for r in cases.values() for row in r['updates'] for entry in row['methods'].values()]
    flags=[[s['methods']['projected']['profile']['right_exceeds_interior'][ch] for s in summary.values()] for ch in (0,1)]
    degraded=[name for name,s in summary.items() if s['methods']['projected']['position_rmse_m']>s['forward']['position_rmse_m']]
    cache_events=json.loads((root/'cache_events.json').read_text()) if (root/'cache_events.json').exists() else []
    lines += [f"- Completed {sum(r['window_count'] for r in cases.values())} windows and {len(fits)} fits; {sum(f['success'] for f in fits)} fits report convergence. No solver option or noise parameter was tuned.",
        f"- The right endpoint exceeds the defined interior in {sum(flags[0])}/{len(flags[0])} linear and {sum(flags[1])}/{len(flags[1])} angular warm-start profiles. The case-by-case table below tests consistency rather than presuming it.",
        f"- Warm-started position RMSE is worse than the forward EKF in: {', '.join(degraded) if degraded else 'none of the completed cases'}. Lower innovation cost does not establish improved truth accuracy.",
        f"- {len(set(e['key'] for e in cache_events))} structural replay program(s) served {len(cache_events)} preparation calls. Remaining warm objective/gradient/SciPy work still dominates many updates; this is not real-time performance.",
        '- All 132 fits terminate by the unchanged function-reduction criterion; none meets projected-gradient gtol=1e-6. Solver success is therefore reported separately from first-order stationarity.',
        '- The two boundary basis energies are symmetric and small relative to interior functions. Sensitivity, measurement timing, arrival information and fitted bounds must be examined together; basis support alone cannot explain a right/left error asymmetry.','',
        '## Position-dependent disturbance reconstruction','',
        'The 501-point grid includes sigma=0,5 and all knots 0,1.25,2.5,3.75,5. '
        'E(sigma)=sqrt(mean_windows(error(sigma)^2)). The primary interior is [1.25,3.75] s; its scalar RMS '
        'is sqrt(mean_sigma(E(sigma)^2)). The secondary open-interior RMS excludes only the two endpoints '
        'and is also retained in JSON. Endpoints and every window are included without discarding empty scans.','',
        '| Case | Start | Left / interior / right w_v RMSE (m/s) | Right > interior? | Left / interior / right w_omega RMSE (rad/s) | Right > interior? |',
        '|---|---|---|---|---|---|']
    for name,s in summary.items():
        for mode,m in s['methods'].items():
            p=m['profile'];v=[p[k][0] for k in ('left','interior','right')];w=[p[k][1] for k in ('left','interior','right')]
            lines.append(f"| {name} | {mode} | {' / '.join(f'{x:.6g}' for x in v)} | {p['right_exceeds_interior'][0]} | "
                         f"{' / '.join(f'{x:.6g}' for x in w)} | {p['right_exceeds_interior'][1]} |")
    for ch,label in [(0,'linear'),(1,'angular')]:
        flags=[s['methods']['projected']['profile']['right_exceeds_interior'][ch] for s in summary.values()]
        lines+=['',f"Warm-started {label}: right exceeds the defined interior in **{sum(flags)}/{len(flags)}** completed scenarios. "
                'This is a deterministic comparison; it is not a statistical test of an independent population.']
    lines+=['','![Linear profiles](linear_error_profiles.png)','', '![Angular profiles](angular_error_profiles.png)','',
            '## Basis support and exact whitened coefficient sensitivity','',
            '| j | Support inside window (s) | Integral of b_j squared (s) | Right endpoint weight |',
            '|---:|---|---:|---:|']
    for j in range(7):lines.append(f"| {j} | {basis['support_in_window'][j]} | {basis['integrated_squared_magnitude'][j]:.9g} | {basis['right_endpoint_weights'][j]:.7g} |")
    lines+=['','![Basis and support](basis_support.png)','',
            'Basis energies are symmetric at the two boundaries. That representation fact alone cannot explain a '
            'right/left asymmetry in estimation error. The right endpoint uses coefficients 4,5,6 with weights '
            '[1/6,4/6,1/6]; focusing solely on the last coefficient would miss the larger weight of coefficient 5.','',
            'For every fit, q=L(c)^-1 r(c), where S(c)=L(c)L(c).T is lower Cholesky in the existing range/bearing '
            'and observation order. D=dq/dc differentiates both residuals **and covariance whitening**. This differs '
            'from the earlier fixed-S sensitivity diagnostic. Only correction rows enter q; padded/birth rows '
            'have zero diagnostic residual and are removed before SVD. q.T q=J and 2 D.T q=grad J are checked '
            'for every optimized window. Numerical rank uses max(rows,14)*eps64*s_max; practical rank at 1e-3 '
            'relative threshold is separately reported. Condition number is null for numerical deficiency. '
            'Columns retain physical coefficient units; cross-channel spectra are parameterization dependent. '
            'When S depends on coefficients, alternative valid whitening conventions can introduce coefficient-dependent '
            'rotations, so D spectra are also whitening-convention dependent. Lower forward-sensor column norms partly '
            'reflect fewer correction rows; no per-observation normalization is applied. This local analysis is not '
            'formal nonlinear observability.','',
            '| Case | Start | Rank range | Finite condition min–max / deficient windows | Median rightmost v / omega column norm | gtol met / windows | Max projected gradient |',
            '|---|---|---|---|---|---|---:|']
    for name,s in summary.items():
        for mode,m in s['methods'].items():
            finite=[v for v in m['condition_numbers'] if v is not None]
            condition=f'{min(finite):.4g}–{max(finite):.4g}' if finite else 'none'
            deficient=sum(v is None for v in m['condition_numbers'])
            lines.append(f"| {name} | {mode} | {min(m['numerical_ranks'])}–{max(m['numerical_ranks'])} | {condition} / {deficient} | {m['median_column_norms'][6]:.7g} / {m['median_column_norms'][13]:.7g} | {m['gtol_met_count']}/{s['window_count']} | {m['max_projected_gradient']:.6g} |")
    lines+=['','| Case | Warm channel | Median column norms: left j0 / middle j2–4 / right j6 | Right weaker than left? |',
            '|---|---|---|---|']
    for name,s in summary.items():
        norms=np.asarray(s['methods']['projected']['median_column_norms'])
        for offset,label in [(0,'v'),(7,'omega')]:
            left,mid,right=norms[offset],np.median(norms[offset+2:offset+5]),norms[offset+6]
            lines.append(f"| {name} | {label} | {left:.6g} / {mid:.6g} / {right:.6g} | {right<left} |")
    lines+=['','## Visibility, last-correction age and endpoint errors','',
            'Every update retains scheduled scans, observed/empty scans, corrections, births, mapped counts, '
            'window-relative observation times/IDs and time since the latest valid correction. The latter includes '
            'a preceding forward correction if the window itself has none. Births do not count as corrections.','',
            '| Case | Scheduled scans | Observed scans | Empty scans | Corrections | Births | Active map | Latest-correction age (s) |',
            '|---|---|---|---|---|---|---|---|']
    for name,r in cases.items():
        vs=[x['methods']['projected']['visibility'] for x in r['updates']]
        def span(key):
            values=[v[key] for v in vs if v[key] is not None]
            return f'{min(values):.5g}–{max(values):.5g}' if values else 'none'
        lines.append(f"| {name} | {span('scheduled_scans')} | {span('scans_with_observations')} | {span('empty_scans')} | {span('correction_records')} | {span('birth_records')} | {span('active_landmarks')} | {span('seconds_since_latest_correction')} |")
    lines+=['',
            '| Case | Start | Observed endpoints N / right RMSE v,omega | Empty endpoints N / right RMSE v,omega |',
            '|---|---|---|---|']
    for name,s in summary.items():
        for mode,m in s['methods'].items():
            a=m['endpoint_groups']['observed'];b=m['endpoint_groups']['empty']
            lines.append(f"| {name} | {mode} | {a['count']} / {a['right_rmse']} | {b['count']} / {b['right_rmse']} |")
    lines+=['','![Latest correction age and endpoint errors](correction_age_endpoint_error.png)','',
            'Group counts matter: a scenario with only empty endpoints has no within-scenario observed/empty '
            'contrast. Cross-scenario differences also change motion, map coverage and residual information. '
            'Correlations in diagnostics.json are descriptive only; no p-values, confidence intervals or causal '
            'claims are inferred from overlapping windows.','',
            '## Pose/map degradation, arrivals and numerical optimization','',
            '| Case | Method | Position RMSE m | Heading RMSE rad | Landmark RMSE m | Mapped count range |',
            '|---|---|---:|---:|---:|---|']
    for name,s in summary.items():
        for mode,m in [('forward',s['forward']),*s['methods'].items()]:
            lines.append(f"| {name} | {mode} | {m['position_rmse_m']:.7g} | {m['heading_rmse_rad']:.7g} | {m['landmark_rmse_m']:.7g} | {min(m['mapped_counts'])}–{max(m['mapped_counts'])} |")
    lines+=['','Each update stores arrival pose/map errors and full active covariance diagnostics, all reported '
            'errors, active/near-bound coefficients (1e-10/1e-3 distance thresholds), projected-gradient norms, '
            'termination reasons, and consecutive endpoint/overlap revisions. Arrival error is evaluated at t-L, '
            'not t. Maps use known IDs and the original anchored frame with no post-hoc alignment. '
            '**Conditional EKF covariance; spline coefficients treated as fixed.**','',
            'An innovation-cost reduction is not a reduction in truth error. In addition to J, each update stores '
            'logdet diagnostics, innovation variances and the optimized residual quadratic using baseline S. '
            'These are diagnostic decompositions, not alternate optimized objectives. A lower cost can coexist '
            'with fitting a particular noise realization, weak input directions, representation error, or pose/map '
            'coupling. Nonzero arrival error may contribute, but the first window has zero arrival mean error '
            'at t=0; degradation there cannot require arrival mean bias. These observations do not establish '
            'a unique causal explanation. SciPy convergence often uses function reduction and need not meet gtol.','',
            'The zero-disturbance cases have exactly zero true disturbance and an exact zero-input forward model. '
            'Any fitted nonzero input is therefore reconstruction error for this realization. Low-cost nonzero '
            'solutions demonstrate that minimizing the innovation quadratic is not an input-accuracy criterion. '
            'The smooth input is outside the exact piecewise-cubic spline family. None of these observations '
            'justifies changing the objective during this diagnostic stage.','',
            '### Focused degradation evidence','']
    for name in ('regression_smooth','regression_zero','forward_zero'):
        if name not in cases:continue
        rows=cases[name]['updates'];m=summary[name]['methods']['projected'];b=summary[name]['forward']
        arrivals=np.array([x['arrival_error']['position_error_m'] for x in rows])
        residual_part=np.array([x['baseline_cost']-x['methods']['projected']['objective_diagnostics']['J_with_baseline_S'] for x in rows])
        weight_part=np.array([x['methods']['projected']['objective_diagnostics']['J_with_baseline_S']-x['methods']['projected']['optimization']['objective_final'] for x in rows])
        differences=[abs(x['methods']['projected']['error']['position_error_m']-x['methods']['zero']['error']['position_error_m']) for x in rows]
        lines += [f"**{name}:** forward/warm position RMSE {b['position_rmse_m']:.7g}/{m['position_rmse_m']:.7g} m; "
            f"arrival position error ranges {arrivals.min():.7g}–{arrivals.max():.7g} m. "
            f"The first arrival has position/heading error {rows[0]['arrival_error']['position_error_m']:.4g} m / "
            f"{rows[0]['arrival_error']['heading_error_rad']:.4g} rad. Local D rank ranges "
            f"{min(m['numerical_ranks'])}–{max(m['numerical_ranks'])}; active-bound counts range "
            f"{min(m['active_bound_counts'])}–{max(m['active_bound_counts'])}; gtol is met in {m['gtol_met_count']}/{len(rows)} fits. "
            f"Warm/zero report-position-error difference is at most {max(differences):.5g} m. "
            f"Mean cost reduction decomposes algebraically into {residual_part.mean():.6g} using optimized residuals "
            f"with baseline S, plus {weight_part.mean():.6g} from changing S at those residuals. "
            'This decomposition is conditional and order dependent; it does not identify a causal mechanism.','']
    lines+=['| Case | Start | Maximum overlap pose revision m | Maximum matched-landmark overlap revision m |',
            '|---|---|---:|---:|']
    for name,s in summary.items():
        for mode,m in s['methods'].items():
            lines.append(f"| {name} | {mode} | {m['max_overlap_pose_revision_m']:.7g} | {m['max_map_revision_m']:.7g} |")
    lines+=['','Overlap revisions compare two window solutions at the same physical time and matched landmark '
            'IDs. They differ from ordinary motion between report times. All report-to-report pose/map deltas '
            'are also retained in updates.json. The three focused cases degrade despite observed endpoint scans '
            'and numerical full rank. Their near-identical warm/zero errors provide little evidence that the '
            'warm-start policy causes that degradation. Nonzero fitted inputs in zero-truth cases, weak boundary '
            'columns, active bounds and function-reduction termination are supported observations; noise fitting '
            'and pose/map/input coupling remain plausible explanations, rather than proven isolated causes.','']
    lines+=['## Warm starts and computational profile','',
            '| Case | Start | Iterations | Function / gradient calls | Converged | Clipped updates | Update mean / median / p95 s | SciPy optimization mean s |',
            '|---|---|---:|---|---:|---:|---|---:|']
    for name,s in summary.items():
        for mode,m in s['methods'].items():
            t=m['timings']['update_seconds'];opt=m['timings']['optimization_seconds']
            lines.append(f"| {name} | {mode} | {m['iterations']} | {m['function_evaluations']} / {m['gradient_evaluations']} | {m['converged_count']} | {m['clipped_updates']} | {t['mean']:.6g} / {t['median']:.6g} / {t['p95']:.6g} | {opt['mean']:.6g} |")
    lines+=['','| Case | Start | Mean initial / final J | Dense-grid w_v / w_omega RMSE | Maximum clipping magnitude |',
            '|---|---|---|---|---:|']
    for name,s in summary.items():
        for mode,m in s['methods'].items():
            lines.append(f"| {name} | {mode} | {m['initial_cost']['mean']:.7g} / {m['final_cost']['mean']:.7g} | "
                         f"{m['disturbance_rmse_v_omega'][0]:.7g} / {m['disturbance_rmse_v_omega'][1]:.7g} | {m['max_clipping']:.7g} |")
    lines+=['','Every per-window initial/final cost, call count, termination message, gradient norm, projected '
            'gradient norm, clipping count/magnitude and measured runtime remains in each case\'s updates.json. '
            'Totals and means above summarize the same paired windows; they do not select either solution.']
    for name,r in cases.items():
        worse=[i for i,x in enumerate(r['updates']) if x['methods']['projected']['optimization']['iterations']>x['methods']['zero']['optimization']['iterations']]
        slower=[i for i,x in enumerate(r['updates']) if x['methods']['projected']['timing']['update_seconds']>x['methods']['zero']['timing']['update_seconds']]
        lines+=['',f"{name}: projected starts use more iterations in {len(worse)}/{len(r['updates'])} windows "
                f"(indices {worse}) and higher measured update time in {len(slower)}/{len(r['updates'])}. "
                'The two controlled solutions are retained; neither is automatically selected.']
    lines+=['','The nested timing table in diagnostics.json records mean, median, p95 and totals for window '
            'preparation, synchronized arrival retrieval, buffer/basis remainder, host birth-plan '
            'preparation, warm projection, clipping, tracing, lowering, compilation, checked objective evaluation, '
            'joint value/gradient execution, SciPy optimization, final replay, diagnostics and artifacts. '
            'Some categories are nested: SciPy time includes trial objective/gradient work; window preparation '
            'includes retrieval/buffers/basis; factory time includes birth plans. They must not be summed twice. '
            'Value/gradient timing includes the value, as required by the unchanged SciPy pair. '
            'Update timing includes host initialization and final replay, excludes sensitivity/diagnostic/IO work. '
            'Preparation profiling uses cProfile and has instrumentation overhead; plotting is timed separately.','',
            '**Host-profiler limitation and correction:** cProfile omitted the completed UniformBasis.evaluate '
            'frame in this Python 3.12/pybind environment. Its raw basis_seconds=0 is unresolved and excluded '
            'from timing summaries; the original observation_buffer_seconds is explicitly relabeled '
            'buffer_plus_basis_remainder_seconds. Separate direct post-run timers in '
            'host_profile_remeasurement.json measure basis work and the remaining host buffer work on all '
            '66 saved windows, with three repetitions each and bitwise comparison of every window/arrival '
            'field. These are labeled remeasurements and never replace online fit timings.','',
            f"Plot generation wall time for this report: {plotting:.4g} s. Cache events and detailed lower/compile "
            'measurements remain in cache_events.json. The same operation graph is reused only on an exact '
            'structural fingerprint match. Any new shape/structure compiles separately; no shape is forced. '
            'The adapter is pinned to JAX 0.4.35 and is not a production API/refactor. A major runtime-parameter '
            'redesign remains deferred pending approval.','',
            '| Timing category | N | Mean s | Median s | p95 s | Total s |',
            '|---|---:|---:|---:|---:|---:|']
    for key,t in runtime.items():
        lines.append(f"| {key} | {t['count']} | {t['mean']:.7g} | {t['median']:.7g} | {t['p95']:.7g} | {t['total']:.7g} |")
    lines+=['','NPZ artifact timings are from the fit run; JSON serialization timings are separate post-run '
            'remeasurements. Figure-publication samples include layout/render/write/close for PNG and PDF; '
            'the overall plot timer also includes figure construction. Cold compilation appears as nonzero '
            'events once per new structural program, with zeros for cache hits; medians can therefore be zero. '
            'One structural replay graph still has separate evaluator, value/gradient, sensitivity and checked-AD '
            'auxiliary executables. It does not mean a single XLA compilation for all operations.','',
            '## Representative window and time-series figures','']
    for name,r in cases.items():
        lines += [f'### {name}','',f'![{name} arrival and estimation]({name}/pose_map_arrival.png)','',
                  f'![{name} sensitivity]({name}/sensitivity.png)','']
        for i in sorted(set((0,len(r['updates'])//2,len(r['updates'])-1))):
            lines += [f'![{name} window {i}]({name}/window_{i:03d}_disturbances_observations.png)','']
    lines+=['## Numerical validation, preservation and reproduction','',
            f"Complete repository test result: {tests['passed_tests']} passed, exit {tests['exit_code']}. "
            'Initial diagnostic tests and full regression stdout/XML are retained. The tests include Stage 5A '
            'projection, Stage 5B windows/births/empty scans, Stage 4C regression behavior, independent profile '
            'truth/time checks, covariance, exact compiled outputs/gradients, whitening finite differences and '
            'optimizer-orchestration identity. Selected full-size FD and repeat-fit checks are in saved_validation.json.','',
            '```sh',tests['command'],'```','',
            f"All {len(snapshot)} pre-task protected files preserved: {not changed}. Changed protected files: {changed}. "
            'Stage 4C, 5A, 5B artifacts and all original mathematical implementations are unchanged.','',
            '```json',json.dumps(checks,indent=2),'```','',
            '```sh',
            'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.stage5c_feasibility',
            'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c',
            'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c --validate',
            'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.stage5c_host_profile',
            'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5c --report-only',
            '```','',
            '## Recommended next steps and stopping condition','',
            'Review the unconstrained terminal-input directions, noise-fit evidence and function-reduction '
            'termination before changing estimator design. The saved feasible probes provide a reproducible '
            'test case for any future proposal concerning poorly informed terminal inputs. Review diagnostic '
            'parameter scaling and profile a longer deterministic duration before selecting a future paired '
            'seed/Monte Carlo budget; such runs require approval and are not performed here. If production '
            'compilation reuse is desired, approve a separate runtime-argument API change rather than adopting '
            'the pinned internal-JAX benchmark adapter without review. '
            'No Monte Carlo, nearest-neighbor association, Victoria Park, covariance tuning, regularization or '
            'objective change was started. Work stops after this diagnostic stage.']
    if selected:
        probes=selected['checks'].get('forward_constant',{}).get('feasible_coefficient_probes',[])
        if probes:
            block=['','## Feasible terminal-coefficient perturbation diagnostic','',
                    'These fixed-model evaluations neither reoptimize nor replace any reported solution. '
                    'They directly compare cost and reported endpoint state under feasible coefficient changes.','',
                    '| Coefficient | Delta | Absolute cost change | Max q change | Endpoint pose change (x,y,theta) |',
                    '|---:|---:|---:|---:|---|']
            for p in probes:
                block.append(f"| {p['coefficient']} | {p['delta']} | {p['cost_absolute_change']:.8g} | {p['q_max_absolute_change']:.8g} | {p['endpoint_pose_change']} |")
            last=cases['forward_constant']['updates'][-1]
            age=last['methods']['projected']['visibility']['seconds_since_latest_correction']
            relative=5.-age
            block+=['',f"The final forward/constant window is [{last['start']:g},{last['time']:g}] s. Its latest "
                    f"correction is at sigma={relative:.6g} s, while coefficient j=6 has support beginning at "
                    'sigma=3.75 s. Both channel-j6 columns of D are exactly zero and the numerical rank is 12. '
                    'The feasible probes above leave all correction residuals and J unchanged while changing the '
                    'endpoint pose. Thus these terminal input components are unconstrained by the recorded '
                    'innovation objective in this particular window; warm/zero differences in that interval '
                    'need not signal a failed optimizer. This finding is specific to the saved observation '
                    'schedule. Other cases have larger endpoint errors with full local rank and observed '
                    'endpoint scans, so this mechanism cannot explain all boundary error.']
            location=lines.index('## Numerical validation, preservation and reproduction')
            lines[location:location]=block+['']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    if changed:raise RuntimeError('Protected files changed')
