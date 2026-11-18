"""Extended deterministic diagnostics with an isolated compilation cache."""

import argparse
import cProfile
import hashlib
import json
import pstats
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.milestone4c import configuration,covariance_summary
from benchmarks.stage5c_cache import ProgramCache,optimize_cached,whitened_residual
from benchmarks.stage5c_diagnostics import (
    grid,disturbance_error,error_profile,basis_diagnostics,timed_initial,visibility,
    covariance_diagnostics,state_error,sensitivity_summary,
)
from estimation.fixed_window_objective import checked_evaluation,InvalidCandidateError
from estimation.moving_horizon import ReplayData,ForwardArrivals,canonical_time,pose_difference,map_difference
from estimation.slam import create_state,SlamState
from estimation.slam_window_objective import make_slam_window_objective
from simulation.moving_horizon import simulate_moving
from simulation.slam_window import sensor_settings


ROOT=Path('artifacts/milestone5c')
CASES=[s+'_'+k for s in ('regression','forward') for k in ('zero','constant','smooth')]


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def profiled(fn):
    profile=cProfile.Profile();begin=time.perf_counter();profile.enable()
    value=fn();profile.disable();seconds=time.perf_counter()-begin
    return value,seconds,pstats.Stats(profile).stats


class TimedForward(ForwardArrivals):
    def at(self,t):
        start=time.perf_counter();state=super().at(t);jax.block_until_ready(state)
        self.arrival_seconds=time.perf_counter()-start
        return state


def make_evaluator(forward,w):
    return make_slam_window_objective(w.arrival,w.times,forward.command,w.basis,forward.Qc,
        w.measurements,w.ids,forward.data.covariance,w.mask)


def run_case(name,cache,duration=6.):
    sensor,kind=name.split('_');out=ROOT/name;out.mkdir(exist_ok=True)
    cfg=configuration('smooth' if kind=='smooth' else 'constant')
    cfg.update(sensor=sensor_settings(sensor),disturbance_kind=kind,total_duration_seconds=duration,
        initialization_seconds=0.,disturbance_start_seconds=0.,
        optimization_observations='a_k < t_i <= t_k; boundary posterior belongs to the forward arrival',
        first_full_horizon_seconds=5.)
    cfg.pop('arrival_a',None)
    if kind=='zero':cfg['true_disturbance_v_omega']=[0.,0.]
    if kind=='smooth':cfg['true_disturbance_functions']=['0.15*sin(2*pi*t/5)','0.05*cos(2*pi*t/5)']
    begin=time.perf_counter();data,scheduled,truth_check=simulate_moving(cfg)
    generation_seconds=time.perf_counter()-begin
    ids=np.broadcast_to(np.arange(len(data.landmark_map),dtype=np.int64),data.mask.shape)
    replay=ReplayData(data.times,data.measurements,ids,data.measurement_covariance,data.mask,scheduled)
    command_value=jnp.asarray(cfg['command_v_omega'])
    initial=create_state(cfg['initial_pose'],cfg['arrival_covariance'],cfg['capacity'])
    begin=time.perf_counter()
    forward=TimedForward(replay,initial,lambda t:command_value,cfg['continuous_input_noise_spectral_density_Qc'])
    forward_seconds=time.perf_counter()-begin
    np.savez_compressed(out/'dataset.npz',**data._asdict(),scheduled=scheduled,ids=ids)
    np.savez_compressed(out/'forward.npz',**{k:np.asarray(v) for k,v in forward.result.states._asdict().items()},
                        statuses=np.asarray(forward.result.observations.status))
    previous={mode:None for mode in ('projected','zero')};rows=[]
    all_errors={mode:[] for mode in previous};sigma=grid();report_states={mode:[] for mode in previous}
    for end in data.times[scheduled]:
        if canonical_time(end-5.,data.times)<data.times[0]:continue
        index=len(rows)
        w,prep,stat=profiled(lambda:forward.window(float(end),5.))
        arrival_seconds=forward.arrival_seconds
        basis_seconds=sum(v[3] for (file,line,func),v in stat.items() if file.endswith('uniform_basis.py') and func=='evaluate')
        evaluate,factory_seconds,stat=profiled(lambda:make_evaluator(forward,w))
        plan_seconds=sum(v[3] for (file,line,func),v in stat.items() if file.endswith('slam_window_objective.py') and func=='fixed_observation_plan')
        arrival_index=int(np.argmin(np.abs(data.times-w.start)))
        end_index=int(np.argmin(np.abs(data.times-w.end)))
        base=forward.at(w.end)
        row={'time':float(w.end),'start':float(w.start),
            'preparation':{'window_total_seconds':prep,'arrival_retrieval_seconds':arrival_seconds,
                'basis_seconds':basis_seconds,'observation_buffer_seconds':max(0.,prep-arrival_seconds-basis_seconds),
                'objective_factory_seconds':factory_seconds,'landmark_observation_plan_seconds':plan_seconds},
            'arrival_error':state_error(w.arrival,data.truth[arrival_index],data.landmark_map),
            'arrival_covariance':covariance_diagnostics(w.arrival),
            'forward_error':state_error(base,data.truth[end_index],data.landmark_map),'methods':{}}
        rows.append(row)
        begin_artifact=time.perf_counter()
        np.savez_compressed(out/f'window_{index:03d}.npz',times=w.times,basis=w.basis,measurements=w.measurements,
            ids=w.ids,mask=w.mask,source_indices=w.source_indices,
            **{'arrival_'+k:np.asarray(v) for k,v in w.arrival._asdict().items()})
        row['window_artifact_seconds']=time.perf_counter()-begin_artifact
        baseline_result=None
        for mode in previous:
            total_begin=time.perf_counter();prior=previous[mode]
            delta=0. if prior is None else float(end)-prior['time']
            initial,clipping,init_timing=timed_initial(None if prior is None else prior['coefficients'],delta,mode)
            fit=optimize_cached(cache,evaluate,initial)
            record={'optimization':fit,'clipping':clipping,'initialization_timing':init_timing}
            row['methods'][mode]=record
            if fit['invalid_trials'] or fit['coefficients'] is None:
                write_json(out/'failure.json',{'case':name,'time':float(end),'mode':mode,'fit':fit})
                raise InvalidCandidateError(f'{name}/{mode}/{end}: {fit["status"]}')
            c=np.asarray(fit['coefficients'])
            entry,constants,extra_trace,key=cache.prepare(evaluate,c)
            begin=time.perf_counter();result=checked_evaluation(lambda x:entry['compiled_evaluate'](x,constants),c)
            final_replay=time.perf_counter()-begin
            record['timing']={**init_timing,'additional_trace_seconds':extra_trace['program_trace_seconds'],
                'final_replay_seconds':final_replay,
                'update_seconds':time.perf_counter()-total_begin+prep+factory_seconds}
            begin=time.perf_counter()
            if baseline_result is None:
                baseline_result=checked_evaluation(lambda x:entry['compiled_evaluate'](x,constants),np.zeros(14))
                for x,y in zip(baseline_result.final_state,base):np.testing.assert_allclose(x,y,rtol=0,atol=2e-14)
                row['baseline_cost']=float(baseline_result.innovation_quadratic)
            record['error']=state_error(result.final_state,data.truth[end_index],data.landmark_map)
            record['visibility']=visibility(w,result,replay,forward.result)
            record['covariance_history']=covariance_summary(result.states)
            record['final_covariance']=covariance_diagnostics(result.final_state)
            error,estimate,truth=disturbance_error(c,w.start,kind,sigma)
            all_errors[mode].append(error);report_states[mode].append(result.final_state)
            D,sens_timing=cache.sensitivity(entry,constants,c)
            q=np.asarray(whitened_residual(result))
            value,gradient=entry['compiled_pair'](c,constants);gradient=np.asarray(gradient)
            gradient_identity=float(np.max(np.abs(2*D.T@q-gradient)))
            np.testing.assert_allclose(2*D.T@q,gradient,rtol=2e-9,atol=2e-8)
            np.testing.assert_allclose(q@q,value,rtol=2e-12,atol=2e-12)
            record['sensitivity']=sensitivity_summary(D,result.valid_mask)
            record['sensitivity_timing']=sens_timing
            record['whitening_gradient_identity_error']=gradient_identity
            record['gradient']=gradient.tolist()
            record['active_bound_indices']=np.flatnonzero(np.abs(np.abs(c)-.5)<=1e-10).tolist()
            record['near_bound_indices_1e3']=np.flatnonzero(np.abs(np.abs(c)-.5)<=1e-3).tolist()
            record['objective_diagnostics']={
                'logdet':float(result.logdet_diagnostic),'baseline_logdet':float(baseline_result.logdet_diagnostic),
                'J_with_baseline_S':float(np.sum(np.asarray(result.innovations)[result.valid_mask]*np.linalg.solve(
                    np.asarray(baseline_result.innovation_covariances)[baseline_result.valid_mask],
                    np.asarray(result.innovations)[result.valid_mask][...,None])[...,0])),
                'mean_innovation_variances':(np.diagonal(np.asarray(result.innovation_covariances)[result.valid_mask],axis1=1,axis2=2).mean(axis=0).tolist()
                                             if np.asarray(result.valid_mask).any() else None)}
            if prior is not None:
                prev_index=int(np.argmin(np.abs(w.times-prior['time'])))
                revised=SlamState(*(v[prev_index] for v in result.states))
                record['jumps']={'pose_revision':pose_difference(revised.x,prior['state'].x).tolist(),
                    'map_revision':map_difference(revised,prior['state']),
                    'reported_pose_delta':pose_difference(result.final_state.x,prior['state'].x).tolist(),
                    'reported_map_delta':map_difference(result.final_state,prior['state'])}
            record['diagnostic_seconds']=time.perf_counter()-begin
            begin=time.perf_counter()
            valid=np.asarray(result.valid_mask)
            np.savez_compressed(out/f'{mode}_{index:03d}.npz',
                **{k:np.asarray(v) for k,v in result.states._asdict().items()},
                statuses=np.asarray(result.statuses),slots=np.asarray(result.slots),
                correction_mask=valid,initialization_mask=np.asarray(result.initialization_mask),
                innovations=np.asarray(result.innovations),innovation_covariances=np.asarray(result.innovation_covariances),
                coefficients=c,disturbance_error=error,disturbance_estimate=estimate,disturbance_truth=truth,
                sensitivity=D[np.repeat(valid.ravel(),2)],whitened_residual=q[np.repeat(valid.ravel(),2)])
            record['artifact_seconds']=time.perf_counter()-begin
            previous[mode]={'time':float(end),'coefficients':c,'state':result.final_state}
        write_json(out/'updates.json',rows)
        print(json.dumps({'case':name,'time':float(end),'window':index,
            'fits':{m:{'iterations':r['optimization']['iterations'],'J':r['optimization']['objective_final'],
                       'rank':r['sensitivity']['numerical_rank'],'empty':r['visibility']['empty_endpoint'],
                       'update_seconds':r['timing']['update_seconds']} for m,r in row['methods'].items()}}),flush=True)
    profiles={m:error_profile(all_errors[m],sigma) for m in previous}
    np.savez_compressed(out/'profiles.npz',sigma=sigma,**{m+'_errors':np.asarray(all_errors[m]) for m in previous})
    payload={'configuration':cfg,'truth_validation':truth_check,'duration':duration,'window_count':len(rows),
        'generation_seconds':generation_seconds,'forward_history_seconds_including_compile':forward_seconds,
        'sigma':sigma.tolist(),'profiles':profiles,'updates':rows,
        'dataset_sha256':hashlib.sha256(data.truth.tobytes()+data.measurements.tobytes()+data.mask.tobytes()).hexdigest()}
    write_json(out/'results.json',payload)
    return payload


def validate_saved(cache):
    from benchmarks.stage5c_feasibility import load_window
    records={}
    # Each check uses all columns of the exact, covariance-dependent whitening.
    for name,index in [('regression_zero',0),('regression_smooth',10),('forward_constant',10)]:
        root=ROOT/name
        run=json.loads((root/'results.json').read_text());cfg=run['configuration']
        with np.load(root/f'window_{index:03d}.npz') as saved:w={k:saved[k] for k in saved.files}
        arrival=SlamState(*(w['arrival_'+k] for k in SlamState._fields))
        evaluate=make_slam_window_objective(arrival,w['times'],lambda t:jnp.asarray(cfg['command_v_omega']),
            w['basis'],cfg['continuous_input_noise_spectral_density_Qc'],w['measurements'],w['ids'],
            cfg['measurement_covariance'],w['mask'])
        c=np.asarray(run['updates'][index]['methods']['projected']['optimization']['coefficients'])
        entry,data,timing,key=cache.prepare(evaluate,c)
        D,ts=cache.sensitivity(entry,data,c)
        q=lambda x:np.asarray(entry['residual'](x,data))
        h=1e-5;fd=np.column_stack([(q(c+d)-q(c-d))/(2*h) for d in np.eye(14)*h])
        np.testing.assert_allclose(D,fd,rtol=2e-5,atol=2e-6)
        records[name]={'window':index,'max_absolute_D_error':float(np.max(np.abs(D-fd))),
                       'step':h,'atol':2e-6,'rtol':2e-5,'passed':True}
        if name=='forward_constant':
            reference=checked_evaluation(lambda x:entry['compiled_evaluate'](x,data),c)
            probes=[]
            for coefficient in (6,13):
                shifted=c.copy();delta=-.1 if c[coefficient]>.3 else .1
                shifted[coefficient]+=delta
                assert np.all(np.abs(shifted)<=.5+1e-14)
                perturbed=checked_evaluation(lambda x:entry['compiled_evaluate'](x,data),shifted)
                probes.append({'coefficient':coefficient,'delta':delta,
                    'cost_absolute_change':abs(float(perturbed.innovation_quadratic-reference.innovation_quadratic)),
                    'q_max_absolute_change':float(np.max(np.abs(q(shifted)-q(c)))),
                    'endpoint_pose_change':pose_difference(perturbed.final_state.x,reference.final_state.x).tolist()})
            records[name]['feasible_coefficient_probes']=probes
        if name=='regression_smooth':
            repeat=optimize_cached(cache,evaluate,np.zeros(14))
            original=run['updates'][index]['methods']['zero']['optimization']
            np.testing.assert_array_equal(repeat['coefficients'],original['coefficients'])
            assert repeat['objective_final']==original['objective_final']
            records[name]['zero_start_exact_reproduction']=True
    write_json(ROOT/'saved_validation.json',{'checks':records,'passed':True})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES+['all'],default='all')
    parser.add_argument('--duration',type=float,default=6.)
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args();jax.config.update('jax_enable_x64',True)
    cache=ProgramCache()
    if not args.report_only and not args.validate:
        gate=json.loads((ROOT/'test_record.json').read_text())
        if not gate['passed']:raise RuntimeError('Regression gate must pass before extension')
        for name in CASES if args.case=='all' else [args.case]:
            if (ROOT/name/'results.json').exists():continue
            run_case(name,cache,args.duration)
            write_json(ROOT/'cache_events.json',cache.events)
    if args.validate:validate_saved(cache)
    from benchmarks.stage5c_report import write_report
    write_report(ROOT)


if __name__=='__main__':main()
