"""Stage 5B time accounting, causal arrivals, map continuity and complete replay."""

from dataclasses import replace
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.fixed_window_objective import checked_evaluation, InvalidCandidateError
from estimation.moving_horizon import ReplayData,ForwardArrivals,MovingHorizonSLAM,initial_guess
from estimation.slam import create_state,filter_slam,SlamState,INITIALIZED,UPDATED
from estimation.slam_window_objective import make_slam_window_objective
from simulation.continuous_unicycle import analytical_unicycle
from splines.warm_start import make_warm_start_projection


@pytest.fixture(scope='module',autouse=True)
def precision():
    with jax.experimental.enable_x64():yield


@pytest.fixture(scope='module')
def example():
    times=np.linspace(0.,.5,21);command=lambda t:jnp.array([.4,.15])
    truth=analytical_unicycle(times,[0.,0.,0.],.4,.15)
    landmarks=np.array([[3.,1.],[2.,-2.]])
    delta=landmarks-truth[:,None,:2]
    y=np.stack((np.linalg.norm(delta,axis=2),np.arctan2(delta[:,:,1],delta[:,:,0])-truth[:,None,2]),axis=-1)
    mask=np.zeros((21,2),bool);mask[4,0]=True;mask[8,0]=True;mask[12]=True;mask[20,0]=True
    scheduled=np.arange(21)%4==0;scheduled[0]=False
    data=ReplayData(times,y,np.broadcast_to([42,7],mask.shape).astype(np.int64),
                    np.diag([.0025,.0003]),mask,scheduled)
    initial=create_state([0.,0.,0.],np.diag([.0025,.0025,.001]),3)
    forward=ForwardArrivals(data,initial,command,np.diag([.001,.0002]))
    return data,initial,command,forward


def evaluator(forward,window):
    return make_slam_window_objective(window.arrival,window.times,forward.command,window.basis,
        forward.Qc,window.measurements,window.ids,forward.data.covariance,window.mask)


def test_boundary_births_ids_and_covariance_not_double_counted(example):
    data,initial,command,forward=example
    for end in (.3,.4,.5):
        w=forward.window(end,.2)
        assert w.start==data.times[round((end-.2)/.025)]
        assert not w.mask[0].any()
        assert np.all(data.times[w.source_indices]>w.start)
        assert np.all(data.times[w.source_indices]<=w.end)
        np.testing.assert_array_equal(w.measurements[1:],data.measurements[w.source_indices])
        np.testing.assert_array_equal(w.arrival.covariance,forward.at(w.start).covariance)
    early=forward.window(.3,.2)
    assert list(np.asarray(early.arrival.truth_ids)[early.arrival.active])==[42]
    assert np.linalg.norm(np.asarray(early.arrival.covariance)[:3,3:5])>1e-3
    result=checked_evaluation(jax.jit(evaluator(forward,early)),np.zeros(14))
    assert int(result.initialization_count)==1
    statuses=np.asarray(result.statuses)
    assert np.sum(statuses==INITIALIZED)==1 and np.sum(statuses==UPDATED)==2
    later=forward.window(.5,.2)
    assert list(np.asarray(later.arrival.truth_ids)[later.arrival.active])==[42,7]
    assert np.linalg.norm(np.asarray(later.arrival.covariance)[3:5,5:7])>1e-3
    final=checked_evaluation(jax.jit(evaluator(forward,later)),np.zeros(14))
    assert int(final.initialization_count)==0 and int(final.final_state.active.sum())==2
    np.testing.assert_array_equal(final.final_state.truth_ids,forward.at(.5).truth_ids)


def test_nonaligned_arrival_uses_no_future_observations(example):
    data,initial,command,forward=example
    t=.1375;actual=forward.at(t);prior=forward.at(.1)
    grid=np.array([.1,.125,t])
    expected=jax.jit(lambda:filter_slam(prior,grid,command,np.zeros((2,4,7)),np.zeros(14),forward.Qc,
        np.zeros((3,1,2)),np.zeros((3,1),np.int64),data.covariance,np.zeros((3,1),bool)))()
    for a,b in zip(actual,expected.states):np.testing.assert_allclose(a,b[-1],rtol=0,atol=2e-14)
    changed=data.measurements.copy();changed[data.times>t]+=[.2,.02]
    repeated=ForwardArrivals(replace(data,measurements=changed),initial,command,forward.Qc)
    for a,b in zip(actual,repeated.at(t)):np.testing.assert_array_equal(a,b)
    window=forward.window(.3,.1625)
    assert window.start==pytest.approx(t,abs=1e-16)
    assert not window.mask[0].any()
    for a,b in zip(window.arrival,actual):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)


@pytest.mark.parametrize('end,L',[(.3,.3),(.4,.3),(.5,.3),(.3,.1625)])
def test_zero_replay_matches_same_arrival_baseline(example,end,L):
    data,initial,command,forward=example;w=forward.window(end,L)
    result=checked_evaluation(jax.jit(evaluator(forward,w)),np.zeros(14))
    baseline=jax.jit(lambda:filter_slam(w.arrival,w.times,command,w.basis,np.zeros(14),forward.Qc,
        w.measurements,w.ids,data.covariance,w.mask))()
    for a,b in zip(result.states,baseline.states):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)
    if L==.3:
        for a,b in zip(result.final_state,forward.at(end)):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)
    assert int(result.valid_count)==int(w.mask.sum())


def test_moving_windows_complete_gradients_against_central_differences(example):
    forward=example[-1]
    for end in (.3,.4):
        w=forward.window(end,.3);evaluate=evaluator(forward,w)
        value=jax.jit(lambda c:evaluate(c).innovation_quadratic)
        gradient=jax.jit(jax.grad(value));c=np.linspace(-.01,.02,14)
        h=1e-5
        fd=np.array([(float(value(c+d))-float(value(c-d)))/(2*h) for d in np.eye(14)*h])
        np.testing.assert_allclose(gradient(c),fd,rtol=2e-5,atol=2e-6)
        assert np.linalg.norm(np.asarray(gradient(c))[:7])>1e-6
        assert np.linalg.norm(np.asarray(gradient(c))[7:])>1e-6


def test_projection_then_componentwise_clipping_is_separate():
    old=np.array([.5,-.5,.5,-.5,.5,-.5,.5]*2);before=old.copy()
    raw=make_warm_start_projection(3,4,5.,.3).apply(old)
    actual,record=initial_guess(old,.3)
    np.testing.assert_array_equal(actual,np.clip(raw,-.5,.5))
    np.testing.assert_array_equal(record['raw_projection'],raw)
    assert record['clipped_count']==np.count_nonzero(raw!=actual)>0
    assert record['clipping_max_absolute']==np.max(np.abs(raw-actual))
    np.testing.assert_array_equal(old,before)
    for mode,previous in [('projected',None),('zero',old)]:
        np.testing.assert_array_equal(initial_guess(previous,.1,mode=mode)[0],np.zeros(14))


def test_empty_whole_window_has_no_invented_map_or_cost(example):
    data,initial,command,forward=example
    empty=ForwardArrivals(replace(data,mask=np.zeros_like(data.mask)),initial,command,forward.Qc)
    w=empty.window(.4,.3);evaluate=evaluator(empty,w)
    result=checked_evaluation(jax.jit(evaluate),np.zeros(14))
    assert result.requested_count==result.initialization_count==result.correction_count==0
    assert result.innovation_quadratic==0 and not result.final_state.active.any()
    np.testing.assert_array_equal(jax.jit(jax.grad(lambda c:evaluate(c).innovation_quadratic))(np.zeros(14)),np.zeros(14))


def test_scheduled_empty_scan_is_optimized_and_reported_deterministically(example):
    data,initial,command,forward=example
    before=[np.asarray(v).copy() for v in forward.result.states]
    first=list(MovingHorizonSLAM(forward,.3).run())
    second=list(MovingHorizonSLAM(forward,.3).run())
    np.testing.assert_allclose([r['time'] for r in first],[.3,.4,.5],rtol=0,atol=1e-15)
    assert not data.mask[16].any()  # t=.4 is still an optimization/report time.
    for one,two in zip(first,second):
        assert set(one['fits'])=={'projected','zero'}
        for mode in one['fits']:
            a,b=one['fits'][mode],two['fits'][mode]
            assert a['optimization']['success'] and not a['optimization']['invalid_trials']
            np.testing.assert_array_equal(a['optimization']['coefficients'],b['optimization']['coefficients'])
            assert a['optimization']['objective_final']==b['optimization']['objective_final']
            assert a['replay'].candidate_valid
    assert 'jumps' in first[-1]['fits']['projected']
    assert first[-1]['fits']['projected']['jumps']['reported_map_delta']['ids']==[7,42]
    for a,b in zip(before,forward.result.states):np.testing.assert_array_equal(a,b)


def test_invalid_optimizer_trial_aborts_without_reporting_estimate(example,monkeypatch):
    import estimation.moving_horizon as module
    def fail(*args,**kwargs):
        return {'invalid_trials':[{'invalid':True}],'coefficients':None,'status':'stopped_invalid_trial'}
    monkeypatch.setattr(module,'optimize_coefficients',fail)
    engine=MovingHorizonSLAM(example[-1],.3)
    with pytest.raises(InvalidCandidateError):list(engine.run())
    assert 'replay' not in engine.records[0]['fits']['projected']


def test_data_are_captured_and_invalid_times_rejected(example):
    data,initial,command,forward=example
    with pytest.raises(ValueError):data.mask[0,0]=True
    with pytest.raises(ValueError):forward.window(.3,.5)
    with pytest.raises(ValueError):forward.window(.325,.2)
    with pytest.raises(ValueError):forward.at(-.1)
    with pytest.raises(ValueError):initial_guess(np.zeros(14),.1,mode='invalid')
    with pytest.raises(ValueError):list(MovingHorizonSLAM(forward,.3).run(('zero','zero')))


def test_stage5a_math_unchanged():
    import json
    before=json.loads(Path('artifacts/milestone5b/before_sha256.json').read_text())
    for path in ('splines/warm_start.py','estimation/slam_window_objective.py',
                 'estimation/slam.py','estimation/coefficient_optimization.py'):
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==before[path]


@pytest.mark.parametrize('kind',['zero','constant','smooth'])
def test_seed7_truth_visibility_and_schedule_are_reproducible(kind):
    from benchmarks.milestone4c import configuration
    from simulation.moving_horizon import simulate_moving,true_disturbance
    from simulation.slam_window import sensor_settings,visibility_geometry
    cfg=configuration();cfg.update(disturbance_kind=kind,total_duration_seconds=5.2,sensor=sensor_settings('forward'))
    data,scheduled,check=simulate_moving(cfg)
    repeated,again,_=simulate_moving(cfg)
    for a,b in zip(data,repeated):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(scheduled,again)
    assert check['passed'] and np.max(check['max_absolute_pose_difference'])<1e-10
    for i in np.flatnonzero(scheduled):
        _,visible=visibility_geometry(data.truth[i],data.landmark_map,cfg['sensor'])
        np.testing.assert_array_equal(data.mask[i],visible)
    assert len(np.flatnonzero(scheduled & (data.times>=5.-1e-14)))==3
    if kind=='constant':assert np.any(scheduled & ~data.mask.any(axis=1))
    assert true_disturbance(data.times,kind).shape==(len(data.times),2)
