"""Visibility boundaries and information availability in Stage 4C replays."""

import jax
import numpy as np
import pytest

from benchmarks.milestone4c import configuration, prepare_a, prepare_b, evaluator_and_baseline
from benchmarks.milestone4c_visibility import initialization_times
from benchmarks.slam_window_plots import plot_slam_map
from estimation.fixed_window_objective import checked_evaluation
from estimation.slam import INITIALIZED, MASKED
from simulation.slam_window import sensor_settings, simulate_slam_window, visibility_geometry


@pytest.fixture(scope='module',autouse=True)
def precision():
    with jax.experimental.enable_x64():
        yield


def test_inclusive_range_and_fov_boundaries():
    sensor=sensor_settings('forward')
    points=np.array([[.2,0.],[6.,0.],[np.nextafter(.2,0.),0.],
                     [np.nextafter(6.,np.inf),0.],[0.,2.],[0.,-2.],[-1e-9,2.],[-1e-9,-2.]])
    _,visible=visibility_geometry([0.,0.,0.],points,sensor)
    np.testing.assert_array_equal(visible,[True,True,False,False,True,True,False,False])


def test_wrap_and_pose_dependent_visibility():
    sensor=sensor_settings('forward')
    points=np.array([[-2.,1e-8],[-2.,-1e-8],[2.,0.],[20.,0.]])
    clean,visible=visibility_geometry([0.,0.,np.pi],points,sensor)
    np.testing.assert_array_equal(visible,[True,True,False,False])
    assert np.all(np.abs(clean[:2,1])<1e-7)
    _,repeated=visibility_geometry([0.,0.,-np.pi],points,sensor)
    np.testing.assert_array_equal(visible,repeated)
    _,turned=visibility_geometry([0.,0.,0.],points,sensor)
    assert turned[2] and not turned[0]
    _,moved=visibility_geometry([18.,0.,0.],points,sensor)
    assert moved[3]


@pytest.mark.parametrize('changes',[{'range_min':-1.},{'range_max':.1},
    {'field_of_view_rad':0.},{'field_of_view_rad':7.},{'translation_xy':[1.,0.]},{'yaw_rad':.1}])
def test_invalid_or_unsupported_sensor_rejected(changes):
    with pytest.raises(ValueError):
        visibility_geometry([0.,0.,0.],[[1.,0.]],{**sensor_settings('forward'),**changes})


def test_recorded_visibility_precedes_noise_and_regression_is_preserved():
    for kind in ('constant','smooth'):
        cfg=configuration(kind)
        old,_=simulate_slam_window(cfg)
        with np.load(f'artifacts/milestone4c/dataset_{kind}.npz') as saved:
            for key,value in old._asdict().items():np.testing.assert_array_equal(value,saved[key])
        cfg['sensor']=sensor_settings('forward')
        data,_=simulate_slam_window(cfg)
        cfg['measurement_covariance']=(np.eye(2)*4.).tolist()
        noisy,_=simulate_slam_window(cfg)
        np.testing.assert_array_equal(data.mask,noisy.mask)
        for node in range(4,len(data.times),4):
            _,visible=visibility_geometry(data.truth[node],data.landmark_map,cfg['sensor'])
            np.testing.assert_array_equal(data.mask[node],visible)
        np.testing.assert_array_equal(data.measurements[data.mask],old.measurements[data.mask])
        assert np.all(data.measurements[~data.mask]==0.)


def test_arrival_only_uses_available_observations_and_window_births(tmp_path):
    cfg=configuration();cfg['sensor']=sensor_settings('forward')
    data,_=simulate_slam_window(cfg)
    arrival,window,ids,adapter,basis,prefix=prepare_a(cfg,data)
    seen=np.flatnonzero(data.mask[:81].any(axis=0))
    np.testing.assert_array_equal(np.sort(np.asarray(arrival.truth_ids)[arrival.active]),seen)
    assert 0<len(seen)<6
    # Unobserved rows and ground-truth coordinates are not arrival information.
    changed=data.measurements.copy();changed[~data.mask]=1e6
    poison=data._replace(measurements=changed,landmark_map=np.full_like(data.landmark_map,999.),
                         landmarks_at_times=np.full_like(data.landmarks_at_times,999.))
    repeated=prepare_a(cfg,poison)[0]
    for a,b in zip(arrival,repeated):np.testing.assert_array_equal(a,b)
    assert not np.asarray(prepare_b(cfg).active).any()
    evaluator,baseline=evaluator_and_baseline(cfg,arrival,window,ids,basis)
    result=checked_evaluation(jax.jit(evaluator),np.zeros(14))
    for a,b in zip(result.states,baseline.states):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)
    births=initialization_times(window.times,np.asarray(result.statuses),ids)
    assert births  # This scenario includes a genuine first observation inside the window.
    for truth_id,time in births.items():
        column=int(truth_id);first=np.flatnonzero(window.mask[:,column])[0]
        assert time==window.times[first]
        assert not data.mask[:81,column].any()
        active=np.asarray(result.states.active)
        metadata=np.asarray(result.states.truth_ids)
        assert not np.any(active[:first] & (metadata[:first]==column))
    assert int(result.valid_count)==window.mask.sum()
    arrays={'times':window.times,'truth':window.truth,'landmark_truth':data.landmark_map,
            'mask':window.mask,'measurements':window.measurements,'observation_ids':ids,
            'scan_scheduled':np.arange(len(window.times)) % 4 == 0}
    arrays['scan_scheduled'][0]=False
    for name in ('baseline','optimized'):
        arrays.update({name+'_'+key:np.asarray(value) for key,value in result.states._asdict().items()})
    paths=plot_slam_map(tmp_path,'a_constant',arrays,cfg['sensor'])
    assert len(paths)==2 and all((tmp_path/p).stat().st_size>1000 for p in paths)


def test_empty_scans_and_delayed_births_match_baseline():
    cfg=configuration();data,_=simulate_slam_window(cfg)
    _,window,ids,adapter,basis,_=prepare_a(cfg,data)
    mask=np.zeros_like(window.mask)
    mask[40,2]=True;mask[80,2]=True;mask[120,0]=True
    window=window._replace(mask=mask)
    arrival=prepare_b(cfg)
    evaluator,baseline=evaluator_and_baseline(cfg,arrival,window,ids,basis)
    result=checked_evaluation(jax.jit(evaluator),np.zeros(14))
    assert result.initialization_count==2 and result.correction_count==1
    assert not np.asarray(result.states.active)[:40].any()
    assert np.all(np.asarray(result.statuses)[~mask]==MASKED)
    assert result.statuses[40,2]==INITIALIZED and result.statuses[120,0]==INITIALIZED
    for a,b in zip(result.states,baseline.states):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)
    empty=window._replace(mask=np.zeros_like(mask))
    evaluator,baseline=evaluator_and_baseline(cfg,arrival,empty,ids,basis)
    evaluate=jax.jit(evaluator)
    result=checked_evaluation(evaluate,np.zeros(14))
    assert result.requested_count==result.initialization_count==result.correction_count==0
    assert result.innovation_quadratic==0 and not result.states.active.any()
    np.testing.assert_array_equal(jax.jit(jax.grad(lambda c:evaluator(c).innovation_quadratic))(np.zeros(14)),np.zeros(14))
    for a,b in zip(result.states,baseline.states):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)
