"""Stage 4C metric definitions, independent smooth truth and figure artifacts."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.milestone4c import configuration, metrics
from benchmarks.milestone4c_reporting import completion_checks
from benchmarks.slam_window_plots import CONDITIONAL_LABEL, plot_experiment
from estimation.fixed_window_objective import checked_evaluation
from estimation.propagation import rk4_stage_times
from estimation.slam import create_state
from estimation.slam_window_objective import make_slam_window_objective
from simulation.continuous_unicycle import SimulationData
from simulation.slam_window import simulate_slam_window
from splines import UniformBasis


@pytest.fixture(scope='module',autouse=True)
def precision():
    with jax.experimental.enable_x64():yield


@pytest.fixture(scope='module')
def mini_experiment():
    times=np.array([2.,2.125,2.25])
    truth=np.array([[0.,0.,0.],[.125,0.,0.],[.25,0.,0.]])
    landmark=np.array([[4.,2.]])
    delta=landmark-truth[:,None,:2]
    observations=np.stack((np.linalg.norm(delta,axis=-1),np.arctan2(delta[...,1],delta[...,0])),axis=-1)
    observations[1,0]+=[.02,-.01]
    mask=np.array([[False],[True],[True]])
    data=SimulationData(times,truth,observations,np.broadcast_to(landmark,observations.shape),
                        mask,np.eye(2)*.01,landmark)
    adapter=UniformBasis(3,4)
    B=adapter.precompute_jax(rk4_stage_times(times)-times[0],times[-1]-times[0])
    evaluator=make_slam_window_objective(create_state([0.,0.,0.],np.eye(3)*.01,2),times,
        lambda t:jnp.array([1.,0.]),B,np.eye(2)*.001,observations,np.zeros(mask.shape,dtype=np.int64),
        data.measurement_covariance,mask)
    result=checked_evaluation(jax.jit(evaluator),np.zeros(14))
    dense=np.linspace(0.,.25,51)
    dense_basis=adapter.evaluate(dense,.25)
    arrays={'times':times,'truth':truth,'landmark_truth':landmark,'elapsed':dense,
        'disturbance_truth':np.zeros((51,2)),'optimized_disturbance':np.zeros((51,2)),
        'optimization_iteration':np.array([0,1]),'optimization_objective':np.array([1.,.5])}
    for estimator in ('baseline','optimized'):
        arrays.update({estimator+'_'+key:np.asarray(value) for key,value in result.states._asdict().items()})
    return data,result,dense_basis,arrays


def test_saved_metrics_match_independent_pose_map_and_disturbance_calculations(mini_experiment,tmp_path):
    data,result,basis,arrays=mini_experiment
    summary=metrics(result,np.zeros(14),basis,np.zeros((51,2)),data)
    error=np.asarray(result.states.x)[:,:3]-data.truth
    error[:,2]=np.arctan2(np.sin(error[:,2]),np.cos(error[:,2]))
    assert summary['localization']['position_rmse_m']==pytest.approx(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1))))
    assert summary['localization']['heading_rmse_rad']==pytest.approx(np.sqrt(np.mean(error[:,2]**2)))
    map_error=np.asarray(result.final_state.x)[3:5]-data.landmark_map[0]
    assert summary['final_landmark_rmse_m']==pytest.approx(np.linalg.norm(map_error))
    assert summary['initialization_observations']==summary['correction_observations']==1
    assert summary['disturbance']['rmse_v_omega']==[0.,0.]
    target=tmp_path/'metrics.json'
    target.write_text(json.dumps(summary,allow_nan=False))
    assert json.loads(target.read_text())==summary
    np.savez_compressed(tmp_path/'histories.npz',**arrays)
    with np.load(tmp_path/'histories.npz') as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(value,saved[key])


def test_publication_figures_exist_and_do_not_modify_covariance(mini_experiment,tmp_path):
    data,result,basis,arrays=mini_experiment
    original={key:value.copy() for key,value in arrays.items()}
    paths=plot_experiment(tmp_path,'b_constant',arrays,data.truth,data.times)
    assert len(paths)==10 and len(set(paths))==10
    assert CONDITIONAL_LABEL=='Conditional EKF covariance; spline coefficients treated as fixed.'
    for name in paths:
        contents=(tmp_path/name).read_bytes()
        assert len(contents)>1000
        assert contents.startswith(b'%PDF') if name.endswith('.pdf') else contents.startswith(b'\x89PNG')
    for key,value in original.items():np.testing.assert_array_equal(value,arrays[key])


def test_plot_rejects_bad_active_covariance_without_repair(mini_experiment,tmp_path):
    data,_,_,original=mini_experiment
    arrays={key:value.copy() for key,value in original.items()}
    arrays['optimized_covariance'][1,0,0]=-1.
    with pytest.raises(ValueError):plot_experiment(tmp_path,'b_constant',arrays,data.truth,data.times)


def test_smooth_truth_refinement_and_common_pre_window_noise():
    constant,check=simulate_slam_window(configuration())
    smooth,smooth_check=simulate_slam_window(configuration('smooth'))
    assert check['passed'] and smooth_check['passed']
    assert max(smooth_check['max_absolute_pose_difference'])<1e-10
    np.testing.assert_array_equal(constant.truth[:81],smooth.truth[:81])
    np.testing.assert_array_equal(constant.measurements[:81],smooth.measurements[:81])
    np.testing.assert_array_equal(constant.mask,smooth.mask)
    sigma=smooth.times[80:]-2.
    expected=.4+.2*sigma+.05*5/(2*np.pi)*np.sin(2*np.pi*sigma/5)
    np.testing.assert_allclose(smooth.truth[80:,2],expected,rtol=0,atol=1e-11)
    assert 'true_disturbance_v_omega' not in configuration('smooth')


def test_completion_needs_all_evidence_but_no_performance_winner():
    experiments={name:{'validation_checks':{'valid':True},'dataset_sha256':kind,
        'baseline_b':{'requested_observations':300},
        'optimized_mhe':{'valid_observations':300,'initialization_observations':0 if config=='a' else 6,
                         'correction_observations':300 if config=='a' else 294},
        'truth_validation':{'passed':True},'figures':[str(i) for i in range(10)]}
        for config in ('a','b') for kind in ('constant','smooth') for name in (config+'_'+kind,)}
    payload={'experiments':experiments,
        'configuration_a_gate':{'validated_before_configuration_b_implementation':True,'test_exit_code':0},
        'configuration_b_constant_gate':{'validated_before_smooth_extension':True,'test_exit_code':0},
        'current_test_record':{'exit_code':0,'scope':'complete repository suite'},'preservation':{'all_preserved':True}}
    assert all(completion_checks(payload).values())
    payload['current_test_record']=None
    assert not completion_checks(payload)['current_complete_test_suite']
    payload['experiments']['b_constant']['optimized_mhe']['correction_observations']=300
    assert not completion_checks(payload)['arrival_and_update_accounting']
