"""Stage 4C in-window map births, fixed decisions and differentiable augmentation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.milestone4b_smooth import gradient_validation
from benchmarks.milestone4c import configuration, evaluator_and_baseline, prepare_a, prepare_b
from estimation.fixed_window_objective import InvalidCandidateError, checked_evaluation
from estimation.propagation import propagate, rk4_stage_times
from estimation.slam import INITIALIZED, MASKED, UPDATED, LandmarkCapacityError, create_state, initialize_landmark
from estimation.slam_window_objective import fixed_observation_plan, make_slam_window_objective
from simulation.slam_window import simulate_slam_window
from splines import UniformBasis


@pytest.fixture(scope='module',autouse=True)
def precision():
    with jax.experimental.enable_x64():yield


@pytest.fixture(scope='module')
def experiment():
    cfg=configuration()
    data,_=simulate_slam_window(cfg)
    arrival_a,window,ids,adapter,basis,prefix=prepare_a(cfg,data)
    arrival=prepare_b(cfg)
    evaluator,baseline=evaluator_and_baseline(cfg,arrival,window,ids,basis)
    return cfg,arrival,arrival_a,window,evaluator,baseline


def test_independent_command_only_prior_and_empty_map(experiment):
    cfg,arrival,arrival_a,window,evaluator,baseline=experiment
    assert not np.asarray(arrival.active).any()
    assert np.all(np.asarray(arrival.x)[3:]==0.)
    assert np.all(np.asarray(arrival.covariance)[3:]==0.)
    assert not np.array_equal(arrival.covariance[:3,:3],arrival_a.covariance[:3,:3])
    times=np.linspace(0.,2.,81)
    B=UniformBasis(3,4).precompute_jax(rk4_stage_times(times),2.)
    pose,P=propagate((np.array(cfg['initial_pose']),np.array(cfg['arrival_covariance'])),times,
        lambda t:jnp.asarray(cfg['command_v_omega']),B,np.zeros(14),cfg['continuous_input_noise_spectral_density_Qc'])
    np.testing.assert_allclose(arrival.x[:3],pose[-1],rtol=0,atol=1e-14)
    np.testing.assert_allclose(arrival.covariance[:3,:3],P[-1],rtol=0,atol=1e-14)
    # No data argument exists on this construction: changing every window
    # observation cannot affect the prior, which also consumes no pre-window y.
    before=np.asarray(arrival.x).copy()
    changed=window.measurements.copy();changed[:]=1e6
    repeated=prepare_b(cfg)
    np.testing.assert_array_equal(repeated.x,before)
    np.testing.assert_array_equal(repeated.covariance,arrival.covariance)


def test_birth_correction_accounting_fixed_plans_and_original_replay(experiment):
    cfg,arrival,arrival_a,data,evaluator,baseline=experiment
    evaluate=jax.jit(evaluator)
    zero=checked_evaluation(evaluate,np.zeros(14))
    assert int(zero.initialization_count)==6 and int(zero.correction_count)==294
    assert int(zero.valid_count)==int(zero.requested_count)==300
    assert int(zero.final_state.active.sum())==6
    assert not np.asarray(zero.valid_mask)[4].any()
    assert np.asarray(zero.initialization_mask)[4].all()
    assert np.asarray(zero.valid_mask)[8].all()
    assert not data.mask[0].any()
    for a,b in zip(zero.states,baseline.states):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)
    for c in (np.linspace(-.1,.1,14),np.repeat([.15,-.05],7)):
        result=checked_evaluation(evaluate,c)
        np.testing.assert_array_equal(result.requested_initialization_mask,zero.initialization_mask)
        np.testing.assert_array_equal(result.requested_correction_mask,zero.valid_mask)
        np.testing.assert_array_equal(result.statuses,zero.statuses)
        np.testing.assert_array_equal(result.slots,zero.slots)
        np.testing.assert_array_equal(result.states.active,zero.states.active)
        np.testing.assert_array_equal(result.states.truth_ids,zero.states.truth_ids)


def test_full_objective_all_coefficients_against_central_differences(experiment):
    evaluator=experiment[4]
    evaluate=jax.jit(evaluator)
    pair=jax.jit(jax.value_and_grad(lambda c:evaluator(c).innovation_quadratic))
    result=gradient_validation(evaluate,pair,{'zero':np.zeros(14),'nonconstant':np.linspace(-.03,.05,14)})
    assert result['passed']


def test_landmark_mean_covariance_and_cross_covariance_differentiate_through_birth(experiment):
    evaluator=experiment[4]
    # The first observation is at +0.1 s, after four propagation steps; all six
    # landmarks initialize in order here, with no correction innovations.
    def selected(c):
        state=evaluator(c).states
        return jnp.r_[state.x[4,3:7],state.covariance[4,3:7,3:7].ravel(),
                      state.covariance[4,:3,3:7].ravel()]
    evaluate=jax.jit(selected)
    jacobian=jax.jit(jax.jacfwd(selected))
    c=np.linspace(-.02,.03,14)
    D=np.asarray(jacobian(c))
    step=1e-5
    fd=np.column_stack([(np.asarray(evaluate(c+d))-np.asarray(evaluate(c-d)))/(2*step)
                        for d in np.eye(14)*step])
    np.testing.assert_allclose(D,fd,rtol=2e-5,atol=2e-7)
    assert np.linalg.norm(D[:4,:7])>1e-3 and np.linalg.norm(D[:4,7:])>1e-3
    assert np.linalg.norm(D[4:,:7])>1e-6 and np.linalg.norm(D[4:,7:])>1e-6
    result=checked_evaluation(jax.jit(evaluator),c)
    P=np.asarray(result.states.covariance)[4]
    assert np.linalg.norm(P[:3,3:7])>1e-3 and np.linalg.norm(P[3:5,5:7])>1e-3


def repeated_id_problem(bad_initial=False):
    state=create_state([0.,0.,0.],np.diag([.02,.03,.01]),2)
    times=np.array([2.,2.125])
    B=UniformBasis(3,4).precompute_jax(rk4_stage_times(times)-2.,.125)
    ids=np.array([[42,42,-9,-9],[42,42,-9,-9]],dtype=np.int64)
    measurements=np.array([[[4.,.2],[4.1,.21],[3.,-.5],[2.95,-.51]]]*2)
    if bad_initial:measurements[1,0,0]=-1.
    mask=np.array([[False]*4,[True]*4])
    evaluate=make_slam_window_objective(state,times,lambda t:jnp.array([1.,.2]),B,
        np.diag([.01,.0025]),measurements,ids,np.eye(2)*.01,mask)
    return state,ids,mask,evaluate


def test_later_same_timestamp_observation_corrects_without_double_counting():
    state,ids,mask,evaluator=repeated_id_problem()
    result=checked_evaluation(jax.jit(evaluator),np.zeros(14))
    np.testing.assert_array_equal(result.statuses[1],[INITIALIZED,UPDATED,INITIALIZED,UPDATED])
    np.testing.assert_array_equal(result.slots[1],[0,0,1,1])
    assert result.initialization_count==result.correction_count==2
    e=np.asarray(result.innovations)[np.asarray(result.valid_mask)]
    S=np.asarray(result.innovation_covariances)[np.asarray(result.valid_mask)]
    reference=sum(r@np.linalg.solve(cov,r) for r,cov in zip(e,S))
    assert float(result.innovation_quadratic)==pytest.approx(reference,abs=1e-12)
    assert float(result.innovation_quadratic)>0
    assert not np.any(np.asarray(result.initialization_mask)&np.asarray(result.valid_mask))
    np.testing.assert_array_equal(result.final_state.truth_ids,[42,-9])


def test_invalid_initialization_aborts_whole_candidate():
    _,_,_,evaluator=repeated_id_problem(bad_initial=True)
    result=jax.jit(evaluator)(np.zeros(14))
    assert not result.candidate_valid and np.isnan(result.innovation_quadratic)
    assert int(result.requested_count)==4 and int(result.rejected_count)==4
    with pytest.raises(InvalidCandidateError):checked_evaluation(jax.jit(evaluator),np.zeros(14))


def test_fixed_plan_capacity_and_masked_slots():
    state=create_state([0.,0.,0.],np.eye(3),1)
    ids=np.array([[5,5,8]],dtype=np.int64)
    statuses,slots=fixed_observation_plan(state,ids,np.array([[True,True,False]]))
    np.testing.assert_array_equal(statuses,[[INITIALIZED,UPDATED,MASKED]])
    np.testing.assert_array_equal(slots,[[0,0,-1]])
    with pytest.raises(LandmarkCapacityError):fixed_observation_plan(state,ids,np.ones_like(ids,dtype=bool))


def test_invalid_correction_geometry_does_not_drop_cost_term():
    state=create_state([0.,0.,0.],np.eye(3)*.01,1)
    state=initialize_landmark(state,0,3,np.array([.125,0.]),np.eye(2)*.01)
    times=np.array([0.,.125])
    B=UniformBasis(3,4).precompute_jax(rk4_stage_times(times),.125)
    evaluator=make_slam_window_objective(state,times,lambda t:jnp.array([1.,0.]),B,
        np.zeros((2,2)),np.ones((2,1,2)),np.full((2,1),3,dtype=np.int64),np.eye(2)*.01,
        np.array([[False],[True]]))
    result=jax.jit(evaluator)(np.zeros(14))
    assert not result.candidate_valid and np.isnan(result.innovation_quadratic)
    assert int(result.invalid_geometry_count)==1
