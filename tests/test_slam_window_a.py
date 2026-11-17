"""Stage 4C Configuration A intermediate validation before in-window births."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.milestone4b_smooth import gradient_validation
from benchmarks.milestone4c import configuration, evaluator_and_baseline, prepare_a
from benchmarks.slam_window_plots import covariance_ellipse
from estimation.coefficient_optimization import optimize_coefficients
from estimation.fixed_window_objective import InvalidCandidateError, checked_evaluation
from estimation.propagation import rk4_stage_times
from estimation.slam import INITIALIZED, UPDATED, SlamState, create_state, initialize_landmark
from estimation.slam_window_objective import fixed_observation_plan, make_slam_window_objective
from simulation.slam_window import simulate_slam_window
from splines import UniformBasis


@pytest.fixture(scope='module', autouse=True)
def precision():
    with jax.experimental.enable_x64():
        yield


@pytest.fixture(scope='module')
def experiment():
    cfg = configuration()
    data, truth_check = simulate_slam_window(cfg)
    arrival, window, ids, adapter, basis, prefix = prepare_a(cfg, data)
    evaluator, baseline = evaluator_and_baseline(cfg, arrival, window, ids, basis)
    return cfg, data, arrival, window, ids, basis, prefix, evaluator, baseline, truth_check


def small_problem(*, covariance=None, R=None):
    arrival = create_state([0.,0.,0.], np.eye(3)*.01, 2)
    arrival = initialize_landmark(arrival, 0, 91, np.array([4.,.2]), np.eye(2)*.001)
    if covariance is not None:
        arrival = arrival._replace(covariance=jnp.asarray(covariance))
    times = np.array([2.,2.1,2.2])
    B = UniformBasis(3,4).precompute_jax(rk4_stage_times(times)-times[0], times[-1]-times[0])
    ids = np.full((3,1),91,dtype=np.int64)
    y = np.broadcast_to([4.,.2],(3,1,2))
    mask = np.array([[False],[True],[True]])
    evaluator = make_slam_window_objective(arrival,times,lambda t:jnp.array([1.,.2]),B,
        np.diag([.01,.0025]),y,ids,np.eye(2)*.01 if R is None else R,mask)
    return arrival,evaluator


def test_arrival_contains_uncertain_map_and_full_cross_correlations(experiment):
    cfg,data,arrival,window,ids,basis,prefix,evaluator,baseline,_ = experiment
    assert int(arrival.active.sum()) == 6
    assert np.sum(np.asarray(prefix.observations.status)==INITIALIZED)==6
    assert np.sum(np.asarray(prefix.observations.status)==UPDATED)==114
    np.testing.assert_array_equal(np.asarray(arrival.truth_ids)[:6],np.arange(6))
    P=np.asarray(arrival.covariance)
    assert np.linalg.norm(P[:3,3:15]) > 1e-3
    assert np.linalg.norm(P[3:5,5:7]) > 1e-3
    for a,b in zip(arrival,baseline.states):
        np.testing.assert_array_equal(a,b[0])
    for c in (np.zeros(14),np.linspace(-.1,.1,14)):
        result=checked_evaluation(jax.jit(evaluator),c)
        for a,b in zip(arrival,result.states):
            np.testing.assert_array_equal(a,b[0])


def test_strict_boundary_accounting_and_identical_paired_filter(experiment):
    cfg,data,arrival,window,ids,basis,prefix,evaluator,baseline,_ = experiment
    assert data.times[80] == window.times[0] == 2.
    assert data.mask[80].all() and not window.mask[0].any()
    assert int(window.mask.sum()) == 300
    assert np.all(window.times[np.any(window.mask,axis=1)] > 2.)
    result=checked_evaluation(jax.jit(evaluator),np.zeros(14))
    for actual,expected in zip(result.states,baseline.states):
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-14)
    np.testing.assert_array_equal(result.statuses,baseline.observations.status)
    np.testing.assert_array_equal(result.valid_mask,window.mask)
    assert int(result.initialization_count)==0 and int(result.correction_count)==300


def test_objective_numpy_reference_and_all_coefficient_gradients(experiment):
    evaluator=experiment[7]
    evaluate=jax.jit(evaluator)
    result=checked_evaluation(evaluate,np.zeros(14))
    e=np.asarray(result.innovations)[np.asarray(result.valid_mask)]
    S=np.asarray(result.innovation_covariances)[np.asarray(result.valid_mask)]
    reference=sum(r@np.linalg.solve(s,r) for r,s in zip(e,S))
    assert float(result.innovation_quadratic)==pytest.approx(reference,abs=2e-10)
    assert float(result.logdet_diagnostic)==pytest.approx(np.linalg.slogdet(S)[1].sum(),abs=2e-10)
    gradients=jax.jit(jax.value_and_grad(lambda c:evaluator(c).innovation_quadratic))
    check=gradient_validation(evaluate,gradients,{'zero':np.zeros(14),'nonconstant':np.linspace(-.03,.04,14)})
    assert check['passed']


def test_truth_and_noise_repeat_independently(experiment,monkeypatch):
    import estimation.slam as slam
    cfg,data,*_=experiment
    def forbidden(*args,**kwargs):
        pytest.fail('truth must be independent of spline and estimator')
    monkeypatch.setattr(UniformBasis,'evaluate',forbidden)
    monkeypatch.setattr(slam,'rk4_slam_step',forbidden)
    repeated,check=simulate_slam_window(cfg)
    for a,b in zip(data,repeated):
        np.testing.assert_array_equal(a,b)
    assert check['passed'] and max(check['max_absolute_pose_difference'])<1e-10


def test_arrival_and_data_are_captured_immutably():
    arrival,evaluator=small_problem()
    # Construct from mutable host storage, then mutate every input after capture.
    times=np.array([2.,2.1])
    B=UniformBasis(3,4).evaluate(rk4_stage_times(times)-times[0],times[-1]-times[0])
    host=SlamState(*(np.asarray(v).copy() for v in arrival))
    y=np.array([[[4.,.2]],[[3.9,.18]]])
    ids=np.full((2,1),91,dtype=np.int64)
    mask=np.array([[False],[True]])
    evaluate=jax.jit(make_slam_window_objective(host,times,lambda t:jnp.array([1.,.2]),B,
        np.eye(2)*.01,y,ids,np.eye(2)*.01,mask))
    before=checked_evaluation(evaluate,np.zeros(14))
    host.x[:]=999.;host.covariance[:]=999.;y[:]=999.;mask[:]=False;ids[:]=0;B[:]=0.;times[:]=0.
    after=checked_evaluation(evaluate,np.zeros(14))
    np.testing.assert_array_equal(before.states.x,after.states.x)
    assert float(before.innovation_quadratic)==float(after.innovation_quadratic)


@pytest.mark.parametrize('R',[np.array([[1.,0.],[0.,-1.]]),np.full((2,2),np.nan)])
def test_invalid_noise_rejected_without_partial_cost(R):
    _,evaluator=small_problem(R=R)
    result=jax.jit(evaluator)(np.zeros(14))
    assert not result.candidate_valid and np.isnan(result.innovation_quadratic)
    with pytest.raises(InvalidCandidateError):
        checked_evaluation(jax.jit(evaluator),np.zeros(14))


def test_inactive_pollution_and_covariance_invalidity_are_not_repaired():
    arrival,_=small_problem()
    for kind in ('negative','asymmetric','inactive'):
        P=np.asarray(arrival.covariance).copy()
        if kind=='negative':P[0,0]=-1.
        if kind=='asymmetric':P[0,1]+=.1
        if kind=='inactive':P[-1,-1]=.1
        _,evaluate=small_problem(covariance=P)
        result=jax.jit(evaluate)(np.zeros(14))
        assert not result.candidate_valid and np.isnan(result.innovation_quadratic)


def test_invalid_optimizer_trial_has_no_value_or_gradient(monkeypatch):
    import estimation.coefficient_optimization as optimizer
    _,evaluate=small_problem()
    returned=[]
    def request_bad(fun,initial,**kwargs):
        returned.append(fun(np.full(14,np.nan)))
        pytest.fail('invalid point returned a pair to scipy')
    monkeypatch.setattr(optimizer,'minimize',request_bad)
    result=optimize_coefficients(evaluate,np.zeros(14))
    assert not returned and result['status']=='stopped_invalid_trial'
    assert not result['success'] and result['gradient_evaluations']==0
    json.dumps(result,allow_nan=False)


def test_jit_scan_repeatable_optimization_and_fixed_ids():
    arrival,evaluator=small_problem()
    c=np.linspace(-.1,.1,14)
    eager=evaluator(c)
    compiled=jax.jit(evaluator)(c)
    np.testing.assert_allclose(eager.states.x,compiled.states.x,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(compiled.states.truth_ids,np.broadcast_to(arrival.truth_ids,(3,2)))
    first=optimize_coefficients(evaluator,np.zeros(2),constant=True)
    second=optimize_coefficients(evaluator,np.zeros(2),constant=True)
    assert first['success'] and second['success']
    np.testing.assert_array_equal(first['coefficients'],second['coefficients'])


def test_covariance_ellipse_threshold_rotation_and_rejection():
    angle=.4
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    P=rotation@np.diag([4.,1.])@rotation.T
    width,height,degrees=covariance_ellipse(P)
    assert width==pytest.approx(2*np.sqrt(5.991*4))
    assert height==pytest.approx(2*np.sqrt(5.991))
    assert abs(np.cos(np.deg2rad(degrees)-angle))==pytest.approx(1.)
    for bad in (np.diag([1.,-1e-15]),np.array([[1.,.1],[0.,1.]]),np.full((2,2),np.nan)):
        with pytest.raises(ValueError):covariance_ellipse(bad)


def test_all_preexisting_artifacts_and_math_preserved():
    # These independently pin the protected mathematical implementation and
    # validated constant experiment; the full task audit covers every artifact.
    hashes={'estimation/fixed_window_objective.py':'0034031a436d113501f1b14b3e25eb72ec5979e4bc64820d9cbdf4e6d906d153',
            'artifacts/milestone4b/results.json':'1de4435b60a2d43e744a75cf1c6e07e460f4253852acda6f5f168cac1c548237'}
    for name,digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
