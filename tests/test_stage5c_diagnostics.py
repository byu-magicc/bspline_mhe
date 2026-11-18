"""Diagnostics and compilation reuse must preserve the numerical estimator."""

import ast
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from benchmarks.stage5c_cache import ProgramCache,optimize_cached,whitened_residual
from benchmarks.stage5c_diagnostics import (
    grid,error_profile,disturbance_error,basis_diagnostics,independent_truth,
    timed_initial,sensitivity_summary,
)
from benchmarks.stage5c_feasibility import load_window
from estimation.coefficient_optimization import optimize_coefficients
from estimation.fixed_window_objective import checked_evaluation
from estimation.moving_horizon import initial_guess
from splines import UniformBasis


@pytest.fixture(scope='module',autouse=True)
def precision():
    with jax.experimental.enable_x64():yield


def test_optimizer_orchestration_is_identical_except_bridge_injection():
    original=ast.parse(inspect.getsource(optimize_coefficients))
    copied=ast.parse(inspect.getsource(optimize_cached))
    fn=copied.body[0];fn.name='optimize_coefficients';fn.args.args.pop(0)
    for node in ast.walk(fn):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='CachedBridge':
            node.func.id='CheckedScipyObjective';node.args.pop(0)
    assert ast.dump(original)==ast.dump(copied)


def test_grid_profile_and_independent_global_time_truth():
    sigma=grid();assert sigma[0]==0 and sigma[-1]==5.
    assert all(k in sigma for k in (0.,1.25,2.5,3.75,5.))
    errors=[]
    for start in (.1,.6,1.):
        e,estimate,truth=disturbance_error(np.zeros(14),start,'smooth')
        expected=np.stack((.15*np.sin(2*np.pi*(start+sigma)/5),.05*np.cos(2*np.pi*(start+sigma)/5)),axis=1)
        np.testing.assert_array_equal(truth,expected)
        np.testing.assert_array_equal(e,-expected);errors.append(e)
    E=np.asarray(error_profile(errors,sigma)['rmse'])
    np.testing.assert_allclose(E,np.sqrt(np.mean(np.array(errors)**2,axis=0)),rtol=0,atol=0)
    e,_,_=disturbance_error(np.repeat([.15,-.05],7),1.,'constant')
    np.testing.assert_allclose(e,0.,atol=1e-16)


def test_endpoint_comparison_does_not_assume_it_is_larger():
    sigma=grid();e=np.ones((3,len(sigma),2));e[:,-1,0]=2.;e[:,-1,1]=.5
    p=error_profile(e,sigma)
    assert p['right_exceeds_interior']==[True,False]
    assert p['interior']==[1.,1.]


def test_basis_energy_support_and_endpoint_through_cpp():
    d=basis_diagnostics();b=UniformBasis(3,4)
    assert d['support_in_window'][0]==[0.,1.25] and d['support_in_window'][-1]==[3.75,5.]
    for j in range(7):
        value=quad(lambda t:b.evaluate(t,5.)[j]**2,0.,5.,points=[1.25,2.5,3.75],epsabs=1e-13)[0]
        assert value==pytest.approx(d['integrated_squared_magnitude'][j],abs=2e-14)
    np.testing.assert_allclose(d['integrated_squared_magnitude'],d['integrated_squared_magnitude'][::-1],atol=3e-16)
    np.testing.assert_allclose(d['right_endpoint_weights'],[0.,0.,0.,0.,1/6,4/6,1/6],atol=1e-16)


def test_timed_initial_preserves_projection_clipping_exactly():
    previous=np.array([.5,-.5,.5,-.5,.5,-.5,.5]*2)
    for mode in ('projected','zero'):
        for old in (None,previous):
            c,record,timing=timed_initial(old,.3,mode)
            expected,audit=initial_guess(old,.3,mode=mode)
            np.testing.assert_array_equal(c,expected);assert record==audit
            assert all(t>=0 for t in timing.values())


def test_whitening_derivative_includes_candidate_covariance():
    def q(c):
        result=SimpleNamespace(innovations=jnp.array([[[1.+c,2.-c]]]),
            innovation_covariances=jnp.exp(c)*jnp.eye(2)[None,None],valid_mask=jnp.ones((1,1),bool))
        return whitened_residual(result)
    c=.2;expected=jnp.exp(-c/2)*(jnp.array([1.,-1.])-.5*jnp.array([1.+c,2.-c]))
    np.testing.assert_allclose(jax.jacfwd(q)(c),expected,atol=2e-15)
    assert not np.allclose(expected,jnp.exp(-c/2)*jnp.array([1.,-1.]))


def test_cached_windows_masks_and_births_reuse_exact_graph_and_gradients():
    cache=ProgramCache()
    for case,index in [('regression_constant',0),('regression_constant',1),('forward_constant',2)]:
        evaluator,_=load_window(case,index);c=np.linspace(-.01,.02,14)
        entry,data,timing,key=cache.prepare(evaluator,c)
        a=checked_evaluation(jax.jit(evaluator),c)
        b=checked_evaluation(lambda c:entry['compiled_evaluate'](c,data),c)
        for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(x,y)
        _,g=entry['compiled_pair'](c,data)
        reference=jax.jit(jax.grad(lambda c:evaluator(c).innovation_quadratic))(c)
        np.testing.assert_array_equal(g,reference)
    assert len(cache.entries)==1
    assert [e['program_cache_hit'] for e in cache.events]==[False,True,True]
    with pytest.raises(ValueError):
        checked_evaluation(lambda c:entry['compiled_evaluate'](c,data),np.full(14,np.nan))


def test_full_whitening_jacobian_finite_difference_and_cost_gradient():
    cache=ProgramCache();evaluator,_=load_window('forward_constant',2)
    c=np.linspace(-.02,.03,14);entry,data,_,_=cache.prepare(evaluator,c)
    D,timing=cache.sensitivity(entry,data,c)
    q=lambda x:np.asarray(entry['residual'](x,data))
    h=1e-5
    fd=np.column_stack([(q(c+d)-q(c-d))/(2*h) for d in np.eye(14)*h])
    np.testing.assert_allclose(D,fd,rtol=2e-5,atol=2e-6)
    result=entry['compiled_evaluate'](c,data);value,g=entry['compiled_pair'](c,data)
    np.testing.assert_allclose(q(c)@q(c),value,rtol=2e-12,atol=2e-12)
    np.testing.assert_allclose(2*D.T@q(c),g,rtol=2e-9,atol=2e-8)
    summary=sensitivity_summary(D,result.valid_mask)
    assert summary['shape'][1]==14 and np.isfinite(summary['column_norms']).all()
    jax.clear_caches()
