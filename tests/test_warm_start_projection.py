"""Stage 5A integral projection, with independent adaptive integration checks."""

import numpy as np
import pytest
from scipy.integrate import quad

from splines import UniformBasis
from splines.warm_start import (
    make_warm_start_projection, clear_projection_cache, projection_cache_info,
)


CONFIGS = [(0,3),(1,3),(2,4),(3,4),(5,2)]


def independent_error(degree,M,L,delta,old,new):
    """Adaptive Gauss-Kronrod on physical time, independent of projection GL."""
    adapter=UniformBasis(degree,M)
    end=L-delta
    knots=np.arange(M+1)*L/M
    points=np.unique(np.r_[knots,knots-delta])
    points=points[(points>0)&(points<end)]
    def error(t):
        e=new@adapter.evaluate(t,L)-old@adapter.evaluate(t+delta,L)
        return float(e@e)
    return quad(error,0.,end,points=points,epsabs=2e-12,epsrel=2e-11,limit=200)[0]


@pytest.mark.parametrize('degree,M',CONFIGS)
def test_zero_shift_exact_identity(degree,M):
    n=degree+M;c=np.random.default_rng(7).normal(size=(2,n))
    p=make_warm_start_projection(degree,M,5.,0.)
    np.testing.assert_array_equal(p.transform,np.eye(n))
    np.testing.assert_array_equal(p.apply(c),c)
    np.testing.assert_array_equal(p.apply(c.ravel()),c.ravel())
    assert p.diagnostics['method']=='zero_shift_identity'


@pytest.mark.parametrize('degree,M',CONFIGS)
@pytest.mark.parametrize('fraction',[0.,.37,1.,1.4])
def test_constant_both_channels_preserved(degree,M,fraction):
    L=5.;delta=fraction*L/M;n=degree+M
    p=make_warm_start_projection(degree,M,L,delta)
    c=np.repeat([[.3],[-.17]],n,axis=1)
    shifted=p.apply(c)
    # Evaluate disturbances away from degree-zero's half-open right endpoint.
    basis=UniformBasis(degree,M).evaluate(np.linspace(0,L,99,endpoint=False),L)
    np.testing.assert_allclose(basis@shifted.T,np.broadcast_to([.3,-.17],(99,2)),atol=2e-11,rtol=0)
    np.testing.assert_allclose(shifted,c,atol=2e-11,rtol=0)


@pytest.mark.parametrize('degree,M',CONFIGS)
@pytest.mark.parametrize('fraction',[.13,.5,1.3])
def test_fractional_projection_normal_equation_and_independent_optimality(degree,M,fraction):
    L=5.;delta=fraction*L/M;n=degree+M
    old=np.random.default_rng(17).normal(scale=.3,size=(2,n))
    p=make_warm_start_projection(degree,M,L,delta)
    new=p.apply(old)
    np.testing.assert_allclose(new@p.gram,old@p.cross.T,atol=4e-12,rtol=4e-12)
    error=independent_error(degree,M,L,delta,old,new)
    retained=independent_error(degree,M,L,delta,old,old)
    assert error <= retained+5e-11
    if fraction<1:
        assert error>1e-10  # Generic target has interior knots off the new grid.
    elif M==2:
        # The shortened overlap lies in one old AND one new polynomial span;
        # the shifted polynomial can be represented exactly even fractionally.
        assert error<1e-20


@pytest.mark.parametrize('degree,M',CONFIGS)
def test_one_and_multiple_whole_knot_translation(degree,M):
    L=5.;n=degree+M;adapter=UniformBasis(degree,M)
    old=np.random.default_rng(9).normal(size=(2,n))
    for q in range(1,M):
        delta=q*(L/M)
        p=make_warm_start_projection(degree,M,L,delta)
        expected=old[:,np.minimum(np.arange(n)+q,n-1)]
        np.testing.assert_array_equal(p.apply(old),expected)
        t=np.linspace(0.,L-delta,101,endpoint=degree>0)
        np.testing.assert_allclose(adapter.evaluate(t,L)@expected.T,
                                   adapter.evaluate(t+delta,L)@old.T,atol=2e-13,rtol=0)
        assert p.diagnostics['gram_rank']==n-q
        assert p.diagnostics['numerically_rank_deficient']
        assert p.diagnostics['gram_condition_number'] is None
        assert p.diagnostics['method']=='integer_knot_translation_repeat_terminal'


@pytest.mark.parametrize('degree,M',CONFIGS)
def test_full_rank_gram_before_one_knot_and_structural_rank_after(degree,M):
    L=5.;h=L/M;n=degree+M
    p=make_warm_start_projection(degree,M,L,.25*h)
    np.testing.assert_array_equal(p.gram,p.gram.T)
    assert np.linalg.eigvalsh(p.gram)[0]>0
    assert p.diagnostics['gram_rank']==n and p.diagnostics['gram_condition_number']>0
    if degree==0:assert p.diagnostics['gram_condition_number']>=1.
    for shift in (h,1.25*h):
        p=make_warm_start_projection(degree,M,L,shift)
        assert p.diagnostics['gram_rank']==n-1
        assert np.all(p.gram[-1]==0.)
        assert p.diagnostics['gram_condition_number'] is None


def test_conditioning_approaching_one_knot_and_documented_fallback():
    rows=[make_warm_start_projection(3,4,5.,(1-eps)*1.25)
          for eps in (.5,.1,.01,1e-3,1e-5)]
    cond=[p.diagnostics['gram_condition_number'] for p in rows]
    assert all(a<b for a,b in zip(cond,cond[1:]))
    assert rows[0].diagnostics['gram_rank']==7
    assert rows[-1].diagnostics['gram_rank']<7
    assert rows[-1].diagnostics['mathematically_full_rank']
    assert rows[-1].diagnostics['method']=='rank_deficient_svd_reference_continuation'
    old=np.random.default_rng(7).normal(size=(2,7))
    for p in rows:
        new=p.apply(old)
        assert np.isfinite(new).all()
        assert independent_error(3,4,5.,p.delta,old,new) <= independent_error(3,4,5.,p.delta,old,old)+1e-11


@pytest.mark.parametrize('degree,M',[(1,3),(2,4),(3,4),(5,2)])
def test_quadrature_converges_at_polynomial_exact_order(degree,M):
    L=5.;delta=.37*L/M
    reference=make_warm_start_projection(degree,M,L,delta,quadrature_order=2*degree+5)
    exact=make_warm_start_projection(degree,M,L,delta)
    coarse=make_warm_start_projection(degree,M,L,delta,quadrature_order=1)
    np.testing.assert_allclose(exact.gram,reference.gram,atol=2e-15,rtol=3e-14)
    np.testing.assert_allclose(exact.cross,reference.cross,atol=2e-15,rtol=3e-14)
    np.testing.assert_allclose(exact.transform,reference.transform,atol=2e-10,rtol=2e-10)
    assert np.linalg.norm(exact.gram-reference.gram)<np.linalg.norm(coarse.gram-reference.gram)*1e-5


def test_independently_integrated_matrices_and_per_channel_solves():
    d,M,L,delta=3,4,5.,.1
    adapter=UniformBasis(d,M);n=d+M
    knots=np.arange(M+1)*L/M
    edges=np.unique(np.r_[knots,knots-delta]);points=edges[(edges>0)&(edges<L-delta)]
    G=np.empty((n,n));B=G.copy()
    for i in range(n):
        for j in range(n):
            G[i,j]=quad(lambda t:adapter.evaluate(t,L)[i]*adapter.evaluate(t,L)[j],
                         0,L-delta,points=points,epsabs=1e-12,epsrel=1e-12)[0]
            B[i,j]=quad(lambda t:adapter.evaluate(t,L)[i]*adapter.evaluate(t+delta,L)[j],
                         0,L-delta,points=points,epsabs=1e-12,epsrel=1e-12)[0]
    p=make_warm_start_projection(d,M,L,delta)
    np.testing.assert_allclose(p.gram,G,atol=8e-16,rtol=2e-13)
    np.testing.assert_allclose(p.cross,B,atol=8e-16,rtol=2e-13)
    old=np.random.default_rng(71).normal(size=(2,n))
    separate=np.stack([np.linalg.solve(G,B@channel) for channel in old])
    np.testing.assert_allclose(p.apply(old),separate,atol=3e-12,rtol=3e-12)


def test_cache_readonly_inputs_and_reproducibility():
    clear_projection_cache()
    p=make_warm_start_projection(3,4,5.,.1)
    assert make_warm_start_projection(3,4,5.,.1) is p
    assert projection_cache_info().hits==1
    for array in (p.gram,p.cross,p.transform,p.breakpoints):
        with pytest.raises(ValueError):array.flat[0]=0
        with pytest.raises(ValueError):array.setflags(write=True)
    old=np.arange(14,dtype=float)/100;before=old.copy();a=p.apply(old)
    np.testing.assert_array_equal(old,before)
    clear_projection_cache();repeated=make_warm_start_projection(3,4,5.,.1)
    np.testing.assert_array_equal(repeated.apply(old),a)
    assert make_warm_start_projection(3,4,5.,.2) is not repeated
    assert make_warm_start_projection(3,4,6.,.1) is not repeated
    assert make_warm_start_projection(2,4,5.,.1) is not repeated
    assert make_warm_start_projection(3,5,5.,.1) is not repeated


def test_empty_overlap_and_input_validation():
    p=make_warm_start_projection(3,4,5.,5.)
    old=np.arange(14).reshape(2,7)
    np.testing.assert_array_equal(p.apply(old),np.repeat(old[:,-1:],7,axis=1))
    assert p.diagnostics['gram_rank']==0 and p.diagnostics['method']=='empty_overlap_repeat_terminal'
    for args in [(3,4,0.,0.),(3,4,5.,-.1),(3,4,5.,5.1),(3,4,np.inf,0.),
                 (-1,4,5.,.1),(3,0,5.,.1),(3,4,5.,np.nan)]:
        with pytest.raises(ValueError):make_warm_start_projection(*args)
    with pytest.raises(TypeError):make_warm_start_projection(2.5,4,5.,.1)
    with pytest.raises(ValueError):make_warm_start_projection(3,4,5.,.1,quadrature_order=0)
    for bad in (np.zeros(7),np.zeros((7,2)),np.full(14,np.nan)):
        with pytest.raises(ValueError):p.apply(bad)


def test_projection_does_not_silently_enforce_optimizer_bounds():
    p=make_warm_start_projection(3,4,5.,.3)
    old=np.array([.5,-.5,.5,-.5,.5,-.5,.5]*2)
    new=p.apply(old)
    assert np.max(np.abs(new))>.5
    assert not p.diagnostics['coefficient_bounds_enforced']
