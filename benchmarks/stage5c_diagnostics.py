"""Deterministic window-position, information and arrival diagnostics."""

import time
import numpy as np

from splines import UniformBasis
from splines.warm_start import make_warm_start_projection


def grid(L=5.,M=4):
    return np.unique(np.r_[np.linspace(0.,L,501),np.arange(M+1)*(L/M)])


def independent_truth(t,kind):
    t=np.asarray(t,dtype=np.float64)
    if kind=='zero':return np.zeros(t.shape+(2,))
    if kind=='constant':return np.broadcast_to([.15,-.05],t.shape+(2,))
    if kind=='smooth':return np.stack((.15*np.sin(2*np.pi*t/5.),.05*np.cos(2*np.pi*t/5.)),axis=-1)
    raise ValueError('unknown disturbance')


def disturbance_error(c,start,kind,sigma=None):
    sigma=grid() if sigma is None else np.asarray(sigma)
    estimate=UniformBasis(3,4).evaluate(sigma,5.)@np.asarray(c).reshape(2,7).T
    truth=independent_truth(start+sigma,kind)
    return estimate-truth,estimate,truth


def error_profile(errors,sigma):
    errors=np.asarray(errors);sigma=np.asarray(sigma)
    if errors.ndim!=3 or errors.shape[1:]!=(len(sigma),2) or len(errors)==0:
        raise ValueError('errors must be (windows,grid,2)')
    E=np.sqrt(np.mean(errors**2,axis=0))
    inside=(sigma>=1.25)&(sigma<=3.75)
    interior=np.sqrt(np.mean(E[inside]**2,axis=0))
    open_interior=np.sqrt(np.mean(E[1:-1]**2,axis=0))
    return {'rmse':E.tolist(),'left':E[0].tolist(),'right':E[-1].tolist(),
        'interior':interior.tolist(),'interior_interval_seconds':[1.25,3.75],
        'open_interior':open_interior.tolist(),'right_exceeds_interior':(E[-1]>interior).tolist(),
        'right_exceeds_open_interior':(E[-1]>open_interior).tolist(),'window_count':len(errors)}


def basis_diagnostics():
    p=make_warm_start_projection(3,4,5.,0.)
    return {'degree':3,'segments':4,'L':5.,'knots':np.linspace(0.,5.,5).tolist(),
        'support_in_window':[[max(0.,(j-3)*1.25),min(5.,(j+1)*1.25)] for j in range(7)],
        'integrated_squared_magnitude':np.diag(p.gram).tolist(),
        'right_endpoint_weights':UniformBasis(3,4).evaluate(5.,5.).tolist()}


def timed_initial(previous,delta,mode):
    begin=time.perf_counter()
    raw=np.zeros(14) if previous is None or mode=='zero' else make_warm_start_projection(3,4,5.,delta).apply(previous)
    projected=time.perf_counter()-begin
    begin=time.perf_counter();feasible=np.clip(raw,-.5,.5)
    clipped=time.perf_counter()-begin
    difference=feasible-raw
    return (feasible,{'raw_projection':raw.tolist(),'feasible_initial':feasible.tolist(),
        'clipped_count':int(np.count_nonzero(difference)),
        'clipping_max_absolute':float(np.max(np.abs(difference))),
        'clipping_l1':float(np.sum(np.abs(difference))),'first_window_zero':previous is None},
        {'projection_seconds':projected,'clipping_seconds':clipped})


def visibility(window,result,data,forward):
    mask=np.asarray(result.valid_mask);births=np.asarray(result.initialization_mask)
    scheduled=data.scheduled[window.source_indices]
    scans=np.flatnonzero(np.r_[False,scheduled])
    corrected=np.any(mask,axis=1)
    past=np.asarray(forward.observations.status)==2
    prior_times=data.times[(data.times<=window.start)&past.any(axis=1)]
    correction_times=np.r_[prior_times,window.times[corrected]]
    latest=float(np.max(correction_times)) if len(correction_times) else None
    return {'scheduled_scans':len(scans),'scans_with_observations':int(np.sum(window.mask[scans].any(axis=1))),
        'empty_scans':int(np.sum(~window.mask[scans].any(axis=1))),
        'correction_records':int(mask.sum()),'birth_records':int(births.sum()),
        'active_landmarks':int(np.asarray(result.final_state.active).sum()),
        'empty_endpoint':not bool(window.mask[-1].any()),
        'seconds_since_latest_correction':None if latest is None else float(window.end-latest),
        'scheduled_relative_times':(window.times[scans]-window.start).tolist(),
        'observation_rows':[{'sigma':float(window.times[i]-window.start),
            'observed_ids':window.ids[i,window.mask[i]].tolist(),
            'correction_ids':window.ids[i,mask[i]].tolist(),'birth_ids':window.ids[i,births[i]].tolist()} for i in scans]}


def covariance_diagnostics(state):
    active=np.asarray(state.active);P=np.asarray(state.covariance)
    indices=np.r_[np.arange(3),3+np.flatnonzero(np.repeat(active,2))]
    block=P[np.ix_(indices,indices)]
    eigen=np.linalg.eigvalsh((block+block.T)/2)
    asymmetry=float(np.max(np.abs(block-block.T)))
    if not np.isfinite(np.asarray(state.x)).all() or not np.isfinite(P).all() or asymmetry>1e-10 or eigen[0]<-1e-10:
        raise ValueError('invalid active covariance; no repair allowed')
    return {'minimum_active_eigenvalue':float(eigen[0]),'maximum_active_eigenvalue':float(eigen[-1]),
        'maximum_asymmetry':asymmetry,'robot_trace':float(np.trace(P[:3,:3])),
        'robot_variances':np.diag(P[:3,:3]).tolist(),'active_landmarks':int(active.sum())}


def state_error(state,truth_pose,landmarks):
    e=np.asarray(state.x)[:3]-truth_pose
    e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi
    active=np.asarray(state.active);ids=np.asarray(state.truth_ids)[active]
    points=np.asarray(state.x)[3:].reshape(-1,2)[active]
    map_errors=np.linalg.norm(points-landmarks[ids],axis=1)
    return {'position_error_m':float(np.linalg.norm(e[:2])),'heading_error_rad':float(e[2]),
        'landmark_ids':ids.tolist(),'landmark_errors_m':map_errors.tolist(),
        'landmark_rmse_m':float(np.sqrt(np.mean(map_errors**2))) if len(map_errors) else None,
        'mapped_landmarks':len(ids),'pose_error':e.tolist()}


def sensitivity_summary(D,valid_mask):
    selected=np.repeat(np.asarray(valid_mask).ravel(),2)
    A=np.asarray(D)[selected]
    singular=np.linalg.svd(A,compute_uv=False)
    largest=float(singular[0]) if len(singular) else 0.
    tolerance=max(A.shape)*np.finfo(float).eps*largest
    rank=int(np.sum(singular>tolerance))
    columns=np.linalg.norm(A,axis=0)
    return {'shape':list(A.shape),'column_norms':columns.tolist(),'singular_values':singular.tolist(),
        'rank_tolerance':float(tolerance),'numerical_rank':rank,
        'condition_number':float(singular[0]/singular[-1]) if rank==14 else None,
        'practical_rank_relative_1e_minus3':int(np.sum(singular>largest*1e-3)),
        'right_column_weaker_than_middle':[bool(columns[j+6]<np.median(columns[j+2:j+5])) for j in (0,7)],
        'whitening':'q=L(c)^-1 r(c), S(c)=L(c)L(c).T, differentiate L and r; correction rows only',
        'formal_observability_claim':False}
