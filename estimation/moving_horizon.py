"""Full-length moving windows around the preserved Stage 4C EKF objective.

The independent forward filter owns arrival information. Optimized histories
are reports only and are never fed back into that filter or its persistent map.
"""

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from estimation.coefficient_optimization import optimize_coefficients
from estimation.fixed_window_objective import InvalidCandidateError, checked_evaluation
from estimation.propagation import rk4_stage_times
from estimation.slam import SlamState, INVALID, filter_slam, raise_on_overflow
from estimation.slam_window_objective import make_slam_window_objective, _state_flags
from splines import UniformBasis
from splines.warm_start import make_warm_start_projection


def frozen(value):
    a=np.asarray(value)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def canonical_time(t,times):
    """Resolve arithmetic roundoff only (eight ULPs at the time scale)."""
    i=int(np.argmin(np.abs(times-t)))
    tolerance=8*np.spacing(max(1.,abs(float(t)),abs(float(times[i]))))
    return float(times[i]) if abs(times[i]-t)<=tolerance else float(t)


@dataclass(frozen=True)
class ReplayData:
    times: np.ndarray
    measurements: np.ndarray
    ids: np.ndarray
    covariance: np.ndarray
    mask: np.ndarray
    scheduled: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self,name,frozen(getattr(self,name)))
        rk4_stage_times(self.times)
        if (self.measurements.shape!=self.mask.shape+(2,) or self.ids.shape!=self.mask.shape
                or self.mask.ndim!=2 or self.mask.shape[0]!=len(self.times)
                or self.scheduled.shape!=self.times.shape or self.scheduled.dtype!=bool
                or self.mask.dtype!=bool or not np.issubdtype(self.ids.dtype,np.signedinteger)
                or self.covariance.shape!=(2,2) or np.any(self.mask & ~self.scheduled[:,None])):
            raise ValueError('inconsistent replay shapes, IDs, masks or scan schedule')


@dataclass(frozen=True)
class Window:
    arrival: SlamState
    times: np.ndarray
    basis: np.ndarray
    measurements: np.ndarray
    ids: np.ndarray
    mask: np.ndarray
    source_indices: np.ndarray
    start: float
    end: float


class ForwardArrivals:
    """Causal zero-disturbance posteriors from observations alone, offline replay."""

    def __init__(self,data,initial,command,Qc,*,degree=3,M=4,max_step=.025):
        if not jax.config.x64_enabled:
            raise RuntimeError('Moving horizon requires JAX float64')
        if max_step<=0 or np.max(np.diff(data.times),initial=0.)>max_step+1e-12:
            raise ValueError('forward replay grid must resolve max_step')
        self.data,self.command=data,command
        self.Qc=jnp.array(Qc,dtype=jnp.float64,copy=True)
        self.degree,self.M,self.max_step=degree,M,max_step
        n=degree+M
        result=jax.jit(lambda:filter_slam(initial,data.times,command,
            np.zeros((len(data.times)-1,4,n)),np.zeros(2*n),self.Qc,
            data.measurements,data.ids,data.covariance,data.mask))()
        jax.block_until_ready(result)
        raise_on_overflow(result)
        if np.any(np.asarray(result.observations.status)==INVALID):
            raise InvalidCandidateError('Invalid observation in independent forward EKF')
        flags=jax.jit(jax.vmap(lambda state:_state_flags(state,1e-10)))(result.states)
        if not np.asarray(flags).all():
            raise InvalidCandidateError('Invalid independent forward EKF state/covariance')
        self.result=result

    def at(self,t):
        t=canonical_time(t,self.data.times)
        if not self.data.times[0]<=t<=self.data.times[-1]:
            raise ValueError('arrival query outside forward history')
        i=int(np.searchsorted(self.data.times,t,side='right')-1)
        state=SlamState(*(v[i].copy() for v in self.result.states))
        if self.data.times[i]==t:
            return state
        # Start from the last posterior and use no future measurement.
        steps=int(np.ceil((t-self.data.times[i])/self.max_step))
        grid=np.linspace(self.data.times[i],t,steps+1)
        n=self.degree+self.M
        result=jax.jit(lambda:filter_slam(state,grid,self.command,
            np.zeros((steps,4,n)),np.zeros(2*n),self.Qc,np.zeros((len(grid),1,2)),
            np.zeros((len(grid),1),dtype=np.int64),self.data.covariance,
            np.zeros((len(grid),1),dtype=bool)))()
        jax.block_until_ready(result)
        arrival=SlamState(*(v[-1].copy() for v in result.states))
        if not np.asarray(_state_flags(arrival,1e-10)).all():
            raise InvalidCandidateError('Invalid propagated arrival')
        return arrival

    def window(self,end,L):
        end=canonical_time(end,self.data.times)
        if not np.isfinite(L) or L<=0:
            raise ValueError('horizon must be finite and positive')
        start=canonical_time(end-L,self.data.times)
        if start<self.data.times[0] or end>self.data.times[-1]:
            raise ValueError('a complete fixed-length horizon is unavailable')
        selected=np.flatnonzero((self.data.times>start)&(self.data.times<=end))
        if not len(selected) or self.data.times[selected[-1]]!=end or not self.data.scheduled[selected[-1]]:
            raise ValueError('optimization endpoint must be a scheduled scan')
        times=np.r_[start,self.data.times[selected]]
        y=np.concatenate((np.zeros_like(self.data.measurements[:1]),self.data.measurements[selected]))
        ids=np.concatenate((np.zeros_like(self.data.ids[:1]),self.data.ids[selected]))
        mask=np.concatenate((np.zeros_like(self.data.mask[:1]),self.data.mask[selected]))
        elapsed=times-start
        # The mathematical horizon has length L; pin its exact spline endpoints
        # rather than evaluating L+roundoff outside the C++ adapter's domain.
        elapsed[0],elapsed[-1]=0.,L
        B=UniformBasis(self.degree,self.M).evaluate(rk4_stage_times(elapsed),L)
        return Window(self.at(start),*(frozen(a) for a in (times,B,y,ids,mask,selected)),start,end)


def initial_guess(previous,delta,*,degree=3,M=4,L=5.,mode='projected',bounds=None):
    n=degree+M
    bounds=np.tile([-.5,.5],(2*n,1)) if bounds is None else np.asarray(bounds)
    if bounds.shape!=(2*n,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>=bounds[:,1]):
        raise ValueError('invalid coefficient bounds')
    if mode not in ('projected','zero'):
        raise ValueError('initialization must be projected or zero')
    if previous is None or mode=='zero':
        raw=np.zeros(2*n)
    else:
        raw=make_warm_start_projection(degree,M,L,delta).apply(previous)
    feasible=np.clip(raw,bounds[:,0],bounds[:,1])
    difference=feasible-raw
    return feasible,{'raw_projection':raw.tolist(),'feasible_initial':feasible.tolist(),
        'clipped_count':int(np.count_nonzero(difference)),
        'clipping_max_absolute':float(np.max(np.abs(difference))),
        'clipping_l1':float(np.sum(np.abs(difference))),
        'first_window_zero':previous is None}


def pose_difference(left,right):
    e=np.asarray(left)[:3]-np.asarray(right)[:3]
    e[2]=(e[2]+np.pi)%(2*np.pi)-np.pi
    return e


def map_difference(left,right):
    left_map={int(i):np.asarray(left.x)[3+2*s:5+2*s] for s,i in enumerate(np.asarray(left.truth_ids)) if left.active[s]}
    right_map={int(i):np.asarray(right.x)[3+2*s:5+2*s] for s,i in enumerate(np.asarray(right.truth_ids)) if right.active[s]}
    ids=sorted(left_map.keys()&right_map.keys())
    return {'ids':ids,'position_differences':[list(left_map[i]-right_map[i]) for i in ids]}


class MovingHorizonSLAM:
    """Compare projected and zero initializations on the very same windows.

    Before the first full horizon only the independent forward EKF is defined.
    Failed numerical candidates stop estimation; there is no silent retry.
    Valid but unconverged optimizer iterates are explicitly labeled as such.
    """

    def __init__(self,forward,L=5.,bounds=None):
        self.forward,self.L=forward,float(L)
        n=forward.degree+forward.M
        self.bounds=np.tile([-.5,.5],(2*n,1)) if bounds is None else np.asarray(bounds).copy()
        self.records=[]

    def run(self,modes=('projected','zero')):
        if len(set(modes))!=len(modes) or not set(modes)<= {'projected','zero'}:
            raise ValueError('distinct projected/zero modes required')
        previous={mode:None for mode in modes}
        schedule=self.forward.data.times[self.forward.data.scheduled]
        schedule=[t for t in schedule if canonical_time(t-self.L,self.forward.data.times)>=self.forward.data.times[0]]
        n=self.forward.degree+self.forward.M
        for end in schedule:
            start_time=time.perf_counter()
            window=self.forward.window(float(end),self.L)
            evaluate=make_slam_window_objective(window.arrival,window.times,self.forward.command,
                window.basis,self.forward.Qc,window.measurements,window.ids,
                self.forward.data.covariance,window.mask,coefficient_count=2*n)
            preparation=time.perf_counter()-start_time
            row={'time':float(end),'window':window,'window_preparation_seconds':preparation,'fits':{}}
            self.records.append(row)
            for mode in modes:
                prior=previous[mode]
                delta=0. if prior is None else float(end)-prior['time']
                initial,clipping=initial_guess(None if prior is None else prior['coefficients'],delta,
                    degree=self.forward.degree,M=self.forward.M,L=self.L,mode=mode,bounds=self.bounds)
                begin=time.perf_counter()
                fit=optimize_coefficients(evaluate,initial,coefficients_per_channel=n,bounds=self.bounds)
                entry={'optimization':fit,'clipping':clipping}
                row['fits'][mode]=entry
                if fit['invalid_trials'] or fit['coefficients'] is None:
                    raise InvalidCandidateError(f'{mode} at t={end}: {fit["status"]}; no estimate reported')
                replay=checked_evaluation(jax.jit(evaluate),np.asarray(fit['coefficients']))
                entry.update(replay=replay,total_update_seconds=preparation+time.perf_counter()-begin)
                if prior is not None:
                    index=int(np.argmin(np.abs(window.times-prior['time'])))
                    if abs(window.times[index]-prior['time'])>1e-12:
                        raise ValueError('previous report time absent from overlapping replay')
                    revised=SlamState(*(v[index] for v in replay.states))
                    entry['jumps']={'reported_pose_delta':pose_difference(replay.final_state.x,prior['state'].x).tolist(),
                        'overlap_pose_revision':pose_difference(revised.x,prior['state'].x).tolist(),
                        'reported_map_delta':map_difference(replay.final_state,prior['state']),
                        'overlap_map_revision':map_difference(revised,prior['state'])}
                previous[mode]={'time':float(end),'coefficients':np.asarray(fit['coefficients']),
                                'state':replay.final_state}
            yield row
