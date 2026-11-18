"""Seeded Stage 5B truth/scan records, independent of estimator and splines."""

import numpy as np
from scipy.integrate import solve_ivp

from simulation.continuous_unicycle import SimulationData,analytical_unicycle
from simulation.smooth_unicycle import smooth_disturbance
from simulation.slam_window import visibility_geometry


def true_disturbance(times,kind,L=5.):
    times=np.asarray(times)
    if kind=='zero':return np.zeros(times.shape+(2,))
    if kind=='constant':return np.broadcast_to([.15,-.05],times.shape+(2,))
    if kind=='smooth':return smooth_disturbance(times,L)
    raise ValueError('zero, constant or smooth disturbance required')


def simulate_moving(cfg):
    """Disturbances start at t=0; no artificial initialization prefix.

    Scheduled scans exist independently of detections. All estimators consume
    one saved dataset. Range/FOV gating and the full-slot seed-7 noise stream
    follow the preserved Stage 4C visibility convention.
    """
    dt,T,L=(cfg[k] for k in ('integration_step_seconds','total_duration_seconds','window_length_seconds'))
    if dt<=0 or T<L or not np.isclose(T/dt,round(T/dt),atol=1e-10,rtol=0):
        raise ValueError('duration must cover the full horizon and align with the integration grid')
    times=np.linspace(0.,T,round(T/dt)+1)
    command=np.asarray(cfg['command_v_omega']);initial=np.asarray(cfg['initial_pose'])
    kind=cfg['disturbance_kind']
    def rhs(t,x):
        v,omega=command+true_disturbance(t,kind,L)
        return [v*np.cos(x[2]),v*np.sin(x[2]),omega]
    reference=solve_ivp(rhs,(0.,T),initial,t_eval=times,method='DOP853',max_step=.0125,rtol=1e-12,atol=1e-14)
    if not reference.success:raise RuntimeError(reference.message)
    if kind in ('zero','constant'):
        truth=analytical_unicycle(times,initial,*(command+true_disturbance(0.,kind,L)))
    else:
        solution=solve_ivp(rhs,(0.,T),initial,t_eval=times,method='DOP853',max_step=.025,rtol=1e-11,atol=1e-13)
        if not solution.success:raise RuntimeError(solution.message)
        truth=solution.y.T
    error=np.max(np.abs(truth-reference.y.T),axis=0)
    if not np.all(error<1e-10):raise RuntimeError('Independent truth refinement failed')
    landmarks=np.asarray(cfg['landmark_coordinates']);R=np.asarray(cfg['measurement_covariance'])
    measurements=np.zeros((len(times),len(landmarks),2));mask=np.zeros(measurements.shape[:2],bool)
    interval=cfg['measurement_interval_seconds'];stride=round(interval/dt)
    if stride<1 or not np.isclose(interval/dt,stride,atol=1e-10,rtol=0):
        raise ValueError('measurement schedule must align with saved grid')
    scheduled=np.arange(len(times))%stride==0;scheduled[0]=False
    rng=np.random.default_rng(cfg['seed'])
    for i in np.flatnonzero(scheduled):
        clean,visible=visibility_geometry(truth[i],landmarks,cfg['sensor'])
        noise=rng.multivariate_normal(np.zeros(2),R,size=len(landmarks))
        noisy=clean[visible]+noise[visible]
        noisy[:,1]=(noisy[:,1]+np.pi)%(2*np.pi)-np.pi
        measurements[i,visible]=noisy;mask[i,visible]=True
    data=SimulationData(times,truth,measurements,np.broadcast_to(landmarks,measurements.shape).copy(),mask,R,landmarks)
    return data,scheduled,{'passed':True,'max_absolute_pose_difference':error.tolist(),
        'truth':'independent analytic constant-input unicycle or DOP853 sinusoid',
        'reference':'DOP853 max_step=.0125 rtol=1e-12 atol=1e-14','disturbance_start_seconds':0.}
