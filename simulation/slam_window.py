"""Independent piecewise unicycle truth and fixed observations for Stage 4C."""

import numpy as np
from scipy.integrate import solve_ivp

from simulation.continuous_unicycle import SimulationData, analytical_unicycle
from simulation.smooth_unicycle import integrate_truth, smooth_disturbance


def sensor_settings(scenario='regression'):
    """Stage 4C assumptions; None denotes unlimited maximum range in JSON."""
    if scenario not in ('regression', 'forward'):
        raise ValueError('sensor scenario must be regression or forward')
    return dict(scenario=scenario, range_min=0. if scenario == 'regression' else .2,
                range_max=None if scenario == 'regression' else 6.,
                field_of_view_rad=2*np.pi if scenario == 'regression' else np.pi,
                translation_xy=[0., 0.], yaw_rad=0.)


def visibility_geometry(pose, landmarks, sensor):
    """Noise-free range/bearing and inclusive visibility in the true robot frame.

    The existing EKF models a colocated aligned sensor. Reject unsupported
    extrinsics rather than generate measurements inconsistent with that model.
    """
    if np.any(np.asarray(sensor.get('translation_xy', [0., 0.])) != 0.) or sensor.get('yaw_rad', 0.) != 0.:
        raise ValueError('Stage 4C measurement model requires zero sensor extrinsics')
    lower, upper, fov = sensor['range_min'], sensor['range_max'], sensor['field_of_view_rad']
    upper = np.inf if upper is None else upper
    if not (np.isfinite(lower) and 0 <= lower <= upper and upper > 0 and
            np.isfinite(fov) and 0 < fov <= 2*np.pi):
        raise ValueError('invalid sensor range or field of view')
    delta = np.asarray(landmarks)-np.asarray(pose)[:2]
    ranges = np.hypot(delta[:, 0], delta[:, 1])
    raw = np.arctan2(delta[:, 1], delta[:, 0])-pose[2]
    bearings = (raw+np.pi) % (2*np.pi)-np.pi
    visible = (ranges >= lower) & (ranges <= upper) & (np.abs(bearings) <= fov/2)
    return np.stack((ranges, bearings), axis=-1), visible


def simulate_slam_window(cfg):
    """Two disturbance-free seconds, followed by five disturbed seconds.

    Truth is independent NumPy closed form. A split DOP853 reference validates
    the trajectory without integrating across the input discontinuity.
    """
    dt, t0, L = cfg['integration_step_seconds'], cfg['initialization_seconds'], cfg['window_length_seconds']
    times = np.linspace(0., t0+L, round((t0+L)/dt)+1)
    split = round(t0/dt)
    command, disturbance = np.asarray(cfg['command_v_omega']), np.array([.15, -.05])
    initial = np.asarray(cfg['initial_pose'])
    before = analytical_unicycle(times[:split+1], initial, *command)
    kind = cfg['disturbance_kind']
    if kind == 'constant':
        after = analytical_unicycle(times[split:]-t0, before[-1], *(command+disturbance))
    elif kind == 'smooth':
        after, _ = integrate_truth(times[split:]-t0, before[-1], command, L=L)
    else:
        raise ValueError('disturbance_kind must be constant or smooth')
    truth = np.concatenate((before[:-1], after))
    references = []
    for phase, (local_times, pose, inputs) in enumerate([(times[:split+1], initial, command),
                                       (times[split:]-t0, before[-1], command+disturbance)]):
        def rhs(t, x):
            effective = command+smooth_disturbance(t,L) if phase == 1 and kind == 'smooth' else inputs
            return [effective[0]*np.cos(x[2]), effective[0]*np.sin(x[2]), effective[1]]
        solution = solve_ivp(rhs, (0., local_times[-1]), pose, method='DOP853', t_eval=local_times,
                             max_step=.025, rtol=1e-12, atol=1e-14)
        if not solution.success:
            raise RuntimeError(solution.message)
        references.append(solution.y.T)
    reference = np.concatenate((references[0][:-1], references[1]))
    discrepancy = np.max(np.abs(truth-reference), axis=0)
    if not np.all(discrepancy < 1e-10):
        raise RuntimeError('Independent ground-truth integration check failed')
    landmarks, R = np.asarray(cfg['landmark_coordinates']), np.asarray(cfg['measurement_covariance'])
    measurements = np.zeros((len(times), len(landmarks), 2))
    mask = np.zeros(measurements.shape[:2], dtype=bool)
    sensor = cfg.get('sensor', sensor_settings())
    rng = np.random.default_rng(cfg['seed'])
    stride = round(cfg['measurement_interval_seconds']/dt)
    for index in range(stride, len(times), stride):
        _, visible = visibility_geometry(truth[index], landmarks, sensor)
        # Keep the original full-slot noise stream, including invisible slots,
        # so the regression is bitwise preserved and shared observations across
        # scenarios use identical noise. Only visible rows are recorded.
        delta = landmarks-truth[index, :2]
        clean = np.stack((np.linalg.norm(delta, axis=1),
                          np.arctan2(delta[:, 1], delta[:, 0])-truth[index, 2]), axis=-1)
        noise = rng.multivariate_normal(np.zeros(2), R, size=len(landmarks))
        noisy = clean[visible]+noise[visible]
        noisy[:, 1] = (noisy[:, 1]+np.pi) % (2*np.pi)-np.pi
        measurements[index, visible], mask[index, visible] = noisy, True
    data = SimulationData(times, truth, measurements, np.broadcast_to(landmarks, measurements.shape).copy(),
                          mask, R, landmarks.copy())
    return data, {'method': 'independent piecewise NumPy analytic unicycle' if kind == 'constant'
                   else 'analytic prefix and independent DOP853 smooth truth, max_step=.05, rtol=1e-11, atol=1e-13',
                   'reference_method': 'split SciPy DOP853, max_step=.025, rtol=1e-12, atol=1e-14',
                   'max_absolute_pose_difference': discrepancy.tolist(), 'passed': True}
