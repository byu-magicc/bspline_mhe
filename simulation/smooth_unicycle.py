"""Independent sinusoidal truth for the controlled Stage 4B extension.

SciPy DOP853 integrates only the deterministic ground truth. This module does
not import JAX, the estimator, or the spline adapter.
"""

import numpy as np
from scipy.integrate import solve_ivp

from simulation.continuous_unicycle import SimulationData


def smooth_disturbance(sigma, L=5.0):
    """Return [..., (linear, angular)] = [.15 sin(2pi sigma/L), .05 cos(...)]."""
    phase = 2 * np.pi * np.asarray(sigma, dtype=np.float64) / L
    return np.stack((.15 * np.sin(phase), .05 * np.cos(phase)), axis=-1)


def integrate_truth(times, initial_pose, command, *, L=5.0, max_step=.05,
                    rtol=1e-11, atol=1e-13):
    """Integrate the analytic inputs independently of EKF/RK4/spline code.

    ``times`` are elapsed window times starting at zero. The internal adaptive
    integration grid is independent of the estimator's output sample grid.
    """
    times = np.asarray(times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or times[-1] > L or L <= 0):
        raise ValueError('times must increase from zero within [0,L]')

    def rhs(sigma, pose):
        disturbance = smooth_disturbance(sigma, L)
        speed, turn = np.asarray(command) + disturbance
        return [speed * np.cos(pose[2]), speed * np.sin(pose[2]), turn]

    solution = solve_ivp(rhs, (0., times[-1]), initial_pose, method='DOP853',
                         t_eval=times, max_step=max_step, rtol=rtol, atol=atol)
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(f'Independent truth integration failed: {solution.message}')
    return solution.y.T, {'method': 'SciPy DOP853', 'max_step_seconds': max_step,
                          'rtol': rtol, 'atol': atol, 'nfev': solution.nfev,
                          'message': solution.message}


def simulate_smooth_landmarks(cfg):
    """Same observation policy and seeded draw order as constant Stage 4B."""
    L, dt = cfg['window_length_seconds'], cfg['integration_step_seconds']
    interval = cfg['measurement_interval_seconds']
    ratios = np.array([L / dt, L / interval, interval / dt])
    if min(L, dt, interval) <= 0 or not np.isfinite(ratios).all() or not np.allclose(
            ratios, np.round(ratios), rtol=0, atol=1e-10):
        raise ValueError('window and measurement interval must align with dt')
    times = np.linspace(0., L, round(L / dt) + 1)
    truth, integration = integrate_truth(times, cfg['initial_pose'], cfg['command_v_omega'], L=L)
    refined, refinement = integrate_truth(times, cfg['initial_pose'], cfg['command_v_omega'],
                                         L=L, max_step=.025, rtol=1e-12, atol=1e-14)
    differences = np.max(np.abs(truth-refined), axis=0)
    integration['refinement'] = {**refinement, 'max_absolute_pose_difference': differences.tolist(),
                                 'acceptance_atol': 1e-10, 'passed': bool(np.all(differences < 1e-10))}
    if not integration['refinement']['passed']:
        raise RuntimeError('Ground-truth step/tolerance refinement failed')
    landmarks = np.asarray(cfg['landmark_coordinates'], dtype=np.float64)
    R = np.asarray(cfg['measurement_covariance'], dtype=np.float64)
    slots = len(landmarks)
    observations = np.zeros((len(times), slots, 2))
    mask = np.zeros((len(times), slots), dtype=bool)
    rng = np.random.default_rng(cfg['seed'])
    for index in range(round(interval / dt), len(times), round(interval / dt)):
        delta = landmarks-truth[index, :2]
        clean = np.stack((np.hypot(delta[:, 0], delta[:, 1]),
                          np.arctan2(delta[:, 1], delta[:, 0])-truth[index, 2]), axis=-1)
        noisy = clean+rng.multivariate_normal(np.zeros(2), R, size=slots)
        noisy[:, 1] = (noisy[:, 1]+np.pi) % (2*np.pi)-np.pi
        observations[index] = noisy
        mask[index] = True
    return SimulationData(times, truth, observations,
                          np.broadcast_to(landmarks, observations.shape).copy(),
                          mask, R, landmarks.copy()), integration
