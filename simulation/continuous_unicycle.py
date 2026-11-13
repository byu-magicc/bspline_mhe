"""Seeded continuous-time unicycle and known-landmark observation experiment."""

from typing import NamedTuple

import numpy as np


class SimulationData(NamedTuple):
    times: np.ndarray
    truth: np.ndarray
    measurements: np.ndarray
    landmarks_at_times: np.ndarray
    mask: np.ndarray
    measurement_covariance: np.ndarray
    landmark_map: np.ndarray


DEFAULT_LANDMARKS = np.array([
    [5.0, 2.0], [7.0, -4.0], [1.0, 7.0], [-4.0, 4.0],
    [-5.0, -3.0], [3.0, -7.0],
], dtype=np.float64)


def analytical_unicycle(times, initial_pose, speed, turn_rate):
    """Independent NumPy closed-form trajectory for constant effective inputs."""
    times = np.asarray(times, dtype=np.float64)
    x0, y0, theta0 = np.asarray(initial_pose, dtype=np.float64)
    angle = turn_rate * times
    distance = speed * times * np.sinc(angle / (2 * np.pi))
    midpoint_heading = theta0 + angle / 2
    return np.stack((x0 + distance * np.cos(midpoint_heading),
                     y0 + distance * np.sin(midpoint_heading),
                     theta0 + angle), axis=-1)


def simulate_known_landmarks(
    *,
    L=5.0,
    dt=0.025,
    measurement_interval=0.1,
    command=(1.0, 0.2),
    disturbance=(0.15, -0.05),
    initial_pose=(0.0, 0.0, 0.0),
    landmark_map=DEFAULT_LANDMARKS,
    measurement_covariance=None,
    seed=7,
):
    """Generate fixed-known-association range-bearing records on a static map.

    Measurements are emitted at positive multiples of ``measurement_interval``;
    the arrival time at zero is deliberately measurement-free. Landmark order
    is fixed and every map landmark is observed at each emitted timestamp.
    """
    if not np.isfinite([L, dt, measurement_interval]).all() or min(L, dt, measurement_interval) <= 0:
        raise ValueError("L, dt, and measurement_interval must be finite and positive")
    ratio, observation_ratio = L / dt, L / measurement_interval
    measurement_stride = measurement_interval / dt
    if (not np.isclose(ratio, round(ratio), rtol=0, atol=1e-10)
            or not np.isclose(observation_ratio, round(observation_ratio), rtol=0, atol=1e-10)
            or not np.isclose(measurement_stride, round(measurement_stride), rtol=0, atol=1e-10)):
        raise ValueError("L must be an integer multiple of dt and measurement_interval")
    times = np.linspace(0., L, int(round(ratio)) + 1, dtype=np.float64)
    landmarks = np.asarray(landmark_map, dtype=np.float64)
    if landmarks.ndim != 2 or landmarks.shape[1] != 2 or len(landmarks) == 0:
        raise ValueError("landmark_map must have shape (N,2), N > 0")
    if measurement_covariance is None:
        measurement_covariance = np.diag([0.05**2, np.deg2rad(1.0)**2])
    R = np.asarray(measurement_covariance, dtype=np.float64)
    if R.shape != (2, 2) or not np.allclose(R, R.T) or np.min(np.linalg.eigvalsh(R)) <= 0:
        raise ValueError("measurement_covariance must be symmetric positive definite (2,2)")

    v_eff = float(command[0]) + float(disturbance[0])
    omega_eff = float(command[1]) + float(disturbance[1])
    truth = analytical_unicycle(times, initial_pose, v_eff, omega_eff)
    slots = len(landmarks)
    measurements = np.zeros((len(times), slots, 2), dtype=np.float64)
    landmarks_at_times = np.broadcast_to(landmarks, (len(times), slots, 2)).copy()
    mask = np.zeros((len(times), slots), dtype=bool)
    observation_stride = int(round(measurement_stride))
    observed_nodes = np.arange(len(times)) % observation_stride == 0
    observed_nodes[0] = False

    rng = np.random.default_rng(seed)
    for i in np.flatnonzero(observed_nodes):
        pose = truth[i]
        delta = landmarks - pose[:2]
        clean = np.stack((np.hypot(delta[:, 0], delta[:, 1]),
                          np.arctan2(delta[:, 1], delta[:, 0]) - pose[2]), axis=-1)
        noisy = clean + rng.multivariate_normal(np.zeros(2), R, size=slots)
        noisy[:, 1] = (noisy[:, 1] + np.pi) % (2 * np.pi) - np.pi
        measurements[i] = noisy
        mask[i] = True
    return SimulationData(times, truth, measurements, landmarks_at_times, mask, R, landmarks.copy())
