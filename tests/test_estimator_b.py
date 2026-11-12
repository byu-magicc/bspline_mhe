"""Saved-data command/noise adapter checks, independent of Baseline A code."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path

from benchmarks.estimator_b import odometry_phases, replay_cycle, replay_dataset
from benchmarks.estimator_a import Dataset
from estimation.propagation import rk4_stage_times
from estimation.slam import create_state, initialize_landmark
from splines import UniformBasis


@pytest.fixture(scope="module", autouse=True)
def float64():
    with jax.experimental.enable_x64():
        yield


@pytest.mark.parametrize("u", [[0., 10., 0.], [np.pi/4, 10., np.pi/4], [np.pi/4, 0., np.pi/4]])
def test_rtr_phase_mapping_matches_discrete_first_order_prediction(u):
    u = np.array(u)
    alphas = np.array([.05, .001, .05, .01])**2
    controls, durations, Qc = odometry_phases(u, .1, alphas)
    starts = np.r_[0., np.cumsum(durations)[:-1]]
    edges = np.r_[0., np.cumsum(durations)]
    adapter = UniformBasis(3, 4)
    B = adapter.precompute_jax(rk4_stage_times(edges), edges[-1])
    state = create_state([1., 2., .3], np.diag([.03, .04, .02]), capacity=3)
    state = initialize_landmark(state, 0, -9, np.array([3., .2]), np.eye(2)*.01)
    updated, _ = jax.jit(replay_cycle)(state, starts, durations, controls, Qc, B,
                                      np.zeros(14), np.empty((0, 2)), np.empty(0, dtype=np.int64),
                                      np.eye(2)*.01, np.empty(0, dtype=bool))
    angle = float(state.x[2]) + u[0]
    co, si = np.cos(angle), np.sin(angle)
    expected_x = np.asarray(state.x).copy()
    expected_x[:3] += [u[1]*co, u[1]*si, u[0]+u[2]]
    F = np.eye(len(expected_x)); F[0, 2] = -u[1]*si; F[1, 2] = u[1]*co
    R = np.array([[-u[1]*si, co, 0.], [u[1]*co, si, 0.], [1., 0., 1.]])
    variances = [alphas[0]*u[0]**2+alphas[1]*u[1]**2,
                 alphas[2]*u[1]**2+alphas[3]*(u[0]**2+u[2]**2),
                 alphas[0]*u[2]**2+alphas[1]*u[1]**2]
    expected_P = F@np.asarray(state.covariance)@F.T
    expected_P[:3, :3] += R@np.diag(variances)@R.T
    np.testing.assert_allclose(updated.x, expected_x, rtol=0, atol=1e-11)
    np.testing.assert_allclose(updated.covariance, expected_P, rtol=0, atol=1e-11)
    print(f"\nRTR mapping {u.tolist()}: mean error={np.max(np.abs(updated.x-expected_x)):.6e}, "
          f"covariance error={np.max(np.abs(updated.covariance-expected_P)):.6e}")


def test_short_saved_replay_uses_data_and_preserves_inputs():
    saved = Dataset.load(Path('artifacts/milestone1/dataset.npz'))
    metadata = {**saved.metadata, 'config': {**saved.config, 'steps': 12}}
    dataset = Dataset(saved.raw[:12].copy(), saved.landmarks.copy(), metadata)
    before = dataset.raw.copy()
    arrays, metrics = replay_dataset(dataset, mode='noisy', capacity=16, repeats=2)
    assert metrics['exact_replays']
    assert metrics['initialized'] > 0 and metrics['updated'] > 0
    assert metrics['invalid'] == 0 and metrics['overflow'] == 0
    assert metrics['max_inactive_storage'] == 0
    np.testing.assert_array_equal(dataset.raw, before)
    np.testing.assert_array_equal(arrays['used_observations'], dataset.raw[:, 9:].reshape(12, 2, 5)[:, :, 3:5])
    # Truth coordinates are evaluation-only: changing the evaluation map must
    # leave the complete filtered numerical trajectory and covariance unchanged.
    changed = Dataset(dataset.raw, dataset.landmarks + 10., dataset.metadata)
    other, _ = replay_dataset(changed, mode='noisy', capacity=16, repeats=1)
    np.testing.assert_array_equal(other['state'], arrays['state'])
    np.testing.assert_array_equal(other['covariance'], arrays['covariance'])
