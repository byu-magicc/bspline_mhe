"""Validation of Estimator A preservation, data semantics, replay and metrics."""

from contextlib import redirect_stdout
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from benchmarks.estimator_a import (
    Dataset, accuracy_metrics, association_metrics, generate_dataset,
    identical_replays, legacy_modules, observations, replay, run_experiment,
    source_hashes,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(seed=7, steps=60)


@pytest.mark.parametrize("association", ["known", "nn"])
@pytest.mark.parametrize("method", ["batch", "seq"])
@pytest.mark.parametrize("mode", ["original", "noisy"])
def test_exact_parity_with_original_runner(dataset, association, method, mode):
    """Execute legacy run(), not a rewritten reference loop, including generation."""
    legacy = legacy_modules()
    before = source_hashes()
    captured_states, captured_covariances, captured_assignments, generated = [], [], [], []

    def capture_plot(data, step, initial, mu, covariance, history):
        captured_states.append(mu.copy())
        captured_covariances.append(covariance.copy())

    def capture_generate(*args, **kwargs):
        data = legacy.generator.generate(*args, **kwargs)
        generated.append(data.copy())
        if mode == "noisy":
            # Adapt the runner's input slots independently of observations().
            # The estimator itself still executes the original run() unchanged.
            data = data.copy()
            for start in range(9, data.shape[1], 5):
                data[:, start+1:start+3] = generated[-1][:, start+3:start+5]
        return data

    association_module = getattr(legacy, f"da_{association}")
    original_associate = association_module.associateData

    def capture_associate(*args, **kwargs):
        result = original_associate(*args, **kwargs)
        captured_assignments.append(result[0].copy())
        return result

    rng_state = np.random.get_state()
    try:
        np.random.seed(dataset.config["seed"])
        with (patch.object(legacy.simSLAM, "plt", MagicMock()),
              patch.object(legacy.simSLAM, "plotsim", side_effect=capture_plot),
              patch.object(legacy.simSLAM, "generate", side_effect=capture_generate),
              patch.object(association_module, "associateData", side_effect=capture_associate),
              patch.object(np, "savez"), redirect_stdout(io.StringIO())):
            legacy.simSLAM.run(len(dataset.raw), association, method, 0, False)
    finally:
        np.random.set_state(rng_state)
    result = replay(dataset, mode, association, method)
    np.testing.assert_array_equal(generated[0], dataset.raw)
    for step, (state, cov, assigned) in enumerate(zip(
            captured_states, captured_covariances, captured_assignments)):
        np.testing.assert_array_equal(result.arrays["state"][step, :len(state)], state)
        np.testing.assert_array_equal(result.arrays["covariance"][step, :len(state), :len(state)], cov)
        np.testing.assert_array_equal(result.arrays["association_indices"][step, :len(assigned)], assigned)
    assert before == source_hashes()


def test_seed_generation_and_rng_restoration(dataset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # No legacy savedData directory or partial writes required.
    before = np.random.get_state()
    same = generate_dataset(seed=7, steps=60)
    after = np.random.get_state()
    for original, restored in zip(before, after):
        np.testing.assert_array_equal(original, restored)
    np.testing.assert_array_equal(same.raw, dataset.raw)
    different = generate_dataset(seed=8, steps=60)
    assert not np.array_equal(different.raw[:, 3:6], dataset.raw[:, 3:6])
    assert not list(tmp_path.iterdir())


def test_observation_semantics_against_geometry(dataset):
    legacy = legacy_modules()
    residuals = []
    for row in dataset.raw:
        clean = observations(row, "original")
        noisy = observations(row, "noisy")
        np.testing.assert_array_equal(clean[:, 2], noisy[:, 2])
        for measurement in clean:
            predicted = legacy.generator.observation(row[3:6], int(measurement[2]))
            np.testing.assert_array_equal(measurement, predicted[[1, 2, 0]])
        residuals.extend(noisy[:, :2] - clean[:, :2])
    rms = np.sqrt(np.mean(np.array(residuals)**2, axis=0))
    assert rms[0] > 1
    assert rms[1] > 0.01
    with pytest.raises(ValueError, match="Unknown observation"):
        observations(dataset.raw[0], "typo")


@pytest.mark.parametrize("mode", ["original", "noisy"])
@pytest.mark.parametrize("association", ["known", "nn"])
def test_saved_replay_exact_and_noisy_adapter(dataset, mode, association, tmp_path):
    path = tmp_path / "data.npz"
    dataset.save(path)
    first = replay(Dataset.load(path), mode, association)
    np.random.seed(123)  # Replay must not depend on RNG state.
    second = replay(Dataset.load(path), mode, association)
    assert identical_replays(first, second)
    assert first.metrics == {**second.metrics,
                             "mean_update_ms": first.metrics["mean_update_ms"],
                             "p95_update_ms": first.metrics["p95_update_ms"],
                             "estimator_total_seconds": first.metrics["estimator_total_seconds"]}
    for step, row in enumerate(dataset.raw):
        z = observations(row, mode)
        np.testing.assert_array_equal(first.arrays["used_observations"][step, :len(z)], z)
    assert np.isfinite(first.arrays["estimated_pose"]).all()
    assert first.metrics["max_relative_covariance_asymmetry"] < 1e-10
    assert first.metrics["min_relative_covariance_eigenvalue"] > -1e-10
    if association == "known":
        assert first.metrics["association_accuracy"] == 1
        assert first.metrics["duplicate_initializations"] == 0
    with np.load(path, allow_pickle=False) as saved:
        assert "source_sha256" in str(saved["metadata_json"])
        np.testing.assert_array_equal(saved["true_executed_pose"], dataset.raw[:, 3:6])
        np.testing.assert_array_equal(saved["commands"], dataset.raw[:, :3])


def test_noisy_adapter_changes_estimates(dataset):
    clean = replay(dataset, "original")
    noisy = replay(dataset, "noisy")
    assert not np.array_equal(clean.arrays["estimated_pose"], noisy.arrays["estimated_pose"])


def test_metric_definitions_and_wrapped_heading():
    truth = np.array([[0., 0., np.pi - 0.1], [0., 0., -np.pi + 0.1]])
    estimated = np.array([[3., 4., -np.pi + 0.1], [0., 0., np.pi - 0.1]])
    landmarks = np.array([[0., 0.], [10., 10.]])
    metrics = accuracy_metrics(truth, estimated, landmarks,
                               np.array([[13., 14.], [0., 0.]]), np.array([1, 0]))
    assert metrics["position_rmse"] == pytest.approx(np.sqrt(12.5))
    assert metrics["heading_rmse_rad"] == pytest.approx(0.2)
    assert metrics["landmark_rmse"] == pytest.approx(np.sqrt(12.5))
    # Three matches: two correct, one incorrect; three new, one duplicate, one reject.
    assigned = np.array([[-1, -1], [0, 0], [-1, 1], [-2, -3]])
    ids = np.array([[0, 1], [0, 1], [0, 1], [0, -1]])
    scored = association_metrics(assigned, ids, np.array([0, 1, 0]))
    assert scored["association_accuracy"] == pytest.approx(2 / 3)
    assert scored["association_coverage"] == pytest.approx(3 / 7)
    assert scored["used_observation_fraction"] == pytest.approx(6 / 7)
    assert scored["duplicate_initializations"] == 1
    assert scored["incorrect_existing_matches"] == 1
    assert scored["rejected_measurements"] == 1
    assert scored["unmatched_measurements"] == 4
    empty = association_metrics(np.array([[-1]]), np.array([[0]]), np.array([0]))
    assert empty["association_accuracy"] is None


def test_motion_jacobians_against_central_difference():
    legacy = legacy_modules()
    pose = np.array([2., -3., 0.3])
    control = np.array([0.2, 10., -0.1])
    _, state_jacobian, input_jacobian = legacy.simSLAM.g(control, pose.copy())
    eps = 1e-6
    for column in range(3):
        offset = np.eye(3)[column] * eps
        state_fd = (legacy.simSLAM.g(control, pose + offset)[0] -
                    legacy.simSLAM.g(control, pose - offset)[0]) / (2 * eps)
        input_fd = (legacy.simSLAM.g(control + offset, pose.copy())[0] -
                    legacy.simSLAM.g(control - offset, pose.copy())[0]) / (2 * eps)
        np.testing.assert_allclose(state_jacobian[:, column], state_fd, rtol=1e-7, atol=2e-9)
        np.testing.assert_allclose(input_jacobian[:, column], input_fd, rtol=1e-7, atol=2e-9)


def test_known_measurement_jacobian_against_central_difference():
    legacy = legacy_modules()
    state = np.array([2., -3., 0.3, 5., 10.])

    def h(x):
        delta = x[3:5] - x[:2]
        return np.array([np.linalg.norm(delta), np.arctan2(delta[1], delta[0]) - x[2]])

    z = np.array([[*h(state), 0]])
    _, jacobian, innovation = legacy.da_known.associateData(
        z, np.eye(2), state, np.eye(5), np.array([0]))
    np.testing.assert_allclose(innovation, 0, atol=1e-14)
    eps = 1e-6
    fd = np.column_stack([(h(state + np.eye(5)[j] * eps) -
                           h(state - np.eye(5)[j] * eps)) / (2 * eps) for j in range(5)])
    np.testing.assert_allclose(jacobian[0], fd, rtol=1e-7, atol=2e-9)


def test_empty_observations_fail_explicitly(dataset):
    sparse = Dataset(dataset.raw.copy(), dataset.landmarks.copy(), dataset.metadata)
    sparse.raw[3, [9, 14]] = -1
    with pytest.raises(ValueError, match="empty-observation"):
        replay(sparse)


def test_end_to_end_artifacts_and_saved_dataset_replay(tmp_path):
    output = tmp_path / "first"
    result = run_experiment(output, steps=50, repeats=2)
    assert result["validation_passed"]
    assert all(result["checks"].values())
    for filename in ("dataset.npz", "report.md", "summary.json", "trajectories_known.png",
                     "trajectories_nn.png", "original_known.npz", "noisy_known.npz"):
        assert (output / filename).stat().st_size > 0
    again = run_experiment(tmp_path / "again", dataset_path=output / "dataset.npz", repeats=2)
    assert again["validation_passed"]
    assert result["dataset_sha256"] == again["dataset_sha256"]
    for case in result["results"]:
        assert result["results"][case]["position_rmse"] == again["results"][case]["position_rmse"]


def test_reject_invalid_validation_config(tmp_path):
    with pytest.raises(ValueError, match="two"):
        generate_dataset(7, 1)
    with pytest.raises(ValueError, match="two"):
        run_experiment(tmp_path, repeats=1)
