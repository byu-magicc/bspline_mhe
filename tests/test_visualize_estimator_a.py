"""Saved-data adapter and headless playback checks; never run an estimator."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
import numpy as np
import pytest

from benchmarks.visualize_estimator_a import ROOT, ReplayData, ReplayViewer


@pytest.fixture
def synthetic_files(tmp_path):
    """Three frames exercise new, incorrect, duplicate, rejected and absent slots."""
    truth = np.array([[1, 0, np.pi - .1], [2, 0, -np.pi + .1], [3, 0, 0]])
    pose = np.array([[1.5, 0, -np.pi + .1], [2.5, 0, np.pi - .1], [3.5, 0, .2]])
    landmarks = np.array([[10., 0.], [0., 10.]])
    raw = np.zeros((3, 19))
    raw[:, 3:6] = truth
    slots = raw[:, 9:].reshape(3, 2, 5)
    slots[:, :, 0] = [[0, 1], [0, -1], [0, 1]]
    used = np.full((3, 2, 3), np.nan)
    for index in range(3):
        for slot in range(2):
            truth_id = int(slots[index, slot, 0])
            if truth_id < 0:
                continue
            delta = landmarks[truth_id] - truth[index, :2]
            bearing = (np.arctan2(delta[1], delta[0]) - truth[index, 2] + np.pi) % (2*np.pi) - np.pi
            slots[index, slot, 1:3] = [np.linalg.norm(delta), bearing]
            slots[index, slot, 3:5] = slots[index, slot, 1:3] + [.2, .02]
            used[index, slot] = [*slots[index, slot, 3:5], truth_id]
    maps = np.full((3, 3, 2), np.nan)
    maps[:, :2] = landmarks
    maps[2, 2] = [11., 0.]
    dataset = tmp_path / "dataset.npz"
    replay = tmp_path / "replay.npz"
    dataset_values = dict(raw=raw, landmark_positions=landmarks, time=np.array([.1, .2, .3]),
                          metadata_json=np.array(json.dumps({"config": {"initial_pose": [0, 0, 0]}})))
    replay_values = dict(
        metrics_json=np.array(json.dumps({"estimator": "A", "observation_mode": "noisy", "association": "nn"})),
        estimated_pose=pose, landmark_estimates=maps, landmark_birth_truth_ids=np.array([0, 1, 0]),
        state_dimensions=np.array([7, 7, 9]), used_observations=used,
        association_indices=np.array([[-1, -1], [1, -3], [-1, -2]]),
        observation_truth_ids=np.array([[0, 1], [0, -1], [0, 1]]),
        covariance=np.tile(np.eye(9), (3, 1, 1)),
    )
    np.savez(dataset, **dataset_values)
    np.savez(replay, **replay_values)
    return SimpleNamespace(dataset=dataset, replay=replay,
                           dataset_values=dataset_values, replay_values=replay_values)


def test_frame_adapter_geometry_identity_and_padding(synthetic_files):
    data = ReplayData(synthetic_files.dataset, synthetic_files.replay)
    first, middle, last = [data.frame(index) for index in range(3)]
    assert first.time == .1
    assert first.position_error == pytest.approx(.5)
    assert first.heading_error == pytest.approx(.2)
    assert middle.heading_error == pytest.approx(-.2)
    assert first.true_path.shape == (2, 3)  # Initial pose plus current row.
    np.testing.assert_array_equal(last.true_path[-1], data.truth[2])
    assert len(first.landmarks) == 2  # No future third track leaks into first frame.
    assert [o.status for o in first.observations] == ["new landmark", "new landmark"]
    assert [o.track for o in first.observations] == [0, 1]
    assert len(middle.observations) == 1  # Absent padding is not an observation.
    bad = middle.observations[0]
    assert (bad.status, bad.truth_id, bad.track, bad.birth_truth_id) == ("incorrect", 0, 1, 1)
    np.testing.assert_array_equal(bad.target, data.maps[1, 1])
    assert [o.status for o in last.observations] == ["duplicate initialization", "rejected"]
    assert last.observations[0].track == 2
    assert last.observations[1].target is None
    assert last.duplicate_tracks.tolist() == [False, False, True]
    assert data.issue_frames() == (1, 2)
    # Projection is visualization-only, from the consumed noisy measurement.
    radius, bearing, _ = data.measurements[1, 0]
    expected = data.truth[1, :2] + radius * np.array([
        np.cos(data.truth[1, 2] + bearing), np.sin(data.truth[1, 2] + bearing)])
    np.testing.assert_allclose(bad.truth_endpoint, expected)
    assert np.linalg.norm(bad.truth_endpoint - data.true_landmarks[0]) > .01
    with pytest.raises(ValueError, match="read-only"):
        first.landmarks[0, 0] = 99
    with pytest.raises(IndexError):
        data.frame(-1)


def test_adapter_without_recorded_associations(synthetic_files):
    values = synthetic_files.replay_values.copy()
    del values["association_indices"]
    del values["covariance"]
    np.savez(synthetic_files.replay, **values)
    data = ReplayData(synthetic_files.dataset, synthetic_files.replay)
    assert data.robot_covariance is None
    assert all(o.status == "unavailable" and o.target is None for o in data.frame(0).observations)
    assert data.issue_frames() == ()


@pytest.mark.parametrize("corruption", ["mode", "frame_count", "track", "birth", "padding"])
def test_reject_mismatched_or_corrupt_replay(synthetic_files, corruption):
    values = synthetic_files.replay_values
    if corruption == "mode":
        values["metrics_json"] = np.array(json.dumps({"estimator": "A", "observation_mode": "original", "association": "nn"}))
    elif corruption == "frame_count":
        values["estimated_pose"] = values["estimated_pose"][:-1]
    elif corruption == "track":
        values["association_indices"][1, 0] = 50
    elif corruption == "birth":
        values["landmark_birth_truth_ids"][2] = 1
    else:
        values["landmark_estimates"][0, 2] = [9., 9.]
    np.savez(synthetic_files.replay, **values)
    with pytest.raises(ValueError):
        ReplayData(synthetic_files.dataset, synthetic_files.replay)


@pytest.mark.parametrize("mode", ["original", "noisy"])
@pytest.mark.parametrize("association", ["known", "nn"])
def test_existing_saved_experiments(mode, association):
    directory = ROOT / "artifacts" / "milestone1"
    data = ReplayData(directory / "dataset.npz", directory / f"{mode}_{association}.npz")
    statuses = [o.status for index in range(len(data)) for o in data.frame(index).observations]
    assert statuses.count("incorrect") == data.metrics["incorrect_existing_matches"]
    assert statuses.count("duplicate initialization") == data.metrics["duplicate_initializations"]
    assert len(data.frame(len(data) - 1).landmarks) == data.metrics["mapped_landmark_count"]
    if association == "known":
        assert data.issue_frames() == ()


def test_controls_render_and_preserve_inputs(synthetic_files, tmp_path):
    source = ROOT / "baseline" / "ekf_slam" / "code"
    paths = list(source.glob("*.py")) + [synthetic_files.dataset, synthetic_files.replay]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rng = np.random.get_state()
    with patch("numpy.random.seed", side_effect=AssertionError("No simulation allowed")):
        data = ReplayData(synthetic_files.dataset, synthetic_files.replay)
        viewer = ReplayViewer(data)
        try:
            # Confirm reuse of the original drawing helper source.
            assert Path(viewer.helpers.plotRobot.__code__.co_filename) == source / "helpers.py"
            assert Path(viewer.helpers.plotCov2D.__code__.co_filename) == source / "helpers.py"
            KeyEvent("key_press_event", viewer.figure.canvas, key="right")._process()
            assert viewer.index == 1
            assert "INCORRECT" in "\n".join(t.get_text() for t in viewer.panel.texts)
            viewer.slider.set_val(2)
            assert viewer.index == 2
            assert "DUPLICATE INITIALIZATION" in "\n".join(t.get_text() for t in viewer.panel.texts)
            viewer.figure.savefig(tmp_path / "frame.png")
            assert (tmp_path / "frame.png").stat().st_size > 1000
            KeyEvent("key_press_event", viewer.figure.canvas, key="home")._process()
            assert viewer.index == 0
            # A real Matplotlib button mouse event activates the playback timer.
            x, y = viewer.buttons["Play"].ax.transAxes.transform((.5, .5))
            MouseEvent("button_press_event", viewer.figure.canvas, x, y, button=1)._process()
            MouseEvent("button_release_event", viewer.figure.canvas, x, y, button=1)._process()
            assert viewer.playing
            viewer.tick()
            assert viewer.index == 1
            viewer.tick()
            assert viewer.index == 2 and not viewer.playing
            KeyEvent("key_press_event", viewer.figure.canvas, key=" ")._process()
            assert viewer.playing and viewer.index == 0  # Restart from the end.
            KeyEvent("key_press_event", viewer.figure.canvas, key=" ")._process()
            assert not viewer.playing
            viewer.next_issue()
            assert viewer.index == 1
            viewer.step(-50)
            assert viewer.index == 0
            viewer.step(50)
            assert viewer.index == 2
        finally:
            viewer.pause()
            plt.close(viewer.figure)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for original, current in zip(rng, np.random.get_state()):
        np.testing.assert_array_equal(original, current)
