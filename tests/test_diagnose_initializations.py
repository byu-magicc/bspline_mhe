"""Read-only duplicate tracking and live original-NN decision evidence."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.diagnose_initializations import (
    AssociationTrace, collect_initializations, run_diagnostics, summary_counts,
)
from benchmarks.estimator_a import Dataset, ROOT, legacy_modules


@pytest.mark.parametrize("range_measurement,expected_distance,assignment", [
    (11., 1., 0), (12., 4., -1), (13., 9., -1)])
def test_exact_original_candidate_distance_and_strict_gate(range_measurement, expected_distance, assignment):
    legacy = legacy_modules()
    original = legacy.da_nn.associateData
    observer = AssociationTrace(original)
    z = np.array([[range_measurement, 0., 0.]])
    mu = np.array([0., 0., 0., 10., 0.])
    covariance = np.zeros((5, 5))
    R = np.diag([1., .01])
    signatures = np.array([0])
    before = [value.copy() for value in (z, mu, covariance, R, signatures)]
    expected = original(z, R, mu, covariance, signatures)
    prior_trace = sys.gettrace()
    actual = observer(z, R, mu, covariance, signatures)
    assert sys.gettrace() is prior_trace
    assert legacy.da_nn.associateData is original
    assert legacy.da_nn.startingThresh == 4
    for value, unmodified in zip((z, mu, covariance, R, signatures), before):
        np.testing.assert_array_equal(value, unmodified)
    for value, untraced in zip(actual, expected):
        np.testing.assert_array_equal(value, untraced)
    row = observer.steps[0]["observations"][0]
    assert row["assignment"] == assignment
    assert row["best_candidate"]["squared_mahalanobis_distance"] == expected_distance
    assert row["best_candidate"]["passes_gate"] is (expected_distance < 4)
    if assignment == -1:
        assert row["reason"] == "all_candidate_squared_distances_at_or_above_gate"


def test_tied_candidates_keep_original_first_track_rule():
    original = legacy_modules().da_nn.associateData
    observer = AssociationTrace(original)
    mu = np.array([0., 0., 0., 10., 0., 12., 0.])
    assigned, _, _ = observer(np.array([[11., 0., 0.]]), np.diag([1., .01]),
                              mu, np.zeros((7, 7)), np.array([0, 1]))
    assert assigned.tolist() == [0]
    assert observer.steps[0]["observations"][0]["best_candidate"]["track_id"] == 0


def test_no_candidates_has_explicit_reason():
    observer = AssociationTrace(legacy_modules().da_nn.associateData)
    assigned, _, _ = observer(np.array([[11., 0., 0.]]), np.eye(2),
                              np.zeros(3), np.zeros((3, 3)), [])
    assert assigned.tolist() == [-1]
    decision = observer.steps[0]["observations"][0]
    assert decision["reason"] == "no_existing_candidates"
    assert decision["best_candidate"] is None
    assert decision["candidates"] == []


def test_trace_restored_after_original_association_raises():
    observer = AssociationTrace(legacy_modules().da_nn.associateData)
    previous = sys.gettrace()
    with np.errstate(divide="raise", invalid="raise"), pytest.raises(FloatingPointError):
        observer(np.array([[1., 0., 0.]]), np.eye(2), np.zeros(5), np.eye(5), np.array([0]))
    assert sys.gettrace() is previous
    assert observer.steps == []


def identity_fixture():
    # Repeated same-truth observations in one augmentation batch, then a later
    # duplicate. Earlier same-step births are not pre-association candidates.
    ids = np.array([[0, 0, 1], [0, -1, -1]])
    raw = np.zeros((2, 24))
    raw[:, 9:].reshape(2, 3, 5)[:, :, 0] = ids
    raw[:, 9:].reshape(2, 3, 5)[:, :, 1:5] = [10., 0., 10., 0.]
    measurements = np.full((2, 3, 3), np.nan)
    for step, obs in zip(*np.where(ids >= 0)):
        measurements[step, obs] = [10., 0., ids[step, obs]]
    dataset = Dataset(raw, np.array([[10., 0.], [20., 0.]]),
                      {"config": {"delta_t": .1, "betas": [10., .1]}})
    arrays = {"association_indices": np.array([[-1, -1, -1], [-1, -3, -3]]),
              "observation_truth_ids": ids, "estimated_pose": np.zeros((2, 3)),
              "used_observations": measurements, "landmark_birth_truth_ids": np.array([0, 0, 1, 0]),
              "state_dimensions": np.array([9, 11])}
    return dataset, arrays


def test_chronological_identity_tracking_and_same_step_duplicates():
    dataset, arrays = identity_fixture()
    events = collect_initializations(dataset, arrays)
    assert [e["estimated_landmark_id"] for e in events] == [0, 1, 2, 3]
    assert [e["is_duplicate"] for e in events] == [False, True, False, True]
    assert events[1]["existing_same_truth_track_ids"] == [0]
    assert events[1]["same_step_prior_track_ids"] == [0]
    assert events[1]["same_truth_track_ids_available_at_association"] == []
    assert events[3]["existing_same_truth_track_ids"] == [0, 1]
    assert events[3]["same_truth_track_ids_available_at_association"] == [0, 1]
    assert events[3]["same_step_prior_track_ids"] == []
    assert events[3]["step_index"] == 1 and events[3]["step_number"] == 2
    assert events[3]["time_seconds"] == .2
    assert all(e["association_diagnostic"] is None and e["decision_evidence"] == "unavailable" for e in events)
    summary = summary_counts(events, truth_count=3)
    assert summary["total_estimated_tracks"] == 4
    assert summary["unique_truth_landmarks_represented"] == 2
    assert summary["duplicate_initializations"] == 2
    assert summary["duplicate_counts_by_truth_id"] == {"0": 2, "1": 0, "2": 0}
    assert summary["duplicate_step_indices"] == [0, 1]


@pytest.mark.parametrize("corruption", ["birth", "dimension", "measurement"])
def test_identity_consistency_failures(corruption):
    dataset, arrays = identity_fixture()
    if corruption == "birth":
        arrays["landmark_birth_truth_ids"][0] = 1
    elif corruption == "dimension":
        arrays["state_dimensions"][0] = 7
    else:
        arrays["used_observations"][0, 0, 2] = 1
    with pytest.raises(ValueError):
        collect_initializations(dataset, arrays)


@pytest.fixture(scope="module")
def diagnostic(tmp_path_factory):
    folder = ROOT / "artifacts" / "milestone1"
    output = tmp_path_factory.mktemp("initialization-audit")
    legacy = legacy_modules()
    with patch.object(legacy.generator, "generate", side_effect=AssertionError("Do not regenerate data")):
        document = run_diagnostics(folder / "dataset.npz", folder / "noisy_nn.npz", output)
    return document, output


def test_saved_seed7_500_reproduction_and_numerical_preservation(diagnostic):
    document, output = diagnostic
    assert document["validation_passed"]
    assert all(document["validation_checks"].values())
    summary = document["summary"]
    assert summary["total_estimated_tracks"] == 48
    assert summary["unique_truth_landmarks_represented"] == 8
    assert summary["duplicate_initializations"] == 40
    assert summary["duplicate_counts_by_truth_id"] == {"0": 6, "1": 3, "2": 5, "3": 5, "4": 6, "5": 4, "6": 5, "7": 6}
    assert summary["first_incorrect_match"]["step_index"] == 22
    assert summary["duplicate_step_indices_before_first_incorrect_match"] == [2, 3]
    assert summary["duplicates_whose_nearest_candidate_has_same_birth_truth_id"] == 37
    assert len(document["association_decisions"]) == 1000
    saved = json.loads((output / "landmark_initializations.json").read_text())
    assert saved == document
    report = (output / "landmark_initializations.md").read_text()
    assert "48 estimated tracks" in report and "40 duplicate initializations" in report
    assert "hypotheses" in report


def test_live_candidates_support_rejection_conclusions(diagnostic):
    document, _ = diagnostic
    for event in document["initializations"]:
        decision = event["association_diagnostic"]
        assert event["decision_evidence"] == "live_original_nn_locals"
        candidates = decision["candidates"]
        assert [c["track_id"] for c in candidates] == list(range(decision["candidate_count"]))
        same_truth = [c["track_id"] for c in candidates if c["birth_truth_id"] == event["truth_landmark_id"]]
        assert same_truth == event["same_truth_track_ids_available_at_association"]
        for candidate in candidates:
            residual = np.array(candidate["innovation_range_bearing"])
            covariance = np.array(candidate["innovation_covariance"])
            distance = residual @ np.linalg.solve(covariance, residual)
            assert candidate["squared_mahalanobis_distance"] == pytest.approx(distance, rel=1e-12, abs=1e-10)
        if event["is_duplicate"]:
            assert same_truth
            assert all(not c["passes_gate"] and c["squared_mahalanobis_distance"] >= 4 for c in candidates)
            assert decision["reason"] == "all_candidate_squared_distances_at_or_above_gate"
    assert document["summary"]["duplicates_with_noise_only_squared_norm_ge_gate"] == 25
    assert document["summary"]["duplicates_with_noise_only_squared_norm_lt_gate"] == 15
