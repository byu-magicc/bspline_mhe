"""Audit landmark creation by replaying saved data through unchanged Estimator A.

Read-only Python line tracing captures the actual nearest-neighbor candidate
calculations. No alternative association implementation decides matches.
Diagnostic replay timings include instrumentation and are never benchmarks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from benchmarks.estimator_a import Dataset, ROOT, environment, legacy_modules, replay, source_hashes


def finite_number(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def best_candidate(candidates: list[dict]) -> dict | None:
    finite = [c for c in candidates if c["squared_mahalanobis_distance"] is not None]
    return min(finite, key=lambda c: c["squared_mahalanobis_distance"]) if finite else None


class AssociationTrace:
    """Observe original da_nn.associateData locals at its decision line.

    The caller's trace hook is restored even if the original function raises.
    Tracing is confined to the original function's code object in this thread.
    Source markers are checked explicitly; an incompatible revision fails rather
    than silently inventing candidate values. Use standalone, outside a debugger
    or coverage session: temporary tracing may suppress their association events.
    """

    def __init__(self, original):
        self.original = original
        source, first_line = inspect.getsourcelines(original)

        def locate(marker):
            matches = [first_line + i for i, line in enumerate(source) if line.strip().startswith(marker)]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot instrument this NN revision: expected one {marker!r} line.")
            return matches[0]

        self.observation_line = locate("dist, bearing, _ = z")
        self.distance_line = locate("if mahalonobisDistance < startingThresh")
        self.steps = []

    def __call__(self, measurements, R, mu, Sigma, landmarkSignatures):
        rows = []
        birth_ids = np.asarray(landmarkSignatures, dtype=int).copy()
        pose = mu[:3].copy().tolist()

        def trace(frame, event, arg):
            if frame.f_code is not self.original.__code__:
                return None
            if event == "line":
                if frame.f_lineno == self.observation_line:
                    rows.append({"observation_index": len(rows),
                                 "measurement": frame.f_locals["z"].copy().tolist(),
                                 "candidates": []})
                elif frame.f_lineno == self.distance_line:
                    local = frame.f_locals
                    track = int(local["j"])
                    distance = finite_number(local["mahalonobisDistance"])
                    threshold = float(frame.f_globals["startingThresh"])
                    rows[-1]["candidates"].append({
                        "track_id": track, "birth_truth_id": int(birth_ids[track]),
                        "squared_mahalanobis_distance": distance,
                        "distance_is_finite": distance is not None,
                        "passes_gate": distance is not None and distance < threshold,
                        "threshold": threshold,
                        "predicted_measurement": local["zHat"].copy().tolist(),
                        "innovation_range_bearing": local["innovationTemp"].copy().tolist(),
                        "innovation_covariance": local["P"].copy().tolist(),
                    })
            return trace

        previous_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            result = self.original(measurements, R, mu, Sigma, landmarkSignatures)
        finally:
            sys.settrace(previous_trace)
        assigned = result[0]
        if len(rows) != len(measurements) or len(assigned) != len(rows):
            raise RuntimeError("NN tracing did not capture every observation.")
        for row, assignment in zip(rows, assigned):
            candidates = row["candidates"]
            if len(candidates) != len(birth_ids):
                raise RuntimeError("NN tracing did not capture every candidate.")
            truth_id = int(row["measurement"][2])
            threshold = float(self.original.__globals__["startingThresh"])
            row.update({"truth_id": truth_id, "assignment": int(assignment),
                        "threshold": threshold, "gate_rule": "squared_mahalanobis_distance < threshold",
                        "best_candidate": best_candidate(candidates),
                        "best_same_truth_candidate": best_candidate([
                            c for c in candidates if c["birth_truth_id"] == truth_id]),
                        "candidate_count": len(candidates)})
            eligible = [c for c in candidates if c["passes_gate"]]
            predicted_choice = best_candidate(eligible)
            expected_assignment = predicted_choice["track_id"] if predicted_choice else -1
            if int(assignment) != expected_assignment:
                raise RuntimeError("Captured candidates disagree with the actual NN decision.")
            if assignment >= 0:
                row["assigned_birth_truth_id"] = int(birth_ids[assignment])
                row["reason"] = "minimum_distance_candidate_inside_strict_gate"
                # Full candidate tables are retained for all initializations;
                # ordinary matches retain their best-candidate context only.
                del row["candidates"]
            elif not candidates:
                row["reason"] = "no_existing_candidates"
            elif not all(c["distance_is_finite"] for c in candidates):
                row["reason"] = "no_candidate_passed_gate_nonfinite_distances_present"
            else:
                row["reason"] = "all_candidate_squared_distances_at_or_above_gate"
        self.steps.append({"predicted_robot_pose": pose, "birth_ids_at_association": birth_ids.tolist(),
                           "observations": rows})
        return result  # Return the original arrays unchanged.


def collect_initializations(dataset: Dataset, result_arrays: dict,
                            association_steps: list[dict] | None = None) -> list[dict]:
    """Track immutable initialization identities in chronological measurement order.

    Existing same-truth IDs include earlier creations in the same time step.
    Those are labeled separately from candidates available during association,
    because legacy association runs once before the whole batch is augmented.
    """
    assignments = result_arrays["association_indices"]
    ids = result_arrays["observation_truth_ids"]
    poses = result_arrays["estimated_pose"]
    measurements = result_arrays["used_observations"]
    birth_ids = result_arrays["landmark_birth_truth_ids"]
    dimensions = result_arrays["state_dimensions"]
    by_truth = {}
    events = []
    next_track = 0
    if len(assignments) != len(dataset.raw) or (association_steps is not None and len(association_steps) != len(assignments)):
        raise ValueError("Dataset/replay/trace step counts disagree.")
    for step, row in enumerate(assignments):
        count_before_step = next_track
        for observation_index, assignment in enumerate(row):
            if assignment != -1:
                continue
            truth_id = int(ids[step, observation_index])
            if (next_track >= len(birth_ids) or birth_ids[next_track] != truth_id
                    or not 0 <= truth_id < len(dataset.landmarks)):
                raise ValueError("Initialization order disagrees with recorded birth truth IDs.")
            previous_ids = by_truth.setdefault(truth_id, [])
            existing = previous_ids.copy()
            prior_candidates = [track for track in existing if track < count_before_step]
            z = measurements[step, observation_index]
            if not np.isfinite(z).all() or z[2] != truth_id:
                raise ValueError("Initialization observation identity mismatch.")
            raw_slots = dataset.raw[step, 9:].reshape(-1, 5)
            selected = raw_slots[raw_slots[:, 0] != -1]
            if observation_index >= len(selected) or selected[observation_index, 0] != truth_id:
                raise ValueError("Saved dataset and initialization observation disagree.")
            noise = z[:2] - selected[observation_index, 1:3]
            noise[1] = (noise[1] + np.pi) % (2 * np.pi) - np.pi
            standardized_noise = noise / np.array(dataset.config["betas"])
            decision = None
            prior_pose = None
            if association_steps is not None:
                frame = association_steps[step]
                decision = frame["observations"][observation_index]
                if decision["assignment"] != -1 or decision["truth_id"] != truth_id:
                    raise ValueError("Association trace and replay initialization disagree.")
                prior_pose = frame["predicted_robot_pose"]
            events.append({
                "step_index": step, "step_number": step + 1,
                "time_seconds": float((step + 1) * dataset.config["delta_t"]),
                "observation_index": observation_index, "estimated_landmark_id": next_track,
                "truth_landmark_id": truth_id, "is_duplicate": bool(existing),
                "existing_same_truth_track_ids": existing,
                "same_truth_track_ids_available_at_association": prior_candidates,
                "same_step_prior_track_ids": [track for track in existing if track >= count_before_step],
                "robot_pose_at_initialization": poses[step].tolist(),
                "predicted_robot_pose_at_association": prior_pose,
                "measurement": {"range": float(z[0]), "bearing_rad": float(z[1]), "truth_id": truth_id},
                "measurement_noise_against_clean_truth": noise.tolist(),
                "measurement_noise_in_nominal_sigma": standardized_noise.tolist(),
                "measurement_noise_squared_standardized_norm": float(standardized_noise @ standardized_noise),
                "association_diagnostic": decision,
                "decision_evidence": "live_original_nn_locals" if decision is not None else "unavailable",
            })
            previous_ids.append(next_track)
            next_track += 1
        if int(dimensions[step]) != 3 + 2 * next_track:
            raise ValueError("Initialization count disagrees with joint-state dimensions.")
    if next_track != len(birth_ids):
        raise ValueError("Final initialization count disagrees with map size.")
    return events


def summary_counts(events: list[dict], truth_count: int) -> dict:
    duplicates = [e for e in events if e["is_duplicate"]]
    counts = Counter(e["truth_landmark_id"] for e in duplicates)
    return {
        "total_estimated_tracks": len(events),
        "unique_truth_landmarks_represented": len({e["truth_landmark_id"] for e in events}),
        "duplicate_initializations": len(duplicates),
        "duplicate_counts_by_truth_id": {str(truth_id): counts[truth_id] for truth_id in range(truth_count)},
        "duplicate_step_indices": sorted({e["step_index"] for e in duplicates}),
        "duplicate_step_numbers": sorted({e["step_number"] for e in duplicates}),
    }


def compare_saved(result, path: Path) -> bool:
    """Compare all estimator numerical arrays; recorded runtime is not numerical state."""
    with np.load(path, allow_pickle=False) as saved:
        return all(key in saved and np.array_equal(value, saved[key], equal_nan=True)
                   for key, value in result.arrays.items() if key != "update_seconds")


def audit_replay(dataset: Dataset, mode: str = "noisy"):
    """External wrapper around the existing replay harness; no harness edits."""
    legacy = legacy_modules()
    observer = AssociationTrace(legacy.da_nn.associateData)
    with patch.object(legacy.da_nn, "associateData", observer):
        traced = replay(dataset, mode, "nn")
    return traced, observer.steps


def run_diagnostics(dataset_path: Path, replay_path: Path, output: Path) -> dict:
    paths = sorted((ROOT / "baseline" / "ekf_slam" / "code").glob("*.py")) + [dataset_path, replay_path]

    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    before = hashes()
    dataset = Dataset.load(dataset_path)
    if dataset.config["seed"] != 7 or len(dataset.raw) != 500:
        raise ValueError("This validation targets the saved seed-7, 500-step Milestone 1 experiment.")
    # Import before taking the source snapshot; all imports remain outside
    # numerical-equality checks. Do not regenerate simulation or edit inputs.
    legacy = legacy_modules()
    current_sources = source_hashes()
    original_function = legacy.da_nn.associateData
    threshold_before = float(legacy.da_nn.startingThresh)
    baseline = replay(dataset, "noisy", "nn")
    traced, steps = audit_replay(dataset)
    unchanged = all(np.array_equal(value, traced.arrays[key], equal_nan=True)
                    for key, value in baseline.arrays.items() if key != "update_seconds")
    events = collect_initializations(dataset, traced.arrays, steps)
    summary = summary_counts(events, len(dataset.landmarks))
    duplicates = [event for event in events if event["is_duplicate"]]
    finite_best = [event["association_diagnostic"]["best_candidate"]["squared_mahalanobis_distance"]
                   for event in duplicates if event["association_diagnostic"]["best_candidate"] is not None]
    incorrect_matches = [
        {"step_index": step, "observation_index": row["observation_index"],
         "truth_id": row["truth_id"], "assigned_track_id": row["assignment"],
         "assigned_birth_truth_id": row["assigned_birth_truth_id"]}
        for step, frame in enumerate(steps) for row in frame["observations"]
        if row["assignment"] >= 0 and row["assigned_birth_truth_id"] != row["truth_id"]]
    first_incorrect_step = incorrect_matches[0]["step_index"] if incorrect_matches else None
    summary.update({
        "duplicate_reason_counts": dict(Counter(e["association_diagnostic"]["reason"] for e in duplicates)),
        "duplicate_best_candidate_distance_min": min(finite_best) if finite_best else None,
        "duplicate_best_candidate_distance_max": max(finite_best) if finite_best else None,
        "duplicates_with_noise_only_squared_norm_ge_gate": sum(
            e["measurement_noise_squared_standardized_norm"] >= threshold_before for e in duplicates),
        "duplicates_with_noise_only_squared_norm_lt_gate": sum(
            e["measurement_noise_squared_standardized_norm"] < threshold_before for e in duplicates),
        "duplicates_without_same_truth_candidate_at_association": sum(
            not e["same_truth_track_ids_available_at_association"] for e in duplicates),
        "incorrect_existing_matches": len(incorrect_matches),
        "first_incorrect_match": incorrect_matches[0] if incorrect_matches else None,
        "duplicate_step_indices_before_first_incorrect_match": sorted({e["step_index"] for e in duplicates
            if first_incorrect_step is None or e["step_index"] < first_incorrect_step}),
        "duplicates_whose_nearest_candidate_has_same_birth_truth_id": sum(
            e["association_diagnostic"]["best_candidate"] is not None and
            e["association_diagnostic"]["best_candidate"]["birth_truth_id"] == e["truth_landmark_id"]
            for e in duplicates),
        "steps_with_multiple_observations_matched_to_same_track": [index for index, frame in enumerate(steps)
            if len([r["assignment"] for r in frame["observations"] if r["assignment"] >= 0]) >
            len({r["assignment"] for r in frame["observations"] if r["assignment"] >= 0})],
    })
    checks = {
        "reproduced_48_tracks": summary["total_estimated_tracks"] == 48,
        "reproduced_8_unique_truth_ids": summary["unique_truth_landmarks_represented"] == 8,
        "reproduced_40_duplicates": summary["duplicate_initializations"] == 40,
        "diagnostics_on_off_numerical_arrays_exact": unchanged,
        "plain_replay_matches_saved_numerical_arrays": compare_saved(baseline, replay_path),
        "diagnostic_replay_matches_saved_numerical_arrays": compare_saved(traced, replay_path),
        "legacy_source_and_inputs_unchanged": before == hashes(),
        "association_callable_and_threshold_restored": legacy.da_nn.associateData is original_function and
            float(legacy.da_nn.startingThresh) == threshold_before,
        "source_files_unchanged": source_hashes() == current_sources,
    }
    # These summaries retain all decisions; complete candidate lists are needed
    # only for new landmarks and are stored in the initialization event records.
    decisions = [{"step_index": step, "observation_index": row["observation_index"],
                  "truth_id": row["truth_id"], "assignment": row["assignment"],
                  "assigned_birth_truth_id": row.get("assigned_birth_truth_id"),
                  "reason": row["reason"]}
                 for step, frame in enumerate(steps) for row in frame["observations"]]
    document = {
        "schema": 1, "experiment": {"estimator": "A", "mode": "noisy", "association": "nn",
            "seed": 7, "steps": 500, "threshold_squared_mahalanobis": threshold_before,
            "acceptance_rule": "d_squared < threshold (strict)", "indexing": "step_index/track_id/truth_id are zero-based; step_number is one-based"},
        "validation_passed": all(checks.values()), "validation_checks": checks,
        "summary": summary, "initializations": events, "association_decisions": decisions,
        "environment": environment(), "source_sha256": current_sources, "input_sha256": before,
        "evidence": "Candidate distances, innovations and S were copied from live original da_nn locals before the gate condition; posterior replay alone does not store this evidence.",
        "timing_policy": "Diagnostic replay update timings include tracing overhead and are not saved or reported as estimator performance.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "landmark_initializations.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    write_report(document, output / "landmark_initializations.md")
    return document


def write_report(document: dict, path: Path) -> None:
    summary = document["summary"]
    events = document["initializations"]
    gate = document["experiment"]["threshold_squared_mahalanobis"]
    lines = ["# Estimator A landmark initialization diagnostic", "",
             f"Validation: **{'PASS' if document['validation_passed'] else 'FAIL'}**. Saved seed 7, 500 steps, corrected noisy observations, nearest-neighbor association; no random data regenerated.", "",
             f"**{summary['total_estimated_tracks']} estimated tracks; {summary['unique_truth_landmarks_represented']} unique true landmarks; {summary['duplicate_initializations']} duplicate initializations.** A duplicate means a new track whose observation truth ID already has at least one earlier initialized track. Birth IDs are immutable evaluation labels, not proof that subsequent matches keep tracks unmixed.", "",
             "| True landmark | Duplicate initializations |", "|---|---:|"]
    lines.extend(f"| L{truth_id} | {count} |" for truth_id, count in summary["duplicate_counts_by_truth_id"].items())
    lines += ["", f"Duplicate step indices (zero-based): `{summary['duplicate_step_indices']}`.", "",
              "## Confirmed observations", "",
              f"- The original NN accepts only squared Mahalanobis distance **strictly below {gate:g}** (not Euclidean distance, and not a squared threshold of {gate**2:g}). If no candidate passes, it returns -1 and the original augmentation immediately creates a permanent new track. No separate rejection or confirmation stage is used by this NN policy.",
              f"- Duplicate decision reasons: `{json.dumps(summary['duplicate_reason_counts'], sort_keys=True)}`. The smallest candidate distance at duplicate events ranges from **{summary['duplicate_best_candidate_distance_min']:.6f} to {summary['duplicate_best_candidate_distance_max']:.6f}**.",
              f"- **{summary['duplicates_without_same_truth_candidate_at_association']}** duplicates lacked a same-birth-truth candidate at association time. This separates batch-order effects from rejection of already existing same-truth tracks.",
              f"- In **{summary['duplicates_whose_nearest_candidate_has_same_birth_truth_id']}** duplicate events, the nearest candidate itself had the correct birth truth ID but failed the gate.",
              f"- At duplicate events, **{summary['duplicates_with_noise_only_squared_norm_ge_gate']}** triggering observations have noise-only squared standardized norm >= {gate:g}, and **{summary['duplicates_with_noise_only_squared_norm_lt_gate']}** have norm < {gate:g}. This statistic uses stored noisy-minus-clean measurements divided by the configured measurement standard deviations. It is **not** the EKF association distance, which also includes state/map errors and innovation covariance.",
              f"- **{summary['incorrect_existing_matches']}** existing-track matches disagree with birth truth IDs; **{len(summary['steps_with_multiple_observations_matched_to_same_track'])}** steps match multiple observations to the same track. NN considers each observation independently and does not enforce one-to-one matching. Full decision indices are in JSON.", "",
              f"The first incorrect existing-track match is `{summary['first_incorrect_match']}`. Duplicate births at steps `{summary['duplicate_step_indices_before_first_incorrect_match']}` precede it, so wrong associations are not required to explain the onset of duplication.", "",
              "## Interpretation: evidence versus hypotheses", "",
              "The immediate confirmed mechanism is failure of the innovation gate followed by unconditional track creation. Repeated observations of an already mapped true landmark can therefore produce new tracks. The noisy measurement realizations, current predicted robot/map state, and their covariances jointly determine each gate failure; exact candidate residuals and innovation covariances are provided in JSON.", "",
              "The most likely explanation is noisy measurements and noisy existing map/pose predictions crossing this fixed innovation gate, followed by immediate new-track creation. A noise-only norm below the gate does not imply the full innovation should pass: the prediction may already be biased. Incorrect matches can additionally contaminate later estimates, but this audit does not establish which earlier event caused each later failure. Isolating covariance calibration, linearization effects, noise outliers, map bias, or feedback from wrong matches requires controlled counterfactual experiments; none were performed and no threshold was tuned. For an ideal correctly normalized two-dimensional Gaussian innovation, this gate alone rejects exp(-4/2) = 13.53% of samples; the actual nonlinear correlated SLAM innovations need not follow that ideal model.", "",
              "## Chronological initialization log", "",
              "IDs and step indices below are zero-based. Time is after execution of command k, at (k+1)*0.1 s. Existing track IDs list all prior births of that truth ID, including earlier observations in this same augmentation batch. The JSON separately lists candidates available before batch correction/augmentation. Pose and triggering measurement are expanded below the table.", "",
              "| Step | Time (s) | New track | True ID | Duplicate? | Existing same-truth tracks | Best candidate / squared distance | Reason |",
              "|---:|---:|---|---|---|---|---|---|"]
    for event in events:
        decision = event["association_diagnostic"]
        best = decision["best_candidate"]
        best_text = f"T{best['track_id']}/L{best['birth_truth_id']} / {best['squared_mahalanobis_distance']:.6f}" if best else "none"
        lines.append(f"| {event['step_index']} | {event['time_seconds']:.1f} | T{event['estimated_landmark_id']} | L{event['truth_landmark_id']} | {'**YES**' if event['is_duplicate'] else 'no'} | {event['existing_same_truth_track_ids']} | {best_text} | {decision['reason']} |")
    duplicates = [event for event in events if event["is_duplicate"]]
    if duplicates:
        first = duplicates[0]
        same = first["association_diagnostic"]["best_same_truth_candidate"]
        if same is not None:
            birth = events[same["track_id"]]
            lines += ["", "## First duplicate example", "",
                      f"At step {first['step_index']} (t={first['time_seconds']:.1f} s), T{first['estimated_landmark_id']} duplicates L{first['truth_landmark_id']}. Existing T{same['track_id']} has d_squared={same['squared_mahalanobis_distance']:.6f} and innovation `[range={same['innovation_range_bearing'][0]:.6f}, bearing={same['innovation_range_bearing'][1]:.6f}]`; the recorded bearing innovation standard deviation is {np.sqrt(same['innovation_covariance'][1][1]):.6f} rad. T{same['track_id']}'s initial observation had bearing noise {birth['measurement_noise_against_clean_truth'][1]:+.6f} rad, while the duplicate-triggering observation has bearing noise {first['measurement_noise_against_clean_truth'][1]:+.6f} rad and noise-only squared norm {first['measurement_noise_squared_standardized_norm']:.6f}. These are measured facts showing prediction/measurement disagreement; they do not by themselves isolate one cause.", ""]
    lines += ["", "## Initialization details", "",
              "Pose is [x, y, theta] after the step's existing-landmark correction, which is the pose actually used for augmentation. The separate association pose in JSON is the predicted pose before correction. Range units are simulator coordinates; bearing and heading are radians. Full candidate tables include all gated-out landmarks, the best candidate sharing the true ID, innovations and S.", ""]
    for event in events:
        z = event["measurement"]
        lines.append(f"- Step {event['step_index']}, t={event['time_seconds']:.1f}, T{event['estimated_landmark_id']}/L{event['truth_landmark_id']} {'**DUPLICATE**' if event['is_duplicate'] else 'first initialization'}: pose `{event['robot_pose_at_initialization']}`; measurement `[range={z['range']:.9f}, bearing={z['bearing_rad']:.9f}]`.")
    lines += ["", "## Evidence and validation", "",
              "The original saved NPZ replay contains posterior states and final decisions, not pre-correction candidate innovation statistics. This tool uses the existing replay harness, wrapping only the NN callable and observing its actual local variables with a temporary Python trace. It does not recalculate candidate distances using posterior poses, reimplement association, or modify the original function's code, thresholds or return arrays. The previous trace hook and original callable are restored. Run outside a debugger/coverage session to avoid interfering with their trace events.", "",
              "Without an instrumented replay, candidate statistics must be marked unavailable. Long-term audit logs would need the pre-association state/covariance, consumed measurement, candidate H/r/S/distance and gate, or the live candidate values captured here. The initialization collector supports unavailable decision evidence explicitly.", "",
              f"Validation checks: `{json.dumps(document['validation_checks'], sort_keys=True)}`.", "",
              "All numerical state, covariance, landmark, observation and association arrays are compared exactly against diagnostics-disabled replay and the original saved output (NaN padding equal). Timing arrays are excluded: tracing adds overhead, and no diagnostic timing replaces the baseline benchmark timings. Original source and input archives are hash-checked for preservation. JSON contains source hashes, dependency versions, the full event log and all 1,000 association decisions.", "",
              "Reproduce from the repository root:", "", "```sh",
              "python3 -B -m benchmarks.diagnose_initializations",
              "python3 -B -m pytest -q tests/test_diagnose_initializations.py", "```", "",
              "The existing animation already highlights duplicate births in magenta and supports Next issue / N. Use `python3 -B -m benchmarks.visualize_estimator_a --mode noisy --association nn --backend QtAgg --frame 2` to inspect the first duplicate frame. No visualization or estimator changes are made by this diagnostic.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT / "artifacts" / "milestone1"
    parser.add_argument("--dataset", type=Path, default=default / "dataset.npz")
    parser.add_argument("--replay", type=Path, default=default / "noisy_nn.npz")
    parser.add_argument("--output", type=Path, default=default)
    args = parser.parse_args()
    document = run_diagnostics(args.dataset, args.replay, args.output)
    print(json.dumps({"validation_passed": document["validation_passed"], "summary": document["summary"]}, indent=2))
    if not document["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
