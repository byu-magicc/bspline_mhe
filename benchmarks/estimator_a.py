"""External reproduction/replay harness for the unchanged discrete EKF-SLAM.

Run from the repository root:
    python -m benchmarks.estimator_a --seed 7 --steps 500 --repeats 3

The legacy generator's first observation pair is CLEAN despite its docstring.
``original`` selects that pair; ``noisy`` selects the second, generated noisy pair.
Only I/O and the GUI backend are intercepted. No estimator equations are replaced.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "baseline" / "ekf_slam" / "code"
MODES = ("original", "noisy")
ASSOCIATIONS = ("known", "nn")


def source_hashes() -> dict[str, str]:
    paths = sorted(LEGACY.glob("*.py")) + sorted((ROOT / "benchmarks").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("test_estimator_a.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def environment() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": {p: importlib.metadata.version(p)
                         for p in ("numpy", "scipy", "matplotlib", "pytest")},
        "float_dtype": "float64 (legacy initial pose intentionally retains integer dtype)",
        "thread_environment": {k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


@lru_cache(maxsize=1)
def legacy_modules() -> SimpleNamespace:
    """Import original source, replacing only its hard-coded Qt backend choice.

    Imports and rcParams changes occur before any estimator timing. The legacy
    modules use unqualified sibling imports, so validate their resolved locations.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    old_path = sys.path[:]
    old_bytecode = sys.dont_write_bytecode
    sys.path.insert(0, str(LEGACY))
    sys.dont_write_bytecode = True
    try:
        with matplotlib.rc_context(), patch.object(matplotlib, "use"):
            modules = {name: importlib.import_module(name) for name in
                       ("fieldSettings", "helpers", "generator", "simSLAM", "da_known", "da_nn")}
        for module in modules.values():
            if Path(module.__file__).resolve().parent != LEGACY:
                raise RuntimeError(f"Legacy module name collision: {module.__file__}")
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
    return SimpleNamespace(**modules)


def default_config(seed: int, steps: int) -> dict:
    if steps < 2:
        raise ValueError("At least two steps are required by legacy generateMotion.")
    if not 0 <= seed < 2**32:
        raise ValueError("NumPy legacy seed must be in [0, 2**32).")
    return {"seed": seed, "steps": steps, "initial_pose": [180, 50, 0],
            "initial_covariance": (1e-3 * np.eye(3)).tolist(),
            "alphas": (np.array([0.05, 0.001, 0.05, 0.01])**2).tolist(),
            "betas": [10, float(10 * np.pi / 180)], "delta_t": 0.1,
            "max_observations": 2, "selection": "fov", "fov": float(np.pi)}


@dataclass
class Dataset:
    raw: np.ndarray
    landmarks: np.ndarray
    metadata: dict

    @property
    def config(self) -> dict:
        return self.metadata["config"]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        slots = self.raw[:, 9:].reshape(len(self.raw), -1, 5)
        # Explicit semantic arrays accompany the untouched legacy layout.
        np.savez_compressed(
            path, raw=self.raw, commands=self.raw[:, :3],
            true_executed_pose=self.raw[:, 3:6], nominal_pose=self.raw[:, 6:9],
            time=(np.arange(len(self.raw)) + 1) * self.config["delta_t"],
            landmark_positions=self.landmarks, observation_ids=slots[:, :, 0],
            clean_observations=slots[:, :, 1:3], noisy_observations=slots[:, :, 3:5],
            metadata_json=np.array(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: Path) -> Dataset:
        with np.load(path, allow_pickle=False) as saved:
            result = cls(saved["raw"].copy(), saved["landmark_positions"].copy(),
                         json.loads(str(saved["metadata_json"])))
        expected = (result.config["steps"], 9 + 5 * result.config["max_observations"])
        if result.raw.shape != expected or not np.isfinite(result.raw).all():
            raise ValueError("Dataset shape or finite-value validation failed.")
        return result


def generate_dataset(seed: int = 7, steps: int = 500) -> Dataset:
    legacy = legacy_modules()
    config = default_config(seed, steps)
    before = source_hashes()
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        # The original writes a partial .npz every step. Suppress those writes;
        # preserve every RNG draw (including zero-variance draws) and all math.
        with patch.object(legacy.generator.np, "savez"):
            start = perf_counter()
            raw = legacy.generator.generate(
                np.array(config["initial_pose"]), steps, np.array(config["alphas"]),
                np.array(config["betas"]), config["delta_t"], config["max_observations"],
                landmark=config["selection"], landmarkControl=config["fov"], forceNew=True,
            )
            generation_seconds = perf_counter() - start
    finally:
        np.random.set_state(rng_state)
    if source_hashes() != before:
        raise RuntimeError("Source files changed during generation.")
    field = legacy.fieldSettings.field
    landmarks = np.column_stack((field["markerPosX"], field["markerPosY"]))
    return Dataset(raw, landmarks, {
        "schema": 1, "config": config, "environment": environment(),
        "source_sha256": before, "generation_seconds_without_file_io": generation_seconds,
        "raw_layout": "u[3], executed_pose[3], nominal_pose[3], slots[id, clean_r, clean_b, noisy_r, noisy_b]",
        "time_convention": "row k is after command k; initial pose at time 0; first row at delta_t",
    })


def observations(row: np.ndarray, mode: str) -> np.ndarray:
    """Return [range, bearing, ID] rows, retaining stored observation order."""
    if mode not in MODES:
        raise ValueError(f"Unknown observation mode: {mode}")
    slots = row[9:].reshape(-1, 5)
    slots = slots[slots[:, 0] != -1]
    pair = slots[:, 1:3] if mode == "original" else slots[:, 3:5]
    return np.column_stack((pair, slots[:, 0]))


@dataclass
class Replay:
    arrays: dict[str, np.ndarray]
    metrics: dict

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.arrays,
                            metrics_json=np.array(json.dumps(self.metrics, sort_keys=True)))


def association_metrics(assignments: np.ndarray, ids: np.ndarray,
                        birth_ids: np.ndarray) -> dict:
    """Score existing-landmark matches against immutable birth IDs, outside EKF."""
    present = ids >= 0
    matched = present & (assignments >= 0)
    attempted = int(matched.sum())
    total = int(present.sum())
    correct = int(np.sum(birth_ids[assignments[matched]] == ids[matched]))
    new = present & (assignments == -1)
    # Report duplicate initializations separately, including within a timestep.
    seen = set()
    duplicate = 0
    for true_id in ids[new]:
        duplicate += int(int(true_id) in seen)
        seen.add(int(true_id))
    return {
        "observations": total, "attempted_existing_matches": attempted,
        "correct_existing_matches": correct, "incorrect_existing_matches": attempted - correct,
        "association_accuracy": correct / attempted if attempted else None,
        "association_coverage": attempted / total if total else None,
        "new_landmark_measurements": int(new.sum()),
        "duplicate_initializations": duplicate,
        "rejected_measurements": int(np.sum(present & (assignments == -2))),
        "unmatched_measurements": int(np.sum(present & (assignments < 0))),
        "used_observation_fraction": float(np.sum(present & (assignments >= -1)) / total) if total else None,
    }


def accuracy_metrics(truth: np.ndarray, pose: np.ndarray, landmarks: np.ndarray,
                     map_estimate: np.ndarray, birth_ids: np.ndarray) -> dict:
    heading_error = (pose[:, 2] - truth[:, 2] + np.pi) % (2 * np.pi) - np.pi
    landmark_error = map_estimate - landmarks[birth_ids]
    return {
        "position_rmse": float(np.sqrt(np.mean(np.sum((pose[:, :2] - truth[:, :2])**2, axis=1)))),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading_error**2))),
        "landmark_rmse": float(np.sqrt(np.mean(np.sum(landmark_error**2, axis=1)))) if len(birth_ids) else None,
        "mapped_landmark_count": len(birth_ids), "unique_mapped_truth_count": len(np.unique(birth_ids)),
    }


def replay(dataset: Dataset, mode: str = "original", association: str = "known",
           update_method: str = "batch") -> Replay:
    if association not in ASSOCIATIONS or update_method not in ("batch", "seq"):
        raise ValueError("Supported associations: known/nn; updates: batch/seq.")
    legacy = legacy_modules()
    associate = getattr(legacy, f"da_{association}").associateData
    config = dataset.config
    mu = np.array(config["initial_pose"])  # Preserve original integer initialization.
    covariance = np.array(config["initial_covariance"])
    signatures = []
    alphas = np.array(config["alphas"])
    measurement_covariance = np.diag(np.array(config["betas"])**2)
    measurements = [observations(row, mode) for row in dataset.raw]
    # The original runner fails before prediction on empty observations. Do not
    # silently invent a new prediction-only policy under Estimator A's name.
    if any(len(z) == 0 for z in measurements):
        raise ValueError("Legacy runner does not support empty-observation steps; replay refused.")
    states, covariances, assignments = [], [], []
    update_times = []
    symmetry, min_eigenvalues, covariance_scales = [], [], []
    for row, z in zip(dataset.raw, measurements):
        u = row[:3]
        start = perf_counter()
        motion_covariance = np.diag([
            alphas[0]*u[0]**2 + alphas[1]*u[1]**2,
            alphas[2]*u[1]**2 + alphas[3]*u[0]**2 + alphas[3]*u[2]**2,
            alphas[0]*u[2]**2 + alphas[1]*u[1]**2,
        ])
        predicted, predicted_covariance = legacy.simSLAM.predict(mu, covariance, u, motion_covariance)
        assigned, jacobian, innovation = associate(
            z, measurement_covariance, predicted, predicted_covariance, signatures)
        mu, covariance = legacy.simSLAM.update(
            predicted, predicted_covariance, assigned, jacobian,
            measurement_covariance, innovation, update_method)
        mu, covariance, signatures = legacy.simSLAM.augmentState(
            assigned, z, mu, covariance, measurement_covariance, signatures)
        elapsed = perf_counter() - start
        # Copy immediately: the original g() mutates its input pose view.
        update_times.append(elapsed)
        states.append(mu.copy())
        covariances.append(covariance.copy())
        assignments.append(assigned.copy())
        symmetry.append(float(np.max(np.abs(covariance - covariance.T))))
        min_eigenvalues.append(float(np.linalg.eigvalsh((covariance + covariance.T) / 2)[0]))
        covariance_scales.append(float(max(1, np.linalg.norm(covariance, ord=2))))
        if not np.isfinite(mu).all() or not np.isfinite(covariance).all():
            raise FloatingPointError("Nonfinite legacy state/covariance.")

    steps = len(states)
    dimension = len(states[-1])
    max_obs = config["max_observations"]
    padded_state = np.full((steps, dimension), np.nan)
    padded_covariance = np.full((steps, dimension, dimension), np.nan)
    padded_assignments = np.full((steps, max_obs), -3, dtype=int)  # -3 = absent slot
    used_observations = np.full((steps, max_obs, 3), np.nan)
    truth_ids = np.full((steps, max_obs), -1, dtype=int)
    for k, (state, cov, assigned, z) in enumerate(zip(states, covariances, assignments, measurements)):
        padded_state[k, :len(state)] = state
        padded_covariance[k, :len(state), :len(state)] = cov
        padded_assignments[k, :len(z)] = assigned
        used_observations[k, :len(z)] = z
        truth_ids[k, :len(z)] = z[:, 2].astype(int)
    birth_ids = np.asarray(signatures, dtype=int)
    metrics = accuracy_metrics(dataset.raw[:, 3:6], padded_state[:, :3],
                               dataset.landmarks, states[-1][3:].reshape(-1, 2), birth_ids)
    metrics.update(association_metrics(padded_assignments, truth_ids, birth_ids))
    metrics.update({
        "estimator": "A", "observation_mode": mode, "association": association,
        "update_method": update_method,
        "mean_update_ms": float(np.mean(update_times) * 1000),
        "p95_update_ms": float(np.percentile(update_times, 95) * 1000),
        "estimator_total_seconds": float(np.sum(update_times)),
        "max_covariance_asymmetry": max(symmetry),
        "min_covariance_eigenvalue": min(min_eigenvalues),
        "max_relative_covariance_asymmetry": float(np.max(np.array(symmetry) / covariance_scales)),
        "min_relative_covariance_eigenvalue": float(np.min(np.array(min_eigenvalues) / covariance_scales)),
    })
    arrays = {"state": padded_state, "covariance": padded_covariance,
              "state_dimensions": np.array([len(s) for s in states]),
              "estimated_pose": padded_state[:, :3].copy(),
              "landmark_estimates": padded_state[:, 3:].reshape(steps, -1, 2).copy(),
              "landmark_birth_truth_ids": birth_ids,
              "association_indices": padded_assignments, "observation_truth_ids": truth_ids,
              "used_observations": used_observations, "update_seconds": np.array(update_times)}
    return Replay(arrays, metrics)


def identical_replays(first: Replay, other: Replay) -> bool:
    """Require exact numerical replay, excluding inherently variable timings."""
    return all(np.array_equal(value, other.arrays[key], equal_nan=True)
               for key, value in first.arrays.items() if key != "update_seconds")


def make_plots(dataset: Dataset, results: dict[str, Replay], output: Path) -> None:
    import matplotlib.pyplot as plt

    for association in ASSOCIATIONS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, mode in zip(axes, MODES):
            result = results[f"{mode}_{association}"]
            pose = np.vstack((dataset.config["initial_pose"], result.arrays["estimated_pose"]))
            truth = np.vstack((dataset.config["initial_pose"], dataset.raw[:, 3:6]))
            ax.plot(truth[:, 0], truth[:, 1], color="black", label="Executed truth", lw=1)
            ax.plot(pose[:, 0], pose[:, 1], label="Estimator A", lw=1)
            ax.scatter(*dataset.landmarks.T, marker="*", s=90, color="black", label="True landmarks")
            estimated_map = result.arrays["landmark_estimates"][-1]
            ax.scatter(*estimated_map.T, marker="x", color="tab:red", label="Estimated landmarks")
            for i, point in enumerate(dataset.landmarks):
                ax.annotate(str(i), point, xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set(title=f"{mode}: {association} association", xlabel="x (simulation units)",
                   ylabel="y (simulation units)", aspect="equal")
            ax.legend(fontsize=8)
        fig.savefig(output / f"trajectories_{association}.png", dpi=160)
        plt.close(fig)


def write_report(output: Path, dataset: Dataset, results: dict[str, Replay], manifest: dict) -> None:
    config = dataset.config
    lines = [
        "# Milestone 1 — Estimator A reproduction", "",
        f"Validation status: **{'PASS' if manifest['validation_passed'] else 'FAIL'}** for this fixed-seed simulation/replay gate.", "",
        f"Seed {config['seed']}; {config['steps']} steps; dt={config['delta_t']} s; float64 arithmetic after legacy integer initialization; batch updates; three estimator replays per case are the default.", "",
        f"Actual repeats per case: {manifest['repeats']}. Each repeat reloads `dataset.npz` from disk. All numerical state, covariance, map, observation and association arrays must be exactly equal (NaN padding equal). Timings are excluded from equality.", "",
        "## Results", "",
        "Errors use the noisy executed trajectory, never the nominal path. Position/map units are the simulator's coordinate units; heading is radians. Map RMSE is final-map error against each estimate's initialization (birth) truth ID, including duplicate tracks. No fitted frame alignment is applied: truth and estimate share the same anchored initial pose. IDs are evaluation labels only for NN.", "",
        "| Observation mode | Association | Position RMSE | Heading RMSE (rad) | Map RMSE | Map tracks / unique IDs | Association accuracy | Coverage | Mean / p95 update (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        m = result.metrics
        accuracy = f"{m['association_accuracy']:.2%}" if m['association_accuracy'] is not None else "N/A"
        lines.append(f"| {m['observation_mode']} | {m['association']} | {m['position_rmse']:.6f} | {m['heading_rmse_rad']:.6f} | {m['landmark_rmse']:.6f} | {m['mapped_landmark_count']} / {m['unique_mapped_truth_count']} | {accuracy} | {m['association_coverage']:.2%} | {m['mean_update_ms']:.3f} / {m['p95_update_ms']:.3f} |")
    lines += [
        "", "Timing columns pool all repeated estimator updates, including the first update (no warm-up removed). They include motion-noise construction, prediction, association, correction and map augmentation. Imports, observation adaptation, diagnostics, history copies, metrics, plotting, printing, generation and file I/O are outside the timer. Exact timings vary with machine load and are not deterministic. Per-run timings and separate generation duration are retained in JSON/NPZ.", "",
        "Association accuracy = correct existing-track matches / attempted existing-track matches. Coverage = attempted existing-track matches / all observations. New landmarks are excluded from accuracy and count as unmatched for this definition of coverage. Counts of new, duplicate, incorrect and rejected decisions, plus the fraction used for either correction or initialization, are in `summary.json`. Birth-ID scoring cannot certify a track remains unmixed after incorrect associations; NN map error should be read with these counts.", "",
        "## Implementation and preserved behavior", "",
        "The external harness calls the original `predict`, association, `update`, and `augmentState` functions. The joint state is robot pose plus stationary 2D landmarks. Prediction uses rotate–translate–rotate odometry with command-dependent Gaussian increment covariance. Observations are range and relative bearing with wrapped bearing innovations. Existing landmarks correct the joint state; new landmarks use inverse range–bearing geometry and full covariance/cross-covariance augmentation. Known association uses IDs; NN selects the smallest squared Mahalanobis distance below 4, without enforcing one-to-one matches.", "",
        "The original source is unchanged, including integer initial pose, input-view mutation in `g`, explicit inverses, simplified covariance correction, and lack of heading wrapping after correction. The first prediction copies into an integer state array; its truncation is preserved. Sequential updates also retain their precomputed innovations/Jacobians. These are properties of A, not changes to make it resemble future B/C.", "",
        "The generator uses fixed legacy `np.random.seed`; its caller RNG state is restored afterward. All original draws, including zero-variance draws for nominal truth, are retained. Only its per-step partial-file writes are suppressed; the completed raw array is saved once. Qt backend selection is intercepted at import for headless Agg plotting. These hooks alter I/O only. Automated parity tests execute the original `simSLAM.run` control flow with plotting/printing/writes intercepted and compare every captured state/covariance and association. They cover both observation modes, both association methods and both update methods; the noisy runner input is adapted independently of the harness observation selector.", "",
        "## Observation mismatch and separate adapter", "",
        "Despite comments/docstrings, generated slots are `[ID, clean range, clean bearing, noisy range, noisy bearing]`. The original runner consumes the clean pair while assuming range standard deviation 10 and bearing standard deviation 10 degrees. `original` is the original reproduction. `noisy` is the corrected observation replay benchmark: it selects the stored noisy pair without changing any EKF logic or nominal covariance assumptions. Both use identical commands, truth and observation opportunities from one saved dataset. They must not be merged into a single baseline result.", "",
        "## Files and reproduction", "",
        "- `dataset.npz`: untouched raw layout plus explicit commands, executed/nominal truth, timestamps, true map, clean/noisy measurements, IDs, seed, configuration, versions and source SHA-256 hashes.",
        "- `<mode>_<association>.npz`: complete padded state/covariance histories, pose/map estimates, landmark birth IDs, observations consumed, associations and update timings. State dimensions identify active entries; NaN means uninitialized and association -3 means absent observation slot (-1=new, -2=ignore).",
        "- `summary.json`: validation checks, source hashes, environment, per-repeat metrics and pooled timing metrics.",
        "- `trajectories_known.png`, `trajectories_nn.png`: executed truth, estimates, and true/estimated maps.", "",
        "From the repository root:", "", "```sh",
        f"python3 -B -m benchmarks.estimator_a --seed {config['seed']} --steps {config['steps']} --repeats {manifest['repeats']} --output artifacts/milestone1",
        "python3 -B -m benchmarks.estimator_a --dataset artifacts/milestone1/dataset.npz --repeats 3 --output /tmp/estimator_a_replay",
        "python3 -B -m pytest -q tests/test_estimator_a.py", "```", "",
        "## Validation and limitations", "",
        f"Validation checks: `{json.dumps(manifest['checks'], sort_keys=True)}`.", "",
        "The covariance diagnostic tolerance is 1e-10 relative to max(1, spectral norm), applied to asymmetry and negative eigenvalues of the symmetrized covariance; it is a numerical roundoff check, not an accuracy target. Accuracy has no imposed pass threshold in Milestone 1. Source preservation and repeated numerical replay are required independently of performance.", "",
        "The original runner cannot handle zero-observation steps; the harness explicitly refuses such data instead of silently changing its prediction-only policy. Legacy double-gated association can drop ambiguous output entries, and JCBB is unfinished with a syntax error; neither is used. Generator cache load/save directories disagree, so the harness uses its own explicit dataset path. The legacy association-history array has fixed dimensions; replay stores histories sized to this experiment.", "",
        "These are single-seed baseline results, not a Monte Carlo comparison or evidence of MHE benefit. Large/noisy NN failures must be reported rather than tuned away. No continuous process-noise mapping is assumed here. Estimators B/C, spline integration, and Victoria Park remain unimplemented by this task. The next gate is Milestone 2 spline-library validation; B remains Milestone 3.", "",
    ]
    (output / "report.md").write_text("\n".join(lines))


def run_experiment(output: Path, *, seed: int = 7, steps: int = 500,
                   repeats: int = 3, dataset_path: Path | None = None) -> dict:
    if repeats < 2:
        raise ValueError("At least two saved-dataset replays are required for validation.")
    before = source_hashes()
    output.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.load(dataset_path) if dataset_path else generate_dataset(seed, steps)
    saved_path = output / "dataset.npz"
    if dataset_path is None:
        dataset.save(saved_path)
    elif dataset_path.resolve() != saved_path.resolve():
        shutil.copyfile(dataset_path, saved_path)  # Preserve archive identity/provenance.
    all_results, repeat_metrics, identical = {}, {}, {}
    for mode in MODES:
        for association in ASSOCIATIONS:
            name = f"{mode}_{association}"
            runs = [replay(Dataset.load(saved_path), mode, association) for _ in range(repeats)]
            identical[name] = all(identical_replays(runs[0], run) for run in runs[1:])
            repeat_metrics[name] = [run.metrics.copy() for run in runs]
            result = runs[0]
            pooled = np.concatenate([run.arrays["update_seconds"] for run in runs])
            result.metrics["mean_update_ms"] = float(np.mean(pooled) * 1000)
            result.metrics["p95_update_ms"] = float(np.percentile(pooled, 95) * 1000)
            result.arrays["repeated_update_seconds"] = np.stack([run.arrays["update_seconds"] for run in runs])
            result.save(output / f"{name}.npz")
            all_results[name] = result
    make_plots(dataset, all_results, output)
    checks = {
        "repeated_saved_replay_exact": all(identical.values()),
        "source_files_unchanged": source_hashes() == before,
        "generation_source_matches_current_legacy": all(
            dataset.metadata["source_sha256"].get(key) == value for key, value in before.items()
            if key.startswith("baseline/")),
        "covariance_roundoff_checks": all(
            r.metrics["max_relative_covariance_asymmetry"] <= 1e-10 and
            r.metrics["min_relative_covariance_eigenvalue"] >= -1e-10 for r in all_results.values()),
        "finite_active_states": all(np.isfinite(r.arrays["estimated_pose"]).all() for r in all_results.values()),
    }
    manifest = {"validation_passed": all(checks.values()), "checks": checks,
                "repeats": repeats, "exact_replay_by_case": identical,
                "dataset_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
                "environment": environment(), "source_sha256": before,
                "dataset_metadata": dataset.metadata,
                "results": {k: r.metrics for k, r in all_results.items()},
                "per_repeat_metrics": repeat_metrics}
    (output / "summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(output, dataset, all_results, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dataset", type=Path, help="Replay an existing harness dataset; seed/steps come from it.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "milestone1")
    args = parser.parse_args()
    manifest = run_experiment(args.output, seed=args.seed, steps=args.steps,
                              repeats=args.repeats, dataset_path=args.dataset)
    print(json.dumps({"validation_passed": manifest["validation_passed"],
                      "results": manifest["results"]}, indent=2))
    if not manifest["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
