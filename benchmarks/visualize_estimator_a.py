"""Inspect saved Estimator A replays using the legacy Matplotlib drawing helpers.

No simulation or estimator is imported or executed. Example:
    python -m benchmarks.visualize_estimator_a --mode noisy --association nn
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "artifacts" / "milestone1"
STATUS_COLORS = {"correct": "#218739", "incorrect": "#c51b28",
                 "new landmark": "#1976b9", "duplicate initialization": "#a91bb5",
                 "rejected": "#777777", "unavailable": "#777777"}


@dataclass(frozen=True)
class Observation:
    slot: int
    truth_id: int
    assignment: int | None
    track: int | None
    birth_truth_id: int | None
    status: str
    truth_endpoint: np.ndarray
    estimate_endpoint: np.ndarray
    target: np.ndarray | None


@dataclass(frozen=True)
class Frame:
    index: int
    time: float
    true_pose: np.ndarray
    estimated_pose: np.ndarray
    true_path: np.ndarray
    estimated_path: np.ndarray
    landmarks: np.ndarray
    birth_truth_ids: np.ndarray
    duplicate_tracks: np.ndarray
    observations: tuple[Observation, ...]
    position_error: float
    heading_error: float


class ReplayData:
    """Read-only adapter for the saved raw dataset and estimator output.

    Track identities are evaluation labels from initialization, not a guarantee
    that a nearest-neighbor track remains unmixed after incorrect associations.
    Measurements are projected from the recorded posterior poses for display;
    these rays are not the pre-correction innovations used by the EKF.
    """

    def __init__(self, dataset_path: Path, replay_path: Path):
        with np.load(dataset_path, allow_pickle=False) as saved:
            raw = saved["raw"]
            self.true_landmarks = saved["landmark_positions"]
            config = json.loads(str(saved["metadata_json"]))["config"]
            self.times = saved["time"]
        with np.load(replay_path, allow_pickle=False) as saved:
            self.metrics = json.loads(str(saved["metrics_json"]))
            self.pose = saved["estimated_pose"]
            self.maps = saved["landmark_estimates"]
            self.birth_ids = saved["landmark_birth_truth_ids"]
            dimensions = saved["state_dimensions"]
            self.measurements = saved["used_observations"]
            self.assignments = saved["association_indices"] if "association_indices" in saved else None
            self.robot_covariance = saved["covariance"][:, :3, :3].copy() if "covariance" in saved else None
            recorded_ids = saved["observation_truth_ids"] if "observation_truth_ids" in saved else None
        self.mode = self.metrics.get("observation_mode")
        self.association = self.metrics.get("association")
        if self.mode not in ("original", "noisy") or self.association not in ("known", "nn"):
            raise ValueError("Replay must identify original/noisy observations and known/nn association.")
        if self.metrics.get("estimator") != "A":
            raise ValueError("This viewer expects an Estimator A replay.")
        self.initial_pose = np.asarray(config["initial_pose"], dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 14 or (raw.shape[1] - 9) % 5:
            raise ValueError("Invalid legacy dataset dimensions.")
        steps, slots = len(raw), (raw.shape[1] - 9) // 5
        if steps < 2 or self.pose.shape != (steps, 3) or self.times.shape != (steps,):
            raise ValueError("Dataset/replay frame counts or pose dimensions disagree.")
        if self.true_landmarks.ndim != 2 or self.true_landmarks.shape[1] != 2:
            raise ValueError("True landmarks must have shape (N, 2).")
        if (self.maps.ndim != 3 or self.maps.shape[0] != steps or self.maps.shape[2] != 2
                or self.birth_ids.shape != (self.maps.shape[1],) or dimensions.shape != (steps,)
                or self.measurements.shape != (steps, slots, 3)):
            raise ValueError("Invalid replay map/measurement dimensions.")
        if not np.issubdtype(dimensions.dtype, np.integer) or np.any((dimensions - 3) % 2):
            raise ValueError("Invalid joint-state dimensions.")
        self.counts = (dimensions - 3) // 2
        if (np.any(self.counts < 0) or np.any(np.diff(self.counts) < 0)
                or self.counts[-1] != len(self.birth_ids)):
            raise ValueError("Landmark counts must grow monotonically to the saved map size.")
        if (not np.issubdtype(self.birth_ids.dtype, np.integer) or np.any(self.birth_ids < 0)
                or np.any(self.birth_ids >= len(self.true_landmarks))):
            raise ValueError("Invalid landmark birth truth IDs.")
        self.truth = raw[:, 3:6].copy()
        if (not np.isfinite(raw).all() or not np.isfinite(self.pose).all()
                or not np.isfinite(self.true_landmarks).all() or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0) or not np.isfinite(self.initial_pose).all()):
            raise ValueError("Nonfinite poses/map or nonmonotone timestamps.")
        self.valid = np.isfinite(self.measurements[:, :, 2])
        self.ids = np.full((steps, slots), -1, dtype=int)
        if self.assignments is not None:
            if (self.assignments.shape != (steps, slots)
                    or not np.issubdtype(self.assignments.dtype, np.integer)):
                raise ValueError("Invalid association array dimensions/type.")
        raw_slots = raw[:, 9:].reshape(steps, slots, 5)
        previous_count = 0
        for index in range(steps):
            selected = raw_slots[index][raw_slots[index, :, 0] != -1]
            pair = selected[:, 1:3] if self.mode == "original" else selected[:, 3:5]
            expected = np.column_stack((pair, selected[:, 0]))
            n = len(selected)
            if (not np.array_equal(self.measurements[index, :n], expected)
                    or not np.isnan(self.measurements[index, n:]).all()):
                raise ValueError(f"Dataset/replay observations disagree at frame {index}; check mode and dataset.")
            self.ids[index, :n] = expected[:, 2].astype(int)
            if np.any(self.ids[index, :n] < 0) or np.any(self.ids[index, :n] >= len(self.true_landmarks)):
                raise ValueError("Invalid observed truth ID.")
            count = self.counts[index]
            if (not np.isfinite(self.maps[index, :count]).all()
                    or not np.isnan(self.maps[index, count:]).all()):
                raise ValueError(f"Active/padded landmark coordinates disagree at frame {index}.")
            if self.assignments is not None:
                assigned = self.assignments[index, :n]
                if (np.any(assigned < -2) or np.any(assigned >= previous_count)
                        or np.any(self.assignments[index, n:] != -3)):
                    raise ValueError(f"Invalid existing-track association at frame {index}.")
                new_ids = self.ids[index, :n][assigned == -1]
                if (count != previous_count + len(new_ids)
                        or not np.array_equal(new_ids, self.birth_ids[previous_count:count])):
                    raise ValueError(f"Initialization/track identity mismatch at frame {index}.")
            previous_count = count
        if recorded_ids is not None and not np.array_equal(recorded_ids, self.ids):
            raise ValueError("Recorded observation IDs disagree with dataset.")
        if self.robot_covariance is not None and (
                self.robot_covariance.shape != (steps, 3, 3) or not np.isfinite(self.robot_covariance).all()):
            raise ValueError("Invalid robot covariance history.")
        seen = set()
        self.duplicate = np.zeros(len(self.birth_ids), dtype=bool)
        for track, truth_id in enumerate(self.birth_ids):
            self.duplicate[track] = int(truth_id) in seen
            seen.add(int(truth_id))
        # Includes the current row; the legacy plot omitted it from path lines.
        self.true_path = np.vstack((self.initial_pose, self.truth))
        self.estimated_path = np.vstack((self.initial_pose, self.pose))
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def __len__(self) -> int:
        return len(self.pose)

    def frame(self, index: int) -> Frame:
        if not 0 <= index < len(self):
            raise IndexError(index)
        count = self.counts[index]
        previous_count = self.counts[index - 1] if index else 0
        new_track = int(previous_count)
        entries = []
        for slot in np.flatnonzero(self.valid[index]):
            radius, bearing, truth_id = self.measurements[index, slot]
            assignment = int(self.assignments[index, slot]) if self.assignments is not None else None
            track = None
            if assignment is None:
                status = "unavailable"
            elif assignment == -2:
                status = "rejected"
            elif assignment == -1:
                track = new_track
                new_track += 1
                status = "duplicate initialization" if self.duplicate[track] else "new landmark"
            else:
                track = assignment
                status = "correct" if self.birth_ids[track] == truth_id else "incorrect"

            def endpoint(pose):
                angle = pose[2] + bearing
                return pose[:2] + radius * np.array([np.cos(angle), np.sin(angle)])

            entries.append(Observation(
                slot=int(slot), truth_id=int(truth_id), assignment=assignment, track=track,
                birth_truth_id=int(self.birth_ids[track]) if track is not None else None,
                status=status, truth_endpoint=endpoint(self.truth[index]),
                estimate_endpoint=endpoint(self.pose[index]),
                target=self.maps[index, track] if track is not None else None,
            ))
        delta = self.pose[index] - self.truth[index]
        return Frame(index, float(self.times[index]), self.truth[index], self.pose[index],
                     self.true_path[:index + 2], self.estimated_path[:index + 2],
                     self.maps[index, :count], self.birth_ids[:count], self.duplicate[:count],
                     tuple(entries), float(np.linalg.norm(delta[:2])),
                     float((delta[2] + np.pi) % (2 * np.pi) - np.pi))

    def issue_frames(self) -> tuple[int, ...]:
        return tuple(index for index in range(len(self))
                     if any(o.status in ("incorrect", "duplicate initialization")
                            for o in self.frame(index).observations))


@lru_cache(maxsize=1)
def legacy_plot_helpers():
    """Load only plotting helpers and their field settings, without simSLAM.

    helpers.py uses an unqualified fieldSettings import. Bind that dependency
    temporarily while executing its unchanged file, without forcing a backend.
    """
    directory = ROOT / "baseline" / "ekf_slam" / "code"

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    field = load("_replay_field_settings", directory / "fieldSettings.py")
    previous = sys.modules.get("fieldSettings")
    sys.modules["fieldSettings"] = field
    try:
        return load("_replay_plot_helpers", directory / "helpers.py")
    finally:
        if previous is None:
            del sys.modules["fieldSettings"]
        else:
            sys.modules["fieldSettings"] = previous


class ReplayViewer:
    """Matplotlib replay controls around an adapted legacy frame drawing layout."""

    def __init__(self, data: ReplayData, interval_ms: float | None = None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.data = data
        self.helpers = legacy_plot_helpers()
        self.index = 0
        self.playing = False
        self.issues = data.issue_frames()
        self.figure = plt.figure(figsize=(14, 8))
        self.ax = self.figure.add_axes([0.07, 0.23, 0.58, 0.68])
        self.panel = self.figure.add_axes([0.68, 0.23, 0.31, 0.68])
        self.panel.axis("off")
        self.slider = Slider(self.figure.add_axes([0.12, 0.14, 0.76, 0.035]),
                             "Frame", 0, len(data) - 1, valinit=0, valstep=1, valfmt="%d")
        self.buttons = {}
        for left, label, callback in (
                (0.18, "Previous", lambda event: self.step(-1)),
                (0.31, "Play", self.toggle),
                (0.44, "Next", lambda event: self.step(1)),
                (0.57, "Next issue", lambda event: self.next_issue())):
            button = Button(self.figure.add_axes([left, 0.07, 0.12, 0.045]), label)
            button.on_clicked(callback)
            self.buttons[label] = button
        self.figure.text(0.5, 0.025,
                         "Space: play/pause   Left/Right: step   Home/End: first/last   N: next issue",
                         ha="center", fontsize=10)
        self.slider.on_changed(lambda value: self.draw(int(value)))
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("close_event", lambda event: self.pause())
        interval = interval_ms if interval_ms is not None else float(np.median(np.diff(data.times)) * 1000)
        if not np.isfinite(interval) or interval <= 0:
            raise ValueError("Playback interval must be positive and finite.")
        self.timer = self.figure.canvas.new_timer(interval=max(1, round(interval)))
        self.timer.add_callback(self.tick)
        points = np.vstack((data.truth[:, :2], data.pose[:, :2], data.true_landmarks,
                            data.maps[np.isfinite(data.maps).all(axis=2)], data.initial_pose[:2]))
        lower, upper = points.min(axis=0), points.max(axis=0)
        margin = max(20, float(np.max(upper - lower)) * 0.08)
        self.bounds = (lower - margin, upper + margin)
        self.draw(0)

    def draw(self, index: int) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        self.index = index
        frame = self.data.frame(index)
        ax = self.ax
        # Adapt plotsim's clear-and-redraw composition, clearing only the map
        # axes so Slider/Button widgets survive. plotRobot/plotCov2D are reused.
        ax.clear()
        ax.plot(*frame.true_path[:, :2].T, color="#1864ab", lw=1.3, label="Executed truth")
        ax.plot(*frame.estimated_path[:, :2].T, color="#e67700", lw=1.3, label="Estimate")
        ax.scatter(*self.data.true_landmarks.T, marker="*", s=90, c="black", label="True landmarks")
        for truth_id, position in enumerate(self.data.true_landmarks):
            ax.annotate(f"L{truth_id}", position, xytext=(4, 6), textcoords="offset points", fontsize=8)
        colors = np.where(frame.duplicate_tracks, STATUS_COLORS["duplicate initialization"], "#e67700")
        if len(frame.landmarks):
            ax.scatter(*frame.landmarks.T, marker="x", c=colors, s=40, label="Estimated tracks")
        for track, (position, truth_id) in enumerate(zip(frame.landmarks, frame.birth_truth_ids)):
            ax.annotate(f"T{track}/L{truth_id}", position, xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color=colors[track])
        plt.sca(ax)  # Original helpers draw into pyplot's current axes.
        self.helpers.plotRobot(frame.true_pose, "#1864ab", "#1864ab20", r=6)
        self.helpers.plotRobot(frame.estimated_pose, "#e67700", "#e6770020", r=6)
        if self.data.robot_covariance is not None:
            self.helpers.plotCov2D(frame.estimated_pose[:2], self.data.robot_covariance[index, :2, :2],
                                   color="#e67700", nSigma=3)
        for observation in frame.observations:
            color = STATUS_COLORS[observation.status]
            # Solid: actual consumed measurement at truth pose; dotted: the
            # same measurement at posterior estimated pose; dashed: chosen track.
            ax.plot(*np.vstack((frame.true_pose[:2], observation.truth_endpoint)).T,
                    color=color, lw=1.4, alpha=0.85)
            ax.plot(*np.vstack((frame.estimated_pose[:2], observation.estimate_endpoint)).T,
                    color=color, ls=":", lw=1.5)
            ax.scatter(*observation.estimate_endpoint, marker="+", c=color, s=60)
            if observation.target is not None:
                ax.plot(*np.vstack((observation.estimate_endpoint, observation.target)).T,
                        color=color, ls="--", lw=1.4)
                ax.scatter(*observation.target, facecolors="none", edgecolors=color, s=140, lw=1.4)
            ax.scatter(*self.data.true_landmarks[observation.truth_id],
                       facecolors="none", edgecolors=color, s=160, lw=1.4)
        lower, upper = self.bounds
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal",
               xlabel="x (simulation units)", ylabel="y (simulation units)")
        mode_label = "Original reproduction (clean)" if self.data.mode == "original" else "Corrected replay (noisy)"
        ax.set_title(f"{mode_label} | {self.data.association} association")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper left", fontsize=8)
        self.panel.clear()
        self.panel.axis("off")
        count = len(frame.landmarks)
        self.panel.text(0, 1, f"Frame {index} / {len(self.data) - 1}  |  step {index + 1}\n"
                        f"Time: {frame.time:.2f} s\n"
                        f"Position error: {frame.position_error:.3f}\n"
                        f"Heading error: {frame.heading_error:+.4f} rad\n"
                        f"Tracks: {count}  |  unique IDs: {len(np.unique(frame.birth_truth_ids))}\n"
                        f"Duplicate tracks: {int(frame.duplicate_tracks.sum())}", va="top", fontsize=11)
        self.panel.text(0, 0.73, "Current observations", weight="bold", fontsize=11)
        for row, observation in enumerate(frame.observations):
            target = (f"T{observation.track} (birth L{observation.birth_truth_id})"
                      if observation.track is not None else "no track")
            label = f"Observation {observation.slot}: true L{observation.truth_id}\n  -> {target}\n  {observation.status.upper()}"
            self.panel.text(0, 0.68 - row * 0.16, label, va="top", fontsize=10,
                            color=STATUS_COLORS[observation.status])
        if not frame.observations:
            self.panel.text(0, 0.68, "No observations", va="top")
        legend_handles = [Line2D([], [], color=color, lw=2, label=status)
                          for status, color in STATUS_COLORS.items() if status != "unavailable"]
        self.panel.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(-0.03, 0.06),
                          frameon=False, fontsize=9)
        self.panel.text(0, 0.04, "Solid: measurement at truth pose\nDotted: measurement at estimated pose\nDashed: endpoint to assigned track\n"
                        "T = track index; L = birth truth ID\nRays use posterior poses, not innovations.",
                        va="top", fontsize=8, color="#444444")
        self.figure.canvas.draw_idle()

    def seek(self, index: int) -> None:
        self.slider.set_val(int(np.clip(index, 0, len(self.data) - 1)))

    def pause(self) -> None:
        self.playing = False
        self.timer.stop()
        self.buttons["Play"].label.set_text("Play")

    def toggle(self, event=None) -> None:
        if self.playing:
            self.pause()
        else:
            if self.index == len(self.data) - 1:
                self.seek(0)
            self.playing = True
            self.buttons["Play"].label.set_text("Pause")
            self.timer.start()
        self.figure.canvas.draw_idle()

    def tick(self) -> None:
        if self.playing:
            self.seek(self.index + 1)
            if self.index == len(self.data) - 1:
                self.pause()

    def step(self, offset: int) -> None:
        self.pause()
        self.seek(self.index + offset)

    def next_issue(self) -> None:
        self.pause()
        candidates = [index for index in self.issues if index > self.index]
        if candidates:
            self.seek(candidates[0])
        elif self.issues:
            self.seek(self.issues[0])

    def on_key(self, event) -> None:
        if event.key in (" ", "space"):
            self.toggle()
        elif event.key in ("left", "right"):
            self.step(-1 if event.key == "left" else 1)
        elif event.key in ("home", "end"):
            self.pause()
            self.seek(0 if event.key == "home" else len(self.data) - 1)
        elif event.key in ("n", "N"):
            self.next_issue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RESULTS / "dataset.npz")
    parser.add_argument("--mode", choices=("original", "noisy"), default="noisy")
    parser.add_argument("--association", choices=("known", "nn"), default="nn")
    parser.add_argument("--replay", type=Path, help="Explicit replay NPZ; its metadata determines mode/association.")
    parser.add_argument("--interval-ms", type=float, help="Default: saved simulation time step.")
    parser.add_argument("--backend", help="Interactive Matplotlib backend, e.g. QtAgg or MacOSX.")
    parser.add_argument("--frame", type=int, default=0, help="Initial frame, indexed from zero.")
    parser.add_argument("--snapshot", type=Path, help="Render one frame with Agg and exit (no GUI).")
    args = parser.parse_args()
    import matplotlib

    if args.snapshot:
        matplotlib.use("Agg")
    elif args.backend:
        matplotlib.use(args.backend)
    import matplotlib.pyplot as plt

    replay_path = args.replay or args.dataset.parent / f"{args.mode}_{args.association}.npz"
    data = ReplayData(args.dataset, replay_path)
    if not 0 <= args.frame < len(data):
        parser.error(f"--frame must be in [0, {len(data) - 1}]")
    viewer = ReplayViewer(data, args.interval_ms)
    viewer.seek(args.frame)
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        viewer.figure.savefig(args.snapshot, dpi=150)
        plt.close(viewer.figure)
    else:
        if str(matplotlib.get_backend()).lower() in ("agg", "pdf", "svg", "ps", "template"):
            parser.error("Interactive playback needs a GUI backend; try --backend QtAgg or use --snapshot.")
        plt.show()


if __name__ == "__main__":
    main()
