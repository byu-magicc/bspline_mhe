"""Known-association Baseline B replay of saved Milestone 1 data only.

Use three held-input phases to represent each stored RTR odometry increment.
Map independent increment variances into per-phase input-noise intensities.
This matches A's nominal endpoint and first-order prediction covariance, not
the underlying nonlinear stochastic distribution. No truth coordinates enter
the filter. Truth is used only in post-run metrics.
"""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.estimator_a import Dataset, observations, accuracy_metrics
from estimation.propagation import rk4_stage_times
from estimation.slam import (create_state, rk4_slam_step, sequential_observations,
    INITIALIZED, UPDATED, INVALID, OVERFLOW, LandmarkCapacityError)
from splines import UniformBasis


def odometry_phases(commands, dt, alphas):
    """RTR -> held [v,omega] phases and Qc; durations dt*[1/4,1/2,1/4].

    For independent rotation1/translation/rotation2 increment variances m1,mT,m2,
    set Qc_rot1=diag(0,m1/dt1), Qc_trans=diag(mT/dtT,0),
    Qc_rot2=diag(0,m2/dt2). Thus each integrated input variance equals its
    increment variance, and covariance transport across phases recovers the
    RTR linearized process covariance. Qc entries are NOT increment variances.
    """
    commands, alphas = np.asarray(commands, dtype=float), np.asarray(alphas, dtype=float)
    if commands.shape[-1:] != (3,) or alphas.shape != (4,) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("commands (...,3), alphas (4,), and positive dt are required")
    durations = dt * np.array([.25, .5, .25])
    controls = np.zeros(commands.shape[:-1] + (3, 2))
    controls[..., 0, 1] = commands[..., 0] / durations[0]
    controls[..., 1, 0] = commands[..., 1] / durations[1]
    controls[..., 2, 1] = commands[..., 2] / durations[2]
    variances = np.stack((alphas[0]*commands[..., 0]**2 + alphas[1]*commands[..., 1]**2,
                          alphas[2]*commands[..., 1]**2 + alphas[3]*(commands[..., 0]**2 + commands[..., 2]**2),
                          alphas[0]*commands[..., 2]**2 + alphas[1]*commands[..., 1]**2), axis=-1)
    intensity = np.zeros(commands.shape[:-1] + (3, 2, 2))
    intensity[..., 0, 1, 1] = variances[..., 0] / durations[0]
    intensity[..., 1, 0, 0] = variances[..., 1] / durations[1]
    intensity[..., 2, 1, 1] = variances[..., 2] / durations[2]
    return controls, durations, intensity


def replay_cycle(state, phase_starts, durations, controls, intensities, basis,
                  coefficients, measurements, ids, Sigma, mask):
    """One compiled command/measurement cycle, including three held phases."""
    def advance(current, row):
        t, dt, control, Qc, B = row
        # This phase-local callable holds the same command at all four stages,
        # including the right endpoint: no next-phase command leaks backward.
        return rk4_slam_step(current, t, dt, lambda t: control, B, coefficients, Qc), None

    state, _ = jax.lax.scan(advance, state, (phase_starts, durations, controls, intensities, basis))
    return sequential_observations(state, measurements, ids, Sigma, mask)


def covariance_metrics(states, covariances, active, truth):
    asymmetry, eigenvalue, inactive_max, nees = [], [], [], []
    for x, P, live, real in zip(states, covariances, active, truth):
        selected = np.r_[np.arange(3), np.flatnonzero(np.repeat(live, 2)) + 3]
        block = P[np.ix_(selected, selected)]
        asymmetry.append(np.max(np.abs(block-block.T)))
        eigenvalue.append(np.linalg.eigvalsh((block+block.T)/2)[0])
        idle = np.setdiff1d(np.arange(len(x)), selected)
        inactive_max.append(max(np.max(np.abs(P[idle]), initial=0.),
                                np.max(np.abs(P[:, idle]), initial=0.),
                                np.max(np.abs(x[idle]), initial=0.)))
        e = x[:3] - real
        e[2] = (e[2]+np.pi) % (2*np.pi)-np.pi
        nees.append(e @ np.linalg.solve(P[:3, :3], e))
    return {"max_active_covariance_asymmetry": float(max(asymmetry)),
            "min_active_covariance_eigenvalue": float(min(eigenvalue)),
            "max_inactive_storage": float(max(inactive_max)),
            "mean_robot_nees": float(np.mean(nees))}


def replay_dataset(dataset, mode="noisy", capacity=16, repeats=2):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    config = dataset.config
    dt, steps, slots = config["delta_t"], len(dataset.raw), config["max_observations"]
    controls, durations, intensities = odometry_phases(dataset.raw[:, :3], dt, config["alphas"])
    edges = np.r_[(np.arange(steps)[:, None] * dt + dt*np.array([0., .25, .75])).ravel(), steps*dt]
    stage_times = rk4_stage_times(edges).reshape(steps, 3, 4)
    adapter = UniformBasis(3, 4)
    basis = adapter.precompute_jax(stage_times, steps*dt)
    coefficients = jnp.zeros(2*adapter.num_coefficients, dtype=jnp.float64)
    y, ids, mask = np.zeros((steps, slots, 2)), np.zeros((steps, slots), dtype=np.int64), np.zeros((steps, slots), dtype=bool)
    for k, row in enumerate(dataset.raw):
        z = observations(row, mode)
        y[k, :len(z)] = z[:, :2]
        if not np.all(z[:, 2] == np.rint(z[:, 2])):
            raise ValueError("saved truth IDs must be integers")
        ids[k, :len(z)] = z[:, 2].astype(np.int64)
        mask[k, :len(z)] = True
    Sigma = jnp.asarray(np.diag(np.array(config["betas"])**2))
    initial = create_state(config["initial_pose"], config["initial_covariance"], capacity)
    starts = jnp.asarray(stage_times[:, :, 0])
    controls, durations, intensities, y, ids, mask = map(jnp.asarray, (controls, durations, intensities, y, ids, mask))

    def args(k):
        return (starts[k], durations, controls[k], intensities[k], basis[k], coefficients,
                y[k], ids[k], Sigma, mask[k])

    start = perf_counter()
    compiled = jax.jit(replay_cycle).lower(initial, *args(0)).compile()
    compile_seconds = perf_counter()-start
    jax.block_until_ready(compiled(initial, *args(0)))  # Discard warm-up, restart prior.
    first, all_times = None, []
    for _ in range(repeats):
        state = initial
        history, metadata, diagnostics, update_times = [], [], [], []
        for k in range(steps):
            values = args(k)  # Device slicing/preparation excluded from cycle timer.
            start = perf_counter()
            state, info = compiled(state, *values)
            jax.block_until_ready((state, info))
            update_times.append(perf_counter()-start)
            if np.any(np.asarray(info.status) == OVERFLOW):
                raise LandmarkCapacityError(f"capacity {capacity} exhausted during saved replay at row {k}")
            history.append((np.asarray(state.x), np.asarray(state.covariance)))
            metadata.append((np.asarray(state.active), np.asarray(state.truth_ids), np.asarray(info.status), np.asarray(info.slot)))
            diagnostics.append((np.asarray(info.diagnostics.innovation), np.asarray(info.diagnostics.innovation_covariance)))
        arrays = {"state": np.array([h[0] for h in history]), "covariance": np.array([h[1] for h in history]),
                  "active": np.array([h[0] for h in metadata]), "truth_ids": np.array([h[1] for h in metadata]),
                  "observation_status": np.array([h[2] for h in metadata]), "observation_slots": np.array([h[3] for h in metadata]),
                  "innovation": np.array([h[0] for h in diagnostics]), "innovation_covariance": np.array([h[1] for h in diagnostics])}
        if first is None:
            first = arrays
        elif not all(np.array_equal(first[k], arrays[k]) for k in first):
            raise AssertionError("Repeated Baseline B numerical replay was not exact")
        all_times.append(update_times)
    arrays = first
    active = arrays["active"][-1]
    birth_ids = arrays["truth_ids"][-1, active]
    map_estimate = arrays["state"][-1, 3:].reshape(capacity, 2)[active]
    metrics = accuracy_metrics(dataset.raw[:, 3:6], arrays["state"][:, :3], dataset.landmarks, map_estimate, birth_ids)
    metrics.update(covariance_metrics(arrays["state"], arrays["covariance"], arrays["active"], dataset.raw[:, 3:6]))
    statuses = arrays["observation_status"]
    correction = statuses == UPDATED
    e, S = arrays["innovation"][correction], arrays["innovation_covariance"][correction]
    nis = np.einsum('ni,ni->n', e, np.linalg.solve(S, e[..., None])[..., 0])
    used = np.asarray(mask)
    matches = arrays["observation_slots"][correction]
    matched_ids = np.broadcast_to(arrays["truth_ids"][:, None, :], (steps, slots, capacity))[correction]
    correct = matched_ids[np.arange(len(matches)), matches] == np.asarray(ids)[correction]
    metrics.update({"estimator": "B", "mode": mode, "capacity": capacity, "repeats": repeats,
                    "exact_replays": True, "observations": int(used.sum()),
                    "initialized": int(np.sum(statuses == INITIALIZED)), "updated": int(correction.sum()),
                    "invalid": int(np.sum(statuses == INVALID)), "overflow": int(np.sum(statuses == OVERFLOW)),
                    "association_correctness": float(np.mean(correct)),
                    "association_coverage": float(correction.sum()/used.sum()),
                    "used_observation_fraction": float(np.sum((statuses == INITIALIZED) | correction)/used.sum()),
                    "mean_nis": float(nis.mean()), "compile_seconds": compile_seconds,
                    "mean_update_ms": float(np.mean(all_times)*1000),
                    "p95_update_ms": float(np.percentile(all_times, 95)*1000),
                    "estimator_seconds_per_replay": np.sum(all_times, axis=1).tolist()})
    if (not np.isfinite(arrays["state"]).all() or not np.isfinite(arrays["covariance"]).all()
            or metrics["min_active_covariance_eigenvalue"] < -1e-10
            or metrics["max_active_covariance_asymmetry"] > 1e-10 or metrics["max_inactive_storage"] != 0):
        raise AssertionError("Baseline B replay failed numerical covariance/finite-value gates")
    arrays.update({"update_seconds": np.asarray(all_times), "command_phase_controls": np.asarray(controls),
                   "command_phase_durations": np.asarray(durations), "phase_Qc": np.asarray(intensities),
                   "used_observations": np.asarray(y), "observation_truth_ids": np.asarray(ids),
                   "observation_mask": np.asarray(mask), "time": (np.arange(steps)+1)*dt})
    return arrays, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/milestone1/dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/milestone3b2"))
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    dataset = Dataset.load(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
               "dataset": str(args.dataset), "jax": jax.__version__, "numpy": np.__version__,
               "python": sys.version, "platform": platform.platform(), "jax_enable_x64": True,
               "phase_duration_fractions": [.25, .5, .25], "config": dataset.config,
               "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                   ['models/slam.py', 'estimation/slam.py', 'benchmarks/estimator_b.py',
                    'tests/test_slam.py', 'tests/test_estimator_b.py']}, "cases": {}}
    for mode in ["original", "noisy"]:
        arrays, metrics = replay_dataset(dataset, mode, args.capacity, args.repeats)
        np.savez_compressed(args.output/f"{mode}_known.npz", **arrays,
                            metrics_json=np.array(json.dumps(metrics, sort_keys=True)))
        summary["cases"][mode] = metrics
        print(json.dumps(metrics, sort_keys=True), flush=True)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
