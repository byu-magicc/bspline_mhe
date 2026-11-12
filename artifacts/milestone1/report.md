# Milestone 1 — Estimator A reproduction

Validation status: **PASS** for this fixed-seed simulation/replay gate.

Seed 7; 500 steps; dt=0.1 s; float64 arithmetic after legacy integer initialization; batch updates; three estimator replays per case are the default.

Actual repeats per case: 3. Each repeat reloads `dataset.npz` from disk. All numerical state, covariance, map, observation and association arrays must be exactly equal (NaN padding equal). Timings are excluded from equality.

## Results

Errors use the noisy executed trajectory, never the nominal path. Position/map units are the simulator's coordinate units; heading is radians. Map RMSE is final-map error against each estimate's initialization (birth) truth ID, including duplicate tracks. No fitted frame alignment is applied: truth and estimate share the same anchored initial pose. IDs are evaluation labels only for NN.

| Observation mode | Association | Position RMSE | Heading RMSE (rad) | Map RMSE | Map tracks / unique IDs | Association accuracy | Coverage | Mean / p95 update (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original | known | 2.880460 | 0.030822 | 1.192448 | 8 / 8 | 100.00% | 99.20% | 0.085 / 0.105 |
| original | nn | 2.880460 | 0.030822 | 1.192448 | 8 / 8 | 100.00% | 99.20% | 0.274 / 0.298 |
| noisy | known | 9.446002 | 0.054444 | 15.484536 | 8 / 8 | 100.00% | 99.20% | 0.080 / 0.087 |
| noisy | nn | 25.719168 | 0.090295 | 50.511131 | 48 / 8 | 95.48% | 95.20% | 1.048 / 1.793 |

Timing columns pool all repeated estimator updates, including the first update (no warm-up removed). They include motion-noise construction, prediction, association, correction and map augmentation. Imports, observation adaptation, diagnostics, history copies, metrics, plotting, printing, generation and file I/O are outside the timer. Exact timings vary with machine load and are not deterministic. Per-run timings and separate generation duration are retained in JSON/NPZ.

Association accuracy = correct existing-track matches / attempted existing-track matches. Coverage = attempted existing-track matches / all observations. New landmarks are excluded from accuracy and count as unmatched for this definition of coverage. Counts of new, duplicate, incorrect and rejected decisions, plus the fraction used for either correction or initialization, are in `summary.json`. Birth-ID scoring cannot certify a track remains unmixed after incorrect associations; NN map error should be read with these counts.

## Implementation and preserved behavior

The external harness calls the original `predict`, association, `update`, and `augmentState` functions. The joint state is robot pose plus stationary 2D landmarks. Prediction uses rotate–translate–rotate odometry with command-dependent Gaussian increment covariance. Observations are range and relative bearing with wrapped bearing innovations. Existing landmarks correct the joint state; new landmarks use inverse range–bearing geometry and full covariance/cross-covariance augmentation. Known association uses IDs; NN selects the smallest squared Mahalanobis distance below 4, without enforcing one-to-one matches.

The original source is unchanged, including integer initial pose, input-view mutation in `g`, explicit inverses, simplified covariance correction, and lack of heading wrapping after correction. The first prediction copies into an integer state array; its truncation is preserved. Sequential updates also retain their precomputed innovations/Jacobians. These are properties of A, not changes to make it resemble future B/C.

The generator uses fixed legacy `np.random.seed`; its caller RNG state is restored afterward. All original draws, including zero-variance draws for nominal truth, are retained. Only its per-step partial-file writes are suppressed; the completed raw array is saved once. Qt backend selection is intercepted at import for headless Agg plotting. These hooks alter I/O only. Automated parity tests execute the original `simSLAM.run` control flow with plotting/printing/writes intercepted and compare every captured state/covariance and association. They cover both observation modes, both association methods and both update methods; the noisy runner input is adapted independently of the harness observation selector.

## Observation mismatch and separate adapter

Despite comments/docstrings, generated slots are `[ID, clean range, clean bearing, noisy range, noisy bearing]`. The original runner consumes the clean pair while assuming range standard deviation 10 and bearing standard deviation 10 degrees. `original` is the original reproduction. `noisy` is the corrected observation replay benchmark: it selects the stored noisy pair without changing any EKF logic or nominal covariance assumptions. Both use identical commands, truth and observation opportunities from one saved dataset. They must not be merged into a single baseline result.

## Files and reproduction

- `dataset.npz`: untouched raw layout plus explicit commands, executed/nominal truth, timestamps, true map, clean/noisy measurements, IDs, seed, configuration, versions and source SHA-256 hashes.
- `<mode>_<association>.npz`: complete padded state/covariance histories, pose/map estimates, landmark birth IDs, observations consumed, associations and update timings. State dimensions identify active entries; NaN means uninitialized and association -3 means absent observation slot (-1=new, -2=ignore).
- `summary.json`: validation checks, source hashes, environment, per-repeat metrics and pooled timing metrics.
- `trajectories_known.png`, `trajectories_nn.png`: executed truth, estimates, and true/estimated maps.

From the repository root:

```sh
python3 -B -m benchmarks.estimator_a --seed 7 --steps 500 --repeats 3 --output artifacts/milestone1
python3 -B -m benchmarks.estimator_a --dataset artifacts/milestone1/dataset.npz --repeats 3 --output /tmp/estimator_a_replay
python3 -B -m pytest -q tests/test_estimator_a.py
```

## Validation and limitations

Validation checks: `{"covariance_roundoff_checks": true, "finite_active_states": true, "generation_source_matches_current_legacy": true, "repeated_saved_replay_exact": true, "source_files_unchanged": true}`.

The covariance diagnostic tolerance is 1e-10 relative to max(1, spectral norm), applied to asymmetry and negative eigenvalues of the symmetrized covariance; it is a numerical roundoff check, not an accuracy target. Accuracy has no imposed pass threshold in Milestone 1. Source preservation and repeated numerical replay are required independently of performance.

The original runner cannot handle zero-observation steps; the harness explicitly refuses such data instead of silently changing its prediction-only policy. Legacy double-gated association can drop ambiguous output entries, and JCBB is unfinished with a syntax error; neither is used. Generator cache load/save directories disagree, so the harness uses its own explicit dataset path. The legacy association-history array has fixed dimensions; replay stores histories sized to this experiment.

These are single-seed baseline results, not a Monte Carlo comparison or evidence of MHE benefit. Large/noisy NN failures must be reported rather than tuned away. No continuous process-noise mapping is assumed here. Estimators B/C, spline integration, and Victoria Park remain unimplemented by this task. The next gate is Milestone 2 spline-library validation; B remains Milestone 3.

## Recorded validation execution

`python3 -B -m pytest -q tests/test_estimator_a.py`: **21 passed in 11.83s**. This includes runner parity in both observation modes, known/NN association and batch/sequential correction; fixed-seed generation and RNG restoration; stored-data replay; metric definitions; float64 motion/measurement Jacobians checked by central differences (step 1e-6, rtol 1e-7, atol 2e-9); covariance diagnostics; unsupported empty observations; and end-to-end artifacts.

The delivered 500-step experiment passed all five runtime validation checks. A second Python process replayed the same saved dataset three times per case; every nontiming numerical array exactly matched the delivered artifacts for all four cases. The copied dataset archive SHA-256 was identical. `git diff --exit-code -- baseline/ekf_slam` returned 0. Commands and audit results are saved in [validation.json](validation.json). All requested Milestone 1 fixed-seed reproduction criteria are satisfied; final-comparison Monte Carlo trials remain deferred.
