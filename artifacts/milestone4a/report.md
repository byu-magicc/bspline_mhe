# Stage 4A — Fixed-Window Innovation Objective

## Scope and implementation

Stage 4A evaluates the innovation-quadratic objective and differentiates it with JAX. No coefficient optimizer or moving window is included. The known-landmark continuous-discrete EKF, its RK4 propagation, and the C++ uniform spline adapter are reused without changing their implementations.

For each coefficient vector, the same arrival state/covariance and fixed dataset are passed to the sequential EKF. At each valid prior update, the implementation adds `e.T @ solve(S, e)`; the bearing residual is wrapped by the existing measurement model. Masked records are excluded. Requested records whose geometry is invalid are counted separately and set `candidate_valid=false`; they are not silently accepted as a valid candidate. The two extra sums are `sum(logdet(S))` and `J + sum(logdet(S))`, diagnostic only. No regularization is present.

The evaluated objective is `J(c) = sum_k e_k(c).T @ solve(S_k(c), e_k(c))`. The true candidate is `[0.15 * ones(7); -0.05 * ones(7)]`; the zero candidate is a 14-vector of zeros.

### Files added or changed

- `estimation/fixed_window_objective.py` — fixed-data objective and diagnostic result.
- `simulation/continuous_unicycle.py` and `simulation/__init__.py` — independent analytic truth and seeded observation generation.
- `benchmarks/milestone4a.py` — reproducible run and report/JSON generation.
- `tests/test_fixed_window_objective.py` — objective, simulation, masking, invalid candidate, NumPy reference, JIT, and gradient validation.
- `artifacts/milestone4a/report.md` and `artifacts/milestone4a/results.json` — results and retained configuration.
- `project.md` — Stage 4A completion status and validation record.

## Simulation configuration

- Window: 5.0 s; RK4 step: 0.025 s; measurement interval: 0.1 s.
- Spline: degree `3`, `4` segments, `7` coefficients per channel, ordered `[c_v; c_omega]`.
- Commands `(v, omega)`: `[1.0, 0.2]`; true constant disturbance `(w_v, w_omega)`: `[0.15, -0.05]`.
- Seed: `7`. Truth uses the closed-form constant-input unicycle trajectory, independent of EKF integration.
- Arrival mean: `[0.0, 0.0, 0.0]`; arrival covariance: `[[0.0025000000000000005, 0.0, 0.0], [0.0, 0.0025000000000000005, 0.0], [0.0, 0.0, 0.0012184696791468343]]`.
- Measurement covariance (range, bearing): `[[0.0025000000000000005, 0.0], [0.0, 0.00030461741978670857]]`; continuous input-noise spectral density `Qc`: `[[0.01, 0.0], [0.0, 0.0025]]`. These are explicit experiment choices, not a claimed conversion from Baseline A noise.
- Landmark coordinates (metres):

```text
[5.0, 2.0]
[7.0, -4.0]
[1.0, 7.0]
[-4.0, 4.0]
[-5.0, -3.0]
[3.0, -7.0]
```

- Observation policy: all six fixed known landmarks at positive 0.1 s multiples; no observation at arrival time; each candidate has 300 requested records.
- Constant `c_true` reconstruction max absolute error, including both window endpoints: `5.551e-17`.

## Candidate diagnostics

| Candidate | J: innovation quadratic | Log-det diagnostic | Gaussian diagnostic J + log-det | Valid / requested | Min eigenvalue of S | Final position error |
|---|---:|---:|---:|---:|---:|---:|
| zero | 488.627026 | -4052.24325 | -3563.61622 | 300 / 300 | 3.626e-04 | 0.00945355 m |
| true | 470.362078 | -4051.8847 | -3581.52262 | 300 / 300 | 3.626e-04 | 0.00613201 m |

`J` is the Stage 4A objective. The Gaussian diagnostic is shown separately and is not optimized. All three sums use the same valid set. Innovation mean/std and innovation covariance summaries, final state/covariance, and wrapped pose errors are retained in `results.json`.

## Gradient validation

Central differences cover every coefficient in both channel blocks at zero, true-constant, and nonconstant vectors. Both step sizes are included. Relative error is reported over entries whose finite-difference magnitude is at least `1e-5`; this avoids treating near-zero components as meaningful relative-error denominators. The full vectors are retained in the machine-readable results.

| Coefficient vector | Step | Maximum absolute discrepancy | Maximum relative discrepancy (|FD| >= 1e-5) |
|---|---:|---:|---:|
| zero | 0.0001 | 1.880e-08 | 1.138e-08 |
| zero | 1e-05 | 1.870e-07 | 6.556e-08 |
| true | 0.0001 | 2.066e-08 | 2.587e-07 |
| true | 1e-05 | 1.804e-07 | 1.100e-06 |
| nonconstant | 0.0001 | 2.802e-08 | 9.976e-09 |
| nonconstant | 1e-05 | 1.761e-07 | 4.730e-08 |

The checks use JAX float64, central-difference steps `1e-4` and `1e-5`, and `jax.jit` for both the scalar objective/gradient and batched perturbation evaluations. The tests use `rtol=2e-5` and `atol=2e-5` for each coefficient; observed maximum absolute error is below `2e-7`, with the smaller finite-difference step showing the expected additional subtraction error. Since each perturbed evaluation recomputes the EKF and its innovation covariance, these checks include the coefficient dependence of `S`. Tests also verify JIT compatibility and that `S` changes under coefficient perturbations.

## Numerical consistency

The independent NumPy calculation from the recorded valid innovations and covariances agrees with both `J` and `sum(logdet(S))` to `rtol=2e-13`, `atol=2e-12`. All 300 requested innovation covariances are positive definite; the minimum eigenvalue is shown per candidate in the table. The constant-spline reconstruction agrees with the intended two-channel constant to `5.551e-17` maximum absolute error. The true and zero candidates are reproducible from seed 7 and the saved configuration.

## Validation commands and outcome

- `PYTHONPATH=build/bsplines_library python3 -B -m benchmarks.milestone4a --output artifacts/milestone4a` — completed; wrote this report and `results.json`.
- `PYTHONPATH=build/bsplines_library python3 -B -m pytest -q tests/test_fixed_window_objective.py` — **7 passed**.
- `PYTHONPATH=build/bsplines_library python3 -B -m pytest -q` — **205 passed**; existing milestone tests and Stage 4A tests.

The Stage 4A completion criterion in `project.md` is satisfied: the objective is reproducible, valid updates accumulate correctly against independent NumPy calculations, and JAX gradients agree with central differences across every coefficient in both blocks. This conclusion covers the documented fixed known-landmark experiment; coefficient optimization is outside this stage.

## Limitations

This is a fixed known-landmark, known-association pose filter. It does not optimize coefficients, augment landmark states, shift windows, or handle data association changes. Invalid candidate geometry is explicitly reported; gradients are intended for candidate regions with all requested records valid. The Gaussian diagnostic is an algebraic diagnostic without normalization constants and is not the prescribed objective.
