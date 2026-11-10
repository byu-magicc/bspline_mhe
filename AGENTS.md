# Codex Instructions — B-Spline Moving Horizon Estimation

## Source of truth and scope

Read `project.md` and `README.md` before starting work. `project.md` is authoritative for the research scope, mathematical notation, milestones, and acceptance criteria. If a task conflicts with it, explain the conflict and request a decision rather than silently changing the research plan.

Implement **only input-spline moving horizon estimation (MHE)** from the “Moving Horizon Estimation Using Input Splines” section (pp. 13–14) of *Application of Uniform B-splines to Planning, Control, and Estimation*. Do not implement spline MPC, output-spline MHE, differential-flatness MHE, or full-state trajectory MHE unless explicitly authorized.

The initial target is a **simulated EKF-SLAM comparison**. Victoria Park is strictly deferred until simulation validation gates pass.

## Required repositories and preservation rules

- Main repository: `byu-magicc/bspline_mhe`.
- Spline dependency: `byu-magicc/bsplines_library`, branch **`uniform_development`**, located as a sibling repository (`../bsplines_library`) in the intended local workspace.
- Reuse the existing C++/pybind11 `bspline_module.UniformBSpline(degree, M)` implementation and `basis_vector(s)`. Do **not** recreate the uniform B-spline basis in Python or JAX.
- Keep the original class-project **discrete-time EKF-SLAM** estimator **unchanged** as Baseline A. Add adapters, test harnesses, and instrumentation outside the original implementation. If a baseline change seems necessary, ask first and document it.
- Implement a separate **continuous-discrete JAX/RK4 EKF-SLAM** with zero estimated disturbance as Baseline B. The input-spline MHE is Estimator C and must reuse B's dynamics, numerical integration, measurement model, and EKF updates.
- Do not silently modify the sibling spline repository. Propose changes separately if a genuine library defect is found.

## Required three-estimator comparison

- **A:** Original discrete-time EKF-SLAM, preserved unchanged.
- **B:** Continuous-discrete EKF-SLAM using manually written JAX RK4 for continuous state/covariance propagation and discrete landmark updates; assume `w_v = w_omega = 0`, without spline optimization.
- **C:** Continuous-discrete input-spline MHE using B's EKF implementation, with spline disturbance coefficients optimized over the look-back window.
- Benchmark **A vs. B** to characterize effects of changing the numerical propagation/model, **B vs. C** to isolate the incremental contribution of spline-MHE, and **A vs. C** as the overall practical comparison.
- Use identical saved simulated truth, commands, landmark observations, and noise realizations for all three. Match initial conditions, measurement/association policies, and noise assumptions where possible. Explicitly derive/document any mapping between the original discrete process covariance and continuous process-noise intensity; do not equate them by default.
- Do not claim an improvement is caused by MHE solely because C beats A; B vs. C is the primary controlled comparison.

## Mathematical conventions

The initial continuous-time robot model is

```text
p_x_dot = (v + w_v) cos(theta)
p_y_dot = (v + w_v) sin(theta)
theta_dot = omega + w_omega
landmark_dot = 0
```

`v` and `omega` are known commanded inputs; `w_v` and `w_omega` are unknown disturbances.

At estimation time `t`, use look-back time `sigma in [0,L]` measured from `t-L`, with `alpha=M/L`:

```text
w(t-L+sigma) = [I_2 kron b_M^d(alpha*sigma)^T] c(t)
c(t) = [c_v(t); c_omega(t)] in R^(2(M+d)).
```

Use the **uniform, unclamped** basis convention:

```text
b_M^d(s) = [b^d(s+d), b^d(s+d-1), ..., b^d(s-M+1)]^T.
```

For fixed spline degree, segments, and RK4 evaluation times, evaluate the basis through the C++ binding and pass the resulting arrays to JAX as constants. Differentiation is with respect to `c`, not the basis evaluation times. Verify right-endpoint behavior at `s=M`.

For each candidate `c`, rerun the EKF over the window from a defined arrival estimate `(xhat(t-L), P(t-L))` and accumulate the **innovation-quadratic objective**:

```text
J_t(c) = sum_k r_k(c)^T S_k(c)^(-1) r_k(c)
c*(t) = argmin_c J_t(c).
```

This is **not** the full Gaussian innovation negative log-likelihood; do not add log-determinant terms without discussion.

Use the continuous-discrete EKF: continuous propagation of state and covariance between discrete landmark observations, and Joseph-form covariance updates. Handle heading wrap, landmark initialization, and SLAM gauge/reference-frame consistency.

## Numerical implementation

- **JAX** is mandatory for differentiating the EKF innovation objective with respect to spline coefficients.
- Write **RK4 manually in JAX**; do not use SciPy `solve_ivp` for the differentiable estimator.
- Integrate state and covariance as a coupled tuple/PyTree, evaluating dynamics, Jacobians, commanded input, and spline disturbance at **every RK4 stage**.
- Use `jax.grad` and optionally `jax.jit`; use SciPy for gradient-based coefficient optimization if appropriate.
- Avoid non-JAX NumPy operations and Python-side mutations inside differentiable/JIT-traced computation.
- Prefer stable linear solves over explicit inverses. Monitor covariance symmetry, numerical stability, and JAX precision settings.
- Confirm that the spline basis and stage times are handled correctly when time grids change.
- During each differentiable optimization, **hold data-association assignments fixed**; association choices are discrete.

## Development sequence — do not skip validation gates

1. **EKF-SLAM baseline:** inspect the existing simulation, document motion/measurement models and associations, run reproducibly with fixed seeds, and record errors and timing.
2. **Spline integration:** build the `uniform_development` pybind11 module; test dimensions, support, partition of unity, scaling, endpoint handling, and both disturbance channels.
3. **Differentiable EKF / Baseline B:** implement coupled JAX RK4, measurement updates, and numerical tests with zero disturbance and known associations; validate as a standalone estimator on the same replay datasets as Baseline A.
4. **Fixed-window input-spline optimization:** first use **known landmark locations** and injected known linear/angular disturbances; verify `jax.grad` against finite differences and compare reconstructed inputs with truth, not merely the innovation cost.
5. **Moving horizon:** shift measurement windows, warm-start spline coefficients, and implement consistent arrival-state/covariance handling without double counting measurements.
6. **EKF-SLAM benchmarks:** compare A (original discrete EKF), B (continuous-discrete zero-disturbance EKF), and C (input-spline MHE) on identical simulation datasets, first with known association and then nearest-neighbor association; report all three pairwise comparisons.

**Hard gate:** no Victoria Park integration or benchmarking before the simulated MHE and comparison benchmarks are validated.

## Required testing and benchmarking

- Use fixed random seeds for debugging and retain experiment configurations.
- Test RK4 order/step refinement and analytical/JAX Jacobians against finite differences.
- Check automatic-differentiation gradients against central finite differences with justified tolerances; report numerical precision and step sizes.
- Test zero, constant, smooth, and abrupt disturbances; increased measurement noise; and sparse measurements.
- Include at least one true disturbance outside the estimator's exact spline model family.
- Measure robot position/heading RMSE, landmark RMSE, separate `w_v` and `w_omega` RMSE, association correctness **and coverage**, optimizer iterations, gradient time, update-time mean/p95, and runtime.
- Separate JAX compilation cost from steady-state execution.
- **Monte Carlo trials are required** for final comparisons. Select trial count after profiling runtime and considering statistical precision; use paired data/noise realizations and report means, spread, and confidence intervals.
- Do not assert that MHE must outperform EKF-SLAM everywhere. Report negative results and accuracy–runtime tradeoffs.

## Deferred work and known issues

- **VP-001:** `vpSLAM.py` reportedly uses `tan(theta)` where the input Jacobian should use `tan(alpha)`. Reinspect the exact code and check finite differences before editing; defer until Victoria Park work is authorized.
- **Moving-window reassociation:** revisiting association decisions across the look-back window is a **future research extension**, not part of the initial implementation. Initial required comparisons use known and nearest-neighbor association with fixed assignments during optimization.

## Codex operating procedure

1. Before significant code changes, inspect relevant files and briefly state the proposed plan, assumptions, files to change, and tests.
2. Work on **one milestone or tightly scoped subtask at a time**. Avoid unrelated refactors and speculative features.
3. Prefer small, reviewable changes and meaningful commits/PRs. Do not overwrite user work or force-push.
4. Add tests with new functionality. Run the relevant tests and report exact commands, outcomes, and any unrun tests.
5. Do not fabricate results, benchmarks, or successful executions. Explicitly distinguish implemented, tested, and planned behavior.
6. When numerical results differ from theory, diagnose before relaxing tolerances or changing the model.
7. Ask for a decision when research choices are underspecified (e.g., arrival-cost policy, data-association policy, optimization regularization, or benchmark tolerances).
8. At the end of each task, summarize changes, tests, limitations, and the next validation gate.

## First assignment

Start with **Milestone 1 only**. Inspect and explain the existing **discrete-time** EKF-SLAM simulation (Baseline A), reproduce it with a fixed seed, and establish baseline metrics **without modifying the original estimation logic**. Baseline B is a separate Milestone 3 deliverable. Do not start spline-MHE implementation until this baseline is validated.
