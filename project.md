# B-Spline Moving Horizon Estimation

**Status:** Research and implementation plan (Sections 1–5 agreed, November 2026)

## 1. Project Overview

### Objective
Implement and evaluate continuous-time moving horizon estimation (MHE) in which unknown robot inputs are represented by uniform, unclamped B-splines. Follow only **“Moving Horizon Estimation Using Input Splines”** (pp. 13–14) of *Application of Uniform B-splines to Planning, Control, and Estimation*.

### Primary goals
1. Implement continuous-time state and covariance propagation with discrete EKF measurement updates.
2. Reuse the C++/pybind11 `UniformBSpline` implementation from [`byu-magicc/bsplines_library`, branch `uniform_development`](https://github.com/byu-magicc/bsplines_library/tree/uniform_development).
3. Preserve the original **discrete-time EKF-SLAM** implementation as Baseline A; implement a separate **continuous-discrete JAX/RK4 EKF-SLAM** with zero estimated disturbance as Baseline B, before adding input-spline MHE as Estimator C.
4. Optimize uniform B-spline coefficients of unknown linear and angular velocity disturbances using an EKF innovation objective over a look-back window.
5. **Automatically differentiate the complete EKF innovation objective with respect to the spline control points** using JAX.
6. Quantify robot localization, landmark mapping, disturbance reconstruction, data association, and computation time.
7. **Validate simulation before any Victoria Park experiments.**

### Estimation architecture and scope
For each candidate spline coefficient vector, rerun a continuous-discrete EKF over the window, propagating the robot and landmark state and covariance and accumulating innovations. Optimize the disturbance coefficients; do **not** introduce a separate full-state-trajectory MHE optimization at this stage. Maintain a consistent arrival state and covariance at the beginning of each window.

**In scope:** input-spline MHE, uniform unclamped splines, original discrete-time EKF-SLAM (Baseline A), continuous-discrete JAX/RK4 EKF-SLAM (Baseline B), known and nearest-neighbor data association, automatic differentiation, simulation and benchmarks.

**Out of scope:** spline MPC, trajectory planning, output-spline MHE, differential flatness, full nonlinear MHE-SLAM, and window-wide reassociation in the initial implementation.

## 2. Mathematical Formulation

### 2.0 Three-estimator comparison

- **A — Original discrete-time EKF-SLAM:** preserve the existing class-project estimator and its discrete prediction/covariance model without changing its estimation logic.
- **B — Continuous-discrete EKF-SLAM:** implement a separate JAX/RK4 estimator using the continuous-time dynamics and continuous covariance propagation below, with `w_v = w_omega = 0` (no spline optimization).
- **C — Continuous-discrete input-spline MHE-SLAM:** reuse B's dynamics, integration, measurement model, and EKF updates; optimize the spline coefficients for unknown `w_v` and `w_omega` over each look-back window.

**Interpretation:** A vs. C is the practical original-baseline comparison; **B vs. C is the primary comparison for isolating the contribution of input-spline disturbance estimation and moving-window optimization** from changes in time discretization and implementation. Keep measurement opportunities, simulated truth, and observation noise paired. Compare A vs. B separately to quantify effects of changing the prediction model and numerical integration. Align or explicitly document differences in discrete process-noise covariance versus continuous-time noise spectral density; they are not interchangeable.

### 2.1 Continuous-time SLAM dynamics
For robot pose `r = [p_x, p_y, theta]^T`, commanded speed `v` and turn rate `omega`, and unknown disturbances `w_v`, `w_omega`:

```math
\begin{aligned}
\dot p_x &= (v+w_v)\cos\theta,\\
\dot p_y &= (v+w_v)\sin\theta,\\
\dot\theta &= \omega+w_\omega,\\
\dot\ell_j &= 0 \quad \text{(stationary landmarks)}.
\end{aligned}
```

The augmented SLAM state is `x = [p_x, p_y, theta, ell_1^T, ..., ell_N^T]^T`. Include additive continuous-time process noise with spectral density `Xi` as appropriate to the modeled states. Discrete landmark measurements satisfy `y_k = h(x(t_k)) + nu_k`, with `nu_k ~ N(0, Sigma_k)`. Use the measurement model and noise conventions of the existing simulation, adapting its discrete motion model to the continuous-time benchmark consistently. Handle heading wrapping and SLAM reference-frame gauge freedom.

### 2.2 Unknown-input spline on a look-back window
At estimation time `t`, use `sigma in [0,L]` as elapsed time from `t-L` and `alpha = M/L`. With two disturbance channels,

```math
w(t-L+\sigma) = \left[I_2\otimes \boldsymbol b_M^{d\top}(\alpha\sigma)\right]c(t),
\qquad c(t)=\begin{bmatrix}c_v(t)\\c_\omega(t)\end{bmatrix}\in\mathbb R^{2(M+d)}.
```

Use the canonical uniform, unclamped basis `b_M^d(sigma) = [b^d(sigma+d), ..., b^d(sigma-M+1)]^T` and `M+d` coefficients per channel. Confirm numerical behavior at the right endpoint `alpha*sigma=M`.

### 2.3 Continuous-discrete EKF over each window
For fixed `c`, initialize `(xhat(t-L), P(t-L))` from the arrival estimate. Between measurement times integrate

```math
\dot{\hat x}=f(\hat x,u,w(c)),\qquad
\dot P=AP+PA^\top+\Xi,\qquad
A=\frac{\partial f}{\partial x}(\hat x,u,w(c)).
```

At measurement `k`, compute

```math
r_k=y_k-h(\hat x_k^-),\quad H_k=\frac{\partial h}{\partial x}(\hat x_k^-),\quad
S_k=H_kP_k^-H_k^\top+\Sigma_k,\quad
K_k=P_k^-H_k^\top S_k^{-1}.
```

Then update

```math
\hat x_k^+=\hat x_k^-+K_kr_k,\qquad
P_k^+=(I-K_kH_k)P_k^-(I-K_kH_k)^\top+K_k\Sigma_kK_k^\top.
```

Use appropriate association-specific measurement subsets and consistent landmark initialization. For the first known-landmark identification test, landmark positions are fixed and known; full SLAM map augmentation follows later.

### 2.4 Innovation objective and optimization
Following the paper's proposed innovation-quadratic cost (rather than a full Gaussian negative log-likelihood), minimize

```math
J_t(c)=\sum_{t_k\in[t-L,t]}r_k(c)^\top S_k(c)^{-1}r_k(c),\qquad
c^*(t)=\arg\min_c J_t(c).
```

The EKF must be rerun for each candidate `c`; innovations and innovation covariances depend on `c`. Use JAX to compute `grad_c J_t(c)` through integration, covariance propagation, updates and cost accumulation. Verify gradients against finite differences. Keep association assignments fixed during each differentiable optimization; association selection is discrete and is not differentiated.

## 3. Software Architecture and Dependencies

### Core stack
- **Python / NumPy:** simulation, data handling and analysis.
- **JAX:** differentiable EKF, `jax.grad`, optional `jax.jit` and scan-based propagation.
- **SciPy:** gradient-based coefficient optimization, using JAX-computed objective gradients.
- **C++ / pybind11 `bsplines_library`:** basis evaluation; require the `uniform_development` branch, **not** `main`.
- **Matplotlib:** trajectories, maps, disturbance plots and benchmark figures.

### Integration rules
- Use the existing `bspline_module.UniformBSpline(d, M).basis_vector(s)` Python binding; **do not reimplement the uniform basis in JAX or Python**.
- Precompute basis evaluations at fixed integration stage times using the C++ library, then convert them to JAX arrays. The basis values are independent of the optimized coefficients, so differentiation with respect to `c` remains valid. If integration times vary, provide a compatible basis-evaluation strategy without silently breaking differentiation.
- Implement **RK4 manually in JAX**, not SciPy `solve_ivp`, for the differentiable EKF. Integrate `(xhat, P)` as a coupled tuple/PyTree; recompute dynamics, Jacobians, commanded inputs and spline disturbances at each RK4 stage.
- Preserve the original discrete EKF-SLAM source unchanged as Baseline A. Implement Baseline B (continuous-discrete JAX/RK4 EKF, zero disturbance) separately and reuse its tested propagation/update code for Estimator C (input-spline MHE). All three receive identical simulated truth and measurements in paired comparisons.
- Use stable linear solves instead of explicit matrix inversion where practical. Check covariance symmetry and numerical consistency.
- Initially use known associations; later add standard nearest-neighbor associations for both estimators.

### Suggested module boundaries (not mandatory filenames)
`baseline/` (original EKF-SLAM), `models/` (continuous robot and measurements), `splines/` (C++ binding adapter and basis cache), `estimation/` (JAX EKF and MHE optimizer), `simulation/` (shared truth and measurements), `benchmarks/` (metrics and experiments), `tests/` (basis, RK4, EKF and gradients).

## 4. Implementation Milestones and Validation Gates

### Milestone 1 — Reproduce EKF-SLAM baseline
- Review existing simulated motion, landmark observation, covariance and data association code.
- Run with fixed seeds; record true and estimated robot trajectories, landmark estimates, associations and timing.
- **Done when:** results and metrics are reproducible without changing baseline estimation logic.

### Milestone 2 — Integrate `bsplines_library`
- Build the `uniform_development` C++/pybind11 extension and verify the Python API.
- Test `M+d` output dimensions, finite support, partition of unity, time scaling, endpoint handling, and reconstruction of both disturbance channels.
- **Done when:** known spline coefficients reproduce both injected input functions, and basis arrays work in JAX coefficient differentiation.

### Milestone 3 — Differentiable continuous-discrete EKF (Baseline B)
- Implement continuous robot/landmark dynamics, coupled RK4 propagation of state and covariance, and discrete landmark updates in JAX as an independent **zero-disturbance EKF-SLAM baseline**.
- Validate RK4 against tractable analytical solutions and step-size refinement; check Jacobians against finite differences.
- Start with zero disturbance and known associations.
- **Done when:** state/covariance propagation and updates pass numerical tests, produce stable estimates, and Baseline B can run on the same saved simulation datasets as Baseline A. Document process-noise discretization/continuous-intensity assumptions.

### Milestone 4 — Fixed-window spline optimization
- For a single look-back window, rerun the EKF and minimize the innovation objective over spline coefficients.
- **Required intermediate test:** use **known landmark locations** and known injected linear/angular disturbances before full SLAM. Compare reconstructed disturbances against ground truth, not just reduced innovation cost.
- Verify `jax.grad` against finite differences, and investigate observability/identifiability if reconstruction fails.
- **Done when:** the optimizer converges reliably and reconstructs identifiable disturbances in controlled tests.

### Milestone 5 — Moving-horizon implementation
- Slide the window as measurements arrive; manage buffered measurements and shift/warm-start spline coefficients.
- Define a consistent method for carrying forward the arrival estimate and covariance at `t-L` (avoid inadvertently reusing information twice).
- **Done when:** full simulated runs are stable, with interpretable state and disturbance estimates and no numerical failures.

### Milestone 6 — EKF-SLAM comparison
- Compare **all three** on identical simulation datasets: A (original discrete EKF-SLAM), B (continuous-discrete zero-disturbance JAX EKF-SLAM), and C (continuous-discrete input-spline MHE). Report A vs. B, B vs. C, and A vs. C; treat B vs. C as the principal isolation of MHE's incremental effect.
- First use **known** data association; then use **nearest-neighbor** data association in both estimators.
- **Done when:** accuracy, association outcomes, disturbance reconstruction, and runtime are reproducibly reported, including scenarios in which MHE does not improve performance.

**Hard gate:** Do not begin Victoria Park integration until simulated MHE and its benchmarks have passed validation.

## 5. Benchmark Design and Success Criteria

### Scenarios
1. Zero disturbance.
2. Constant linear/angular velocity biases.
3. Smooth time-varying disturbances (sinusoids or spline-generated).
4. Abrupt/piecewise disturbances.
5. Increased measurement noise.
6. Sparse or infrequent landmark measurements.

Include at least one test where the true disturbance is **not** generated using the exact same spline degree and knot spacing assumed by the estimator. Start with known association and repeat selected scenarios with nearest-neighbor association.

### Required metrics
- **Robot position RMSE:** `sqrt(mean(||p_hat_k - p_true_k||^2))`.
- **Robot heading RMSE:** `sqrt(mean(wrap(theta_hat_k-theta_true_k)^2))`.
- **Landmark position RMSE:** over appropriately matched/initialized landmarks in a consistent SLAM reference frame; also report mapped-landmark counts.
- **Disturbance RMSE:** separately for `w_v` and `w_omega`, using known simulated truth.
- **Association quality:** correct associations / attempted associations, plus incorrect, rejected and unmatched measurements; do not report accuracy alone without coverage.
- **Runtime:** mean and 95th-percentile update time, optimization iterations, gradient time, total runtime, and JAX compile time separately from steady-state execution.
- **Sensitivity:** sweep spline degree `d`, segment count `M`, horizon length `L`, and relevant noise/visibility parameters.

### Monte Carlo protocol — required
- Debug first with a single fixed random seed.
- Profile runtime before selecting trial count; choose the number of Monte Carlo trials based on computational cost **and desired statistical precision**, not an arbitrary fixed count.
- Use paired experiments: the same ground truth, disturbances, measurement noise realizations and observation opportunities for both estimators.
- Report means, standard deviations and confidence intervals of key metrics across trials; retain seeds and configurations for reproducibility.

### Success criteria
- All three estimators run reproducibly on identical simulated datasets.
- The input-spline MHE reconstructs identifiable disturbances better than assuming zero disturbance.
- Evaluate whether C improves localization relative to **B** in meaningful model-mismatch scenarios, while also reporting comparison with A; do **not** require it to win in all scenarios or treat improvement over A alone as evidence that the spline-MHE component caused the gain.
- JAX gradients match finite-difference checks within justified numerical tolerances.
- Both known- and nearest-neighbor-association comparisons are complete.
- Accuracy–runtime tradeoffs and failure cases are documented.

## 6. Deferred Issues and Research Extensions

### VP-001 — Victoria Park input Jacobian (deferred)
**File:** `vpSLAM.py`, previously identified near line 255. The input Jacobian reportedly uses `tan(theta)` where differentiation of the vehicle heading dynamics with respect to velocity requires `tan(alpha)` (steering angle). **Status:** previously identified inconsistency; recheck against the source before editing and test its numerical effect.

**Before Victoria Park benchmarking:** inspect the motion model and Jacobians, correct as needed, check derivatives against finite differences, rerun the original EKF-SLAM baseline, and document the effect. This must not block simulation work.

### Moving-window data association (future research, not a completion gate)
Investigate allowing MHE to reconsider landmark associations within the look-back window rather than permanently committing to nearest-neighbor decisions. This introduces discrete association hypotheses and is **not** included in the initial differentiable coefficient optimization. Consider hypothesis generation, fixed-association optimization for each hypothesis, and comparison of window costs only after the core input-spline MHE is validated.

## 7. Source Material and Repositories

- *Application of Uniform B-splines to Planning, Control, and Estimation*, **“Moving Horizon Estimation Using Input Splines”**, pp. 13–14; Algorithm 1.
- [byu-magicc/bsplines_library — `uniform_development`](https://github.com/byu-magicc/bsplines_library/tree/uniform_development), including `uniform_bspline.cpp`, `uniform_bspline.hpp`, `wrapper.cpp`, and `CMakeLists.txt`.
- Existing uploaded class-project EKF-SLAM implementation (`lab5-ekf-slam.zip`): simulated SLAM baseline and Victoria Park code. Keep baseline source unchanged.
