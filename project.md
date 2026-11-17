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

The augmented SLAM state is `x = [p_x, p_y, theta, ell_1^T, ..., ell_N^T]^T`. Discrete landmark measurements satisfy `y_k = h(x(t_k)) + nu_k`, with `nu_k ~ N(0, Sigma_k)`. Use the measurement model and noise conventions of the existing simulation, adapting its discrete motion model to the continuous-time benchmark consistently. Handle heading wrapping and SLAM reference-frame gauge freedom.

**Process-noise decision:** continuous-time process noise enters through the linear and angular velocity inputs. Let `eta = [eta_v, eta_omega]^T` be zero-mean input noise with continuous-time input-noise spectral density `Qc` (a positive semidefinite `2 x 2` matrix). The noise input Jacobian is

```math
G_r(\theta)=\begin{bmatrix}
\cos\theta & 0\\
\sin\theta & 0\\
0 & 1
\end{bmatrix},\qquad
G(x)=\begin{bmatrix}G_r(\theta)\\0_{2N\times2}\end{bmatrix}.
```

Use `G_r` for robot-only propagation and `G` for the augmented SLAM state; stationary landmarks have no direct process-noise input. The state-space noise intensity is `Xi(x) = G(x) Qc G(x)^T`. The spline disturbances `w_v` and `w_omega` describe unknown inputs in the mean dynamics, while `Qc` describes stochastic input uncertainty. `Qc` is not the original baseline's discrete process covariance; derive and document any mapping before comparing Baselines A and B.

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
\dot P=AP+PA^\top+GQ_cG^\top,\qquad
A=\frac{\partial f}{\partial x}(\hat x,u,w(c)).
```

Here `Q_c` denotes `Qc` from Section 2.1. Evaluate `A` and `G` at the propagated state at every RK4 stage, together with the commanded inputs and spline disturbances.

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

### Milestone 3 — Differentiable continuous-discrete EKF (Baseline B) (Completed)

The completion gate is satisfied for the saved simulation datasets with known data association. The independent **zero-disturbance EKF-SLAM baseline** is implemented and validated through Stages 3A, 3B.1, and 3B.2. Shared propagation supports spline disturbances for coefficient-gradient validation and later reuse by Estimator C; Baseline B replay runs use `w_v = w_omega = 0` without coefficient optimization.

#### Stage 3A — Differentiable Continuous-Time Propagation (Completed)

- Implement continuous-time unicycle dynamics for the robot pose, with linear and angular velocity disturbances represented using the existing uniform B-spline adapter and coefficient ordering `c = [c_v; c_omega]`.
- Use continuous-time process noise entering through the linear and angular velocity inputs, with covariance dynamics `P_dot = AP + PA.T + G_r Qc G_r.T` for the robot-only state, following Section 2.1.
- Implement coupled RK4 state and covariance propagation manually in JAX. Recompute dynamics, Jacobians, noise input Jacobian, commanded inputs, and spline disturbances at every stage using precomputed basis values for the stage times.
- Validate analytical/JAX Jacobians against finite differences, RK4 convergence against tractable analytical solutions and step-size refinement, covariance propagation, and automatic differentiation with respect to both disturbance coefficient blocks. Check coefficient gradients against central finite differences.
- Exclude measurement updates and landmark augmentation from Stage 3A.
- **Completed:** differentiable continuous-time state and covariance propagation using JAX and manually implemented RK4 is validated. See the [Stage 3A validation report](artifacts/milestone3a/report.md).

#### Stage 3B.1 — Differentiable EKF with Known Landmarks (Completed)

- The estimated state contains only robot pose `r = [p_x, p_y, theta]^T` and its `3 x 3` covariance. Landmark coordinates are known and fixed; landmarks are not part of the estimated state.
- Process range-bearing observations sequentially in their supplied deterministic order. Relinearize at the updated pose and covariance after every measurement; sequential results can differ from a stacked update.
- Reuse the validated Stage 3A JAX/RK4 propagation and Joseph-form covariance updates. Keep the internal heading unwrapped and wrap bearing innovations to `[-pi, pi)`.
- Support fixed observation slots with masks for JAX JIT/scan compatibility. Keep known landmark associations fixed during differentiation.
- Validate automatic differentiation through the complete EKF, including innovations, innovation covariances, Kalman gains, and final pose/covariance, against central finite differences for both B-spline coefficient blocks.
- **Completed:** all 19 Stage 3B.1 tests passed, along with all 161 Milestone 1, Milestone 2, and Stage 3A regression tests. See the [Stage 3B.1 validation report](artifacts/milestone3b1/report.md) for numerical results, interfaces, and limitations.

#### Stage 3B.2 — Differentiable EKF-SLAM with Unknown Landmarks (Completed)

- Extend the robot state with estimated 2D landmark positions, using a fixed-capacity augmented state with configurable capacity `N_max` (initial default 16) and an active-landmark mask so JAX JIT and scan retain static array shapes.
- Initially use known data association as a simulation oracle: supplied landmark truth IDs determine whether an observation updates an active landmark or initializes a new landmark. Use the IDs only for association; the estimator must use the measured range-bearing and its uncertainty, not landmark truth coordinates. Process multiple observations at a time in deterministic sequence, so the first occurrence of a new ID initializes a slot and a later occurrence can update it.
- Initialize a new landmark with the inverse range-bearing model `ell = [p_x, p_y]^T + range * [cos(theta + bearing), sin(theta + bearing)]^T`.
- Derive the initialization Jacobians with respect to the robot pose and the range-bearing measurement. Use them to compute the new landmark's covariance, its cross-covariance with the robot, and its cross-covariances with every previously active landmark. New measurement noise is independent of prior state uncertainty and other observations; general correlated range-bearing covariance within an observation is supported.
- Propagate the augmented state and full covariance using the continuous-time dynamics and input-noise covariance formulation established in Stage 3A. Stationary landmarks have zero direct dynamics and zero direct process-noise input.
- Use sequential range-bearing EKF updates and Joseph covariance updates for active known landmarks. Keep the association and landmark-initialization decisions fixed while differentiating.
- Preserve automatic differentiation with respect to both B-spline disturbance coefficient blocks through propagation, measurement updates, and landmark initialization for those fixed decisions.
- Ensure inactive landmark slots do not affect active-state estimates, innovations, covariance updates, or reported metrics. Initialize inactive state storage and covariance rows/columns neutrally, keep inactive cross-covariances zero, and mask inactive columns in measurement Jacobians. Inactive storage must not add artificial uncertainty or correlations to the active state.
- Validate covariance augmentation against independent numerical references. Check initialization Jacobians, landmark marginal covariance and cross-covariances, symmetry, and positive semidefiniteness on the active covariance subspace; check inactive storage separately. Compare coefficient gradients against central finite differences.
- Validate continuous-discrete EKF-SLAM as Baseline B on the same saved simulation datasets used for Baseline A, initially with known association and zero estimated disturbance.
- **Completed:** 14 SLAM tests, four replay tests, and 180 regression tests passed. Baseline B runs exactly reproducibly across two repeats on the saved seed-7, 500-step datasets in both clean (`original`) and noisy observation modes, with known associations and zero estimated disturbance. See the [Stage 3B.2 validation report](artifacts/milestone3b2/report.md) for numerical results, covariance checks, command/process-noise mapping, replay metrics, and runtime.

**Fixed-capacity covariance checks:** the active covariance subspace is the robot block together with the blocks for active landmarks. Evaluate its symmetry and eigenvalues separately from inactive storage. Inactive covariance rows/columns should remain zero, so zero eigenvalues in the padded full matrix are expected and must not be treated as negative active-subspace eigenvalues or artificial uncertainty.

**Implemented Stage 3B.2 policies:** capacity exhaustion preserves existing landmarks and returns an explicit overflow status from compiled code, with an informative exception at the Python boundary. Active-subspace covariance checks use maximum asymmetry `1e-10` and minimum eigenvalue at least `-1e-10`; inactive storage is checked separately for exact zero. Measurement-noise independence and the configurable default capacity are documented above.

- **Done when:** state/covariance propagation and updates pass numerical tests, produce stable estimates, and Baseline B can run on the same saved simulation datasets as Baseline A. Document process-noise discretization/continuous-intensity assumptions.

**Completion scope:** this gate covers the saved simulation datasets with known associations. Monte Carlo comparison benchmarks, nearest-neighbor association, MHE optimization, and Victoria Park validation remain future work under the later milestones and simulation gate.

### Milestone 4 — Fixed-window spline optimization

Implement fixed-window input-spline MHE in three stages, validating the objective before adding optimization and validating known-landmark disturbance reconstruction before full SLAM. Use the formulation in Section 2 unchanged. `Qc` remains the configured continuous-time stochastic input-noise spectral density; the deterministic spline coefficients represent the unknown mean disturbances and do not replace `Qc`.

#### Stage 4A — Fixed-Window Innovation Objective (Completed)

- Use a single fixed estimation window of length `L` and the existing Stage 3B.1 known-landmark EKF. Represent both input disturbances with the existing uniform, unclamped B-spline adapter, degree `d`, segment count `M`, and coefficient ordering `c = [c_v; c_omega]`.
- Keep the arrival state and covariance at the beginning of the window fixed for every candidate coefficient vector and during later optimization. Rerun the complete continuous-discrete EKF over the window for each candidate `c`.
- Define the Section 2.4 innovation-quadratic objective, with `e_k` denoting that section's innovation `r_k`:

  ```math
  J(c)=\sum_{k\in\mathcal K}e_k(c)^\top S_k(c)^{-1}e_k(c).
  ```

  where `K` is the requested measurement set fixed before candidate evaluation (masked records are excluded). Every requested update must be valid for the candidate to be in the objective's domain. Invalid candidates have explicit rejection status and an undefined objective; do not minimize a partial sum obtained by dropping their invalid records. Exclude masked/invalid placeholders from diagnostic statistics. Use stable linear solves. Do not add a log-determinant term or coefficient regularization in the initial implementation.
- Preserve sequential measurement updates and fixed known landmark associations. Differentiate through state propagation, covariance propagation, innovations, innovation covariances, Kalman gains, and measurement updates with JAX.
- Validate the objective against independent numerical calculations, verify correct accumulation of valid measurement contributions, and compare gradients for both coefficient blocks against central finite differences in float64.
- Stage 4A implements objective evaluation and differentiation only; coefficient optimization starts in Stage 4B.

**Initial validation scenario:** generate a new continuous-time unicycle simulation with known landmark positions and constant injected disturbances `w_v(t)=a`, `w_omega(t)=b`. Use directly specified continuous-time commanded linear and angular velocities, rather than the saved Baseline A rotation-translation-rotation commands. Retain fixed random seeds, the measurement-noise model and realization, and the experiment configuration for reproducibility.

**Stage 4A completion criterion:** the innovation objective is reproducible, correctly accumulates valid measurement contributions, and its JAX gradients agree with central finite differences for both coefficient blocks.

**Completion record:** the fixed known-landmark experiment is implemented and validated in [`artifacts/milestone4a/report.md`](artifacts/milestone4a/report.md), with machine-readable configuration and results in `artifacts/milestone4a/results.json`. The innovation quadratic and log-determinant calculations agree with an independent NumPy evaluation; all requested observation geometries are valid for the reported zero and true candidates; float64 JAX gradients agree with central differences across both coefficient blocks at zero, true-constant, and nonconstant vectors. The Stage 4A suite passed 7 tests and the complete suite passed 205 tests. Stage 4B optimization remained planned at that gate.

**Hardening record:** the fixed requested-set domain, checked Python interfaces, numerical node/posterior validation, independent nonconstant analytic references, gradient decomposition, validity-aware statistics, evidence-based reporting, and stable analytical truth are validated in [`artifacts/milestone4a_hardening/validation.md`](artifacts/milestone4a_hardening/validation.md). The complete hardening regression run passed 233 tests. See [`estimation/STAGE4A.md`](estimation/STAGE4A.md) for the valid-domain and future SciPy integration requirements. Stage 4B remained planned at that gate.

#### Stage 4B — B-Spline Coefficient Optimization (Controlled constant and smooth disturbance gates completed)

- Extend the validated Stage 4A objective using SciPy gradient-based optimization with JAX-computed objective values and gradients. Optimize both linear and angular velocity disturbance coefficient blocks while preserving the fixed arrival prior and known associations.
- Begin with constant injected disturbances, then extend to smooth time-varying disturbances. Use the zero-disturbance coefficient vector as the initial optimization guess for the first experiment.
- Compare the optimized objective with its zero-disturbance value. Compare reconstructed disturbance functions with the known injected disturbances using separate RMSE metrics for linear and angular velocity; also evaluate localization accuracy and optimizer convergence.
- Investigate coefficient identifiability using the Jacobian of the stacked innovation sequence with respect to the coefficients, including singular-value analysis when disturbance reconstruction is poor. Reduced innovation cost alone does not establish correct disturbance reconstruction.
- After the initial controlled tests, investigate spline degree, segment count, window length, and measurement frequency.
- Do not add regularization or alter the objective without first documenting the mathematical motivation and validation results.

**Stage 4B completion criterion:** the optimizer converges reliably in controlled tests and reconstructs identifiable linear and angular velocity disturbances, with documented limitations and failure cases.

**Original completion record and scope:** the controlled seed-7 known-landmark constant-disturbance gate is satisfied. Checked SciPy L-BFGS-B fits both constant-constrained and full 14-coefficient parameterizations from zero using exploratory bounds [-0.5,0.5], fixed requested observations/arrival prior, and the unchanged Stage 4A objective. Both fits repeat exactly and reconstruct both disturbance channels more accurately than assuming zero disturbance. Local raw/whitened innovation Jacobians have shape 600 x 14 and numerical rank 14 at zero and the fitted coefficients. The full spline has lower cost but higher reconstruction error than the constant fit; noiseless and tighter-tolerance controls document noise fitting and stopping limitations. All 248 tests passed, and 117 protected files matched preservation checks. See [`artifacts/milestone4b/report.md`](artifacts/milestone4b/report.md), [`artifacts/milestone4b/validation.md`](artifacts/milestone4b/validation.md), and [`estimation/STAGE4B.md`](estimation/STAGE4B.md). This original completion covered the specified constant-disturbance experiments; smooth disturbances, validation across seeds and Monte Carlo remained future extensions at that gate. Stage 4C remained planned.

**Smooth-disturbance extension completion:** a separate controlled seed-7 experiment injects `w_v=0.15*sin(2*pi*sigma/5)` and `w_omega=0.05*cos(2*pi*sigma/5)`, using independent DOP853 truth integration with step/tolerance refinement. All shared configuration, arrival conditions and observation policies match the constant experiment; zero, constant-constrained and full-spline comparisons use one immutable evaluator and identical new observations. The original Stage 4A objective, Stage 4B optimizer and constant artifacts remain unchanged. Full-spline linear/angular disturbance RMSE is approximately `[0.008807, 0.009198]`, compared with `[0.106012, 0.035438]` for the constant constraint; J decreases from `473.352181` to `465.425687`. Position RMSE improves modestly while heading RMSE worsens slightly. Independent sampled least-squares approximation, noiseless controls, covariance/residual audits, finite-difference gradients, repeat fits and raw/whitened sensitivities document representation error, noise fitting and weak boundary directions. All 270 repository tests passed and all 135 pre-existing protected files were preserved. See the [smooth report](artifacts/milestone4b_smooth/report.md), [validation record](artifacts/milestone4b_smooth/validation.md), and [interface notes](estimation/STAGE4B_SMOOTH.md) for the measured validation gate and current test evidence. Stage 4B is ready to transition to Stage 4C within this controlled scope; this extension does not establish reliability across seeds/noise scenarios or complete Milestone 4. At that gate, Stage 4C was unimplemented and its measurement-accounting decision remained pending; the Stage 4C completion record below supersedes that status.

#### Stage 4C — Fixed-Window MHE-SLAM (Controlled known-association validation completed)

- Extend the validated fixed-window optimization to the Stage 3B.2 augmented EKF-SLAM, reusing its fixed-capacity state, landmark initialization, covariance augmentation, and sequential Joseph measurement updates.
- Initially use known data association with fixed landmark truth IDs. Optimize the spline disturbance coefficients while rerunning the augmented EKF-SLAM over the fixed window.
- Keep the arrival state, arrival covariance, measurement sequence, association assignments, and initialization decisions fixed during each differentiable optimization. Preserve automatic differentiation through landmark initialization, propagation, covariance augmentation, and measurement updates for both coefficient blocks.
- Compare fixed-window MHE-SLAM against Baseline B on identical simulated truth, commanded inputs, and measurements. Report localization, landmark estimation, disturbance reconstruction, objective reduction, gradient accuracy, and computation time, separating compilation from steady-state execution.
- Stage 4C excludes moving-window shifting, arrival-cost updates, nearest-neighbor association, and Victoria Park integration. Moving-horizon management remains Milestone 5; comparative Monte Carlo and nearest-neighbor benchmarks remain Milestone 6.

**Stage 4C completion criterion:** the fixed-window spline optimizer runs successfully through the augmented EKF-SLAM and produces reproducible, interpretable estimation results with validated gradients.

**Completion record:** Configuration A was implemented and validated first (12 focused tests, paired repeat fits and all-coefficient finite differences), before Configuration B was implemented. Configuration B's constant experiment then passed the combined 20 focused tests before smooth cases were added. Both configurations now run reproducibly on the same saved seed-7 seven-second datasets, with the preserved Stage 3B.2 augmented EKF-SLAM primitives, Stage 4A checked domain policy and Stage 4B coefficient optimizer. All 295 repository tests passed, including 25 new Stage 4C tests; all 163 pre-existing protected files and earlier artifacts are unchanged. See the [Stage 4C report](artifacts/milestone4c/report.md), [validation record](artifacts/milestone4c/validation.md), and [interface notes](estimation/STAGE4C.md).

**Resolved fixed-window arrival/initialization policy:** Configuration A freezes the complete posterior after observations through t0=2.0 s, with all six landmarks active, and processes only observations strictly after t0 (300 corrections). Configuration B independently propagates the original robot prior and Qc using commanded inputs and no observations to t0, with an empty map; it does not reuse A's posterior robot marginal. Its six observations at 2.1 s initialize the map and the remaining 294 observations correct it. Each first observation enters covariance augmentation once and never the innovation objective; subsequent same-timestamp observations can correct normally. ID/slot/birth/update decisions are frozen before differentiation, while initialization means, covariances and cross-correlations remain differentiable. The original pose prior anchors the reference frame; metrics use no post-hoc map alignment.

**Measured outcome and limits:** both constant cases lower J and improve disturbance, pose and landmark RMSE in this controlled realization. Smooth cases improve disturbance reconstruction and lower J, but the empty-map Configuration B gives worse pose and landmark RMSE than its paired zero-disturbance Baseline B. Conditional EKF covariance figures exclude fitted-coefficient uncertainty. Active coefficient bounds, weak directions, map/pose coupling, finite noisy data and function-reduction stopping limit inference; no global-optimality or statistical-reliability claim is made. Stage 4C's controlled implementation criterion is satisfied, not the final comparative benchmark gate. Before Milestone 5, choose sliding-arrival information handling, map-birth accounting across boundaries and coefficient shift/warm-start policies. These are not implemented here. Monte Carlo comparisons, nearest-neighbor association and Victoria Park remain deferred.

**Stage 4 design questions to resolve before the affected implementation:** select the initial continuous-time experiment configuration (`L,d,M,a,b`, commands, landmark geometry, arrival prior, `Qc`, measurement covariance, and observation times); define how coefficient candidates that change geometric measurement validity are handled while retaining the valid-update-only objective; and document Stage 4C window-boundary/initialization measurement accounting so the arrival prior, landmark initialization, and innovation cost do not reuse information inadvertently. These choices must not silently alter the Section 2.4 objective or introduce regularization.

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
