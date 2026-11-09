# B-Spline Moving Horizon Estimation

Continuous-time moving horizon estimation (MHE) using uniform B-splines to estimate unknown inputs in nonlinear dynamical systems.

This repository develops and evaluates an input-spline MHE algorithm using an Extended Kalman Filter (EKF) and automatic differentiation. The primary application is robot localization and simultaneous localization and mapping (SLAM), with comparisons against conventional EKF-SLAM.

## Overview

The objective is to investigate whether representing unknown disturbances as uniform B-splines and optimizing their coefficients over a moving measurement window improves state estimation compared to conventional EKF-SLAM.

The approach follows the **"Moving Horizon Estimation Using Input Splines"** section of *Application of Uniform B-splines to Planning, Control, and Estimation* by Curtis P. Evans, Alejandro Pelaez, and Randal W. Beard.

The estimator uses:

- Continuous-time nonlinear system dynamics.
- Uniform, unclamped B-splines to parameterize unknown inputs.
- Continuous-time EKF state and covariance propagation.
- Discrete landmark measurement updates.
- Innovation-based optimization over a moving estimation window.
- JAX automatic differentiation with respect to the B-spline control points.

Unlike full nonlinear MHE, this approach optimizes the unknown input trajectory while retaining an EKF-based state estimation structure.

## Mathematical Formulation

Consider the nonlinear system

$$
\dot{x}(t)=f(x(t),u(t),w(t))+\xi(t),
$$

$$
y_k=h(x(t_k))+\nu_k,
$$

where $x(t)$ is the system state, $u(t)$ is the known commanded input, $w(t)$ is an unknown disturbance, and $\xi(t)$ and $\nu_k$ represent process and measurement noise.

Over the estimation window $[t-L,t]$, the unknown disturbance is parameterized as

$$
w(t-L+\sigma)=\left[I_q\otimes\boldsymbol{b}_M^{d\top}(\alpha\sigma)\right]c(t),
$$

where $\sigma\in[0,L]$, $\alpha=M/L$, $d$ is the B-spline degree, $M$ is the number of spline segments, and $c(t)$ contains the spline coefficients.

The coefficients are selected by minimizing the EKF innovation cost over the estimation window:

$$
c^\star(t)=\arg\min_c J_t(c),
$$

$$
J_t(c)=\sum_{t_k\in[t-L,t]}r_k(c)^\top S_k(c)^{-1}r_k(c),
$$

where $r_k(c)$ and $S_k(c)$ are the innovation residual and innovation covariance obtained by propagating the EKF using the candidate disturbance trajectory.

JAX is used to compute the gradient $\nabla_cJ_t(c)$ through the EKF.

## Robot Model

The initial simulation uses a planar unicycle model with unknown linear and angular velocity disturbances:

$$
\begin{aligned}
\dot p_x &= (v+w_v)\cos\theta,\\
\dot p_y &= (v+w_v)\sin\theta,\\
\dot\theta &= \omega+w_\omega.
\end{aligned}
$$

The SLAM state additionally contains stationary landmark positions.

The MHE will estimate both $w_v(t)$ and $w_\omega(t)$ using uniform B-splines.

## Dependencies

The planned implementation uses:

- **Python / NumPy** for numerical computations and simulation.
- **JAX** for automatic differentiation and differentiable EKF propagation.
- **SciPy** for gradient-based optimization.
- **C++ / pybind11** for B-spline basis evaluation.
- **Matplotlib** for visualization and benchmarking.

### B-Spline Library

Uniform B-spline basis functions are provided by the existing [byu-magicc/bsplines_library](https://github.com/byu-magicc/bsplines_library/tree/uniform_development).

**Required branch:** `uniform_development`

The project reuses the existing `UniformBSpline` C++ implementation and Python bindings rather than reimplementing the basis functions.

## Implementation Roadmap

The project is organized into six milestones:

1. **EKF-SLAM baseline:** Reproduce and validate the existing EKF-SLAM simulation.
2. **B-spline integration:** Integrate the C++/pybind11 library and verify disturbance reconstruction.
3. **Differentiable EKF:** Implement continuous-time state and covariance propagation using RK4 in JAX.
4. **Fixed-window optimization:** Estimate spline coefficients by minimizing the EKF innovation cost, initially using known landmark positions.
5. **Moving-horizon estimation:** Implement window shifting, coefficient warm-starting, and arrival-state initialization.
6. **Benchmarking:** Compare B-spline MHE against EKF-SLAM using known and nearest-neighbor data association.

The simulation must be validated before attempting experiments with the Victoria Park dataset.

## Benchmarking

The evaluation will compare:

- Robot position and heading estimation errors.
- Landmark position estimation errors.
- Linear and angular velocity disturbance reconstruction.
- Known and nearest-neighbor data association performance.
- Computational cost and optimization convergence.
- Sensitivity to spline degree, segment count, and estimation horizon length.

Monte Carlo experiments will be used to evaluate statistical performance across different disturbance and measurement-noise scenarios.

## Research Extensions

Potential future work includes moving-window data association, in which landmark associations within the estimation window can be reconsidered using measurement history.

This is not part of the initial input-spline MHE implementation.

## Project Status

**Status: Planning complete — implementation pending.**

The initial development priority is reproducing the existing EKF-SLAM simulation and establishing a validated baseline.

## References

1. C. P. Evans, A. Pelaez, and R. W. Beard, *Application of Uniform B-splines to Planning, Control, and Estimation*, manuscript, November 2026.
2. [BYU MAGICC — B-Spline Library](https://github.com/byu-magicc/bsplines_library/tree/uniform_development).
