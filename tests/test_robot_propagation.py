"""Stage 3A: analytical solutions, RK4 convergence, stage timing, and gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.propagation import propagate, rk4_stage_times, rk4_step
from models.unicycle import dynamics, input_noise_jacobian, state_jacobian
from splines import UniformBasis


@pytest.fixture(scope="module", autouse=True)
def enable_float64():
    with jax.experimental.enable_x64():
        yield


def prepared(times, L=None, window_start=0., degree=3, M=4):
    adapter = UniformBasis(degree, M)
    if L is None:
        L = times[-1] - window_start
    stages = rk4_stage_times(times)
    return adapter, adapter.precompute_jax(stages - window_start, L)


def constant_command(v, omega):
    return lambda t: jnp.array([v, omega], dtype=jnp.float64)


def circular_pose(r0, v, omega, elapsed):
    """Independent closed-form solution, with nonzero constant turn rate."""
    theta = r0[2] + omega * elapsed
    return np.stack((r0[0] + v / omega * (np.sin(theta) - np.sin(r0[2])),
                     r0[1] + v / omega * (np.cos(r0[2]) - np.cos(theta)), theta), axis=-1)


def circular_covariance(r0, P0, v, omega, Qc, T):
    """Exact transition plus independent 64-point Gauss-Legendre noise integral.

    Since A(t1)A(t2)=0, Phi(T,s)=I+integral_s^T A(t)dt. No RK4 or estimator
    Jacobian code is used to construct this reference.
    """
    nodes, weights = np.polynomial.legendre.leggauss(64)
    theta_T = r0[2] + omega * T

    def transition(s):
        theta_s = r0[2] + omega * s
        F = np.eye(3)
        F[0, 2] = v / omega * (np.cos(theta_T) - np.cos(theta_s))
        F[1, 2] = v / omega * (np.sin(theta_T) - np.sin(theta_s))
        return F

    F0 = transition(0)
    result = F0 @ P0 @ F0.T
    for s, weight in zip((nodes + 1) * T / 2, weights * T / 2):
        theta_s = r0[2] + omega * s
        G = np.array([[np.cos(theta_s), 0], [np.sin(theta_s), 0], [0, 1.]])
        FG = transition(s) @ G
        result += weight * FG @ Qc @ FG.T
    return result


def test_model_jacobians_against_autodiff_and_central_differences():
    rng = np.random.default_rng(7)
    max_ad_error, max_fd_error, max_G_error = 0., 0., 0.
    h = 1e-6
    for _ in range(12):
        r, u, w = rng.normal(size=3), rng.normal(size=2), rng.normal(size=2)
        r[2] *= 4
        A = np.asarray(state_jacobian(r, u, w))
        ad = np.asarray(jax.jacfwd(dynamics, argnums=0)(r, u, w))
        fd = np.column_stack([(np.asarray(dynamics(r + e, u, w))
                               - np.asarray(dynamics(r - e, u, w))) / (2 * h)
                              for e in h * np.eye(3)])
        G = np.asarray(input_noise_jacobian(r))
        input_ad = np.asarray(jax.jacfwd(dynamics, argnums=1)(r, u, w))
        disturbance_ad = np.asarray(jax.jacfwd(dynamics, argnums=2)(r, u, w))
        max_ad_error = max(max_ad_error, np.max(np.abs(A - ad)))
        max_fd_error = max(max_fd_error, np.max(np.abs(A - fd)))
        max_G_error = max(max_G_error, np.max(np.abs(G - input_ad)))
        np.testing.assert_allclose(A, ad, rtol=0, atol=5e-15)
        np.testing.assert_allclose(A, fd, rtol=2e-8, atol=2e-9)
        np.testing.assert_allclose(G, disturbance_ad, rtol=0, atol=5e-15)
        np.testing.assert_allclose(G, input_ad, rtol=0, atol=5e-15)
    print(f"\nJacobian max errors: A/autodiff={max_ad_error:.6e}, "
          f"A/FD(h={h})={max_fd_error:.6e}, G/autodiff={max_G_error:.6e}")


@pytest.mark.parametrize("theta", [0., .7, 3.4])
def test_straight_line_analytical_solution(theta):
    times = np.array([0., .03, .37, .51, 1.33, 2.3])
    adapter, B = prepared(times)
    n = adapter.num_coefficients
    # Both disturbance blocks are nonzero; the turn disturbance cancels command.
    coefficients = np.r_[np.full(n, .2), np.full(n, -.15)]
    r0 = np.array([.2, -.4, theta])
    poses, covariances = propagate((r0, np.zeros((3, 3))), times,
                                   constant_command(1.4, .15), B, coefficients, np.zeros((2, 2)))
    expected = r0 + times[:, None] * np.array([1.6 * np.cos(theta), 1.6 * np.sin(theta), 0.])
    error = np.max(np.abs(poses - expected))
    np.testing.assert_allclose(poses, expected, rtol=0, atol=3e-15)
    np.testing.assert_array_equal(covariances, 0)
    assert poses.dtype == jnp.float64 and covariances.dtype == jnp.float64
    print(f"\nStraight motion theta={theta}: max pose error={error:.6e}")


@pytest.mark.parametrize("omega", [-.8, 1.3])
def test_circular_motion_analytical_solution_and_unwrapped_heading(omega):
    times = np.linspace(0, 1.4, 141)
    adapter, B = prepared(times)
    r0 = np.array([.2, -.3, 2.9])
    v = 1.7
    coefficients = np.r_[np.full(adapter.num_coefficients, .25),
                          np.full(adapter.num_coefficients, -.1)]
    poses, _ = propagate((r0, np.eye(3) * .01), times,
                         constant_command(v - .25, omega + .1), B, coefficients,
                         np.diag([.03, .01]))
    expected = circular_pose(r0, v, omega, times)
    error = np.max(np.abs(poses - expected))
    np.testing.assert_allclose(poses, expected, rtol=0, atol=2e-11)
    if omega > 0:
        assert poses[-1, 2] > np.pi
    print(f"\nCircular motion omega={omega}: max pose error={error:.6e}")


def test_rk4_fourth_order_state_and_covariance_convergence():
    r0 = np.array([.2, -.4, .3])
    P0 = np.array([[.04, .003, .002], [.003, .03, -.001], [.002, -.001, .02]])
    v, omega, T = 1.4, 1.1, 1.6
    Qc = np.diag([.04, .015])
    exact_r = circular_pose(r0, v, omega, T)
    reference_P = circular_covariance(r0, P0, v, omega, Qc, T)
    state_errors, covariance_errors = [], []
    symmetry, min_eigenvalue = 0., np.inf
    for steps in [4, 8, 16, 32]:
        times = np.linspace(0, T, steps + 1)
        adapter, B = prepared(times)
        poses, covariances = propagate((r0, P0), times, constant_command(v, omega), B,
                                       np.zeros(2 * adapter.num_coefficients), Qc)
        state_errors.append(float(np.linalg.norm(poses[-1] - exact_r)))
        covariance_errors.append(float(np.linalg.norm(covariances[-1] - reference_P)))
        P = np.asarray(covariances)
        symmetry = max(symmetry, np.max(np.abs(P - P.transpose(0, 2, 1))))
        min_eigenvalue = min(min_eigenvalue, np.min(np.linalg.eigvalsh(P)))
    state_ratios = np.array(state_errors[:-1]) / state_errors[1:]
    covariance_ratios = np.array(covariance_errors[:-1]) / covariance_errors[1:]
    assert np.all((state_ratios > 14) & (state_ratios < 18))
    assert np.all((covariance_ratios > 14) & (covariance_ratios < 18))
    assert symmetry < 2e-15
    assert min_eigenvalue > 0
    print(f"\nRK4 steps=[4,8,16,32], state errors={state_errors}, ratios={state_ratios.tolist()}")
    print(f"Covariance errors={covariance_errors}, ratios={covariance_ratios.tolist()}")
    print(f"Covariance max asymmetry={symmetry:.6e}, minimum eigenvalue={min_eigenvalue:.6e}")


@pytest.mark.parametrize("Qc", [np.zeros((2, 2)), np.diag([.04, .015]),
                               np.array([[.04, .009], [.009, .015]])], ids=["zero", "diagonal", "correlated"])
def test_covariance_against_exact_straight_motion(Qc):
    times = np.array([0., .03, .21, .7, 1.1, 1.8])
    adapter, B = prepared(times)
    theta, v = .7, 1.2
    r0 = np.array([.2, -.4, theta])
    P0 = np.array([[.04, .003, .002], [.003, .03, -.001], [.002, -.001, .02]])
    _, actual = propagate((r0, P0), times, constant_command(v, 0.), B,
                           np.zeros(2 * adapter.num_coefficients), Qc)
    # For fixed heading A^2=0, so Phi=I+A*t and the noise integral is cubic.
    A = np.array([[0., 0., -v * np.sin(theta)], [0., 0., v * np.cos(theta)], [0., 0., 0.]])
    G = np.array([[np.cos(theta), 0.], [np.sin(theta), 0.], [0., 1.]])
    Q = G @ Qc @ G.T
    expected = []
    for t in times:
        F = np.eye(3) + A * t
        expected.append(F @ P0 @ F.T + Q * t + (A @ Q + Q @ A.T) * t**2 / 2
                        + (A @ Q @ A.T) * t**3 / 3)
    expected = np.asarray(expected)
    error = np.max(np.abs(actual - expected))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    print(f"\nExact straight covariance Qc={Qc.tolist()}: max error={error:.6e}")


@pytest.mark.parametrize("channel", [0, 1], ids=["linear", "angular"])
def test_stage_basis_and_time_varying_command_against_exact_quadrature(channel):
    start, L = 7., 2.
    times = start + np.array([0., .125, .5, .625, 1., 1.5, 2.])
    adapter, B = prepared(times, L=L, window_start=start)
    n = adapter.num_coefficients
    active_coefficients = np.array([-.2, .1, .4, -.3, .5, .2, -.1])
    coefficients = np.zeros(2 * n)
    coefficients[channel * n:(channel + 1) * n] = active_coefficients

    def command(t):
        value = (t - start)**3 - .1 * (t - start)
        return jnp.stack((value, jnp.zeros_like(value))) if channel == 0 else jnp.stack((jnp.zeros_like(value), value))

    poses, _ = propagate((np.zeros(3), np.zeros((3, 3))), times, command, B,
                          coefficients, np.zeros((2, 2)))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    expected = [0.]
    for left, right in zip(times[:-1], times[1:]):
        elapsed = left - start + (nodes + 1) * (right - left) / 2
        disturbance = adapter.evaluate(elapsed, L) @ active_coefficients
        integral = weights @ (elapsed**3 - .1 * elapsed + disturbance) * (right - left) / 2
        expected.append(expected[-1] + integral)
    # Each interval lies within a spline segment: both inputs are cubic, so
    # independent Gauss quadrature and RK4's stage-weighted integral are exact.
    np.testing.assert_allclose(poses[:, 0 if channel == 0 else 2], expected,
                               rtol=0, atol=2e-14)
    np.testing.assert_array_equal(poses[:, 1], 0)
    np.testing.assert_array_equal(B[:, 1], B[:, 2])
    assert not np.allclose(B[:, 0], B[:, 1])


def test_nonuniform_intervals_and_restart_at_observation_time():
    start = 10.
    times = start + np.array([0., .07, .23, .47, .9, 1.3])
    L = times[-1] - start
    adapter, B = prepared(times, L, start)
    c = np.random.default_rng(9).normal(scale=.1, size=2 * adapter.num_coefficients)
    command = lambda t: jnp.stack((1 + .1 * t, .2 * jnp.sin(t)))
    Qc = np.array([[.03, .004], [.004, .01]])
    initial = (np.array([.1, .2, .3]), np.eye(3) * .02)
    full = propagate(initial, times, command, B, c, Qc)
    left = propagate(initial, times[:3], command, B[:2], c, Qc)
    terminal = (left[0][-1], left[1][-1])
    right = propagate(terminal, times[2:], command, B[2:], c, Qc)
    for entire, first, second in zip(full, left, right):
        np.testing.assert_allclose(entire, np.concatenate((first, second[1:])), rtol=0, atol=5e-14)
    assert full[0].shape == (len(times), 3)
    assert full[1].shape == (len(times), 3, 3)
    # A single RK4 step is also reusable between observation events.
    one = rk4_step(initial, times[0], times[1] - times[0], command, B[0], c, Qc)
    np.testing.assert_allclose(one[0], full[0][1], rtol=0, atol=5e-14)
    np.testing.assert_allclose(one[1], full[1][1], rtol=0, atol=5e-14)


def test_gradients_of_propagated_state_and_covariance_both_blocks():
    times = np.array([0., .07, .25, .49, .73, 1., 1.24, 1.6])
    adapter, B = prepared(times)
    n = adapter.num_coefficients
    initial = (np.array([.2, -.3, .6]), np.diag([.04, .03, .02]))
    Qc = np.diag([.03, .02])
    command = lambda t: jnp.stack((1 + .2 * jnp.cos(.8 * t), .35 + .15 * jnp.sin(1.3 * t)))
    c = np.random.default_rng(11).normal(scale=.15, size=2 * n)
    W = jnp.array([[1., .1, .2], [.1, .7, -.1], [.2, -.1, .5]])

    def components(coefficients):
        poses, covariances = propagate(initial, times, command, B, coefficients, Qc)
        r, P = poses[-1], covariances[-1]
        return jnp.stack((.5 * r @ W @ r, .6 * jnp.sum(P * W) + .2 * jnp.sum(P**2)))

    objective = jax.jit(lambda coefficients: jnp.sum(components(coefficients)))
    gradient = np.asarray(jax.jit(jax.grad(objective))(c))
    component_gradients = np.asarray(jax.jit(jax.jacrev(components))(c))
    np.testing.assert_allclose(gradient, component_gradients.sum(axis=0), rtol=2e-13, atol=2e-13)
    assert gradient.dtype == np.float64
    for component in component_gradients:
        assert np.linalg.norm(component[:n]) > 1e-5
        assert np.linalg.norm(component[n:]) > 1e-5
    for h in [1e-5, 3e-6]:
        fd = np.array([(float(objective(c + e)) - float(objective(c - e))) / (2 * h)
                       for e in np.eye(c.size) * h])
        np.testing.assert_allclose(gradient, fd, rtol=2e-7, atol=2e-9)
        errors = np.abs(gradient - fd)
        print(f"\nPropagation gradient FD h={h}: max c_v error={np.max(errors[:n]):.6e}, "
              f"max c_omega error={np.max(errors[n:]):.6e}")
    print(f"Covariance-only gradient block norms: c_v={np.linalg.norm(component_gradients[1,:n]):.6e}, "
          f"c_omega={np.linalg.norm(component_gradients[1,n:]):.6e}")


def test_empty_propagation_and_zero_duration_step():
    initial = (np.array([.1, .2, .3]), np.eye(3) * .1)
    adapter, B = prepared(np.array([.5]), L=1.)
    c = np.zeros(2 * adapter.num_coefficients)
    history = propagate(initial, np.array([.5]), constant_command(1., 1.), B, c, np.eye(2))
    for actual, expected in zip(history, initial):
        np.testing.assert_array_equal(actual, expected[None])
    same = rk4_step(initial, .5, 0., constant_command(1., 1.),
                    adapter.precompute_jax(np.full(4, .5), L=1.), c, np.eye(2))
    for actual, expected in zip(same, initial):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("times", [[], [[0., 1.]], [0., np.nan], [0., np.inf], [0., 0.], [1., 0.]])
def test_reject_invalid_host_time_grids(times):
    with pytest.raises(ValueError):
        rk4_stage_times(times)


def test_float64_is_required_without_silent_demotion():
    with jax.experimental.disable_x64():
        with pytest.raises(RuntimeError, match="Enable JAX float64"):
            dynamics(np.zeros(3), np.zeros(2), np.zeros(2))
        with pytest.raises(RuntimeError, match="Enable JAX float64"):
            rk4_step((np.zeros(3), np.eye(3)), 0., .1, constant_command(1., 0.),
                      np.zeros((4, 7)), np.zeros(14), np.eye(2))


def test_shape_validation_and_float32_input_promotion():
    state = (np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))
    B = np.ones((4, 1), dtype=np.float32)
    args = (state, 0., .1, constant_command(1., 0.), B, np.zeros(2), np.eye(2))
    output = rk4_step(*args)
    assert output[0].dtype == jnp.float64 and output[1].dtype == jnp.float64
    with pytest.raises(ValueError, match="Qc"):
        rk4_step(*args[:-1], np.ones(2))
    with pytest.raises(ValueError, match="basis_stages"):
        rk4_step(*args[:4], np.ones((3, 1)), *args[5:])
    with pytest.raises(ValueError, match="coefficients"):
        rk4_step(*args[:5], np.zeros(3), args[-1])
    with pytest.raises(ValueError, match="state"):
        rk4_step((np.zeros(3), np.eye(2)), *args[1:])
