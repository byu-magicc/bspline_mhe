"""Milestone 2: C++ basis conventions, time scaling, and JAX coefficients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bspline_module import UniformBSpline

from splines.uniform_basis import UniformBasis, disturbance_from_coefficients


@pytest.fixture(scope="module", autouse=True)
def enable_float64():
    with jax.experimental.enable_x64():
        yield


@pytest.mark.parametrize("degree,M", [(0, 1), (1, 4), (2, 1), (3, 4), (5, 2)])
@pytest.mark.parametrize("shape", [(), (7,), (2, 3), (2, 3, 4), (0,), (2, 0, 3)])
def test_dimensions_and_ordering(degree, M, shape):
    adapter = UniformBasis(degree, M)
    engine = UniformBSpline(degree, M)
    times = np.linspace(0, 0.9, num=int(np.prod(shape))).reshape(shape)
    # Reverse the last axis to exercise noncontiguous input as well.
    if times.ndim:
        times = times[..., ::-1]
    basis = adapter.evaluate(times, L=1.7)
    assert basis.shape == shape + (M + degree,)
    assert basis.dtype == np.float64
    assert (adapter.degree, adapter.M, adapter.num_coefficients) == (degree, M, M + degree)
    for position in np.ndindex(shape):
        s = (times[position] / 1.7) * M
        expected = [engine.evaluate_central_basis(s + degree - j)
                    for j in range(M + degree)]
        np.testing.assert_allclose(basis[position], expected, rtol=0, atol=2e-15)


def test_cubic_ordering_against_known_values():
    # b^3(3.5), b^3(2.5), b^3(1.5), b^3(.5), followed by zeros.
    actual = UniformBasis(3, 4).evaluate(0.5, L=4)
    np.testing.assert_allclose(actual, np.array([1, 23, 23, 1, 0, 0, 0]) / 48,
                               rtol=0, atol=2e-16)


@pytest.mark.parametrize("degree", range(6))
@pytest.mark.parametrize("M", [1, 4, 9])
def test_nonnegativity_support_and_partition(degree, M):
    s = np.unique(np.r_[np.linspace(0, M, 241, endpoint=False), np.arange(M)])
    if degree > 0:
        s = np.r_[s, float(M)]
    basis = UniformBasis(degree, M).evaluate(s, L=M)
    assert np.isfinite(basis).all()
    assert np.min(basis) >= 0
    np.testing.assert_allclose(basis.sum(axis=-1), 1, rtol=0, atol=2e-15)
    assert np.all(np.count_nonzero(basis, axis=-1) <= degree + 1)
    # Entry j has support [j-d, j+1); test that support without a new basis implementation.
    lower = np.arange(M + degree) - degree
    outside = (s[:, None] < lower) | (s[:, None] >= lower + degree + 1)
    np.testing.assert_array_equal(basis[outside], 0)
    engine = UniformBSpline(degree, M)
    for t in [-2., -1e-10, degree + 1., degree + 2.]:
        assert engine.evaluate_central_basis(t) == 0


@pytest.mark.parametrize("degree", range(1, 6))
@pytest.mark.parametrize("M,L", [(1, 0.7), (4, 3.1), (9, 10.)])
def test_right_endpoint_and_left_continuity(degree, M, L):
    adapter = UniformBasis(degree, M)
    endpoint = adapter.evaluate(L, L)
    np.testing.assert_array_equal(endpoint, UniformBSpline(degree, M).basis_vector(float(M)))
    assert endpoint.sum() == pytest.approx(1, abs=2e-15)
    # All tested bases have |db/ds| <= 1, so a normalized gap of 1e-8
    # bounds each component's endpoint difference by 1e-8 plus roundoff.
    left = adapter.evaluate(L * (1 - 1e-8 / M), L)
    np.testing.assert_allclose(left, endpoint, rtol=0, atol=1.01e-8)
    adjacent = adapter.evaluate(np.nextafter(L, -np.inf), L)
    np.testing.assert_allclose(adjacent, endpoint, rtol=0, atol=5e-15)


def test_cubic_endpoint_values_and_degree_zero_half_open_convention():
    endpoint = UniformBasis(3, 4).evaluate(0.7, L=0.7)
    np.testing.assert_allclose(endpoint, np.array([0, 0, 0, 0, 1, 4, 1]) / 6,
                               rtol=0, atol=2e-16)
    adapter = UniformBasis(0, 4)
    np.testing.assert_array_equal(adapter.evaluate(1., L=1.), np.zeros(4))
    np.testing.assert_array_equal(adapter.evaluate(np.nextafter(1., 0.), L=1.),
                                  [0., 0., 0., 1.])


@pytest.mark.parametrize("L", [0.1, 0.7, 2., 13.7])
def test_time_scaling_and_changed_stage_grid(L):
    adapter = UniformBasis(3, 4)
    for edges in [np.array([0., .2, .55, 1.]), np.linspace(0., 1., 7)]:
        # Four RK4 evaluations per interval, including duplicate midpoints.
        starts, ends = edges[:-1], edges[1:]
        midpoint = (starts + ends) / 2
        fractions = np.stack([starts, midpoint, midpoint, ends], axis=-1)
        actual = adapter.evaluate(fractions * L, L)
        np.testing.assert_allclose(actual, adapter.evaluate(fractions, L=1),
                                   rtol=0, atol=2e-15)
        assert actual.shape == (len(starts), 4, 7)
        np.testing.assert_array_equal(actual[:, 1], actual[:, 2])
        np.testing.assert_array_equal(actual[-1, -1], adapter.evaluate(L, L))


@pytest.mark.parametrize("degree", range(1, 6))
def test_zero_and_constant_disturbances_in_both_channels(degree):
    adapter = UniformBasis(degree, 4)
    basis = adapter.precompute_jax(np.linspace(0., 2., 43), L=2)
    assert basis.dtype == jnp.float64
    n = adapter.num_coefficients
    for v, omega in [(0., 0.), (0.3, 0.), (0., -0.2), (0.3, -0.2)]:
        c = np.r_[np.full(n, v), np.full(n, omega)]
        expected = np.broadcast_to([v, omega], (43, 2))
        np.testing.assert_allclose(disturbance_from_coefficients(basis, c), expected,
                                   rtol=0, atol=2e-15)


def test_reconstruct_independent_known_piecewise_linear_inputs():
    adapter = UniformBasis(1, 4)
    sigma = np.array([[0., .17, .5], [.9, 1.5, 2.]])
    c_v = np.array([.1, .4, -.2, 1., -.5])
    c_omega = np.array([-.3, .2, .6, -.1, .8])
    actual = disturbance_from_coefficients(adapter.precompute_jax(sigma, L=2),
                                            np.r_[c_v, c_omega])
    knot_times = np.linspace(0, 2, 5)
    expected = np.stack([np.interp(sigma, knot_times, c_v),
                         np.interp(sigma, knot_times, c_omega)], axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    assert actual.shape == sigma.shape + (2,)
    scalar = disturbance_from_coefficients(adapter.precompute_jax(.17, L=2),
                                            np.r_[c_v, c_omega])
    np.testing.assert_array_equal(scalar, actual[0, 1])


def test_jax_coefficient_gradients_against_analytic_and_finite_differences():
    adapter = UniformBasis(3, 4)
    edges = np.array([0., .13, .7, 1.1, 1.8])
    midpoints = (edges[:-1] + edges[1:]) / 2
    stages = np.stack([edges[:-1], midpoints, midpoints, edges[1:]], axis=-1)
    basis = adapter.precompute_jax(stages, L=edges[-1])
    assert basis.dtype == jnp.float64
    target = jnp.asarray(np.stack([np.sin(stages), np.cos(2 * stages)], axis=-1))
    c = np.random.default_rng(7).normal(scale=.3, size=2 * adapter.num_coefficients)

    def objective(coefficients):
        residual = disturbance_from_coefficients(basis, coefficients) - target
        return .5 * jnp.sum(residual**2)

    gradient = jax.jit(jax.grad(objective))
    actual = np.asarray(gradient(jnp.asarray(c)))
    assert actual.dtype == np.float64
    flat_basis = np.asarray(basis).reshape(-1, adapter.num_coefficients)
    residual = np.asarray(disturbance_from_coefficients(basis, c) - target).reshape(-1, 2)
    analytic = np.r_[flat_basis.T @ residual[:, 0], flat_basis.T @ residual[:, 1]]
    np.testing.assert_allclose(actual, analytic, rtol=2e-13, atol=2e-13)

    # Unit-scale coefficients/residuals, float64, quadratic objective: h=1e-5
    # keeps subtraction error well below 1e-8 without truncation error here.
    h = 1e-5
    directions = np.eye(c.size) * h
    finite_difference = np.array([(float(objective(c + e)) - float(objective(c - e))) / (2 * h)
                                  for e in directions])
    np.testing.assert_allclose(actual, finite_difference, rtol=1e-8, atol=1e-9)
    # Compiled execution must still respond to new coefficients in each channel.
    shifted = c + np.linspace(-.1, .2, c.size)
    assert not np.allclose(gradient(shifted), actual)


@pytest.mark.parametrize("degree,M", [(-1, 4), (2, 0), (2, -1)])
def test_invalid_spline_parameters(degree, M):
    with pytest.raises(ValueError):
        UniformBasis(degree, M)


@pytest.mark.parametrize("degree,M", [(1.5, 4), (2, 2.5)])
def test_noninteger_spline_parameters(degree, M):
    with pytest.raises(TypeError):
        UniformBasis(degree, M)


@pytest.mark.parametrize("L", [0., -1., np.nan, np.inf])
def test_invalid_horizon(L):
    with pytest.raises(ValueError, match="L must"):
        UniformBasis(3, 4).evaluate(0., L)


@pytest.mark.parametrize("sigma", [-.01, 1.01, np.nan, [0., np.inf]])
def test_invalid_elapsed_times(sigma):
    with pytest.raises(ValueError, match="sigma must"):
        UniformBasis(3, 4).evaluate(sigma, L=1.)


@pytest.mark.parametrize("basis,c", [(1., np.zeros(2)), (np.zeros((2, 0)), np.zeros(0)),
                                    (np.ones(3), np.zeros(5)), (np.ones(3), np.zeros((2, 3)))])
def test_invalid_reconstruction_shapes(basis, c):
    with pytest.raises(ValueError):
        disturbance_from_coefficients(basis, c)
