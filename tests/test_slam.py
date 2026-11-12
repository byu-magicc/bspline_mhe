"""Stage 3B.2: augmentation, fixed-capacity SLAM, propagation, and gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from estimation.slam import (SlamState, create_state, initialize_landmark, neutralize_inactive,
    observe, sequential_observations, rk4_slam_step, filter_slam, raise_on_overflow,
    LandmarkCapacityError, INITIALIZED, UPDATED, INVALID, MASKED, OVERFLOW)
from estimation.propagation import rk4_stage_times, rk4_step
from models.slam import (inverse_range_bearing, augmented_range_bearing,
                         augmented_dynamics, augmented_jacobians)
from splines import UniformBasis


@pytest.fixture(scope="module", autouse=True)
def float64():
    with jax.experimental.enable_x64():
        yield


def active_indices(state):
    return np.r_[np.arange(3), np.flatnonzero(np.repeat(np.asarray(state.active), 2)) + 3]


def assert_covariance(state):
    idx = active_indices(state)
    P = np.asarray(state.covariance)
    selected = P[np.ix_(idx, idx)]
    asymmetry = float(np.max(np.abs(selected - selected.T)))
    minimum = float(np.linalg.eigvalsh(selected)[0])
    assert asymmetry <= 1e-10
    assert minimum >= -1e-10
    inactive = np.setdiff1d(np.arange(len(P)), idx)
    np.testing.assert_array_equal(P[inactive], 0)
    np.testing.assert_array_equal(P[:, inactive], 0)
    np.testing.assert_array_equal(np.asarray(state.x)[inactive], 0)
    return asymmetry, minimum


def numpy_inverse(r, y):
    phi = r[2] + y[1]
    return r[:2] + y[0] * np.array([np.cos(phi), np.sin(phi)])


def numpy_observe(x, P, ids, truth_id, y, Sigma):
    """Independent dynamically sized NumPy SLAM, used only as a reference."""
    if truth_id not in ids:
        phi = x[2] + y[1]
        Jr = np.array([[1, 0, -y[0]*np.sin(phi)], [0, 1, y[0]*np.cos(phi)]])
        Jy = np.array([[np.cos(phi), -y[0]*np.sin(phi)], [np.sin(phi), y[0]*np.cos(phi)]])
        J = np.zeros((2, len(x)))
        J[:, :3] = Jr
        cross = P @ J.T
        marginal = J @ P @ J.T + Jy @ Sigma @ Jy.T
        return np.r_[x, numpy_inverse(x[:3], y)], np.block([[P, cross], [cross.T, marginal]]), ids + [truth_id]
    slot = ids.index(truth_id)
    i = 3 + 2 * slot
    dx, dy = x[i:i+2] - x[:2]
    q = dx*dx + dy*dy
    h = np.array([np.sqrt(q), np.arctan2(dy, dx) - x[2]])
    H = np.zeros((2, len(x)))
    H[:, :3] = [[-dx/np.sqrt(q), -dy/np.sqrt(q), 0], [dy/q, -dx/q, -1]]
    H[:, i:i+2] = -H[:, :2]
    e = y - h
    e[1] = np.arctan2(np.sin(e[1]), np.cos(e[1]))
    S = H @ P @ H.T + Sigma
    K = np.linalg.solve(S, (P @ H.T).T).T
    J = np.eye(len(x)) - K @ H
    return x + K @ e, J @ P @ J.T + K @ Sigma @ K.T, ids


def test_inverse_model_and_initialization_jacobians():
    r, y = np.array([.2, -.1, .4]), np.array([2.3, -.2])
    landmark, Jr, Jy = jax.jit(inverse_range_bearing)(r, y)
    np.testing.assert_allclose(landmark, numpy_inverse(r, y), rtol=0, atol=1e-12)
    np.testing.assert_allclose(Jr, jax.jacfwd(lambda r: inverse_range_bearing(r, y)[0])(r), rtol=0, atol=1e-12)
    np.testing.assert_allclose(Jy, jax.jacfwd(lambda y: inverse_range_bearing(r, y)[0])(y), rtol=0, atol=1e-12)
    # Richardson extrapolation of central differences avoids the O(h^2)
    # truncation error while retaining the requested 1e-12 absolute tolerance.
    errors = []
    for base, function, analytical in [(r, lambda r: numpy_inverse(r, y), Jr),
                                       (y, lambda y: numpy_inverse(r, y), Jy)]:
        def central(h):
            return np.column_stack([(function(base+e)-function(base-e))/(2*h) for e in h*np.eye(len(base))])
        fd = (4*central(.001)-central(.002))/3
        error = np.max(np.abs(fd - analytical))
        errors.append(float(error))
        np.testing.assert_allclose(fd, analytical, rtol=0, atol=1e-12)
    print(f"\nInverse-model Jacobian central/Richardson errors (Jr,Jy): {errors}")


@pytest.mark.parametrize("Sigma", [np.diag([.03, .01]), np.array([[.03, .006], [.006, .01]])], ids=["diagonal", "correlated"])
def test_augmentation_covariance_and_all_cross_correlations(Sigma):
    r = np.array([.2, -.3, .4])
    P0 = np.array([[.04, .002, .001], [.002, .03, -.001], [.001, -.001, .02]])
    state = create_state(r, P0, capacity=4)
    state = initialize_landmark(state, 0, 101, np.array([2., .1]), Sigma)
    before_x, before_P = np.asarray(state.x)[:5], np.asarray(state.covariance)[:5, :5]
    y = np.array([3., -.7])
    reference_x, reference_P, _ = numpy_observe(before_x, before_P, [101], -42, y, Sigma)
    updated = jax.jit(initialize_landmark)(state, 1, -42, y, Sigma)
    error = np.max(np.abs(np.asarray(updated.covariance)[:7, :7] - reference_P))
    np.testing.assert_allclose(updated.x[:7], reference_x, rtol=0, atol=1e-11)
    np.testing.assert_allclose(updated.covariance[:7, :7], reference_P, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(updated.covariance[:5, :5], before_P)
    assert np.linalg.norm(np.asarray(updated.covariance)[3:5, 5:7]) > 1e-4
    asymmetry, minimum = assert_covariance(updated)
    print(f"\nAugmentation covariance max error={error:.6e}, asymmetry={asymmetry:.6e}, active min eigenvalue={minimum:.6e}")


def test_full_augmented_model_jacobians():
    state = create_state([.2, -.3, .4], np.eye(3)*.03, capacity=3)
    state = initialize_landmark(state, 1, 99, np.array([3., .2]), np.eye(2)*.01)
    predicted, H, valid = augmented_range_bearing(state.x, 1, state.active)
    assert valid
    ad = jax.jacfwd(lambda x: augmented_range_bearing(x, 1, state.active)[0])(state.x)
    np.testing.assert_allclose(H, ad, rtol=0, atol=1e-12)
    h = 1e-6
    fd = np.column_stack([(np.asarray(augmented_range_bearing(state.x+e, 1, state.active)[0])
                           - np.asarray(augmented_range_bearing(state.x-e, 1, state.active)[0]))/(2*h)
                          for e in np.eye(len(state.x))*h])
    np.testing.assert_allclose(H, fd, rtol=2e-8, atol=1e-9)
    np.testing.assert_array_equal(H[:, [3, 4, 7, 8]], 0)
    u, w = np.array([1.2, .2]), np.array([.1, -.05])
    A, G = augmented_jacobians(state.x, u, w)
    np.testing.assert_allclose(A, jax.jacfwd(augmented_dynamics, argnums=0)(state.x, u, w), rtol=0, atol=1e-12)
    np.testing.assert_allclose(G, jax.jacfwd(augmented_dynamics, argnums=1)(state.x, u, w), rtol=0, atol=1e-12)
    assert not augmented_range_bearing(state.x, 0, state.active)[2]
    assert not augmented_range_bearing(state.x, -1, state.active)[2]
    print(f"\nAugmented H errors: autodiff={np.max(np.abs(H-ad)):.6e}, FD={np.max(np.abs(H-fd)):.6e}")


def test_sequential_arbitrary_ids_same_timestamp_against_numpy():
    initial = create_state([.2, -.1, .4], np.diag([.05, .04, .02]), capacity=4)
    ids = np.array([42, -7, 1001, 42, -7, 90000000000], dtype=np.int64)
    y = np.array([[2., .1], [3., -.3], [2.5, .7], [2.1, .12], [2.9, -.28], [4., -.6]])
    Sigma = np.array([[.03, .004], [.004, .01]])
    final, info = jax.jit(sequential_observations)(initial, y, ids, Sigma)
    np.testing.assert_array_equal(info.status, [INITIALIZED, INITIALIZED, INITIALIZED, UPDATED, UPDATED, INITIALIZED])
    np.testing.assert_array_equal(info.slot, [0, 1, 2, 0, 1, 3])
    np.testing.assert_array_equal(final.truth_ids, [42, -7, 1001, 90000000000])
    x, P, known = np.asarray(initial.x)[:3], np.asarray(initial.covariance)[:3, :3], []
    for measurement, ident in zip(y, ids):
        x, P, known = numpy_observe(x, P, known, int(ident), measurement, Sigma)
    np.testing.assert_allclose(final.x, x, rtol=0, atol=1e-11)
    np.testing.assert_allclose(final.covariance, P, rtol=0, atol=1e-11)
    assert_covariance(final)


def test_capacity_overflow_compiled_flag_and_python_exception():
    state = create_state(np.zeros(3), np.eye(3)*.01, capacity=2)
    measurements = np.array([[[2., .2], [3., -.2], [4., .5], [2.1, .21]]])
    ids = np.array([[100, 200, 300, 100]], dtype=np.int64)
    result = jax.jit(lambda: filter_slam(state, np.array([0.]), lambda t: jnp.zeros(2),
                      np.empty((0, 4, 7)), np.zeros(14), np.eye(2)*.01,
                      measurements, ids, np.eye(2)*.01))()
    assert result.overflow
    np.testing.assert_array_equal(result.observations.status, [[INITIALIZED, INITIALIZED, OVERFLOW, UPDATED]])
    np.testing.assert_array_equal(result.states.truth_ids[-1], [100, 200])
    with pytest.raises(LandmarkCapacityError, match="capacity 2 exhausted"):
        raise_on_overflow(result)
    filled, _ = sequential_observations(state, measurements[0, :2], ids[0, :2], np.eye(2)*.01)
    unchanged, info = observe(filled, measurements[0, 2], ids[0, 2], np.eye(2)*.01)
    assert info.status == OVERFLOW
    for old, new in zip(filled, unchanged):
        np.testing.assert_array_equal(old, new)


def test_inactive_storage_isolation_and_masked_records():
    clean = create_state([.2, -.1, .3], np.eye(3)*.02, capacity=4)
    clean = initialize_landmark(clean, 0, 123, np.array([2., .2]), np.eye(2)*.01)
    poisoned = SlamState(clean.x.at[5:].set(jnp.nan),
                         clean.covariance.at[5:, :].set(jnp.nan).at[:, 5:].set(jnp.nan),
                         clean.active, clean.truth_ids.at[1:].set(123))
    for state in [clean, poisoned]:
        updated, info = observe(state, np.array([2.1, .21]), 123, np.eye(2)*.01)
        if state is clean:
            expected = updated
        else:
            for actual, reference in zip(updated, expected):
                np.testing.assert_array_equal(actual, reference)
        assert info.status == UPDATED
        assert_covariance(updated)
    ignored, info = observe(poisoned, np.array([np.nan, np.nan]), 999, np.full((2, 2), np.nan), requested=False)
    assert info.status == MASKED
    for actual, reference in zip(ignored, clean):
        np.testing.assert_array_equal(actual, reference)
    empty, info = sequential_observations(clean, np.empty((0, 2)), np.empty(0, dtype=np.int64), np.eye(2))
    assert info.status.shape == (0,)
    np.testing.assert_array_equal(empty.x, clean.x)


def test_augmented_covariance_propagation_exact_constant_heading():
    state = create_state([.2, -.1, .4], np.diag([.03, .04, .02]), capacity=3)
    state = initialize_landmark(state, 0, 42, np.array([3., -.2]), np.eye(2)*.01)
    theta, speed, duration = .4, 1.3, .7
    Qc = np.array([[.03, .004], [.004, .01]])
    adapter = UniformBasis(3, 4)
    B = adapter.precompute_jax(np.array([0., duration/2, duration/2, duration]), duration)
    command = lambda t: jnp.array([speed, 0.])
    c = np.zeros(2*adapter.num_coefficients)
    updated = jax.jit(lambda s: rk4_slam_step(s, 0., duration, command, B, c, Qc))(state)
    D = len(state.x)
    A = np.zeros((D, D)); A[:3, 2] = [-speed*np.sin(theta), speed*np.cos(theta), 0.]
    G = np.zeros((D, 2)); G[:3] = [[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]]
    Q = G @ Qc @ G.T
    F = np.eye(D) + duration*A
    P = np.asarray(state.covariance)
    expected = F@P@F.T + Q*duration + (A@Q+Q@A.T)*duration**2/2 + A@Q@A.T*duration**3/3
    error = np.max(np.abs(updated.covariance - expected))
    np.testing.assert_allclose(updated.covariance, expected, rtol=0, atol=1e-11)
    robot = rk4_step((state.x[:3], state.covariance[:3, :3]), 0., duration, command, B, c, Qc)
    np.testing.assert_allclose(updated.x[:3], robot[0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(updated.covariance[:3, :3], robot[1], rtol=0, atol=1e-13)
    np.testing.assert_array_equal(updated.x[3:], state.x[3:])
    assert_covariance(updated)
    print(f"\nAugmented analytical covariance error={error:.6e}")


def test_stage_basis_and_command_reuse_matches_stage3a_robot():
    times = np.array([0., .13, .37, .63, 1.])
    adapter = UniformBasis(3, 4)
    B = adapter.precompute_jax(rk4_stage_times(times), 1.)
    c = np.random.default_rng(21).normal(scale=.1, size=14)
    command = lambda t: jnp.stack((1 + .2*jnp.sin(t), .3*jnp.cos(t)))
    Qc = np.diag([.02, .01])
    state = create_state([.1, .2, .3], np.eye(3)*.01, capacity=3)
    state = initialize_landmark(state, 1, -9, np.array([3., .2]), np.eye(2)*.02)
    robot = (state.x[:3], state.covariance[:3, :3])
    for k, (left, right) in enumerate(zip(times[:-1], times[1:])):
        state = rk4_slam_step(state, left, right-left, command, B[k], c, Qc)
        robot = rk4_step(robot, left, right-left, command, B[k], c, Qc)
        np.testing.assert_allclose(state.x[:3], robot[0], rtol=0, atol=1e-13)
        np.testing.assert_allclose(state.covariance[:3, :3], robot[1], rtol=0, atol=1e-13)
        assert_covariance(state)


def test_gradients_through_initialization_propagation_and_updates():
    times = np.array([0., .1, .3, .7, 1.1, 1.5])
    adapter = UniformBasis(3, 4)
    B = adapter.precompute_jax(rk4_stage_times(times), 1.5)
    n = adapter.num_coefficients
    initial = create_state([.1, -.2, .3], np.diag([.04, .03, .02]), capacity=4)
    ids = np.array([[0, 0, 0], [101, 0, 0], [-7, 101, 0], [101, -7, 9001],
                    [9001, 101, -7], [-7, 9001, 101]], dtype=np.int64)
    mask = np.array([[False]*3, [True, False, False], [True, True, False], [True]*3, [True]*3, [True]*3])
    landmark_map = {101: np.array([3., 1.]), -7: np.array([-2., 3.]), 9001: np.array([1., -3.])}
    y = np.zeros((len(times), 3, 2))
    for i, t in enumerate(times):
        r = np.array([.1 + t, -.2 + .2*t, .3 + .2*t])
        for j in range(3):
            if mask[i, j]:
                delta = landmark_map[int(ids[i, j])] - r[:2]
                y[i, j] = [np.linalg.norm(delta), np.arctan2(delta[1], delta[0])-r[2]]
    command = lambda t: jnp.stack((1 + .1*jnp.sin(t), .2 + .03*jnp.cos(t)))
    Qc, Sigma = np.diag([.02, .006]), np.array([[.02, .002], [.002, .004]])
    c = np.random.default_rng(22).normal(scale=.08, size=2*n)

    def run(c):
        return filter_slam(initial, times, command, B, c, Qc, y, ids, Sigma, mask)

    def components(c):
        result = run(c)
        x, P = result.states.x, result.states.covariance
        info = result.observations.diagnostics
        e, S = info.innovation, info.innovation_covariance
        score = jnp.sum(e * jnp.linalg.solve(S, e[..., None])[..., 0])
        # First initialized landmark is coefficient-dependent before any corrections.
        init_score = .1*jnp.sum(x[1, 3:5]**2) + .3*jnp.trace(P[1, 3:5, 3:5])
        terminal = .05*jnp.sum(x[-1]**2) + .1*jnp.trace(P[-1]) + .01*jnp.sum(P[-1]**2)
        return jnp.stack((score, init_score, terminal))

    objective = jax.jit(lambda c: jnp.sum(components(c)))
    gradient = np.asarray(jax.jit(jax.grad(objective))(c))
    component_gradients = np.asarray(jax.jit(jax.jacrev(components))(c))
    for row in component_gradients:
        assert np.linalg.norm(row[:n]) > 1e-6 and np.linalg.norm(row[n:]) > 1e-6
    np.testing.assert_allclose(gradient, component_gradients.sum(axis=0), rtol=1e-12, atol=1e-12)
    base = jax.jit(run)(c)
    raise_on_overflow(base)
    for h in [1e-5, 3e-6]:
        fd = np.array([(float(objective(c+e))-float(objective(c-e)))/(2*h) for e in h*np.eye(2*n)])
        np.testing.assert_allclose(gradient, fd, rtol=1e-5, atol=1e-7)
        errors = np.abs(gradient-fd)
        print(f"\nSLAM gradient h={h}: c_v error={errors[:n].max():.6e}, c_omega error={errors[n:].max():.6e}")
    np.testing.assert_array_equal(jax.jit(run)(c + 1e-5).observations.status, base.observations.status)
    for i in range(len(times)):
        assert_covariance(jax.tree_util.tree_map(lambda a: a[i], base.states))
    print('Initialization-only gradient block norms:', np.linalg.norm(component_gradients[1, :n]), np.linalg.norm(component_gradients[1, n:]))


@pytest.mark.parametrize("y", [[0., .2], [-1., .3], [np.nan, .1]])
def test_invalid_initialization_does_not_allocate(y):
    state = create_state(np.zeros(3), np.eye(3), capacity=2)
    updated, info = observe(state, np.array(y), -42, np.eye(2))
    assert info.status == INVALID
    np.testing.assert_array_equal(updated.active, False)


def test_default_capacity_and_shape_precision_errors():
    assert create_state(np.zeros(3), np.eye(3)).x.shape == (35,)
    with pytest.raises(ValueError, match='capacity'):
        create_state(np.zeros(3), np.eye(3), capacity=0)
    state = create_state(np.zeros(3), np.eye(3), capacity=2)
    with pytest.raises(ValueError, match='signed integers'):
        observe(state, np.ones(2), 42., np.eye(2))
    with jax.experimental.disable_x64():
        with pytest.raises(RuntimeError, match='float64'):
            create_state(np.zeros(3), np.eye(3))
