"""Host-side C++ basis evaluation and JAX coefficient reconstruction.

Evaluate fixed elapsed-time grids before entering a JAX transformation. Only
the coefficients are differentiated; changing times, L, degree, or M requires
precomputing a new basis array. No basis values are cached implicitly.
"""

from operator import index

import numpy as np
from numpy.typing import ArrayLike

try:
    from bspline_module import UniformBSpline
except ImportError as error:
    raise ImportError(
        "Build ../bsplines_library into build/bsplines_library and add that "
        "directory to PYTHONPATH; see splines/README.md."
    ) from error


class UniformBasis:
    """Adapter for the uniform, unclamped ``UniformBSpline(degree, M)``.

    Coefficient index j corresponds to b^d(s + d - j). Degree zero retains
    the binding's half-open endpoint convention: its vector is zero at s=M.
    For degree >= 1, the vector sums to one on the closed interval [0, M].
    """

    def __init__(self, degree: int, M: int):
        degree, M = index(degree), index(M)
        if degree < 0 or M < 1:
            raise ValueError("degree must be nonnegative and M must be positive")
        self._spline = UniformBSpline(degree, M)

    @property
    def degree(self) -> int:
        return self._spline.degree

    @property
    def M(self) -> int:
        return self._spline.M

    @property
    def num_coefficients(self) -> int:
        """Number of coefficients per disturbance channel, M + degree."""
        return self.M + self.degree

    def evaluate(self, sigma: ArrayLike, L: float) -> np.ndarray:
        """Return float64 basis values with shape ``sigma.shape + (M+d,)``.

        sigma is elapsed time in [0, L], where L is finite and positive.
        Scalar input returns (M+d,). Arbitrary batches, empty batches, and
        repeated times are supported, including all four RK4 stage times.
        Scaling uses (sigma/L)*M so sigma=L maps exactly to the endpoint M.
        """
        L = float(L)
        if not np.isfinite(L) or L <= 0:
            raise ValueError("L must be finite and positive")
        times = np.asarray(sigma, dtype=np.float64)
        if not np.all(np.isfinite(times)) or np.any((times < 0) | (times > L)):
            raise ValueError("sigma must contain finite elapsed times in [0, L]")
        normalized = (times / L) * self.M
        values = [self._spline.basis_vector(float(s)) for s in np.ravel(normalized)]
        return np.asarray(values, dtype=np.float64).reshape(
            times.shape + (self.num_coefficients,)
        )

    def precompute_jax(self, sigma: ArrayLike, L: float):
        """Evaluate on the host and convert to JAX using its precision policy.

        Call outside grad/jit and reuse the returned constant array. Enable
        jax_enable_x64 in the application before calling for float64 values.
        """
        import jax.numpy as jnp

        return jnp.asarray(self.evaluate(sigma, L))


def disturbance_from_coefficients(basis, coefficients):
    """Return JAX disturbances (..., 2) ordered [w_v, w_omega].

    basis has shape (..., n), n=M+d, and is precomputed independently of
    coefficients. coefficients is a flat vector [c_v; c_omega] of length 2n.
    This function supports grad/jit with respect to that coefficient vector.
    """
    import jax.numpy as jnp

    basis = jnp.asarray(basis)
    coefficients = jnp.asarray(coefficients)
    if basis.ndim < 1 or basis.shape[-1] < 1:
        raise ValueError("basis must have a nonempty coefficient axis")
    n = basis.shape[-1]
    if coefficients.shape != (2 * n,):
        raise ValueError(f"coefficients must have shape ({2 * n},) as [c_v; c_omega]")
    return basis @ coefficients.reshape(2, n).T
