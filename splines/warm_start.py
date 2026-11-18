"""Host-side integral projection of two-channel uniform input splines.

Only basis arrays from the existing C++ adapter are used. QR/SVD solves the
weighted least-squares system, avoiding squared conditioning from solving
normal equations. G and B are retained for audit of C_new G = C_old B.T.
"""

from dataclasses import dataclass
from functools import lru_cache
from operator import index

import numpy as np

from .uniform_basis import UniformBasis


@dataclass(frozen=True)
class WarmStartProjection:
    degree: int
    M: int
    L: float
    delta: float
    quadrature_order: int
    breakpoints: np.ndarray
    gram: np.ndarray
    cross: np.ndarray
    transform: np.ndarray
    diagnostics: dict

    def apply(self, coefficients):
        """Project (2,n) C or flat [c_v;c_omega], preserving the input shape.

        This is an unconstrained initial guess. No bound clipping, coefficient
        penalty, objective term or estimator state update is performed here.
        """
        c = np.asarray(coefficients, dtype=np.float64)
        n = self.M+self.degree
        if c.shape not in ((2,n),(2*n,)) or not np.isfinite(c).all():
            raise ValueError(f'coefficients must be finite (2,{n}) or ({2*n},) [c_v;c_omega]')
        if self.delta == 0 or self.diagnostics['method'] in (
                'integer_knot_translation_repeat_terminal','empty_overlap_repeat_terminal'):
            return (c.reshape(2,n) @ self.transform.T).reshape(c.shape)
        # Constants are exactly invariant by partition of unity. Separate the
        # constant component before an ill-conditioned projection so roundoff
        # in T @ ones cannot amplify a constant input into spurious variation.
        matrix = c.reshape(2,n)
        center = matrix[:,-1:]
        return ((matrix-center) @ self.transform.T+center).reshape(c.shape)


def _readonly(array):
    # Immutable backing storage also prevents callers from re-enabling writes.
    value = np.asarray(array, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def make_warm_start_projection(degree, M, L, delta, *, quadrature_order=None):
    """Cached projection for 0 <= delta <= L (exact float keys).

    Default d+1 Gauss points integrate degree-2d products exactly on each
    interval split at both new knots j and old shifted knots j-delta*M/L.
    Lower orders are available explicitly for convergence diagnostics.
    """
    degree, M = index(degree), index(M)
    L, delta = float(L), float(delta)
    order = degree+1 if quadrature_order is None else index(quadrature_order)
    if degree < 0 or M < 1 or order < 1:
        raise ValueError('degree >= 0, M >= 1 and quadrature_order >= 1 required')
    if not np.isfinite([L,delta]).all() or L <= 0 or not 0 <= delta <= L:
        raise ValueError('finite L > 0 and 0 <= delta <= L required')
    return _cached_projection(degree,M,L,delta,order)


@lru_cache(maxsize=128)
def _cached_projection(degree,M,L,delta,order):
    n = M+degree
    h = L/M
    ratio = delta/h
    integer_shift = ratio.is_integer()
    shift = float(round(ratio)) if integer_shift else delta/L*M
    end = M-shift
    knots = np.arange(M+1,dtype=np.float64)
    breaks = np.unique(np.r_[0.,end,knots[(knots>0)&(knots<end)],
                            (knots-shift)[(knots-shift>0)&(knots-shift<end)]])
    q = min(int(np.floor(ratio)),M)
    # C_ref[:,j] = C_old[:,min(j+q,n-1)]: translate full knots,
    # then repeat the final coefficient in unconstrained terminal slots.
    reference = np.eye(n)[np.minimum(np.arange(n)+q,n-1)]
    if delta == L:
        reference = np.eye(n)[np.full(n,n-1)]
        A, D = np.zeros((0,n)), np.zeros((0,n))
    else:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        left, right = breaks[:-1], breaks[1:]
        sigma = ((left+right)[:,None]/2+(right-left)[:,None]/2*nodes).ravel()
        weights = ((right-left)[:,None]/2*weights*(L/M)).ravel()
        adapter = UniformBasis(degree,M)
        # Evaluate normalized time through the adapter, never a new basis.
        A = np.sqrt(weights)[:,None]*adapter.evaluate(sigma,float(M))
        D = np.sqrt(weights)[:,None]*adapter.evaluate(sigma+shift,float(M))
    G, B = A.T@A, A.T@D
    asymmetry = float(np.max(np.abs(G-G.T)))
    eigenvalues = np.linalg.eigvalsh(G)
    U, singular, Vh = np.linalg.svd(A,full_matrices=False)
    largest = float(singular[0]**2) if len(singular) else 0.
    tolerance = n*np.finfo(np.float64).eps*largest
    resolved = singular**2 > tolerance
    rank = int(resolved.sum())
    if asymmetry > 16*np.finfo(float).eps*max(largest,1.) or eigenvalues[0] < -16*max(tolerance,np.finfo(float).tiny):
        raise ArithmeticError('quadrature Gram matrix failed symmetry/PSD checks')
    if delta == 0:
        transform, method = np.eye(n), 'zero_shift_identity'
    elif delta == L:
        transform, method = reference, 'empty_overlap_repeat_terminal'
    elif integer_shift:
        transform, method = reference, 'integer_knot_translation_repeat_terminal'
    elif rank == n:
        # A T = D implies G T = B, hence C_new G = C_old B.T.
        Q, R = np.linalg.qr(A,mode='reduced')
        transform, method = np.linalg.solve(R,Q.T@D), 'full_rank_qr'
    else:
        # Resolved least-squares correction with minimum distance from C_ref.
        # The threshold is exactly the Gram numerical-rank criterion, not a
        # diagonal regularizer. Unresolved directions retain their reference.
        correction = Vh[resolved].T @ ((U[:,resolved].T@(D-A@reference))/singular[resolved,None])
        transform, method = reference+correction, 'rank_deficient_svd_reference_continuation'
    all_positive = len(singular)==n and singular[-1]>0
    condition = float((singular[0]/singular[-1])**2) if all_positive else None
    # Exact structural zeros in G imply an infinite full condition number.
    if np.any(np.all(A==0,axis=0)):
        condition = None
    diagnostics = {
        'method':method, 'gram_rank':rank, 'gram_dimension':n,
        'gram_rank_tolerance':tolerance, 'rank_rule':'lambda > n*float64_epsilon*lambda_max; lambda from squared weighted-design singular values',
        'gram_condition_number':condition, 'gram_min_eigenvalue':float(eigenvalues[0]),
        'gram_max_eigenvalue':float(eigenvalues[-1]), 'gram_max_asymmetry':asymmetry,
        'gram_eigenvalues':tuple(eigenvalues.tolist()), 'weighted_design_singular_values':tuple(singular.tolist()),
        'numerically_rank_deficient':rank<n, 'mathematically_full_rank':delta<h,
        'normal_equation_operator_residual_inf':float(np.linalg.norm(G@transform-B,ord=np.inf)),
        'quadrature_points':len(A), 'quadrature_exact_for_products':order>=degree+1,
        'terminal_policy':'translate floor(delta/h) whole knots; repeat final coefficient in terminal slots; retain reference in unresolved SVD directions',
        'precision':'float64', 'coefficient_bounds_enforced':False}
    # An immutable mapping prevents corruption of shared cached diagnostics.
    from types import MappingProxyType
    return WarmStartProjection(degree,M,L,delta,order,_readonly(breaks*(L/M)),
                               _readonly(G),_readonly(B),_readonly(transform),MappingProxyType(diagnostics))


def projection_cache_info():
    return _cached_projection.cache_info()


def clear_projection_cache():
    _cached_projection.cache_clear()
