# Milestone 2: uniform input-spline basis

The adapter calls the sibling library's existing C++/pybind11
`bspline_module.UniformBSpline(degree, M).basis_vector(s)`. It does not implement
a basis recurrence. The inspected and tested dependency is
`byu-magicc/bsplines_library`, branch `uniform_development`, commit
`bb334d0f75cd0a244c73c0de3468d049e0185ebe`.

## Build and import

Run from the `bspline_mhe` root with the intended Python environment activated.
The sibling CMake file invokes `python3` to locate pybind11, so that executable
must belong to the same environment used to import the extension. Requirements
are a C++11 compiler, CMake >= 3.12, Python development files, and pybind11.
The adapter uses NumPy; coefficient reconstruction and numerical tests use JAX.

```sh
git -C ../bsplines_library branch --show-current
cmake -S ../bsplines_library -B build/bsplines_library \
  -DCMAKE_BUILD_TYPE=Release -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$(command -v python3)"
cmake --build build/bsplines_library --config Release --parallel 2
PYTHONPATH=build/bsplines_library python3 -B -c \
  'import bspline_module; print(bspline_module.__file__)'
```

All generated files are inside this repository's ignored `build/` directory.
No installation target or modification of the sibling source is needed. The
commands use CMake's default single-configuration generator; if using a generator
that places the extension in a configuration subdirectory, put that directory on
`PYTHONPATH` instead. Rebuild for a different Python ABI or platform.

Validated environment: macOS, AppleClang 15.0.0, CMake 3.30.4, Python 3.12.6,
pybind11 3.0.4, NumPy 2.0.2, JAX/JAXLIB 0.4.35, pytest 8.3.3. The existing
`requirements-baseline.txt` records the Milestone 1 test dependencies.

## Basis arrays and coefficients

For elapsed time `sigma` in `[0, L]`, the adapter uses `s = (sigma/L)*M`, equivalent
to `(M/L)*sigma`, while mapping the exact endpoint `sigma=L` to `s=M`.
`evaluate(sigma, L)` returns a NumPy float64 array with shape
`sigma.shape + (M+degree,)`; a scalar returns `(M+degree,)`. Inputs must be finite,
`L > 0`, `degree >= 0`, and integer `M >= 1`. Arbitrary batch shapes, repeated
times, and empty batches are supported.

Coefficient index `j` corresponds to `b^d(s+d-j)`, matching `project.md`. For each
channel there are `n=M+d` coefficients. The flat coefficient vector has ordering
`c = [c_v; c_omega]`. `disturbance_from_coefficients(B, c)` uses JAX operations and
returns `B.shape[:-1] + (2,)`, with channel order `[w_v, w_omega]`.

```python
import jax
import jax.numpy as jnp
import numpy as np

from splines import UniformBasis, disturbance_from_coefficients

jax.config.update("jax_enable_x64", True)
basis = UniformBasis(degree=3, M=4)
L = 2.0
edges = np.array([0.0, 0.3, 0.9, 2.0])
mid = (edges[:-1] + edges[1:]) / 2
stage_times = np.stack([edges[:-1], mid, mid, edges[1:]], axis=-1)
B_numpy = basis.evaluate(stage_times, L)      # (3, 4, 7)
B = basis.precompute_jax(stage_times, L)     # same values as jnp.asarray(B_numpy)
c = jnp.concatenate([jnp.full(7, 0.2), jnp.full(7, -0.1)])
w = disturbance_from_coefficients(B, c)     # (3, 4, 2)
gradient = jax.grad(lambda c: jnp.sum(disturbance_from_coefficients(B, c)**2))(c)
```

Precompute outside `jax.grad`/`jax.jit` and reuse the returned array as a constant.
The adapter does not change JAX's global precision policy and does not cache
arrays. Recompute when the degree, M, L, or elapsed-time grid changes. Preserve
all four RK4 stages, including the two equal midpoint times. This prepares the
basis data for Milestone 3 without implementing its dynamics or integration.

## Endpoint convention and validation

For tested degrees 1–5, partition of unity holds on `[0, M]`, including `s=M`,
and the right endpoint agrees with the limit from the left. Uniform splines
are unclamped: for example, the cubic endpoint's last three entries are
`[1/6, 4/6, 1/6]`, rather than an interpolating endpoint control point.

Degree zero retains the library's half-open interval convention. At `s=M` all
entries are zero; immediately to the left the last entry is one. Partition of
unity for degree zero therefore applies on `[0, M)`, and left continuity does
not hold at the right endpoint. This behavior is explicitly tested; no endpoint
replacement or sibling-library patch is applied.

```sh
PYTHONPATH=build/bsplines_library python3 -B -m pytest -q tests/test_uniform_basis.py
python3 -B -m pytest -q tests/test_estimator_a.py \
  tests/test_diagnose_initializations.py tests/test_visualize_estimator_a.py
```

The new tests cover dimensions/order, compact support, nonnegativity, partition
of unity, endpoints, changing time grids/horizons, both disturbance channels,
and JAX gradients. They enable float64 with a module-scoped JAX precision
context. Partition/scaling errors are bounded by `2e-15`; continuity is checked
at a normalized distance `1e-8` and at the preceding floating-point time. For a
quadratic reconstruction objective, gradients are checked against the analytic
gradient (`rtol=atol=2e-13`) and central finite differences with step `1e-5`
(`rtol=1e-8`, `atol=1e-9`). This tests coefficient differentiation through the
reconstruction only; EKF innovation-objective gradients belong to later gates.

Validation run: **93 spline tests passed** in 1.16 s, and **45 existing Milestone 1
tests passed** in 15.14 s. Both commands exited successfully. Hash checks
confirmed all saved artifacts, benchmark modules, existing tests, and the root
README were unchanged; 142 captured baseline file hashes also matched.
