"""Isolated JAX constant lifting for profiling the unchanged Stage 4C program.

Pinned to the installed JAX 0.4.35 program interpreter. No EKF equations,
observation shapes, plans or solver policies are replaced. The optimizer below
is a static copy of the reference orchestration with only bridge construction
injected; tests enforce that correspondence at the AST level.
"""

import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from estimation.coefficient_optimization import (
    CheckedScipyObjective, DEFAULT_OPTIONS, feasibility_probes,
    constant_to_spline, gradient_norms,
)
from estimation.fixed_window_objective import InvalidCandidateError


def whitened_residual(result):
    """Lower-Cholesky whitening, fixed range/bearing and record ordering."""
    L=jnp.linalg.cholesky(result.innovation_covariances)
    q=jnp.linalg.solve(L,result.innovations[...,None])[...,0]
    return jnp.where(result.valid_mask[...,None],q,0.).reshape(-1)


class ProgramCache:
    def __init__(self):
        if jax.__version__!='0.4.35':
            raise RuntimeError('Revalidate program lifting against the installed JAX version')
        self.entries={}
        self.events=[]

    def prepare(self,evaluator,initial):
        start=time.perf_counter()
        closed,shape=jax.make_jaxpr(evaluator,return_shape=True)(jnp.asarray(initial))
        trace_seconds=time.perf_counter()-start
        tree=jax.tree_util.tree_structure(shape)
        key=hashlib.sha256((str(closed.jaxpr)+str(tree)).encode()).hexdigest()
        timing={'program_trace_seconds':trace_seconds,'program_cache_hit':key in self.entries,
                'evaluation_lower_seconds':0.,'gradient_lower_seconds':0.,
                'evaluation_compile_seconds':0.,'value_gradient_compile_seconds':0.}
        if key not in self.entries:
            graph=closed.jaxpr
            def function(c,constants):
                return jax.tree_util.tree_unflatten(tree,jax.core.eval_jaxpr(graph,constants,c))
            evaluate=jax.jit(function)
            pair=jax.jit(jax.value_and_grad(lambda c,constants:function(c,constants).innovation_quadratic,argnums=0))
            args=(jnp.asarray(initial),tuple(closed.consts))
            begin=time.perf_counter();lower=evaluate.lower(*args)
            timing['evaluation_lower_seconds']=time.perf_counter()-begin
            begin=time.perf_counter();compiled_evaluate=lower.compile()
            timing['evaluation_compile_seconds']=time.perf_counter()-begin
            begin=time.perf_counter();lower=pair.lower(*args)
            timing['gradient_lower_seconds']=time.perf_counter()-begin
            begin=time.perf_counter();compiled_pair=lower.compile()
            timing['value_gradient_compile_seconds']=time.perf_counter()-begin
            self.entries[key]={'function':function,'evaluate':evaluate,'pair':pair,
                'compiled_evaluate':compiled_evaluate,'compiled_pair':compiled_pair}
        self.events.append(dict(timing,key=key))
        return self.entries[key],tuple(closed.consts),timing,key

    def sensitivity(self,entry,constants,c):
        """D of L(c)^-1 r(c), including Cholesky covariance dependence."""
        if 'jacobian' not in entry:
            def residual(coefficients,data):
                result=entry['function'](coefficients,data)
                return whitened_residual(result)
            fn=jax.jit(jax.jacfwd(residual,argnums=0))
            start=time.perf_counter();lower=fn.lower(jnp.asarray(c),constants)
            lowering=time.perf_counter()-start
            start=time.perf_counter();entry['jacobian']=lower.compile()
            compilation=time.perf_counter()-start
            entry['residual']=jax.jit(residual)
        else:lowering,compilation=0.,0.
        start=time.perf_counter();D=np.asarray(entry['jacobian'](jnp.asarray(c),constants))
        return D,{'lower_seconds':lowering,'compile_seconds':compilation,
                  'execution_seconds':time.perf_counter()-start}


class CachedBridge(CheckedScipyObjective):
    """Inherit every candidate check, reference check and trial accounting rule."""
    def __init__(self,cache,evaluator,mapping,initial):
        self.mapping=mapping
        entry,data,timing,key=cache.prepare(lambda z:evaluator(mapping(z)),initial)
        self.entry,self.constants,self.program_key=entry,data,key
        self.evaluate=lambda z:entry['evaluate'](z,data)
        self.value_gradient=lambda z:entry['pair'](z,data)
        self.compiled_evaluate=lambda z:entry['compiled_evaluate'](z,data)
        self.compiled_value_gradient=lambda z:entry['compiled_pair'](z,data)
        self.timings=timing
        self.invalid_trials=[];self.evaluation_history=[];self.cache={};self.gradient_returns=0


def optimize_cached(program_cache, evaluator, initial, *, coefficients_per_channel=7, constant=False,
                          bounds=None, options=None, random_probe_count=8):
    """L-BFGS-B fit with clean stop on the first invalid probe or SciPy trial.

    The returned point after an invalid trial is the last accepted iterate,
    never an invalid trial or a claimed solution. No objective/gradient is
    returned to SciPy for an invalid request. No automatic domain revision.
    """
    if not jax.config.x64_enabled:
        raise RuntimeError('Stage 4B requires JAX float64')
    initial = np.asarray(initial, dtype=np.float64)
    size = 2 if constant else 2*coefficients_per_channel
    if initial.shape != (size,) or not np.isfinite(initial).all():
        raise ValueError(f'initial must be a finite vector of length {size}')
    bounds = np.tile([-.5, .5], (size, 1)) if bounds is None else np.asarray(bounds, dtype=np.float64)
    probes = feasibility_probes(bounds, random_probe_count)
    if bounds.shape != (size, 2) or np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ValueError('initial must lie inside matching parameter bounds')
    mapping = ((lambda z: constant_to_spline(z, coefficients_per_channel)) if constant else
               (lambda z: jnp.asarray(z, dtype=jnp.float64)))
    preparation_start = time.perf_counter()
    bridge = CachedBridge(program_cache, evaluator, mapping, initial)
    output = {'method': 'L-BFGS-B', 'parameterization': 'constant' if constant else 'full_spline',
              'bounds': bounds.tolist(), 'options': {**DEFAULT_OPTIONS, **(options or {})},
              'initial_parameters': initial.tolist(), 'success': False,
              'feasibility_probes': [], 'invalid_trials': bridge.invalid_trials,
              'history': [], 'evaluation_history': bridge.evaluation_history,
              'timing': bridge.timings, 'reference_consistency': None,
              'parameters': None, 'coefficients': None, 'objective_initial': None,
              'objective_final': None, 'objective_reduction': None,
              'iterations': 0, 'function_evaluations': 0, 'gradient_evaluations': 0,
              'scipy_status': None}
    try:
        bridge.reference_check(initial)
        output['reference_consistency'] = bridge.reference_errors
        initial_value, initial_gradient = bridge(initial, 'initial')
        output['objective_initial'] = initial_value
        output.update(parameters=initial.tolist(), coefficients=np.asarray(mapping(initial)).tolist(),
                      objective_final=initial_value, objective_reduction=0.,
                      **gradient_norms(initial, initial_gradient, bounds))
        for point in probes:
            try:
                result = bridge.check(point, 'feasibility_probe')
            except InvalidCandidateError:
                output['feasibility_probes'].append({'parameters': point.tolist(), 'valid': False})
                raise
            output['feasibility_probes'].append({'parameters': point.tolist(), 'valid': True,
                                                'objective': float(result.innovation_quadratic)})
    except InvalidCandidateError as error:
        output.update(status='stopped_invalid_preflight', message=str(error))
        bridge.timings['preparation_seconds'] = time.perf_counter()-preparation_start
        bridge.timings['optimization_seconds'] = 0.
        return output
    bridge.timings['preparation_seconds'] = time.perf_counter()-preparation_start
    optimization_start = time.perf_counter()
    last_accepted = initial.copy()

    def record(x, value, gradient, iteration):
        return {'iteration': iteration, 'parameters': x.tolist(),
                'coefficients': np.asarray(mapping(x)).tolist(), 'objective': float(value),
                'elapsed_optimization_seconds': time.perf_counter()-optimization_start,
                **gradient_norms(x, gradient, bounds)}

    output['history'].append(record(initial, initial_value, initial_gradient, 0))

    def callback(intermediate_result):
        nonlocal last_accepted
        x = np.asarray(intermediate_result.x).copy()
        value, gradient = bridge.cache[tuple(x)]
        output['history'].append(record(x, value, gradient, len(output['history'])))
        last_accepted = x

    try:
        result = minimize(bridge, initial, method='L-BFGS-B', jac=True,
                          bounds=bounds, callback=callback, options=output['options'])
        last_accepted = np.asarray(result.x)
        output.update(status='converged' if result.success else 'scipy_not_converged',
                      success=bool(result.success), message=str(result.message),
                      iterations=int(result.nit), scipy_status=int(result.status),
                      function_evaluations=int(result.nfev), gradient_evaluations=int(result.njev))
    except InvalidCandidateError as error:
        output.update(status='stopped_invalid_trial', success=False,
                      message=f'{error}. Search region or line-search strategy requires revision.',
                      iterations=len(output['history'])-1,
                      function_evaluations=sum(r['phase']=='optimization' for r in bridge.evaluation_history)+1,
                      gradient_evaluations=bridge.gradient_returns)
    bridge.timings['optimization_seconds'] = time.perf_counter()-optimization_start
    for field in ('validity_check_seconds', 'value_gradient_seconds'):
        samples = [row[field] for row in bridge.evaluation_history if row['phase'] == 'optimization']
        bridge.timings[field+'_mean'] = float(np.mean(samples)) if samples else None
        bridge.timings[field+'_p95'] = float(np.percentile(samples, 95)) if samples else None
    final_value, final_gradient = bridge.cache[tuple(last_accepted)]
    output.update(parameters=last_accepted.tolist(), coefficients=np.asarray(mapping(last_accepted)).tolist(),
                  objective_final=final_value, objective_reduction=initial_value-final_value,
                  **gradient_norms(last_accepted, final_gradient, bounds))
    return output
