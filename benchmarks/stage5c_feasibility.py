"""Evaluate compilation reuse on untouched Stage 5B saved windows first."""

import json,time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.stage5c_cache import ProgramCache,optimize_cached
from estimation.fixed_window_objective import checked_evaluation
from estimation.slam import SlamState
from estimation.slam_window_objective import make_slam_window_objective


def load_window(case,index):
    root=Path('artifacts/milestone5b')/case
    cfg=json.loads((root/'results.json').read_text())['configuration']
    with np.load(root/f'window_{index:03d}.npz') as saved:w={k:saved[k] for k in saved.files}
    arrival=SlamState(*(w['arrival_'+k] for k in SlamState._fields))
    evaluate=make_slam_window_objective(arrival,w['times'],lambda t:jnp.asarray(cfg['command_v_omega']),
        w['basis'],cfg['continuous_input_noise_spectral_density_Qc'],w['measurements'],w['ids'],
        cfg['measurement_covariance'],w['mask'])
    return evaluate,w


def main():
    jax.config.update('jax_enable_x64',True)
    output=Path('artifacts/milestone5c');cache=ProgramCache();records=[]
    for case,index in [('regression_constant',0),('regression_constant',1),('forward_constant',2)]:
        evaluator,w=load_window(case,index)
        reference=jax.jit(evaluator)
        reference_pair=jax.jit(jax.value_and_grad(lambda c:evaluator(c).innovation_quadratic))
        entry,constants,timing,key=cache.prepare(evaluator,np.zeros(14))
        maximum=0.;gradient_error=0.;times={}
        for c in (np.zeros(14),np.linspace(-.04,.06,14)):
            a=checked_evaluation(reference,c)
            b=checked_evaluation(lambda c:entry['compiled_evaluate'](c,constants),c)
            for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
                if np.issubdtype(np.asarray(x).dtype,np.inexact):
                    maximum=max(maximum,float(np.max(np.abs(np.asarray(x)-np.asarray(y)),initial=0.)))
                    np.testing.assert_allclose(x,y,rtol=2e-12,atol=2e-12)
                else:np.testing.assert_array_equal(x,y)
            va,ga=reference_pair(c);vb,gb=entry['compiled_pair'](c,constants)
            gradient_error=max(gradient_error,float(np.max(np.abs(np.asarray(ga)-np.asarray(gb)))))
            np.testing.assert_allclose(ga,gb,rtol=1e-10,atol=1e-9)
        for name,fn in [('reference_objective',reference),('cached_objective',lambda c:entry['compiled_evaluate'](c,constants)),
                        ('reference_gradient',reference_pair),('cached_gradient',lambda c:entry['compiled_pair'](c,constants))]:
            samples=[]
            for _ in range(3):
                begin=time.perf_counter();jax.block_until_ready(fn(np.zeros(14)));samples.append(time.perf_counter()-begin)
            times[name]=samples
        records.append({'case':case,'window':index,'maximum_result_difference':maximum,
                        'maximum_gradient_difference':gradient_error,'timing':timing,'execution_samples':times,'key':key})
        print(json.dumps(records[-1]),flush=True)
    # Compare a complete cached solve to the preserved zero-start first fit.
    evaluate,_=load_window('regression_constant',0)
    fit=optimize_cached(cache,evaluate,np.zeros(14))
    original=json.loads(Path('artifacts/milestone5b/regression_constant/results.json').read_text())['updates'][0]['fits']['zero']['optimization']
    cost_error=abs(fit['objective_final']-original['objective_final'])
    assert fit['success'] and not fit['invalid_trials'] and cost_error<1e-7
    payload={'jax_version':jax.__version__,'windows':records,'program_count':len(cache.entries),
        'cached_fit':fit,'reference_cost':original['objective_final'],'fit_cost_difference':cost_error,
        'fit_coefficient_max_difference':float(np.max(np.abs(np.asarray(fit['coefficients'])-original['coefficients']))),
        'passed':True,'architecture_changed':False}
    (output/'feasibility.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
