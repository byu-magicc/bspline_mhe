"""Check full-size saved moving windows against central finite differences."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.milestone4b_smooth import gradient_validation
from estimation.slam import SlamState
from estimation.slam_window_objective import make_slam_window_objective


def main():
    jax.config.update('jax_enable_x64',True)
    output=Path('artifacts/milestone5b');records={}
    for name,index in [('regression_constant',0),('regression_constant',1),('forward_constant',2)]:
        case=output/name
        run=json.loads((case/'results.json').read_text());cfg=run['configuration']
        with np.load(case/f'window_{index:03d}.npz') as saved:
            w={k:saved[k] for k in saved.files}
        arrival=SlamState(*(w['arrival_'+k] for k in SlamState._fields))
        evaluator=make_slam_window_objective(arrival,w['times'],lambda t:jnp.asarray(cfg['command_v_omega']),
            w['basis'],cfg['continuous_input_noise_spectral_density_Qc'],w['measurements'],w['ids'],
            cfg['measurement_covariance'],w['mask'])
        evaluate=jax.jit(evaluator)
        gradient=jax.jit(jax.value_and_grad(lambda c:evaluator(c).innovation_quadratic))
        optimized=run['updates'][index]['fits']['projected']['optimization']['coefficients']
        record=gradient_validation(evaluate,gradient,{'zero':np.zeros(14),'nonconstant':np.linspace(-.04,.06,14),
                                                      'optimized':np.asarray(optimized)})
        record.update(time=run['updates'][index]['time'],start=run['updates'][index]['start'],
            note='float64, all 14 coefficients, central steps 1e-4 and 1e-5; original rtol/atol 2e-5 and maximum absolute error <2e-5')
        records[f'{name}_{index:03d}']=record
        print(json.dumps({'window':f'{name}_{index:03d}','passed':record['passed']}),flush=True)
        jax.clear_caches()
    payload={'passed':all(r['passed'] for r in records.values()),'windows':records,
             'command':'PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5b_gradients'}
    (output/'gradient_checks.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    if not payload['passed']:raise RuntimeError('Saved-window finite differences failed')


if __name__=='__main__':main()
