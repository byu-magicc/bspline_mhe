"""Direct post-run host profiling, with complete saved-window equivalence checks.

The local Python 3.12/pybind profiler omitted UniformBasis.evaluate from its
completed cProfile frames. Never interpret the resulting zero as zero work.
These separate remeasurements do not overwrite the original online timings.
"""

import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.milestone5c import ROOT, TimedForward, write_json
from estimation.moving_horizon import ReplayData
from estimation.slam import SlamState
from splines import UniformBasis


def profile_saved_windows(root=ROOT, repeats=3):
    if repeats < 1:
        raise ValueError('at least one repeat is required')
    records=[]
    evaluate=UniformBasis.evaluate
    for case in sorted(root.glob('*/results.json')):
        run=json.loads(case.read_text());out=case.parent
        with np.load(out/'dataset.npz') as saved:
            d={k:saved[k] for k in saved.files}
        forward=TimedForward.__new__(TimedForward)
        forward.data=ReplayData(d['times'],d['measurements'],d['ids'],d['measurement_covariance'],d['mask'],d['scheduled'])
        forward.degree,forward.M,forward.max_step=3,4,.025
        with np.load(out/'forward.npz') as saved:
            forward.result=SimpleNamespace(states=SlamState(*(jnp.asarray(saved[k]) for k in SlamState._fields)))
        jax.block_until_ready(forward.result.states)
        # Exclude startup of JAX indexing/copy dispatch from this host remeasurement.
        forward.window(run['updates'][0]['time'],5.)
        for index,row in enumerate(run['updates']):
            with np.load(out/f'window_{index:03d}.npz') as saved:
                expected={k:saved[k] for k in saved.files}
            for repeat in range(repeats):
                basis_times=[]
                def timed_basis(self,*args,**kwargs):
                    begin=time.perf_counter()
                    result=evaluate(self,*args,**kwargs)
                    basis_times.append(time.perf_counter()-begin)
                    return result
                with patch.object(UniformBasis,'evaluate',timed_basis):
                    begin=time.perf_counter();window=forward.window(row['time'],5.)
                    total=time.perf_counter()-begin
                assert len(basis_times)==1 and basis_times[0]>0
                for key in ('times','basis','measurements','ids','mask','source_indices'):
                    np.testing.assert_array_equal(getattr(window,key),expected[key])
                for key in SlamState._fields:
                    np.testing.assert_array_equal(getattr(window.arrival,key),expected['arrival_'+key])
                remainder=total-forward.arrival_seconds-basis_times[0]
                assert remainder>=0
                records.append({'case':out.name,'window':index,'repeat':repeat,
                    'window_seconds':total,'arrival_seconds':forward.arrival_seconds,
                    'basis_seconds':basis_times[0],'buffer_other_seconds':remainder})
    result={'passed':True,'window_count':len(records)//repeats,'repeats':repeats,
        'all_saved_window_fields_bitwise_equal':True,'records':records,
        'interpretation':'Separate direct post-run measurements; original cProfile basis field is unresolved, '
            'and original observation_buffer_seconds includes basis work. No fits or original timings replaced.'}
    write_json(root/'host_profile_remeasurement.json',result)
    return result


if __name__=='__main__':
    jax.config.update('jax_enable_x64',True)
    result=profile_saved_windows()
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))
