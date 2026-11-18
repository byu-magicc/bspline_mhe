"""Numerical evidence for Stage 5A only; no moving-horizon estimator or fitting."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad

from splines import UniformBasis
from splines.warm_start import make_warm_start_projection, projection_cache_info


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def adaptive_error(d,M,L,delta,old,new):
    """Independent physical-time adaptive Gauss-Kronrod integral."""
    basis=UniformBasis(d,M)
    knots=np.arange(M+1)*(L/M)
    points=np.unique(np.r_[knots,knots-delta])
    points=points[(points>0)&(points<L-delta)]
    def residual(t):
        e=new@basis.evaluate(t,L)-old@basis.evaluate(t+delta,L)
        return float(e@e)
    return quad(residual,0.,L-delta,points=points,epsabs=2e-12,epsrel=2e-11,limit=200)


def case(d,M,L,delta,old=None):
    p=make_warm_start_projection(d,M,L,delta)
    n=d+M
    if old is None:
        old=np.random.default_rng(7).normal(scale=.3,size=(2,n))
    projected=p.apply(old)
    error,quad_error=adaptive_error(d,M,L,delta,old,projected)
    retained,_=adaptive_error(d,M,L,delta,old,old)
    constant=np.repeat([[.3],[-.17]],n,axis=1)
    return {'degree':d,'segments':M,'L':L,'delta':delta,'delta_over_knot':delta/(L/M),
        'diagnostics':dict(p.diagnostics),
        'normal_equation_max_absolute_residual':float(np.max(np.abs(projected@p.gram-old@p.cross.T))),
        'constant_coefficient_max_absolute_error':float(np.max(np.abs(p.apply(constant)-constant))),
        'adaptive_integrated_error':error,'adaptive_error_estimate':quad_error,
        'retained_coefficients_integrated_error':retained,
        'previous_coefficients':old.ravel().tolist(),'projected_coefficients':projected.ravel().tolist(),
        'max_absolute_projected_coefficient':float(np.max(np.abs(projected)))}


def projection_figure(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d,M,L=3,4,5.;adapter=UniformBasis(d,M)
    old=np.random.default_rng(7).normal(scale=.3,size=(2,M+d))
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for row,delta in enumerate((.625,1.25)):
        p=make_warm_start_projection(d,M,L,delta);new=p.apply(old)
        overlap=np.linspace(0.,L-delta,401)
        t=np.linspace(0.,L,501)
        reference=adapter.evaluate(overlap+delta,L)@old.T
        for channel in range(2):
            axis=axes[row,channel]
            axis.plot(overlap,reference[:,channel],color='black',label='Previous spline at sigma + delta')
            axis.plot(t,(adapter.evaluate(t,L)@old.T)[:,channel],ls=':',color='#0072B2',label='Retained coefficients')
            axis.plot(t,(adapter.evaluate(t,L)@new.T)[:,channel],ls='--',color='#D55E00',label='Projected coefficients')
            axis.axvspan(L-delta,L,color='.8',alpha=.5,label='Outside overlap: unconstrained continuation')
            axis.set_title(f'delta={delta:g} s ({delta/1.25:g} knot), channel '+('v' if channel==0 else 'omega'))
            axis.set_ylabel('Disturbance (arbitrary units)');axis.grid(alpha=.2)
            if row==1:axis.set_xlabel('New-window elapsed time sigma (s)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    fig.tight_layout(rect=(0,.10,1,1))
    for ext in ('png','pdf'):fig.savefig(output/f'projection_overlap.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    # Retain an existing pre-task snapshot. New output directories capture
    # their own inputs before this numerical report writes any results.
    manifest=output/'before_sha256.json'
    if not manifest.exists():
        paths=[]
        for folder in ('baseline','estimation','simulation','splines','benchmarks','tests','artifacts'):
            paths.extend(p for p in Path(folder).rglob('*') if p.is_file()
                         and '__pycache__' not in p.parts and p.name!='.DS_Store'
                         and output.resolve() not in p.resolve().parents)
        paths.extend((Path('project.md'),Path('README.md')))
        paths.extend(p for p in Path('../bsplines_library').rglob('*') if p.is_file()
                     and p.suffix in ('.cpp','.hpp','.h') and not any(x in p.parts for x in ('.git','build')))
        write_json(manifest,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    configs=[(0,3),(1,3),(2,4),(3,4),(5,2)]
    rows=[case(d,M,5.,fraction*5./M) for d,M in configs
          for fraction in (0.,.08,.37,.5,.9,1.,1.3)]
    conditioning=[case(3,4,5.,(1-eps)*1.25) for eps in (.5,.1,.01,.001,1e-4,1e-5)]
    quadrature=[]
    for d,M in configs:
        reference=make_warm_start_projection(d,M,5.,.37*5./M,quadrature_order=2*d+5)
        for order in sorted(set((1,d+1,2*d+5))):
            p=make_warm_start_projection(d,M,5.,.37*5./M,quadrature_order=order)
            quadrature.append({'degree':d,'segments':M,'order':order,
                'gram_max_absolute_error':float(np.max(np.abs(p.gram-reference.gram))),
                'cross_max_absolute_error':float(np.max(np.abs(p.cross-reference.cross))),
                'transform_max_absolute_error':float(np.max(np.abs(p.transform-reference.transform)))})
    saved=[]
    for path in sorted(Path('artifacts/milestone4c').glob('[ab]_*_results.json')):
        old=np.asarray(json.loads(path.read_text())['optimization']['coefficients']).reshape(2,7)
        saved.append({'source':str(path),**case(3,4,5.,.1,old)})
    tests=json.loads((output/'test_record.json').read_text()) if (output/'test_record.json').exists() else None
    snapshot=json.loads((output/'before_sha256.json').read_text())
    changes=[p for p,h in snapshot.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    preservation={'checked_count':len(snapshot),'changed':changes,'all_preserved':not changes}
    checks={
        'tests_passed':tests is not None and tests['exit_code']==0 and tests['passed']==188,
        'all_existing_files_preserved':not changes,
        'normal_equations':max(r['normal_equation_max_absolute_residual'] for r in rows+conditioning)<1e-11,
        'constants':max(r['constant_coefficient_max_absolute_error'] for r in rows+conditioning)<1e-14,
        'projection_no_worse_than_retained':all(r['adaptive_integrated_error']<=r['retained_coefficients_integrated_error']+5e-11 for r in rows+conditioning+saved),
        'quadrature_exact_order':all(max(r['gram_max_absolute_error'],r['cross_max_absolute_error'])<3e-14 for r in quadrature if r['order']>=r['degree']+1),
        'stage4c_coefficients_examined':len(saved)==4,
        'ill_conditioning_reported':conditioning[-1]['diagnostics']['numerically_rank_deficient']}
    record={'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'precision':'float64'},
            'cases':rows,'conditioning':conditioning,'quadrature':quadrature,'stage4c_saved_coefficients':saved,
            'test_record':tests,'preservation':preservation,'cache':projection_cache_info()._asdict(),
            'checks':checks,'stage5a_gate_satisfied':all(checks.values()),'stage5b_implemented':False}
    projection_figure(output)
    write_json(output/'results.json',record)
    write_json(output/'preservation.json',preservation)
    lines=['# Stage 5A — integral B-spline warm-start projection','',
        f"Stage 5A numerical gate: **{'SATISFIED' if record['stage5a_gate_satisfied'] else 'NOT YET SATISFIED'}**. Stage 5B is not implemented.",'',
        'The C++ uniform unclamped basis supplies every evaluation. Products use piecewise Gauss–Legendre quadrature, '
        'split at the new knots and shifted old knots. The default d+1 points integrate degree-2d products exactly '
        'on each span up to float64 roundoff. Flat coefficients remain [c_v;c_omega], reshaped as C with shape (2,M+d).','',
        'The cached map T solves G T = B, so C_new = C_old T.T satisfies C_new G = C_old B.T. '
        'A weighted-design QR solve avoids forming an inverse and avoids squaring the condition number during solution. '
        'Singular values of the weighted design determine Gram rank using lambda > n*eps*lambda_max. '
        'Condition numbers below are squared weighted-design singular-value ratios; exact structural singularity is infinity.','',
        'The SVD fallback solves only numerically resolved directions and retains an explicitly translated/repeated-terminal '
        'reference in unresolved directions. This is a rank-revealing numerical approximation when the exact Gram is '
        'full rank but ill-conditioned. It is not diagonal regularization or a change to the MHE objective. '
        'A constant channel component is separated before projection to avoid amplifying roundoff in the partition-of-unity identity.','',
        '## Errors and rank across configurations','',
        '| d | M | delta / knot | Numerical rank / n | Gram condition | Method | Normal residual max | Integrated projected / retained error |',
        '|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in rows:
        diag=r['diagnostics'];cond=diag['gram_condition_number']
        lines.append(f"| {r['degree']} | {r['segments']} | {r['delta_over_knot']:.3g} | {diag['gram_rank']}/{diag['gram_dimension']} | "
                     f"{format(cond,'.6g') if cond is not None else 'infinity'} | {diag['method']} | {r['normal_equation_max_absolute_residual']:.4g} | "
                     f"{r['adaptive_integrated_error']:.7g} / {r['retained_coefficients_integrated_error']:.7g} |")
    lines+=['','Errors are independent adaptive Gauss–Kronrod integrals on physical time, '
            'epsabs=2e-12, epsrel=2e-11, split at both sets of knots. Normal residual means max absolute entry of '
            'C_new G - C_old B.T. Random examples use seed 7 and independent coefficients in both channels. '
            'Coefficient constants [0.3,-0.17] are preserved exactly in these runs.','',
            '## Conditioning near one knot (d=3, M=4, L=5 s)','',
            '| delta / knot | Numerical rank | Gram condition | Projected error | Normal residual max |',
            '|---:|---:|---:|---:|---:|']
    for r in conditioning:
        diag=r['diagnostics']
        lines.append(f"| {r['delta_over_knot']:.7g} | {diag['gram_rank']} | {diag['gram_condition_number']:.7g} | "
                     f"{r['adaptive_integrated_error']:.7g} | {r['normal_equation_max_absolute_residual']:.5g} |")
    lines+=['','Strictly below one knot the exact Gram is positive definite. Near that boundary, '
            'the last basis has vanishing overlap and the matrix becomes numerically rank deficient; these cases use the reported SVD fallback. '
            'At one or more whole knots terminal basis columns are exactly unobserved, so the Gram is structurally singular.','',
            '## Continuation, endpoint and bounds policies','',
            'Zero shift returns the coefficient identity. For an exact q-knot shift, '
            'C_new[:,j]=C_old[:,min(j+q,n-1)]: shift left q coefficients and repeat the old final coefficient into terminal slots. '
            'This reconstructs the overlap exactly and selects a continuation not determined by the integral. '
            'Repeating the final coefficient is not a claim of constant extrapolated disturbance near an unclamped endpoint. '
            'For delta=L there is no overlap; all new coefficients repeat the old final coefficient. '
            'Degree zero retains the library\'s half-open endpoint at L; exact overlap equality for that degree excludes its isolated right endpoint.','',
            'Fractional shifts generally leave nonzero approximation error. They can be exact when the entire shortened overlap '
            'lies in a single old and new polynomial span. Numerical fallback retains the reference only in unresolved directions; '
            'no unique continuation is inferred from missing data. Cache keys include degree, M, L, delta and quadrature order; '
            'cached arrays and diagnostics are read-only.','',
            'This projection is unconstrained and can exceed the Stage 4C coefficient bounds. No clipping is performed. '
            'Stage 5B integration must make feasible-start handling explicit while retaining the existing optimization bounds.','',
            '## Quadrature convergence','',
            '| d | M | Gauss points / span | max G difference | max B difference | max T difference |',
            '|---:|---:|---:|---:|---:|---:|']
    for r in quadrature:
        lines.append(f"| {r['degree']} | {r['segments']} | {r['order']} | {r['gram_max_absolute_error']:.5g} | "
                     f"{r['cross_max_absolute_error']:.5g} | {r['transform_max_absolute_error']:.5g} |")
    lines+=['','Reference uses 2d+5 Gauss points per split span. Order 1 intentionally underintegrates positive degrees; '
            'd+1 and higher agree to float64 roundoff in G/B. Coefficient-map errors amplify matrix/basis roundoff when conditioning is poor.','',
            '## Saved Stage 4C coefficient checks (delta=0.1 s)','',
            '| Source | Projected / retained overlap error | Maximum projected coefficient magnitude |',
            '|---|---:|---:|']
    for r in saved:
        lines.append(f"| {r['source']} | {r['adaptive_integrated_error']:.7g} / {r['retained_coefficients_integrated_error']:.7g} | {r['max_absolute_projected_coefficient']:.7g} |")
    lines+=['','These read existing fitted coefficients without rerunning or changing any Stage 4C estimator/artifact. '
            'They evaluate initialization projection only and make no new SLAM accuracy or runtime claim.','',
            '![Overlap projection and unconstrained terminal continuation](projection_overlap.png)','',
            '## Tests, initial failures and preservation','',
            'The first projection-only run had 57 passes and two failures. One test incorrectly required every fractional '
            'shift to have positive error; the d=5, M=2 shortened overlap is one polynomial span and is exactly representable. '
            'That test now distinguishes the two mathematical cases. The other exposed about 3e-11 coefficient error for a '
            'constant channel in an ill-conditioned system. Separating the invariant constant component fixes it without '
            'relaxing tolerances or adding regularization. The final projection-only run passed all 59 tests. '
            'Initial and final XML/stdout are retained.','']
    if tests:
        lines+=['```sh',tests['command'],'```','',f"Outcome: **{tests['passed']} passed**, exit {tests['exit_code']}, "
                f"subprocess wall time {tests['wall_seconds']:.3f} s. See test_record.json, pytest_stdout.txt and pytest.xml.",'']
    else:
        lines+=['No combined test record supplied; the gate remains unsatisfied.','']
    lines += ['Projection tests cover zero/constants, independent integrated normal equations, optimality versus retained coefficients, '
              'whole/fractional shifts, multiple degrees/M, per-channel agreement, conditioning, quadrature convergence, '
              'cache reproducibility/immutability and input validation. The combined run includes all 36 Stage 4C/visibility '
              'tests and all 93 existing basis tests. Unrelated baseline and earlier estimator tests were not rerun; their '
              'sources and artifacts are unchanged.','',
              f"All {len(snapshot)} pre-task files are {'unchanged' if not changes else 'NOT all unchanged'}; "
              'see preservation.json and before_sha256.json. The original baseline, existing spline adapter/C++ dependency, '
              'Stage 4C results, mathematical objective and EKF implementations are preserved.','',
              '```sh','PYTHONPATH=build/bsplines_library MPLCONFIGDIR=/tmp/bspline_mhe_mpl python3 -B -m benchmarks.milestone5a','```','',
              '```json',json.dumps(checks,indent=2),'```','',
              '**Stopped at the Stage 5A gate. Stage 5B requires explicit user approval.** '
              'No moving-window scheduler, arrival replay, optimizer warm-start integration, nearest-neighbor association '
              'or Victoria Park code was implemented.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/milestone5a'))
    args=parser.parse_args()
    record=run(args.output)
    print(json.dumps({'checks':record['checks'],'stage5a_gate_satisfied':record['stage5a_gate_satisfied']}))


if __name__=='__main__':
    main()
