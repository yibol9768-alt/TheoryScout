"""Numerical characteristic-function checks for the R5 tangent envelope.

Post hoc replay at fixed thresholds, with analytic infinite-tail bounds.
Finite-interval quadrature and eigenvalues remain float64, not certified.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import f as f_distribution


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2);handle.write('\n')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def weighted_tail(values,counts,tolerance=1e-10):
    values=np.asarray(values,dtype=float);counts=np.asarray(counts,dtype=int)
    active=counts>0;values=values[active];counts=counts[active]
    if np.all(values==0):return dict(probability=1.,tail_bound=0.,quad_error_estimate=0.,cutoff=0.)
    if np.all(values>=0):return dict(probability=1.,tail_bound=0.,quad_error_estimate=0.,cutoff=0.)
    if np.all(values<=0):return dict(probability=0.,tail_bound=0.,quad_error_estimate=0.,cutoff=0.)
    scale=math.sqrt(float(np.sum(counts*values**2))*2)
    values=values/scale
    candidates=np.concatenate([np.repeat(abs(a),min(int(c),32)) for a,c in zip(values,counts) if a!=0])
    candidates=np.sort(candidates)[::-1][:32]
    choices=[]
    for k in range(1,len(candidates)+1):
        a=float(candidates[k-1])
        log_cutoff=(2/k)*math.log(2/(math.pi*k*(tolerance/4)))-math.log(2*a)
        choices.append((log_cutoff,k,a))
    log_cutoff,k,a=min(choices)
    if log_cutoff>math.log(1e6):raise RuntimeError('Tail budget unsuitable for this low-rank case')
    cutoff=math.exp(log_cutoff)
    def integrand(u):
        if u==0:return float(np.sum(counts*values))
        log_cf=-.5*np.sum(counts*np.log1p(-2j*u*values))
        return float(np.exp(log_cf).imag/u)
    integral,error=quad(integrand,0,cutoff,epsabs=math.pi*tolerance/4,
                        epsrel=tolerance/4,limit=300)
    tail=2/(math.pi*k)*(2*a*cutoff)**(-k/2)
    probability=.5+integral/math.pi
    if not -1e-7<=probability<=1+1e-7:raise RuntimeError('Probability inversion outside range')
    return dict(probability=probability,tail_bound=tail,quad_error_estimate=error/math.pi,
                cutoff=cutoff,tail_factors=k)


def self_checks():
    checks=[]
    for positive,negative,ratio in ((8,8,1.),(8,12,.7),(10,6,1.75),(20,20,2.)):
        got=weighted_tail([1.,-ratio],[positive,negative])
        expected=float(f_distribution.sf(ratio*negative/positive,positive,negative))
        error=abs(got['probability']-expected)
        if error>1e-8:raise RuntimeError('Characteristic-function sign or tail test failed')
        checks.append(dict(positive_df=positive,negative_df=negative,ratio=ratio,
                           expected=expected,error=error,**got))
    for values,expected in (([1.,2.],1.),([-1.,-2.],0.),([0.,0.],1.)):
        got=weighted_tail(values,[2,3]);assert got['probability']==expected
    return checks


def run(source,out):
    source=source.resolve();out=out.resolve();root=Path(__file__).resolve().parents[1]
    generator_path=root/'tools/diagnose_joint_energy.py'
    moments_path=root/'tools/analyze_joint_gaussian.py'
    design=json.loads((source/'design.json').read_text())
    assert sha(generator_path)==design['source_sha256']
    generator=load_module('r3_fixed',generator_path)
    moment_module=load_module('r5_fixed',moments_path)
    inputs=[generator_path,moments_path,source/'design.json',root/'revisions/R5_TANGENT_RECALL_BRIDGE.md']
    inputs+=sorted(source.glob('thresholds_*.npz'))+sorted(source.glob('joint_*_0.npz'))
    hashes={str(p.relative_to(root)):sha(p) for p in inputs}
    out.mkdir(parents=True,exist_ok=False)
    h_grid=[.005,.01,.02,.04,.08,.16,.32,.64]
    save(out/'PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Post hoc fixed-threshold replay; no new training samples',
         thresholds='Median of archived sphere threshold bank for each L, fixed across scatter replay',
         scatters='rep=0 from all six R3 dimension/method cells',h_grid=h_grid,
         source_sha256=sha(Path(__file__)),inputs=hashes,
         certification='Analytic truncation formula only; float64 eigensolver and finite quadrature are not certified'))
    save(out/'self_checks.json',self_checks())
    rows=[]
    with (out/'cases.jsonl').open('x',buffering=1) as stream:
        for cell in design['archived_loads']:
            d,n,method=cell['d'],cell['n'],cell['method'];m=n-1
            rng=generator.seed_rng(design['seed'],d,generator.METHODS.index(method),0,2)
            x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
            u=generator.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
            xy=np.vstack((x,y));chol=np.linalg.cholesky(d/n*xy@xy.T)
            b=chol[d:].T@u@chol[:d]/np.linalg.norm(u)
            k=(b+b.T)/2;nn=b.T@b;mom=moment_module.moments(b,n)
            with np.load(source/f'joint_{d}_{method}_0.npz') as archive:
                correct=archive['D'].copy();energy=archive['V'].copy()
                for key in ('mu','sigma2','expected_V'):assert abs(float(archive[key])-mom[key])<1e-10
            with np.load(source/f'thresholds_{d}_{method}.npz') as archive:thresholds=archive['sphere'].copy()
            v=mom['expected_V'];e2=mom['mu']**2+mom['sigma2']
            var_d2=(4*mom['mu']**2*mom['sigma2']+4*mom['mu']*mom['third_D']
                    +mom['fourth_D']-mom['sigma2']**2)
            if var_d2<=0:raise RuntimeError('Invalid squared-score variance')
            for j,length in enumerate(design['lists']):
                t=float(np.median(thresholds[:,j]));assert t>0
                alpha=t/(2*math.sqrt(v));constant=alpha*e2/m-t*math.sqrt(v)+alpha*v
                eigen=np.linalg.eigvalsh(k-alpha*n/m*nn)
                def probability(level,tol=1e-10):
                    shift=(level-constant)/n
                    return weighted_tail(np.r_[eigen-shift,-shift],np.r_[np.ones(len(eigen),dtype=int),n-len(eigen)],tol)
                zero=probability(0.);tight=probability(0.,1e-12)
                if abs(zero['probability']-tight['probability'])>1e-8:
                    raise RuntimeError('Inversion precision cross-check failed')
                eps_q=alpha/m*math.sqrt(var_d2)
                eps_curv=t*mom['variance_V']/(2*v**1.5)
                bounds=[]
                for h in h_grid:
                    plus,minus=probability(h),probability(-h)
                    bounds.append(dict(h=h,lower=max(0.,plus['probability']-eps_q/h),
                                       upper=min(1.,minus['probability']+(eps_q+eps_curv)/h),
                                       plus=plus,minus=minus))
                a_sample=correct-t*math.sqrt(v)-alpha*(energy-v)
                q_sample=a_sample-alpha/m*(correct**2-e2)
                true_score=correct-t*np.sqrt(energy)
                q_reference=float(np.mean(q_sample>=0))
                actual_reference=float(np.mean(true_score>=0))
                row=dict(**cell,L=length,rep=0,t=t,eps_q=eps_q,eps_curv=eps_curv,
                         quadratic_probability=zero,precision_cross_difference=abs(zero['probability']-tight['probability']),
                         sampled_quadratic_probability=q_reference,sampled_actual_fixed_threshold=actual_reference,
                         sampled_tangent_success=float(np.mean(a_sample>=0)),
                         minimum_curvature_score=float(np.min(true_score-a_sample)),
                         lower=max(r['lower'] for r in bounds),upper=min(r['upper'] for r in bounds),grid=bounds)
                rows.append(row);stream.write(json.dumps(row)+'\n')
                print(f'{d} {method} L={length}: quadratic={zero["probability"]:.6f}, sample={q_reference:.6f}, envelope=[{row["lower"]:.4f},{row["upper"]:.4f}]',flush=True)
            del x,y,xy,chol,b,k,nn,u
    for name,digest in hashes.items():assert sha(root/name)==digest,name
    save(out/'results.json',dict(scope='Float evaluation of exact symbolic envelope, not a certified interval',rows=rows))
    save(out/'FINISHED.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();run(args.source,args.out)
