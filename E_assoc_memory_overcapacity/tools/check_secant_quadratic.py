"""Float evaluation of double-tangent lower and secant upper recall bounds.

Replays fixed R3 scatters; no new training samples or confirmatory inference.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2);handle.write('\n')


def run(source,out):
    source=source.resolve();out=out.resolve();root=Path(__file__).resolve().parents[1]
    quad=load('r6_integrator',root/'tools/check_tangent_quadratic.py')
    generator=load('r3_generator',root/'tools/diagnose_joint_energy.py')
    moments=load('r5_moments',root/'tools/analyze_joint_gaussian.py')
    design=json.loads((source/'design.json').read_text())
    assert sha(root/'tools/diagnose_joint_energy.py')==design['source_sha256']
    inputs=[source/'design.json',root/'tools/check_tangent_quadratic.py',root/'tools/analyze_joint_gaussian.py']
    inputs+=sorted(source.glob('joint_*_0.npz'))+sorted(source.glob('thresholds_*.npz'))
    hashes={str(p.relative_to(root)):sha(p) for p in inputs}
    out.mkdir(parents=True,exist_ok=False)
    factors=[2.,2.5,3.,3.5,4.]
    save(out/'PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Post hoc improved symbolic envelope, float evaluation only',
        inputs=hashes,source_sha256=sha(Path(__file__)),scatter_rule='rep=0 from all six R3 cells',
        threshold_rule='Median of archived sphere order-statistic bank',
        score_radius='5 exact conditional SDs',energy_factors=factors,
        optimization='Minimum over valid deterministic energy intervals; each analytic upper bound is valid separately'))
    save(out/'self_checks.json',quad.self_checks())
    rows=[]
    with (out/'cases.jsonl').open('x',buffering=1) as stream:
        for cell in design['archived_loads']:
            d,n,method=cell['d'],cell['n'],cell['method'];m=n-1
            rng=generator.seed_rng(design['seed'],d,generator.METHODS.index(method),0,2)
            x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
            u=generator.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
            xy=np.vstack((x,y));f=np.linalg.cholesky(d/n*xy@xy.T)
            b=f[d:].T@u@f[:d]/np.linalg.norm(u);k=(b+b.T)/2;nn=b.T@b
            mom=moments.moments(b,n);mu=mom['mu'];v=mom['expected_V']
            with np.load(source/f'joint_{d}_{method}_0.npz') as archive:
                correct=archive['D'].copy();energy=archive['V'].copy()
                for key in ('mu','sigma2','expected_V'):assert abs(float(archive[key])-mom[key])<1e-10
            with np.load(source/f'thresholds_{d}_{method}.npz') as archive:bank=archive['sphere'].copy()
            def tail(eigen,constant=0.,level=0.):
                shift=(level-constant)/n
                return quad.weighted_tail(np.r_[eigen-shift,-shift],np.r_[np.ones(len(eigen),dtype=int),n-len(eigen)])
            eig_d=np.linalg.eigvalsh(k);eig_energy=np.linalg.eigvalsh(n/m*nn)
            radius=5*math.sqrt(mom['sigma2']);maximum_abs=abs(mu)+radius
            low_d=tail(eig_d,level=mu-radius);high_d=tail(eig_d,level=mu+radius)
            score_outside=max(0.,1-low_d['probability']+high_d['probability'])
            tilde_sample=energy+correct**2/m
            for j,length in enumerate(design['lists']):
                t=float(np.median(bank[:,j]));alpha=t/(2*math.sqrt(v))
                lower_h=(1+2*alpha*mu/m)*k-alpha*n/m*nn
                lower_c=-alpha*mu*mu/m-t*math.sqrt(v)/2
                lower=tail(np.linalg.eigvalsh(lower_h),lower_c)
                lower_sample=(1+2*alpha*mu/m)*correct-alpha*tilde_sample+lower_c
                true_score=correct-t*np.sqrt(energy)
                min_lower_gap=float(np.min(true_score-lower_sample))
                if min_lower_gap < -1e-9:raise RuntimeError('Double-tangent lower event identity failed')
                candidates=[]
                for factor in factors:
                    a=v-factor*math.sqrt(mom['variance_V']);upper_energy=v+factor*math.sqrt(mom['variance_V'])
                    if a<=0:
                        candidates.append(dict(factor=factor,status='nonpositive interval lower endpoint'));continue
                    beta=t/(math.sqrt(a)+math.sqrt(upper_energy))
                    upper_h=(1+2*beta*mu/m)*k-beta*n/m*nn
                    upper_c=beta*(radius**2-mu*mu)/m-beta*math.sqrt(a*upper_energy)
                    upper_quad=tail(np.linalg.eigvalsh(upper_h),upper_c)
                    low_energy=tail(eig_energy,level=a+maximum_abs**2/m)
                    high_energy=tail(eig_energy,level=upper_energy)
                    energy_outside=max(0.,1-low_energy['probability'])+max(0.,high_energy['probability'])
                    probability=min(1.,upper_quad['probability']+energy_outside+score_outside)
                    good=(np.abs(correct-mu)<=radius)&(energy>=a)&(energy<=upper_energy)
                    upper_sample=(1+2*beta*mu/m)*correct-beta*tilde_sample+upper_c
                    gap=float(np.min((upper_sample-true_score)[good])) if np.any(good) else None
                    if gap is not None and gap < -1e-9:raise RuntimeError('Secant upper event identity failed')
                    candidates.append(dict(factor=factor,interval=[a,upper_energy],upper=probability,
                        quadratic_probability=upper_quad,energy_outside_bound=energy_outside,
                        score_outside_bound=score_outside,low_energy_probability=low_energy,
                        high_energy_probability=high_energy,minimum_upper_score_gap=gap))
                valid=[r for r in candidates if 'upper' in r]
                best=min(valid,key=lambda r:r['upper'])
                row=dict(**cell,L=length,rep=0,t=t,lower=lower['probability'],upper=best['upper'],
                         width=best['upper']-lower['probability'],lower_probability=lower,
                         best_energy_factor=best['factor'],best_energy_outside=best['energy_outside_bound'],
                         score_outside=score_outside,sampled_actual_fixed_threshold=float(np.mean(true_score>=0)),
                         minimum_lower_score_gap=min_lower_gap,candidates=candidates)
                if row['width']< -1e-7:raise RuntimeError('Float envelope has reversed endpoints')
                rows.append(row);stream.write(json.dumps(row)+'\n')
                print(f'{d} {method} L={length}: [{row["lower"]:.5f}, {row["upper"]:.5f}], width={100*row["width"]:.3f}pp',flush=True)
            del x,y,xy,f,b,k,nn,u
    for name,digest in hashes.items():assert sha(root/name)==digest,name
    save(out/'results.json',dict(scope='Analytic formulas evaluated in float64; no interval arithmetic certification',rows=rows))
    save(out/'FINISHED.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.out)
