"""Post hoc conditional Gaussian/skew diagnostics on the frozen R3 scatters.

All moments come from exact sphere traces. Replay the same seeds without
new training draws. The skew correction is a heuristic without a remainder.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr,roots_hermitenorm,roots_legendre


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2);handle.write('\n')


def moments(b,n):
    k=(b+b.T)/2;k2=k@k;nn=b.T@b
    mu=float(np.trace(k));t2=float(np.sum(k*k));t3=float(np.sum(k2*k));t4=float(np.sum(k2*k2))
    trn=float(np.trace(nn));trn2=float(np.sum(nn*nn))
    kn=float(np.sum(k*nn));k2n=float(np.sum(k2*nn))
    centered2=t2-mu*mu/n
    centered3=t3-3*mu/n*t2+2*mu**3/n**2
    centered4=t4-4*mu/n*t3+6*mu*mu/n**2*t2-3*mu**4/n**3
    variance=2*n/(n+2)*centered2
    third=8*n*n/((n+2)*(n+4))*centered3
    fourth=n**3/((n+2)*(n+4)*(n+6))*(12*centered2**2+48*centered4)
    ev=(n*trn-mu*mu-variance)/(n-1)
    qn2=(trn*trn+2*trn2)/(n*(n+2))
    qnkk=(trn*(mu*mu+2*t2)+4*mu*kn+8*k2n)/(n*(n+2)*(n+4))
    qk4=(mu**4+12*mu*mu*t2+12*t2*t2+32*mu*t3+48*t4)/(n*(n+2)*(n+4)*(n+6))
    varv=n**4/(n-1)**2*(qn2-2*qnkk+qk4)-ev*ev
    covariance=2*n*n/((n-1)*(n+2))*(kn-mu*trn/n)-(2*mu*variance+third)/(n-1)
    return dict(mu=mu,sigma2=variance,third_D=third,fourth_D=fourth,expected_V=ev,
                variance_V=varv,covariance=covariance,skew_D=third/variance**1.5,
                excess_D=fourth/variance**2-3)


def quadrature():
    records=[]
    for n in (2,3):
        theta=np.arange(128)*2*math.pi/128
        if n==2:
            w=np.stack((np.cos(theta),np.sin(theta)));weights=np.full(128,1/128)
        else:
            z,zw=roots_legendre(16);radius=np.sqrt(1-z*z)
            w=np.stack(((radius[:,None]*np.cos(theta)).ravel(),
                        (radius[:,None]*np.sin(theta)).ravel(),np.repeat(z,128)))
            weights=np.repeat(zw/256,128)
        for index,b in enumerate((np.arange(1,n*n+1,dtype=float).reshape(n,n)/7,
                                 np.diag(np.linspace(-.6,.8,n)))):
            exact=moments(b,n);bw=b@w;d=n*np.sum(w*bw,axis=0)
            v=(n*n*np.sum(bw*bw,axis=0)-d*d)/(n-1)
            mu=float(weights@d);ev=float(weights@v)
            numeric=dict(mu=mu,sigma2=float(weights@((d-mu)**2)),
                         third_D=float(weights@((d-mu)**3)),fourth_D=float(weights@((d-mu)**4)),
                         expected_V=ev,variance_V=float(weights@((v-ev)**2)),
                         covariance=float(weights@((d-mu)*(v-ev))))
            error=max(abs(numeric[key]-exact[key]) for key in numeric)
            if error>1e-9:
                raise RuntimeError('Exact joint sphere moments failed quadrature')
            records.append(dict(n=n,matrix=index,max_error=error))
    return records


def gaussian_recall(thresholds,mom,covariance):
    mean,varv=mom['expected_V'],mom['variance_V']
    if varv<=0:
        raise RuntimeError('Nonpositive energy variance')
    slope=covariance/math.sqrt(varv)
    residual=mom['sigma2']-slope*slope
    if residual<=0:
        raise RuntimeError('Invalid Gaussian covariance')
    invalid=float(ndtr(-mean/math.sqrt(varv)))
    previous=None
    for order in (32,64,128,256):
        nodes,weights=roots_hermitenorm(order);weights/=math.sqrt(2*math.pi)
        energy=mean+math.sqrt(varv)*nodes;valid=energy>0
        integrand=np.zeros(order)
        arguments=(mom['mu']+slope*nodes[valid,None]
                   -np.sqrt(energy[valid,None])*thresholds[None,:])/math.sqrt(residual)
        integrand[valid]=ndtr(arguments).mean(axis=1)
        value=float(weights@integrand)
        if previous is not None and abs(value-previous)<1e-8:
            return value,min(1.,value+invalid),invalid,order,abs(value-previous)
        previous=value
    raise RuntimeError('Gaussian quadrature failed fixed precision budget')


def run(source,out):
    source=source.resolve();out=out.resolve()
    root=Path(__file__).resolve().parents[1]
    generator=root/'tools/diagnose_joint_energy.py'
    design=json.loads((source/'design.json').read_text())
    if sha(generator)!=design['source_sha256']:
        raise RuntimeError('Frozen R3 generator source changed')
    spec=importlib.util.spec_from_file_location('r3_frozen',generator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    out.mkdir(parents=True,exist_ok=False)
    inputs=[generator,source/'design.json',source/'scatters.jsonl']
    inputs+=sorted(source.glob('joint_*.npz'))+sorted(source.glob('thresholds_*.npz'))
    hashes={str(p.relative_to(root)):sha(p) for p in inputs}
    save(out/'PLAN.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
         scope='Post hoc replay; no new training samples; conditional proxy diagnostics',
         inputs=hashes,source_sha256=sha(Path(__file__)),
         quadrature_orders=[32,64,128,256],quadrature_tolerance=1e-8,
         skew_correction='Heuristic first skew term; no signed remainder or coverage guarantee'))
    save(out/'geometry_checks.json',quadrature())
    rows=[]
    with (out/'scatters.jsonl').open('x',buffering=1) as stream:
        for cell in design['archived_loads']:
            d,n,method=cell['d'],cell['n'],cell['method']
            with np.load(source/f'thresholds_{d}_{method}.npz') as archive:
                bank=archive['sphere'].copy()
            for rep in range(design['scatters']):
                rng=module.seed_rng(design['seed'],d,module.METHODS.index(method),rep,2)
                x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
                u=module.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
                xy=np.vstack((x,y));f=np.linalg.cholesky(d/n*xy@xy.T)
                b=f[d:].T@u@f[:d]/np.linalg.norm(u);mom=moments(b,n)
                with np.load(source/f'joint_{d}_{method}_{rep}.npz') as archive:
                    for key in ('mu','sigma2','expected_V'):
                        if abs(float(archive[key])-mom[key])>1e-10:
                            raise RuntimeError('Replay differs from frozen R3 moments')
                    correct=archive['D'].copy();energy=archive['V'].copy()
                endpoints=[]
                for j,length in enumerate(design['lists']):
                    thresholds=bank[:,j];sorted_t=np.sort(thresholds)
                    exact_proxy=float(np.searchsorted(sorted_t,correct/np.sqrt(energy),side='right').mean()/len(sorted_t))
                    actual_fixed=float(np.searchsorted(sorted_t,correct/math.sqrt(mom['expected_V']),side='right').mean()/len(sorted_t))
                    z=(thresholds*math.sqrt(mom['expected_V'])-mom['mu'])/math.sqrt(mom['sigma2'])
                    fixed_gauss=float(ndtr(-z).mean())
                    skew_term=float((-mom['skew_D']/6*(1-z*z)*np.exp(-z*z/2)/math.sqrt(2*math.pi)).mean())
                    independent=gaussian_recall(thresholds,mom,0.)
                    joint=gaussian_recall(thresholds,mom,mom['covariance'])
                    endpoints.append(dict(L=length,actual_joint_proxy=exact_proxy,
                        actual_D_fixed_energy=actual_fixed,gaussian_fixed_energy=fixed_gauss,
                        independent_gaussian=independent[0],joint_gaussian=joint[0],
                        joint_gaussian_upper=joint[1],gaussian_nonpositive_energy_mass=joint[2],
                        heuristic_skew_term=skew_term,heuristic_joint_plus_skew=joint[0]+skew_term,
                        quadrature_order=max(independent[3],joint[3]),
                        quadrature_change=max(independent[4],joint[4])))
                row=dict(**cell,rep=rep,moments=mom,endpoints=endpoints)
                rows.append(row);stream.write(json.dumps(row)+'\n')
                print(f'{d} {method} replay {rep+1}/4',flush=True)
                del x,y,xy,f,b,u
    for name,digest in hashes.items():
        if sha(root/name)!=digest:
            raise RuntimeError('Archived input changed during analysis')
    summary=[]
    for cell in design['archived_loads']:
        selected=[r for r in rows if all(r[key]==cell[key] for key in ('d','n','method'))]
        for j,length in enumerate(design['lists']):
            eps=[r['endpoints'][j] for r in selected]
            summary.append(dict(**cell,L=length,skew_D=float(np.mean([r['moments']['skew_D'] for r in selected])),
                excess_D=float(np.mean([r['moments']['excess_D'] for r in selected])),
                mean={key:float(np.mean([e[key] for e in eps])) for key in (
                    'actual_joint_proxy','actual_D_fixed_energy','gaussian_fixed_energy',
                    'independent_gaussian','joint_gaussian','heuristic_skew_term','heuristic_joint_plus_skew')},
                invalid_gaussian_energy_mass=max(e['gaussian_nonpositive_energy_mass'] for e in eps)))
    save(out/'results.json',dict(scope='Post hoc diagnostic, not a proved Edgeworth law',summary=summary))
    save(out/'FINISHED.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source,args.out)
