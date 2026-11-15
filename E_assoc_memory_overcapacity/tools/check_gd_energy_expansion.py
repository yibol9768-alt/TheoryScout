"""Freeze and run a fresh exploratory check of the R4 GD energy expansion."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2);handle.write('\n')


def train(s,z,steps,eta):
    u=np.zeros_like(z)
    for _ in range(steps):
        u+=eta*(z-u@s)
    return u


def energy(u,z,s,t,n):
    norm2=float(np.sum(u*u))
    if norm2<=0:
        raise RuntimeError('Zero endpoint: fail without redraw')
    numerator=float(np.sum((u@s@u.T)*t))
    trn=numerator/norm2
    uz=u@z.T
    cross=float(np.sum(uz*uz.T))/(n*norm2)
    mu=float(np.sum(u*z))/math.sqrt(n*norm2)
    sigma2=n/(n+2)*(trn+cross-2*mu*mu/n)
    average=n/(n-1)*trn-(mu*mu+sigma2)/(n-1)
    return dict(energy=average,trN=trn,mu=mu,sigma2=sigma2,
                numerator=numerator,denominator=norm2)


def freeze(out):
    archive=ROOT/'ac2_checks/full/AC2_results.json'
    cells=[dict(d=r['d'],n=r['n']) for r in json.loads(archive.read_text())['rows'] if r['method']=='GD3']
    branches=[dict(name='GD1_eta07',steps=1,eta=.7),dict(name='GD3_eta07',steps=3,eta=.7),
              dict(name='GD2_eta15',steps=2,eta=1.5),dict(name='GD2_eta10',steps=2,eta=1.)]
    for b in branches:
        a=1-(1-b['eta'])**b['steps']
        b['coefficient']=2*b['steps']*b['eta']*(1-b['eta'])**(b['steps']-1)/a
    design=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),seed=202611071837,
                cells=[r for r in cells if r['d'] in (128,256,512)],branches=branches,
                datasets=8,shared_arrays_across_branches=True,threads=2,
                wall_limit_seconds=120,cpu_limit_seconds=180,
                source_sha256=sha(Path(__file__)),archive_sha256=sha(archive),
                proof_sha256=sha(ROOT/'revisions/R4_GD_ENERGY_PROOF.md'),
                primary='(exact conditional mean wrong-column raw energy - 1)/(d/n)',
                secondary='finite conditional ratio-of-means, distinguished from mean of ratio',
                inference='Exploratory means and dataset SD; no confirmatory intervals or fitted coefficients',
                failure='Stop and retain partial output; never redraw or change cells')
    out.mkdir(parents=True,exist_ok=False);save(out/'design.json',design)
    print('Frozen all branches, datasets, source, proof and archive hashes before sampling.')


def geometry_check(branches):
    rng=np.random.default_rng(202611071836)
    d,n=8,80;a,b=rng.normal(size=(2,d,n))
    s=a@a.T/n;z=b@a.T/math.sqrt(n);t=b@b.T/n
    xy=np.vstack((a,b));f=np.linalg.cholesky(xy@xy.T/n)
    rows=[]
    for branch in branches:
        u=train(s,z,branch['steps'],branch['eta'])
        got=energy(u,z,s,t,n)
        block=f[d:].T@u@f[:d]/np.linalg.norm(u)
        k=(block+block.T)/2
        mu=float(np.trace(k));variance=2*n/(n+2)*(float(np.sum(k*k))-mu*mu/n)
        exact=n/(n-1)*float(np.sum(block*block))-(mu*mu+variance)/(n-1)
        error=abs(got['energy']-exact)
        if error>1e-10:
            raise RuntimeError('Direct energy formula failed conditional-block check')
        rows.append(dict(branch=branch['name'],error=error))
    return rows


def run(out):
    spec=json.loads((out/'design.json').read_text())
    inputs=[(Path(__file__),'source_sha256'),(ROOT/'ac2_checks/full/AC2_results.json','archive_sha256'),
            (ROOT/'revisions/R4_GD_ENERGY_PROOF.md','proof_sha256')]
    for path,key in inputs:
        if sha(path)!=spec[key]:raise RuntimeError('Frozen input changed')
    resource.setrlimit(resource.RLIMIT_CPU,(spec['cpu_limit_seconds'],spec['cpu_limit_seconds']+5))
    save(out/'STARTED.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),geometry=geometry_check(spec['branches'])))
    rows=[];start=time.monotonic()
    with (out/'datasets.jsonl').open('x',buffering=1) as stream:
        for cell in spec['cells']:
            d,n=cell['d'],cell['n'];r=d/n
            for rep in range(spec['datasets']):
                if time.monotonic()-start>spec['wall_limit_seconds']:raise TimeoutError('Frozen wall budget reached')
                rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],d,rep]))
                a,b=rng.normal(size=(2,d,n));s=a@a.T/n;z=b@a.T/math.sqrt(n);t=b@b.T/n
                eigen=np.linalg.eigvalsh(s)
                np.save(out/f'spectrum_{d}_{rep}.npy',eigen,allow_pickle=False)
                for branch in spec['branches']:
                    u=train(s,z,branch['steps'],branch['eta'])
                    got=energy(u,z,s,t,n)
                    p=branch['eta']*sum((1-branch['eta']*eigen)**j for j in range(branch['steps']))
                    ratio=(1+(d+1)/n)*float(np.mean(eigen**2*p**2)/np.mean(eigen*p**2))
                    row=dict(**cell,rep=rep,branch=branch['name'],r=r,**got,
                             coefficient=branch['coefficient'],empirical_coefficient=(got['energy']-1)/r,
                             first_order_energy=1+branch['coefficient']*r,conditional_ratio_of_means=ratio,
                             normalized_trace_error=(got['trN']-ratio)/r)
                    rows.append(row);stream.write(json.dumps(row)+'\n')
                print(f'd={d} completed dataset {rep+1}/{spec["datasets"]} {time.monotonic()-start:.1f}s',flush=True)
                del a,b,s,z,t,u
    for path,key in inputs:
        if sha(path)!=spec[key]:raise RuntimeError('Input changed during run')
    summaries=[]
    for cell in spec['cells']:
        for branch in spec['branches']:
            group=[r for r in rows if r['d']==cell['d'] and r['branch']==branch['name']]
            values=np.array([r['empirical_coefficient'] for r in group])
            summaries.append(dict(**cell,branch=branch['name'],datasets=len(group),
                                  target=branch['coefficient'],mean_coefficient=float(values.mean()),
                                  dataset_SD=float(values.std(ddof=1)),
                                  mean_energy=float(np.mean([r['energy'] for r in group])),
                                  mean_absolute_ratio_error_over_r=float(np.mean([abs(r['normalized_trace_error']) for r in group]))))
    save(out/'results.json',dict(summary=summaries,datasets=len(rows)//len(spec['branches']),
             wall_seconds=time.monotonic()-start,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             all_frozen_hashes_unchanged=True))
    save(out/'FINISHED.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
             files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'));parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();(freeze if args.action=='freeze' else run)(args.out)
