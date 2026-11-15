"""Frozen independent conditional-query check of all twelve R6 cases.

Six existing training scatters, 2**18 new IID conditional queries per scatter.
Hoeffding inference concerns these conditional probabilities only.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as handle:
        json.dump(value,handle,indent=2);handle.write('\n')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def freeze(out):
    inputs=[ROOT/'tools/diagnose_joint_energy.py',ROOT/'tools/analyze_joint_gaussian.py',
            ROOT/'revisions/r3_joint_diagnostic_v2/design.json',ROOT/'revisions/r6_secant_integral/results.json']
    spec=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),seed=202611071907,
              queries_per_scatter=262144,block=2048,scatters=6,threshold_cases=12,
              checked_endpoints_per_case=6,family=72,alpha=.05,
              scope='Conditional-query Monte Carlo validation; not new training datasets or full top-L recall',
              confidence='Two-sided Hoeffding at all 72 endpoints, union-bound total error <= .05 under exact IID sampling/arithmetic',
              endpoints=['actual','lower','upper','low_energy','high_energy','outside_score'],
              source_sha256=sha(Path(__file__)),inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},
              wall_limit_seconds=420,cpu_limit_seconds=600,threads=2,
              stopping='Fixed sample size, no early precision stop, no redraw or outcome-dependent extension')
    out.mkdir(parents=True,exist_ok=False);save(out/'design.json',spec)
    print('Frozen six scatters, 262144 independent queries each and all 72 endpoint checks.')


def run(out):
    spec=json.loads((out/'design.json').read_text())
    assert sha(Path(__file__))==spec['source_sha256']
    for name,digest in spec['inputs'].items():assert sha(ROOT/name)==digest,name
    resource.setrlimit(resource.RLIMIT_CPU,(spec['cpu_limit_seconds'],spec['cpu_limit_seconds']+5))
    generator=module('r3_generator',ROOT/'tools/diagnose_joint_energy.py')
    moments=module('r5_moments',ROOT/'tools/analyze_joint_gaussian.py')
    old_design=json.loads((ROOT/'revisions/r3_joint_diagnostic_v2/design.json').read_text())
    cases=json.loads((ROOT/'revisions/r6_secant_integral/results.json').read_text())['rows']
    save(out/'STARTED.json',dict(time_utc=datetime.now(timezone.utc).isoformat()))
    radius=math.sqrt(math.log(2*spec['family']/spec['alpha'])/(2*spec['queries_per_scatter']))
    start=time.monotonic();result=[]
    with (out/'cases.jsonl').open('x',buffering=1) as stream:
        for cell in old_design['archived_loads']:
            d,n,method=cell['d'],cell['n'],cell['method'];m=n-1
            rng=generator.seed_rng(old_design['seed'],d,generator.METHODS.index(method),0,2)
            x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
            u=generator.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
            xy=np.vstack((x,y));f=np.linalg.cholesky(d/n*xy@xy.T)
            b=f[d:].T@u@f[:d]/np.linalg.norm(u);mom=moments.moments(b,n)
            mu=mom['mu'];v=mom['expected_V'];score_radius=5*math.sqrt(mom['sigma2']);maximum_abs=abs(mu)+score_radius
            selected=[r for r in cases if all(r[key]==cell[key] for key in ('d','n','method'))]
            assert len(selected)==2
            counts=np.zeros((2,6),dtype=np.int64);violations=np.zeros((2,2),dtype=np.int64)
            query_rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],d,generator.METHODS.index(method)]))
            raw=np.lib.format.open_memmap(out/f'D_V_{d}_{method}.npy',mode='w+',dtype='float64',shape=(2,spec['queries_per_scatter']))
            for offset in range(0,spec['queries_per_scatter'],spec['block']):
                if time.monotonic()-start>spec['wall_limit_seconds']:raise TimeoutError('Frozen wall budget exceeded; partial arrays retained')
                size=min(spec['block'],spec['queries_per_scatter']-offset)
                g=query_rng.normal(size=(2*d,size))
                w=g/np.sqrt(np.sum(g*g,axis=0)+query_rng.chisquare(n-2*d,size=size))
                bw=b@w;correct=n*np.sum(w*bw,axis=0)
                energy=(n*n*np.sum(bw*bw,axis=0)-correct**2)/m
                if np.any(energy<=0):raise RuntimeError('Nonpositive query energy: retain failure without redraw')
                raw[:,offset:offset+size]=np.stack((correct,energy))
                tilde=energy+correct**2/m;outside=np.abs(correct-mu)>score_radius
                for j,case in enumerate(selected):
                    t=case['t'];alpha=t/(2*math.sqrt(v))
                    best=next(r for r in case['candidates'] if r['factor']==case['best_energy_factor'])
                    a,bb=best['interval'];beta=t/(math.sqrt(a)+math.sqrt(bb))
                    actual=correct-t*np.sqrt(energy)>=0
                    lower=(1+2*alpha*mu/m)*correct-alpha*tilde-alpha*mu*mu/m-t*math.sqrt(v)/2>=0
                    upper=(1+2*beta*mu/m)*correct-beta*tilde+beta*(score_radius**2-mu*mu)/m-beta*math.sqrt(a*bb)>=0
                    low_energy=tilde<a+maximum_abs**2/m;high_energy=tilde>bb
                    events=np.stack((actual,lower,upper,low_energy,high_energy,outside))
                    counts[j]+=events.sum(axis=1)
                    violations[j,0]+=np.count_nonzero(lower&~actual)
                    violations[j,1]+=np.count_nonzero(actual&~(upper|low_energy|high_energy|outside))
                if (offset+size)%(spec['block']*16)==0:
                    print(f'{d} {method}: {offset+size}/{spec["queries_per_scatter"]} queries; {time.monotonic()-start:.1f}s',flush=True)
            raw.flush();del raw
            for j,case in enumerate(selected):
                best=next(r for r in case['candidates'] if r['factor']==case['best_energy_factor'])
                predictions=dict(lower=case['lower'],upper=best['quadratic_probability']['probability'],
                    low_energy=1-best['low_energy_probability']['probability'],high_energy=best['high_energy_probability']['probability'],
                    outside_score=case['score_outside'])
                endpoints={}
                for index,key in enumerate(spec['endpoints']):
                    estimate=float(counts[j,index]/spec['queries_per_scatter'])
                    ci=[max(0.,estimate-radius),min(1.,estimate+radius)]
                    endpoint=dict(count=int(counts[j,index]),estimate=estimate,hoeffding_interval=ci)
                    if key!='actual':
                        predicted=predictions[key];endpoint.update(integral_prediction=predicted,compatible=ci[0]<=predicted<=ci[1])
                    endpoints[key]=endpoint
                ci=endpoints['actual']['hoeffding_interval']
                row=dict(**cell,L=case['L'],t=case['t'],samples=spec['queries_per_scatter'],endpoints=endpoints,
                         float_envelope=[case['lower'],case['upper']],
                         actual_interval_overlaps_envelope=ci[1]>=case['lower'] and ci[0]<=case['upper'],
                         pointwise_lower_violations=int(violations[j,0]),pointwise_upper_violations=int(violations[j,1]))
                if violations[j].any():raise RuntimeError('Pointwise analytic event inclusion failed')
                result.append(row);stream.write(json.dumps(row)+'\n')
            del x,y,xy,f,b,u,g,w,bw
    for name,digest in spec['inputs'].items():assert sha(ROOT/name)==digest,name
    save(out/'results.json',dict(scope=spec['scope'],rows=result,hoeffding_halfwidth=radius,
        primary_integral_checks=60,compatible_integral_checks=sum(ep['compatible'] for row in result for ep in row['endpoints'].values() if 'compatible' in ep),
        actual_intervals_overlapping_envelope=sum(row['actual_interval_overlaps_envelope'] for row in result),
        wall_seconds=time.monotonic()-start,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    save(out/'FINISHED.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'));parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();(freeze if args.action=='freeze' else run)(args.out.resolve())
