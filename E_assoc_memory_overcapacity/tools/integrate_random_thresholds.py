"""Frozen random-threshold integration of the R6 symbolic probability bounds.

Primary case: existing d=512 Adam2 scatter rep=0. New independent sphere
thresholds and conditional queries, with separate Hoeffding error budgets.
All quadratic-form evaluations remain uncertified float64 calculations.
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


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def freeze(out):
    names=['tools/check_tangent_quadratic.py','tools/diagnose_joint_energy.py','tools/analyze_joint_gaussian.py',
           'revisions/r3_joint_diagnostic_v2/design.json','revisions/R6_DOUBLE_TANGENT_SECANT.md',
           'revisions/R7_MONOTONE_THRESHOLD_TRANSFER.md']
    design=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),seed=202611071917,
        d=512,method='Adam2',rep=0,lists=[1,5],thresholds=131072,threshold_block=128,
        queries=131072,query_block=2048,groups=128,energy_factor=3.,score_sd_radius=5.,
        threshold_alpha=.025,threshold_function_family=4,
        reference_alpha=.025,reference_endpoint_family=6,
        confidence='Two-sided Hoeffding for four threshold functions and six paired reference endpoints; union error <= .05 under exact sampling/evaluation',
        scope='New focused conditional top-L study; not a retrospective pass for old experimental families or a single-dataset guarantee',
        selection='d=512 Adam2 selected before new draws to study the previous approximately 4pp mechanism gap',
        source_sha256=sha(Path(__file__)),inputs={name:sha(ROOT/name) for name in names},
        wall_limit_seconds=600,cpu_limit_seconds=900,threads=2,
        stopping='Fixed sample counts and grouping; retain partial data on failure, no redraw or outcome-dependent extension')
    out.mkdir(parents=True,exist_ok=False);save(out/'design.json',design)
    print('Frozen focused scatter, 131072 fresh thresholds and queries, 128 groups and separate error budgets.')


def run(out):
    design=json.loads((out/'design.json').read_text())
    assert sha(Path(__file__))==design['source_sha256']
    for name,digest in design['inputs'].items():assert sha(ROOT/name)==digest,name
    resource.setrlimit(resource.RLIMIT_CPU,(design['cpu_limit_seconds'],design['cpu_limit_seconds']+5))
    quad=load('r6_cf',ROOT/'tools/check_tangent_quadratic.py')
    generator=load('r3_gen',ROOT/'tools/diagnose_joint_energy.py')
    moment_module=load('r5_mom',ROOT/'tools/analyze_joint_gaussian.py')
    old=json.loads((ROOT/'revisions/r3_joint_diagnostic_v2/design.json').read_text())
    cell=next(r for r in old['archived_loads'] if r['d']==design['d'] and r['method']==design['method'])
    d,n,method=cell['d'],cell['n'],cell['method'];m=n-1;count=design['thresholds']
    start=time.monotonic()
    save(out/'STARTED.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),integrator_checks=quad.self_checks()))
    def budget():
        if time.monotonic()-start>design['wall_limit_seconds']:
            raise TimeoutError('Frozen wall budget reached; partial products retained')
    rng=generator.seed_rng(old['seed'],d,generator.METHODS.index(method),design['rep'],2)
    x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
    u=generator.train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
    xy=np.vstack((x,y));chol=np.linalg.cholesky(d/n*xy@xy.T)
    b=chol[d:].T@u@chol[:d]/np.linalg.norm(u);k=(b+b.T)/2;nn=b.T@b
    mom=moment_module.moments(b,n);mu=mom['mu'];v=mom['expected_V']
    with np.load(ROOT/f'revisions/r3_joint_diagnostic_v2/joint_{d}_{method}_0.npz') as archive:
        for key in ('mu','sigma2','expected_V'):assert abs(float(archive[key])-mom[key])<1e-10
    radius=design['score_sd_radius']*math.sqrt(mom['sigma2']);maximum_abs=abs(mu)+radius
    a=v-design['energy_factor']*math.sqrt(mom['variance_V']);bb=v+design['energy_factor']*math.sqrt(mom['variance_V'])
    assert a>0 and math.sqrt(a*bb)>=radius**2/m
    def tail(eigen,constant=0.,level=0.):
        shift=(level-constant)/n
        return quad.weighted_tail(np.r_[eigen-shift,-shift],np.r_[np.ones(len(eigen),dtype=int),n-len(eigen)])
    eig_d=np.linalg.eigvalsh(k);eig_w=np.linalg.eigvalsh(n/m*nn)
    d0=tail(eig_d)['probability'];dlo=tail(eig_d,level=mu-radius);dhi=tail(eig_d,level=mu+radius)
    wlo=tail(eig_w,level=a+maximum_abs**2/m);whi=tail(eig_w,level=bb)
    delta=max(0.,1-dlo['probability']+dhi['probability'])+max(0.,1-wlo['probability'])+max(0.,whi['probability'])
    save(out/'scatter.json',dict(**cell,moments=mom,energy_interval=[a,bb],score_radius=radius,
         monotone_margin=math.sqrt(a*bb)-radius**2/m,bad_event_bound=delta,
         component_probabilities=dict(dlo=dlo,dhi=dhi,wlo=wlo,whi=whi)))
    del x,y,xy,chol,u
    threshold_rng=np.random.default_rng(np.random.SeedSequence([design['seed'],1]))
    thresholds=np.lib.format.open_memmap(out/'thresholds.npy',mode='w+',dtype='float64',shape=(count,2))
    for offset in range(0,count,design['threshold_block']):
        budget();size=min(design['threshold_block'],count-offset)
        g=threshold_rng.normal(size=(size,m));norm=np.sqrt(np.mean(g*g,axis=1))
        top=np.partition(g,[m-5,m-1],axis=1)[:,[m-1,m-5]]
        thresholds[offset:offset+size]=top/norm[:,None]
        if (offset+size)%8192==0:print(f'thresholds {offset+size}/{count}; {time.monotonic()-start:.1f}s',flush=True)
    thresholds.flush()
    query_rng=np.random.default_rng(np.random.SeedSequence([design['seed'],2]))
    raw=np.lib.format.open_memmap(out/'D_V.npy',mode='w+',dtype='float64',shape=(2,count))
    event_counts=np.zeros((2,3),dtype=np.int64);violations=np.zeros((2,2),dtype=np.int64)
    for offset in range(0,count,design['query_block']):
        budget();size=min(design['query_block'],count-offset)
        g=query_rng.normal(size=(2*d,size));w=g/np.sqrt(np.sum(g*g,axis=0)+query_rng.chisquare(n-2*d,size=size))
        bw=b@w;correct=n*np.sum(w*bw,axis=0);energy=(n*n*np.sum(bw*bw,axis=0)-correct**2)/m
        if np.any(energy<=0):raise RuntimeError('Nonpositive energy; no discard or redraw')
        raw[:,offset:offset+size]=np.stack((correct,energy));tilde=energy+correct**2/m
        bad=(np.abs(correct-mu)>radius)|(tilde<a+maximum_abs**2/m)|(tilde>bb)
        for j in range(2):
            t=np.asarray(thresholds[offset:offset+size,j]);positive=t>0
            alpha=t/(2*math.sqrt(v));beta=t/(math.sqrt(a)+math.sqrt(bb))
            actual=correct>=t*np.sqrt(energy)
            low=(1+2*alpha*mu/m)*correct-alpha*tilde-alpha*mu*mu/m-t*math.sqrt(v)/2>=0
            high=(1+2*beta*mu/m)*correct-beta*tilde+beta*(radius**2-mu*mu)/m-beta*math.sqrt(a*bb)>=0
            low=np.where(positive,low,correct>=0);high=np.where(positive,high,True)
            event_counts[j]+=np.stack((actual,low,high)).sum(axis=1)
            violations[j,0]+=np.count_nonzero(low&~actual)
            violations[j,1]+=np.count_nonzero(actual&~(high|bad))
        if (offset+size)%32768==0:print(f'queries {offset+size}/{count}; {time.monotonic()-start:.1f}s',flush=True)
    raw.flush()
    if violations.any():raise RuntimeError('Paired event inclusion failure')
    stat_radius=math.sqrt(math.log(2*design['threshold_function_family']/design['threshold_alpha'])/(2*count))
    ref_radius=math.sqrt(math.log(2*design['reference_endpoint_family']/design['reference_alpha'])/(2*count))
    results=[]
    with (out/'nodes.jsonl').open('x',buffering=1) as stream:
        for j,length in enumerate(design['lists']):
            ordered=np.sort(np.asarray(thresholds[:,j]));group=count//design['groups']
            indices=np.r_[0,np.arange(group,count+1,group)-1];nodes=ordered[indices]
            lowers=[];uppers=[]
            for index,t in enumerate(nodes):
                budget()
                if t<=0:
                    lower=dict(probability=d0,tail_bound=0.,quad_error_estimate=0.)
                    upper=dict(probability=1.,tail_bound=0.,quad_error_estimate=0.)
                else:
                    alpha=float(t)/(2*math.sqrt(v));beta=float(t)/(math.sqrt(a)+math.sqrt(bb))
                    low_h=(1+2*alpha*mu/m)*k-alpha*n/m*nn
                    low_c=-alpha*mu*mu/m-float(t)*math.sqrt(v)/2
                    high_h=(1+2*beta*mu/m)*k-beta*n/m*nn
                    high_c=beta*(radius**2-mu*mu)/m-beta*math.sqrt(a*bb)
                    lower=tail(np.linalg.eigvalsh(low_h),low_c);upper=tail(np.linalg.eigvalsh(high_h),high_c)
                lowers.append(lower['probability']);uppers.append(upper['probability'])
                stream.write(json.dumps(dict(L=length,index=index,t=float(t),lower=lower,upper_quadratic=upper))+'\n')
                if index%16==0:print(f'L={length} node {index}/{design["groups"]}; {time.monotonic()-start:.1f}s',flush=True)
            if max(np.diff(lowers))>1e-8 or max(np.diff(uppers))>1e-8:raise RuntimeError('Monotonicity check failed')
            llo=float(np.mean(lowers[1:]));lhi=float(np.mean(lowers[:-1]))
            ulo=float(np.mean(uppers[1:]));uhi=float(np.mean(uppers[:-1]))
            enclosure=[max(0.,llo-stat_radius),min(1.,uhi+delta+stat_radius)]
            references={}
            for index,key in enumerate(('actual','lower_quadratic','upper_quadratic')):
                estimate=float(event_counts[j,index]/count)
                references[key]=dict(estimate=estimate,count=int(event_counts[j,index]),
                                     interval=[max(0.,estimate-ref_radius),min(1.,estimate+ref_radius)])
            ci=references['actual']['interval']
            results.append(dict(L=length,threshold_samples=count,conditional_queries=count,
                empirical_lower_mean_bracket=[llo,lhi],empirical_upper_quadratic_mean_bracket=[ulo,uhi],
                lower_grouping_gap=lhi-llo,upper_grouping_gap=uhi-ulo,threshold_hoeffding_radius=stat_radius,
                bad_event_bound=delta,float_statistical_enclosure=enclosure,width=enclosure[1]-enclosure[0],
                references=references,reference_hoeffding_radius=ref_radius,
                actual_reference_compatible=ci[1]>=enclosure[0] and ci[0]<=enclosure[1],
                negative_threshold_count=int(np.count_nonzero(ordered<=0)),event_inclusion_violations=violations[j].tolist()))
    for name,digest in design['inputs'].items():assert sha(ROOT/name)==digest,name
    save(out/'results.json',dict(scope=design['scope'],numerical_caveat='Uncertified eigenvalues and finite quadrature; statistical formulas assume exact evaluations',
        d=d,n=n,method=method,rep=design['rep'],rows=results,wall_seconds=time.monotonic()-start,
        max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    del thresholds,raw
    save(out/'FINISHED.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'));parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();(freeze if args.action=='freeze' else run)(args.out.resolve())
