"""Read-only AD0 audit, original-seed geometry replay and uniform GD checks.

No new experimental samples: the geometry replay follows the archived seed
and draw order exactly. Other AD0 batches are never executed or resumed.
"""
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import statistics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def save(path,value):
    with path.open('x') as h:json.dump(value,h,indent=2);h.write('\n')
def stats(values):return statistics.fmean(values),statistics.stdev(values)/math.sqrt(len(values))
def close(a,b):assert abs(a-b)<2e-12,(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];source=root/'ad0_checks'
    hashes={p.name:sha(p) for p in source.iterdir() if p.is_file()}
    audits={};all_main=None
    for name in ['pilot','main']:
        result=read(source/f'AD0_{name}_results.json');design=read(source/f'AD0_{name}_design.json');assert design==result['design']
        rows=[json.loads(s) for s in (source/f'AD0_{name}_rows.jsonl').read_text().splitlines()]
        primary=[r for r in rows if 'aux' not in r];groups=defaultdict(list)
        for row in primary:groups[(row['d'],row['n'],row['regime'],row['alg'],row['eta'],str(row['T']))].append(row)
        for item in result['summary']:
            key=tuple(item[k] for k in ['d','n','regime','alg','eta','T']);rs=groups[key];assert len(rs)==design['reps']
            for metric in ['top1','top5','top10pct','mean_rank_fraction','mu','cosine_Z']:
                mean,se=stats([r[metric] for r in rs]);close(mean,item[metric]);close(se,item[metric+'_se'])
            assert sum(r['unstable_step'] for r in rs)==item['unstable_reps']
        for item in result['paired']:
            base={r['rep']:r for r in primary if r['d']==item['d'] and r['regime']==item['regime'] and r['alg']=='GD' and r['T']=='infinity'}
            rs=[r for r in primary if r['d']==item['d'] and r['regime']==item['regime'] and r['alg']=='GD' and r['eta']==item['eta'] and r['T']==item['T']]
            mean,se=stats([r['top1']-base[r['rep']]['top1'] for r in rs]);close(mean,item['top1_minus_infinity']);close(se,item['paired_se'])
        audits[name]=dict(records=len(rows),dataset_endpoint_records=len(primary),auxiliary_records=len(rows)-len(primary),summaries_verified=len(groups),paired_summaries_verified=len(result['paired']),
                         unstable_GD_endpoint_records=sum(r['unstable_step'] for r in primary if r['alg']=='GD'))
        if name=='main':all_main=result
    # Load sweep: preserve pairing with least squares, never select seeds.
    result=read(source/'AD0_load_results.json');rows=[json.loads(s) for s in (source/'AD0_load_rows.jsonl').read_text().splitlines()]
    for item in result['summary']:
        rs=[r for r in rows if (r['d'],r['load'],r['T'])==(item['d'],item['load'],item['T'])]
        assert len(rs)==result['design']['reps'];base={r['rep']:r['top1'] for r in rows if r['d']==item['d'] and r['load']==item['load'] and r['T']=='infinity'}
        mean,se=stats([r['top1'] for r in rs]);close(mean,item['top1']);close(se,item['se'])
        mean,se=stats([r['top1']-base[r['rep']] for r in rs]);close(mean,item['minus_infinity']);close(se,item['paired_se'])
    audits['load']=dict(records=len(rows),summaries_verified=len(result['summary']))
    # All-success observations do not have a zero uncertainty bound.
    result=read(source/'AD0_perfect_results.json');rows=[json.loads(s) for s in (source/'AD0_perfect_rows.jsonl').read_text().splitlines()];success=[]
    for item in result['summary']:
        rs=[r for r in rows if (r['d'],r['load'],r['T'])==(item['d'],item['load'],item['T'])];assert len(rs)==16
        mean,se=stats([r['perfect'] for r in rs]);close(mean,item['perfect_probability']);close(se,item['se']);close(statistics.fmean(r['recall'] for r in rs),item['mean_recall'])
        assert sum(not r['step_stable'] for r in rs)==item['unstable_reps']
        if all(r['perfect']==1 for r in rs):success.append({k:item[k] for k in ['d','load','n','T','perfect_probability','se','unstable_reps']})
    point=F(math.floor((1/20)**(1/16)*10**6),10**6);family=F(math.floor((1/900)**(1/16)*10**6),10**6)
    assert point**16<=F(1,20) and family**16<=F(1,900)
    audits['perfect']=dict(records=len(rows),summaries_verified=len(result['summary']),all_success_cells=success,
        pointwise_one_sided95_lower_conservative=str(point),family45_one_sided95_lower_conservative=str(family),
        assumption='IID Bernoulli dataset-perfect outcomes; formulas exact, decimal endpoints verified by rational powers; no proof of perfect recall probability one')
    audits['selected_early_stopping_diagnostic']=[max([r for r in all_main['paired'] if r['d']==d and r['regime']=='critical' and r['eta']==.5],key=lambda r:r['top1_minus_infinity']) for d in [64,128,256]]
    # Uniform-in-T filter inequality on exact rational eigenvalues.
    checks=0
    for eta in [F(1,2),F(1),F(3,2)]:
        radius=F(1,8);rho=max(abs(1-eta*(1-radius)),abs(1-eta*(1+radius)))
        lipschitz=eta*eta/(1-rho)**2;minimum=1-abs(1-eta)
        for index in range(17):
            s=1-radius+index*radius/8
            for T in [1,2,3,4,8,16,64,128,None]:
                p=1/s if T is None else (1-(1-eta*s)**T)/s
                k=F(1) if T is None else 1-(1-eta)**T
                assert k>=minimum and abs(p-k)<=lipschitz*abs(s-1)
                assert abs(p/k-1)<=lipschitz*abs(s-1)/minimum;checks+=1
    audits['uniform_GD_exact_filter_checks']=checks
    save(args.out/'AUDIT.json',audits)
    # Replay exactly the 32 geometry scatters after the three identity examples.
    import numpy as np
    from scipy.special import ndtri
    spec=importlib.util.spec_from_file_location('r5_moments',root/'tools/analyze_joint_gaussian.py');moment=importlib.util.module_from_spec(spec);spec.loader.exec_module(moment)
    rng=np.random.default_rng(20261107005)
    for _ in range(3):
        rng.standard_normal((8,90));rng.standard_normal((8,90))
    d,n=128,1577;b=float(ndtri(1-1/(n-1)));rows=[];identity=[]
    for rep in range(32):
        X=rng.standard_normal((d,n))/np.sqrt(d);Y=rng.standard_normal((d,n))/np.sqrt(d)
        S=d/n*(X@X.T);Sy=d/n*(Y@Y.T);Z=d/np.sqrt(n)*(Y@X.T)
        eig,V=np.linalg.eigh(S);ZV=Z@V
        for T in [1,2,4,8,'infinity']:
            p=1/eig if T=='infinity' else (1-(1-.5*eig)**T)/eig
            U=(ZV*p)@V.T;normsq=np.sum(U*U);mu=float(np.sum(U*Z)/(np.sqrt(n)*np.sqrt(normsq)))
            energy=float(np.trace(U@S@U.T@Sy)/normsq);A=U@Z.T/np.sqrt(n)
            oldvar=energy+float(np.trace(A@A)/normsq)-2*mu*mu/n
            variance=n/(n+2)*oldvar;meanW=n/(n-1)*energy;meanV=meanW-(mu*mu+variance)/(n-1)
            row=dict(rep=rep,T=T,mu=mu,old_v=math.sqrt(meanW),old_sigma=math.sqrt(oldvar),old_effective_center=(mu-math.sqrt(meanW)*b)/math.sqrt(oldvar),
                     sqrt_expected_V=math.sqrt(meanV),sigma=math.sqrt(variance),effective_center=(mu-math.sqrt(meanV)*b)/math.sqrt(variance))
            rows.append(row)
            if rep<2 and T in [1,'infinity']:
                xy=np.vstack((X,Y));factor=np.linalg.cholesky(d/n*(xy@xy.T));block=factor[d:].T@U@factor[:d]/np.sqrt(normsq)
                m=moment.moments(block,n)
                error=max(abs(m['mu']-mu),abs(m['sigma2']-variance),abs(m['expected_V']-meanV));assert error<5e-12
                identity.append(dict(rep=rep,T=T,max_float_trace_disagreement=error))
    summary=[];old=read(source/'AD0_geometry_results.json')
    for T in [1,2,4,8,'infinity']:
        rs=[r for r in rows if r['T']==T];previous=next(r for r in old['summary'] if r['T']==T)
        means={k:statistics.fmean(r[k] for r in rs) for k in rs[0] if k not in ['rep','T']}
        for key,current in [('mu','mu'),('v','old_v'),('sigma','old_sigma'),('effective_center_b','old_effective_center')]:assert abs(previous[key]-means[current])<2e-11
        summary.append(dict(T=T,**means))
    save(args.out/'GEOMETRY.json',dict(seed=20261107005,archived_seed_replay_only=True,d=d,n=n,summary=summary,rows=rows,trace_checks=identity,
         scope='Corrected exact conditional moment formulas evaluated in float; Gaussian effective-center proxy is still only diagnostic'))
    assert {p.name:sha(p) for p in source.iterdir() if p.is_file()}==hashes
    save(args.out/'SOURCES.json',hashes)
    print(json.dumps(dict(audits={k:v for k,v in audits.items() if k!='perfect'},all_success_cells=len(success),perfect_lower_bounds=[str(point),str(family)],geometry_summary=summary),indent=2))


if __name__=='__main__':main()
