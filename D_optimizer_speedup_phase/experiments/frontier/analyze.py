"""Frozen gates and independent reconstruction of stored states."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/frontier'

def main():
    start=time.monotonic()
    mode=sys.argv[1]
    out=BASE/mode
    subprocess.run(['sha256sum','-c',str(BASE/'FROZEN.sha256')],cwd=ROOT,check=True,
                   stdout=(out/'analysis_freeze_check.txt').open('w'))
    props=subprocess.check_output(['systemctl','--user','show',os.environ['FRONTIER_SCOPE'],
                                  '-p','MemoryMax','-p','MemorySwapMax'],text=True)
    assert 'MemoryMax=2147483648' in props and 'MemorySwapMax=0' in props
    data=json.loads((out/'data.json').read_text())
    records=data['records']
    arrays={}
    max_recompute=0.
    max_loc6=0.
    sandwich=[]
    for ri,r in enumerate(records):
        with np.load(out/'arrays'/r['file']) as z:
            e,lam,tail=z['e'],z['lam'],float(z['tail'])
            v=(e*e)@lam
            risks=tail+v
            max_recompute=max(max_recompute,float(np.max(np.abs(risks-z['risk']))),abs(float(risks.mean())-r['risk']))
            arrays[ri]=risks.copy()
            h=e*e
            s=1+tail+v
            ratio=(h*s[:,None]).mean(axis=0)/h.mean(axis=0)
            influence=h*(s[:,None]-ratio[None,:])/h.mean(axis=0)
            se=influence.std(axis=0,ddof=1)/math.sqrt(len(e))
            max_loc6=max(max_loc6,float(np.max(ratio+6*se)))
            if 'lower' in r:
                sandwich.append(r['lower']<=r['risk']+6*r['se'] and r['risk']-6*r['se']<=r['upper'])
    gen=np.random.default_rng(2026110618 if mode=='formal' else 2026110619)
    def fit(ids,heldout=False):
        rr=[records[i] for i in ids]
        x=np.log([r['C'] for r in rr])
        # Critical alpha=2: C/log C is only an asymptotic coordinate; also report exact finite H.
        if rr[0]['algo']=='sign' and rr[0]['alpha']==2:
            x=np.log([r['C']/np.log(r['C']) for r in rr])
        vals=np.array([arrays[i][48:] if heldout else arrays[i] for i in ids])
        y=np.log(vals.mean(axis=1))
        xc=x-x.mean()
        slope=-float(xc@(y-y.mean())/(xc@xc))
        residual=y-(y.mean()-slope*xc)
        r2=1-float(residual@residual/((y-y.mean())@(y-y.mean())))
        boots=[]
        for j in range(2000):
            means=np.array([v[gen.integers(0,len(v),len(v))].mean() for v in vals])
            yy=np.log(means)
            boots.append(-float(xc@(yy-yy.mean())/(xc@xc)))
        lo,hi=np.quantile(boots,[.025,.975])
        alpha=rr[0]['alpha']
        p=alpha/2 if rr[0]['algo']=='sign' else alpha
        expected=1/(3 if rr[0]['algo']=='sign' and alpha<=2 else 2+p)
        return dict(alpha=alpha,algo=rr[0]['algo'],expected=expected,slope=slope,
                    ci95=[float(lo),float(hi)],r2=r2,ids=ids,
                    log_corrected=rr[0]['algo']=='sign' and alpha==2)
    fits=[]
    fronts=[]
    if mode=='formal':
        for alpha in [1.5,2.,3.]:
            for algo in ['sign','polar','sgd']:
                ids=[i for i,r in enumerate(records) if r['alpha']==alpha and r['algo']==algo and r['b']==1 and r['factor']==.6 and r['tau']==2]
                fits.append(fit(ids))
                choices=[]
                for i in ids:
                    cap=records[i]['C']
                    eligible=[j for j,r in enumerate(records) if r['alpha']==alpha and r['algo']==algo and r['C']<=cap]
                    j=min(eligible,key=lambda j:float(arrays[j][:48].mean()))
                    choices.append(j)
                # Use the paid budget axis, including unused compute, not selected candidate's cost.
                original=[records[j]['C'] for j in choices]
                copies=[]
                for j,i in zip(choices,ids):
                    copies.append(dict(records[j],C=records[i]['C']))
                offset=len(records)
                for j,r in zip(choices,copies):
                    arrays[len(records)]=arrays[j]
                    records.append(r)
                ff=fit(list(range(offset,len(records))),heldout=True)
                ff['selected_original_ids']=choices
                ff['selected_costs']=original
                fronts.append(ff)
                del records[offset:]
    batch=[]
    for i,r in enumerate(records):
        if r['b']==4:
            match=[j for j,s in enumerate(records) if s['alpha']==r['alpha'] and s['algo']==r['algo'] and s['n']==r['n'] and s['factor']==r['factor'] and s['tau']==r['tau'] and s['b']==1][0]
            batch.append(dict(algo=r['algo'],n=r['n'],tau=r['tau'],risk_ratio=r['risk']/records[match]['risk'],compute_ratio=r['C']/records[match]['C']))
    q0=data['p0']['one_step']<=1e-10 and data['p0']['sum_k']<=1e-8 and data['p0']['nonfinite']==0 and max_recompute<=1e-10
    q1=max_loc6<=16 and all(sandwich)
    q2=all(abs(f['slope']-f['expected'])<=.08 and f['ci95'][1]-f['ci95'][0]<=.16 and f['r2']>=.95 for f in fits) if fits else None
    q3=all(abs(f['slope']-f['expected'])<=.10 and f['ci95'][1]-f['ci95'][0]<=.20 and f['r2']>=.90 for f in fronts) if fronts else None
    difference=None
    q4=None
    if fits:
        ss=next(f for f in fits if f['alpha']==3 and f['algo']=='sign')
        gg=next(f for f in fits if f['alpha']==3 and f['algo']=='sgd')
        difference=ss['slope']-gg['slope']
        q4=ss['ci95'][0]>gg['ci95'][1] and abs(difference-(1/3.5-1/5))<=.08
    man=json.loads((out/'manifest.json').read_text())
    expected_count=240 if mode=='formal' else 24  # smoke checkpoints can collapse; audited below
    # Formal: 3*6*3*2*2=216 base checkpoints + 3*2*2=12 batch checkpoints.
    expected_count=228 if mode=='formal' else 24
    q5=man['status']=='complete' and (len(records)==expected_count if mode=='formal' else len(records)>=12) and man['peak_rss_mib']<2048 and man['elapsed_seconds']<2700 and all(v=='1' for v in man['threads'].values())
    result=dict(mode=mode,gates=dict(Q0=q0,Q1=q1,Q2=q2,Q3=q3,Q4=q4,Q5=q5),
                reconstruct_max_abs=max_recompute,loc_max_plus_6se=max_loc6,sandwich_pass=sum(sandwich),
                sandwich_total=len(sandwich),constructive=fits,grid_frontier=fronts,
                alpha3_sign_minus_sgd_slope=difference,batch_descriptive=batch,
                analysis_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                scope_properties=props,records=len(records))
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Frozen frontier analysis','',f'Gates: {result["gates"]}', '',
           '| alpha | algorithm | predicted | measured | 95% bootstrap CI | R2 |',
           '|---|---|---|---|---|---|']
    for f in fits:
        lines.append(f'| {f["alpha"]} | {f["algo"]} | {f["expected"]:.6f} | {f["slope"]:.6f} | {f["ci95"]} | {f["r2"]:.6f} |')
    (out/'TABLES.md').write_text('\n'.join(lines)+'\n')
    if mode=='formal':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,alpha in zip(axes,[1.5,2.,3.]):
            for algo in ['sign','polar','sgd']:
                rr=[r for r in records if r['alpha']==alpha and r['algo']==algo and r['b']==1 and r['factor']==.6 and r['tau']==2]
                ax.errorbar([r['C'] for r in rr],[r['risk'] for r in rr],yerr=[2*r['se'] for r in rr],marker='o',label=algo)
            ax.set(xscale='log',yscale='log',xlabel='paid compute n b T',ylabel='excess risk',title=f'alpha={alpha}, beta=1')
            ax.legend()
        fig.tight_layout()
        fig.savefig(BASE/'frontier.png',dpi=170)
        fig.savefig(BASE/'frontier.pdf')
        plt.close(fig)
    result['analysis_seconds']=time.monotonic()-start
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['gates']))
    if mode=='smoke' and not (q0 and q1 and q5):
        raise SystemExit('Smoke failed; do not run formal')

if __name__=='__main__':
    main()
