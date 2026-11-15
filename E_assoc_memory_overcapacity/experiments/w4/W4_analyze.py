import os, sys
from pathlib import Path
BASE=Path(__file__).resolve().parent; ROOT=BASE.parents[1]
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(BASE/'W4_mplconfig'); os.environ['XDG_CACHE_HOME']=str(BASE/'W4_cache')
sys.path.insert(0,str(BASE/'repro'))
from W4_core import *
import csv, hashlib
from collections import defaultdict
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAMILY=2442

def stat(vals,df=None):
    a=np.array(vals); n=len(a); mean=float(a.mean()); se=float(a.std(ddof=1)/math.sqrt(n))
    crit=float(t.ppf(1-.05/(2*FAMILY),n-1 if df is None else df))
    ordinary=float(t.ppf(.975,n-1))*se
    return dict(n=n,mean=mean,se=se,lo=mean-crit*se,hi=mean+crit*se,
                ordinary_lo=mean-ordinary,ordinary_hi=mean+ordinary)

def csvout(name,rows):
    with (BASE/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    resource_info()
    rows=[json.loads(l) for l in (BASE/'W4_raw.jsonl').read_text().splitlines()]
    meta=json.loads((BASE/'W4_metadata.json').read_text())
    assert meta['stop'] is None and meta['completed']==117 and len(rows)==340
    mg=defaultdict(list); fg=defaultdict(list); lg=defaultdict(list)
    for r in rows:
        if r['kind']=='M': mg[(r['d'],r['method'])].append(r)
        elif r['kind']=='FT': fg[(r['d'],r['mode'])].append(r)
        else: lg[r['method']].append(r)
    endpoints=[]; cdfs=[]; cells=[]; grid=np.arange(1,100)/100
    for (d,name),rr in sorted(mg.items()):
        c=rr[0]['c']; cell=dict(d=d,method=name,c=c)
        for field,target,tol in [('score_mean',c,.05),('score_variance',1.,.10),
             ('mean_rank',float(ndtr(-c/math.sqrt(2))),.01),('top10',float(ndtr(c-ndtri(.9))),.015)]:
            ss=stat([r[field]-target for r in rr]); status='equivalent' if ss['lo']>=-tol and ss['hi']<=tol else ('outside' if ss['lo']>tol or ss['hi']<-tol else 'unresolved')
            endpoints.append(dict(d=d,method=name,metric=field,target=target,tolerance=tol,status=status,**ss))
            cell[field]=dict(target=target,**stat([r[field] for r in rr]))
        for what in ['score','rank']:
            arr=np.array([r[what+'_pit_cdf'] for r in rr]); bounds=[]
            for j,p in enumerate(grid):
                ss=stat(arr[:,j]-p); bounds.append(max(abs(ss['lo']),abs(ss['hi'])))
                cdfs.append(dict(d=d,method=name,distribution=what,p=p,**ss))
            pits=np.concatenate([ndtr(np.array(r['scores'])-c) if what=='score' else ndtr(ndtri(np.array(r['ranks']))+c) for r in rr])
            cell[what+'_cdf']=dict(pooled_full_ks=ks_uniform(pits),simultaneous_upper=min(1.,max(bounds)+.01),
                equivalent=max(bounds)+.01<=.04,seed_ks=stat([r[what+'_ks'] for r in rr]))
        cell['full_variance']=stat([r['full_variance'] for r in rr])
        cell['variance128']=stat([r['variance128'] for r in rr]); cell['wrong_ks']=stat([np.mean(r['wrong_ks']) for r in rr])
        cells.append(cell)
    ftstats=[]
    for (d,mode),rr in sorted(fg.items()):
        ftstats.append(dict(d=d,mode=mode,prediction=rr[0]['prediction'],
             error=stat([r['old']['msr']-r['prediction'] for r in rr]),
             loss=stat([r['loss'] for r in rr]),step=stat([r['step'] for r in rr])))
    trends=[]
    for mode in ['fixed','matched']:
        low=next(r for r in ftstats if r['d']==64 and r['mode']==mode)['error']
        high=next(r for r in ftstats if r['d']==320 and r['mode']==mode)['error']
        diff=high['mean']-low['mean']; se=math.hypot(high['se'],low['se']); crit=t.ppf(1-.05/(2*FAMILY),15)
        trends.append(dict(mode=mode,difference=diff,se=se,lo=diff-crit*se,hi=diff+crit*se))
    legacy=[]; s1=[]
    for name,rr in sorted(lg.items()):
        er=np.array([r['original_variance']-r['full_variance'] for r in rr])
        sv=np.array([r['resample_variances'] for r in rr])
        legacy.append(dict(method=name,original=stat([r['original_variance'] for r in rr]),
             full=stat([r['full_variance'] for r in rr]),query_shift=stat(er),
             query_sampling_sd=float(np.mean(np.std(sv,axis=1,ddof=1))),
             resample_mean=float(sv.mean())))
        s1.append(dict(source='old128_minus_full',method=name,**stat(er)))
        s1.append(dict(source='new_full_minus_one',method=name,**stat([r['full_variance']-1 for r in mg[(64,name)]])))
    shifts=np.array([[r['original_variance']-r['full_variance'] for r in lg[name]] for name in NAMES])
    # Check regenerated old samples against the immutable ledger; retain selected provenance only.
    old=[]; names={'GD':'GD','Sign1':'Sign1','Sign2_a0.6_actual':'Sign2','Adam_b0.9_e0.1_T16_actual':'Adam16'}
    for line in (ROOT/'experiments/exp1/formal_rows.jsonl').open():
        r=json.loads(line)
        if r['phase']=='A' and r['d']==64 and r['kappa']==1 and r['name'] in names:
            name=names[r['name']]; new=next(v for v in lg[name] if v['seed']==r['seed'])
            assert abs(new['original_variance']-r['true_variance'])<1e-11
            assert new['ids']==r['query_indices']
            old.append(dict(d=64,seed=r['seed'],method=name,variance=r['true_variance'],scores=r['true_scores'],ids=r['query_indices']))
    assert len(old)==20
    (BASE/'W4_old_selected.json').write_text(json.dumps(old,indent=2))
    result=dict(family_size=FAMILY,complete=True,M=cells,FT=ftstats,trends=trends,legacy=legacy,
         legacy_query_shift_correlation=np.corrcoef(shifts).tolist(),legacy_correlation_methods=NAMES,
         old_failures_retained=dict(EXP1=4,FT_retrieval_equivalence=316,FT_retrieval_total=640),
         cdf_passes=sum(r[w+'_cdf']['equivalent'] for r in cells for w in ['rank','score']),
         endpoint_status_counts={s:sum(r['status']==s for r in endpoints) for s in ['equivalent','outside','unresolved']})
    (BASE/'W4_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    csvout('W4_endpoints.csv',endpoints); csvout('W4_CDF_intervals.csv',cdfs); csvout('W4_S1_intervals.csv',s1)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name in NAMES:
        rr=mg[(256,name)]; c=rr[0]['c']
        for ax,key in zip(axs,['ranks','scores']):
            vals=np.sort(np.concatenate([r[key] for r in rr])); p=np.arange(1,len(vals)+1)/len(vals)
            pit=ndtr(ndtri(vals)+c) if key=='ranks' else ndtr(vals-c)
            ax.plot(pit,p-pit,label=name)
    for ax,title in zip(axs,['Rank CDF at d256','Correct-score CDF at d256']):
        ax.set(xlabel='Theoretical percentile',ylabel='Empirical CDF minus target',title=title); ax.axhline(0,color='k',lw=.5); ax.legend()
    fig.savefig(BASE/'W4_distributions.png',dpi=160); fig.savefig(BASE/'W4_distributions.pdf'); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for name in NAMES:
        rr=[r for r in cells if r['method']==name]
        axs[0].errorbar([r['d'] for r in rr],[r['full_variance']['mean'] for r in rr],yerr=[r['full_variance']['se'] for r in rr],marker='o',label=name)
    axs[0].axhline(1,color='k',ls='--'); axs[0].set(xlabel='d',ylabel='Full-query correct-score variance'); axs[0].legend()
    for mode in ['fixed','matched']:
        rr=[r for r in ftstats if r['mode']==mode]
        axs[1].errorbar([r['d'] for r in rr],[r['error']['mean'] for r in rr],yerr=[r['error']['se'] for r in rr],marker='o',label=mode)
    axs[1].axhline(0,color='k',ls='--'); axs[1].set(xlabel='d',ylabel='Old MSR minus population limit'); axs[1].legend()
    fig.savefig(BASE/'W4_dimension.png',dpi=160); fig.savefig(BASE/'W4_dimension.pdf'); plt.close(fig)
    print(json.dumps(dict(cdf_passes=result['cdf_passes'],endpoints=result['endpoint_status_counts'],FT=ftstats,legacy=legacy),allow_nan=False),flush=True)

if __name__=='__main__': main()
