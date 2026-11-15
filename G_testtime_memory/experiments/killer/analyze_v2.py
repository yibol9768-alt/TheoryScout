"""Prespecified calibration and complete seed-level gate audit."""
import csv
import json
import resource
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from design import DIMS,LOADS,ALPHAS,METHODS,BETAS,CAL_SEEDS,TEST_SEEDS,snr,predict
from run_v2 import check_freeze,resources

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'experiments/killer'

def mean_se(v):
    x=np.asarray(v,dtype=float)
    return float(x.mean()),float(x.std(ddof=1)/np.sqrt(len(x)))

def main():
    start=time.time()
    check_freeze()
    meta=resources()
    assert json.loads((OUT/'formal_meta.json').read_text())['complete']
    groups=defaultdict(dict)
    for line in (OUT/'raw.jsonl').read_text().splitlines():
        r=json.loads(line)
        key=(r['split'],r['d'],r['load'],r['alpha'],r['method'])
        assert r['seed'] not in groups[key]
        groups[key][r['seed']]=r
    assert len(groups)==4*len(LOADS)*len(ALPHAS)*len(METHODS)
    for (split,d,k,a,m),v in groups.items():
        assert set(v)==set(CAL_SEEDS if split=='calibration' else TEST_SEEDS)
    cache={(d,k,a,m):snr(d,k,a,m) for d in DIMS for k in LOADS for a in ALPHAS for m in METHODS}
    cal_keys=[k for k in cache if k[0]==64 and k[2]>=1]
    targets=np.array([mean_se([r['top1'] for r in groups[('calibration',)+k].values()])[0] for k in cal_keys])
    fits=[]
    for c in np.linspace(.5,1.5,201):
        preds=np.array([predict(cache[k],k[0]*k[1],float(c)) for k in cal_keys])
        fits.append(float(np.mean((preds-targets)**2)))
    j=int(np.argmin(fits))
    c=float(np.linspace(.5,1.5,201)[j])
    (OUT/'calibration.json').write_text(json.dumps({'c':c,'mse':fits[j],'endpoint':j in (0,200),
        'c_grid':np.linspace(.5,1.5,201).tolist(),'mse_grid':fits,'seeds':list(CAL_SEEDS)},indent=2))
    rows=[]
    lookup={}
    for key,mu in cache.items():
        d,k,a,m=key
        rs=groups[('validation',)+key]
        av,se=mean_se([rs[s]['top1'] for s in TEST_SEEDS])
        p=predict(mu,d*k,c)
        row={'d':d,'load':k,'alpha':a,'method':m,'mean':av,'se':se,'predicted':p,
             'predicted_c1':predict(mu,d*k,1.),'abs_error':abs(av-p)}
        for metric in ('old','new','top10','mean_rank'):
            row[metric]=mean_se([rs[s][metric] for s in TEST_SEEDS])[0]
        rows.append(row)
        lookup[key]=row
    with (OUT/'method_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cells=[]
    for d in DIMS:
        for a in ALPHAS:
            for k in LOADS:
                delta=[lookup[d,k,a,m] for m in METHODS[1:]]
                j=max(range(len(delta)),key=lambda j:delta[j]['predicted'])
                chosen=delta[j];h=lookup[d,k,a,'hebb']
                diffs=[groups['validation',d,k,a,chosen['method']][s]['top1']-
                       groups['validation',d,k,a,'hebb'][s]['top1'] for s in TEST_SEEDS]
                gain,se=mean_se(diffs)
                best=max(x['mean'] for x in delta)
                near=[BETAS[t] for t,x in enumerate(delta) if best-x['mean']<=.01+1e-12]
                ratio=min(max(BETAS[j]/b,b/BETAS[j]) for b in near)
                jb=max(range(len(delta)),key=lambda j:delta[j]['mean'])
                cells.append({'d':d,'load':k,'alpha':a,'beta_pred':BETAS[j],
                    'beta_best':BETAS[jb], 'best_boundary':jb in (0,len(BETAS)-1),
                    'hebb':h['mean'],'delta':chosen['mean'],'gain':gain,'gain_se':se,
                    'lower_3se':gain-3*se,'pred_gain':chosen['predicted']-h['predicted'],
                    'gain_error':abs(gain-(chosen['predicted']-h['predicted'])),
                    'regret':best-chosen['mean'],'near_factor':ratio,
                    'argmax_factor':max(BETAS[j]/BETAS[jb],BETAS[jb]/BETAS[j])})
    with (OUT/'cell_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(cells[0]));w.writeheader();w.writerows(cells)
    maincells=[r for r in cells if r['alpha']>=1]
    gaincells=[r for r in maincells if r['d']>=128]
    absrows=[r for r in rows if r['alpha']>=1]
    pass_gain=sum(r['gain']>=.05 and r['lower_3se']>0 for r in gaincells)
    gainmean=float(np.mean([r['gain'] for r in gaincells]))
    near_rate=float(np.mean([r['near_factor']<=2+1e-12 for r in maincells]))
    age=[{'d':d,'gap':lookup[d,8,0.,'delta_0.8']['new']-lookup[d,8,0.,'delta_0.8']['old']} for d in DIMS]
    gates={'P1':pass_gain>=9 and gainmean>=.08,
           'P2':all(r['abs_error']<=.03 for r in absrows) and c not in (.5,1.5),
           'P3':all(r['gain_error']<=.03 for r in maincells),
           'P4':all(r['regret']<=.03 for r in maincells) and near_rate>=.8,
           'P5':all(r['gap']>=.15 for r in age)}
    summary={'c':c,'calibration_endpoint':c in (.5,1.5),'gates':gates,
       'H1':gates['P1'],'H2':all(gates[k] for k in ('P2','P3','P4')),
       'killer_pass':all(gates[k] for k in ('P1','P2','P3','P4')),
       'P1_passing_cells':pass_gain,'P1_total_cells':len(gaincells),'P1_mean_gain':gainmean,
       'P2_max_error':max(r['abs_error'] for r in absrows),
       'P2_fail_count':sum(r['abs_error']>.03 for r in absrows),'P2_total':len(absrows),
       'P3_max_error':max(r['gain_error'] for r in maincells),
       'P3_fail_count':sum(r['gain_error']>.03 for r in maincells),
       'P4_max_regret':max(r['regret'] for r in maincells),
       'P4_regret_fail_count':sum(r['regret']>.03 for r in maincells),
       'P4_near_factor_pass_fraction':near_rate,
       'P4_best_boundary_count':sum(r['best_boundary'] for r in maincells),
       'P5_age_gaps':age,'worst_absolute_cells':sorted(absrows,key=lambda r:r['abs_error'],reverse=True)[:10],
       'main_cells':maincells}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    meta.update({'elapsed_seconds':time.time()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    (OUT/'analysis_meta.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('main_cells','worst_absolute_cells')},indent=2))

if __name__=='__main__':main()
