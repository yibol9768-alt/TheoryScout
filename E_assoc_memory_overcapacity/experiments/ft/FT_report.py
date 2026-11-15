import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):assert os.environ[k]=='2'
import json,csv,collections,time,resource
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
start=time.monotonic()
gates=json.loads((out/'FT_gates.json').read_text())
summary=json.loads((out/'FT_summary.json').read_text())
forecasts=json.loads((out/'FT_forecasts.json').read_text())['rows']
rows=[json.loads(s) for s in (out/'FT_formal_rows.jsonl').read_text().splitlines()]
failed=[g for g in gates if not g['passed']]
failure_groups=collections.Counter()
by_dimension=collections.Counter()
for g in failed:
    import ast
    key=ast.literal_eval(g['key'])
    failure_groups[(key[-2],key[-1])]+=1
    by_dimension[key[0]]+=1
def stat(v):
    a=np.array(v);return dict(mean=float(a.mean()),se=float(a.std(ddof=1)/np.sqrt(len(a))))
groups={}
baseline={}
for r in rows:
    if r['method']=='baseline':baseline[(r['d'],r['case'],r['seed'])]=r
    else:groups.setdefault((r['d'],r['case'],r['T'],r['mode'],r['method']),[]).append(r)
forget=[]
for key,rr in sorted(groups.items()):
    for metric,direction in [('msr',-1),('rank',1),('top10',-1)]:
        vals=[direction*(r['old'][metric]-baseline[(r['d'],r['case'],r['seed'])]['old'][metric]) for r in rr]
        forget.append(dict(d=key[0],case=key[1],T=key[2],mode=key[3],method=key[4],metric=metric,**stat(vals)))
with (out/'FT_forgetting.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(forget[0]));w.writeheader();w.writerows(forget)
with (out/'FT_failures.csv').open('x',newline='') as f:
    keys=['gate','key','mean','se','tolerance','bound','passed']
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(failed)
primary=[r for r in summary if r.get('d')==128 and r.get('case')=='independent']
matched=[r for r in rows if r['mode']=='matched']
data=dict(failure_by_metric={str(k):v for k,v in failure_groups.items()},failure_by_dimension=dict(by_dimension),
    failures=len(failed),point_bias_exceeds_tolerance=sum(abs(g['mean'])>g['tolerance'] for g in failed),
    max_absolute_point_bias=max(abs(g['mean']) for g in gates if g['gate']=='G2'),
    max_matching_error=max(abs(r['relative_loss']-.75) for r in matched),
    max_G1_absolute_error=max(abs(g['mean']) for g in gates if g['gate']=='G1'),
    G3=[g for g in gates if g['gate']=='G3'],G5=[g for g in gates if g['gate']=='G5'],
    primary=primary,primary_forecasts=[r for r in forecasts if r['case']=='independent'],
    primary_forgetting=[r for r in forget if r['d']==128 and r['case']=='independent'],
    count_matching=len(matched),counts={g:dict(passed=sum(v['passed'] for v in gates if v['gate']==g),total=sum(v['gate']==g for v in gates)) for g in sorted(set(v['gate'] for v in gates))},
    seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
with (out/'FT_report_numbers.json').open('x') as f:json.dump(data,f,indent=2)
print(json.dumps({k:v for k,v in data.items() if k not in ['primary','primary_forecasts','primary_forgetting']},indent=2))
for r in primary:
    print(r['method'],r['T'],r['mode'],
        'loss',r['relative_loss']['mean'],'oldmsr',r['old']['msr'],'oldrank',r['old']['rank'],
        'oldtop10',r['old']['top10'],'newmsr',r['new']['msr'])
