import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key)=='2'
import json, math, time, resource, hashlib, sys
from pathlib import Path
import numpy as np

start=time.monotonic()
root=Path(__file__).resolve().parents[2]
out=root/'experiments/ft'
for name in ['FT_FREEZE.sha256','FT_INPUTS.sha256']:
    for line in (root/name).read_text().splitlines():
        digest,rel=line.split('  ',1)
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
rows=[json.loads(line) for line in (out/'FT_formal_rows.jsonl').read_text().splitlines()]
assert len(rows)==5000
methods=['GD','Sign','Adam_e0.1','Adam_e1','AdamW_e0.1','Muon']
expected={('baseline',0,'pre')}|{(m,t,b) for m in methods for t in [1,4] for b in ['fixed','matched']}
groups={}
for row in rows:
    config=(row['d'],row['case'],row['seed'])
    groups.setdefault(config,[]).append(row)
assert len(groups)==200
missing=[]
for config,rr in groups.items():
    assert {(r['method'],r['T'],r['mode']) for r in rr}==expected
    assert len(rr)==25
    ids=rr[0]['old']['ids']
    for row in rr:
        if 'missing' in row:
            missing.append([config,row['method'],row['T'],row['mode']]);continue
        for task in ['old','new']:
            item=row[task]
            assert item['ids']==ids
            ranks=np.array(item['ranks'])
            assert len(ranks)==64 and np.all((1<=ranks)&(ranks<=row['n']))
            assert abs(np.mean((ranks-1)/(row['n']-1))-item['rank'])<1e-14
            assert abs(np.mean(ranks<=math.ceil(.1*row['n']))-item['top10'])<1e-14
            assert item['variance']>0
        assert abs(row['q']-(rr[0]['q']+2*row['cross']+row['v']))<1e-12
        if row['mode']=='matched': assert abs(row['relative_loss']-.75)<=.002

# Reconstruct one already existing formal data set, no new statistical seed.
d=128;n=d*d;ent=[20261106,864219,d,0,0,10]
def draw(stream):
    return np.random.default_rng(np.random.SeedSequence(ent+[stream])).standard_normal((d,n))/math.sqrt(d)
x0=draw(0);y0=draw(1);x1=draw(2);y1=draw(3)
ids=np.random.default_rng(np.random.SeedSequence(ent+[4])).choice(n,64,replace=False)
s=x1@x1.T/d
z0=y0@x0.T
z1=y1@x1.T
opt=np.linalg.solve(s,z1.T).T
def err(u):
    diff=u-opt
    return np.einsum('ab,ab->',diff@s,diff)/(d*d)
base=err(z0)
checks=[]
target_rows=[r for r in groups[(128,'independent',0)] if r['method'] in ['GD','Adam_e0.1'] and r['T']==4 and r['mode']=='matched']
for row in target_rows:
    step=row['step'];sigma=math.sqrt(2)
    if row['method']=='GD':
        a=np.linalg.matrix_power(np.eye(d)-step*s,4)
        u=z0@a+opt@(np.eye(d)-a)
    else:
        # Independent implementation in original W units, with raw moments.
        w=z0/math.sqrt(n);m=np.zeros_like(w);v=np.zeros_like(w)
        for it in range(1,5):
            grad=w@s-z1/math.sqrt(n)
            m=.9*m+.1*grad;v=.99*v+.01*grad**2
            w-=step*sigma/math.sqrt(n)*(m/(1-.9**it))/(np.sqrt(v/(1-.99**it))+.1*sigma/math.sqrt(n))
        u=w*math.sqrt(n)
    max_error=abs(err(u)/base-row['relative_loss'])
    assert max_error<1e-12
    for task,x,y in [('old',x0,y0),('new',x1,y1)]:
        diag=np.einsum('ai,ai->i',y,u@x)
        totals=y.sum(axis=1)@u@x
        wm=(totals-diag)/(n-1)
        # Independently evaluate full second moment in candidate blocks.
        projected=u@x
        total2=0.
        for lo in range(0,n,256):
            # Trace over query covariance instead of invoking run.py identity.
            yy=y[:,lo:lo+256]
            temp=yy.T@u
            total2+=float(np.sum((temp@(x@x.T))*temp))
        var=(total2-float(diag@diag))/(n*(n-1))-np.mean(wm*wm)
        margin=np.mean(diag-wm)
        assert abs(var-row[task]['variance'])<1e-10
        assert abs(margin-row[task]['margin'])<1e-10
        score=y.T@u@x[:,ids]
        for j,idx in enumerate(ids):score[idx,j]=-np.inf
        ranks=1+(score>diag[ids]).sum(axis=0)
        assert ranks.tolist()==row[task]['ranks']
        checks.append(dict(method=row['method'],task=task,variance_error=float(var-row[task]['variance']),
            margin_error=float(margin-row[task]['margin']),rank_count=len(ranks)))
result=dict(passed=True,rows=len(rows),configurations=len(groups),missing=missing,replay_checks=checks,
    seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    executable=sys.executable,threads=2,freeze_unchanged=True,original_inputs_unchanged=True)
with (out/'FT_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
