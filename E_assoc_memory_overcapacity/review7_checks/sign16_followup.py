"""Prespecified follow-up after the original d=16 sample variance was +3.29 SE.
Keep the original 512 datasets; report new and combined estimates separately.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
import math,json
from pathlib import Path
import numpy as np
from scipy.stats import t

out=Path(__file__).resolve().parent
entry=next(v[3:] for v in Path('/proc/self/cgroup').read_text().splitlines() if v.startswith('0::'))
cg=Path('/sys/fs/cgroup')/entry.lstrip('/')
assert (cg/'memory.max').read_text().strip()=='2147483648'
assert (cg/'memory.swap.max').read_text().strip()=='0'
threads=int(next(v.split()[1] for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('Threads:')))
assert threads<=2
rng=np.random.default_rng(761903526)
d,n,reps=16,256,8192
values=[]
for batch in range(reps//64):
    x=rng.normal(size=(64,d,n))/math.sqrt(d)
    y=rng.normal(size=(64,d,n))/math.sqrt(d)
    fitted=np.sign(y@x.transpose(0,2,1))@x
    score=np.sum(y*fitted,axis=1)
    values.extend(np.var(score,axis=1,ddof=1).tolist())
with (out/'direct_raw.jsonl').open() as f:
    old=[r['sign_samplevar'] for line in f if (r:=json.loads(line))['d']==16]
prediction=1+8/math.pi*(d-1)/(n+2)
def stat(v):
    a=np.asarray(v)
    se=float(a.std(ddof=1)/math.sqrt(len(a)))
    m=float(a.mean())
    rad=float(t.ppf(.975,len(a)-1))*se
    return dict(reps=len(a),mean=m,se=se,z=(m-prediction)/se,ci95=[m-rad,m+rad])
result=dict(seed=761903526,d=d,n=n,prediction=prediction,original=stat(old),
            followup=stat(values),combined=stat(old+values),values=values,
            reason='Investigate original +3.29 SE; follow-up fixed at 8192 before execution; original retained.',
            memory_peak=(cg/'memory.peak').read_text().strip(),process_threads=threads)
with (out/'sign16_followup.json').open('x') as f: json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='values'},indent=2))
