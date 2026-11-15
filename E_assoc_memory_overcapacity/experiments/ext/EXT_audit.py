"""Post-run audit: no new random seed, no experiment module imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import ctypes
import numpy as np

root=Path.cwd();out=root/'experiments/ext';st=time.monotonic()
rows=[json.loads(s) for s in (out/'EXT_formal_rows.jsonl').read_text().splitlines()]
fc=json.loads((out/'EXT_forecasts.json').read_text())
frozen=json.loads((out/'EXT_frozen.json').read_text())
for name,sha in frozen.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
assert (root/'EXT_PREREG.sha256').read_text().split()[0]==hashlib.sha256((root/'EXT_PREREG.md').read_bytes()).hexdigest()
assert len(rows)==896
assert len({(r['d'],r['gamma'],r['seed']) for r in rows})==128
assert len({(r['d'],r['gamma'],r['seed'],r['method'],r['tau']) for r in rows})==896
for r in rows:
    assert r['entropy']==[20261106,730021,r['d'],r['n'],r['seed'],11]
    assert r['variance']>0
    assert abs(r['msr_scaled']-r['mean']/math.sqrt(r['variance'])/math.sqrt(r['d']/r['gamma']))<1e-12
    assert r['n']==r['d']*r['gamma']

# Replay the already observed primary noise context, including ALL its candidates.
d,gamma,seed=256,16,0;n=d*gamma
ent=[20261106,730021,d,n,seed,11]
x=np.random.default_rng(np.random.SeedSequence(ent+[0])).standard_normal((d,n))
y=np.random.default_rng(np.random.SeedSequence(ent+[1])).standard_normal((d,n))
h=np.random.default_rng(np.random.SeedSequence(ent+[2])).standard_normal((d,d))
# Center rows explicitly; use a different finite-array computation from run.py.
z=(y-y.mean(axis=1,keepdims=True))@(x-x.mean(axis=1,keepdims=True)).T/math.sqrt(n)
aa,_,bb=np.linalg.svd(z,full_matrices=False)
matrices={'GD':z,'polar':math.sqrt(d)*(aa@bb)}
for e in [0.,.01,.1,1.]:
    value=np.sign(z) if e==0 else z/(e+abs(z))
    matrices[str(e)]=value/math.sqrt(fc['moments'][str(e)]['q'])
maxerr=0.;checks=[]
for name,u in matrices.items():
    taus=[0.,1.,2.] if name in ['0.1','polar'] else [0.]
    for tau in taus:
        scores=(y.T@(u+tau*h)@x)/d
        diagonal=scores.diagonal().copy()
        # Masked complete matrix, no Gram trace or noise interpolation.
        mask=~np.eye(n,dtype=bool)
        incorrect=scores.T[mask].reshape(n,n-1)
        wrong_mean=incorrect.mean(axis=1)
        margin=float((diagonal-wrong_mean).mean())
        variance=float(incorrect.var(axis=1,ddof=0).mean())
        expected=next(r for r in rows if (r['d'],r['gamma'],r['seed'],r['method'],r['tau'])==(d,gamma,seed,name,tau))
        error=max(abs(margin-expected['mean']),abs(variance-expected['variance']))
        maxerr=max(maxerr,error)
        assert error<3e-10,(name,tau,error)
        checks.append(dict(method=name,tau=tau,mean=margin,variance=variance,max_error=error))
        del scores,mask,incorrect

meta=[json.loads((out/f'EXT_{s}_metadata.json').read_text()) for s in ['prepare','smoke','formal','analysis']]
for m in meta:
    r=m.get('resource',m.get('end_resource'))
    assert r['controls']['memory.max']=='2147483648'
    assert r['controls']['memory.swap.max']=='0'
    assert all(v['num_threads']<=2 for v in r['threadpools'])
    events=dict(line.split() for line in r['controls']['memory.events'].splitlines())
    assert int(events['oom'])==0 and int(events['oom_kill'])==0
line=next(s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
cg=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
controls={k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.peak','memory.events']}
assert controls['memory.max']=='2147483648' and controls['memory.swap.max']=='0'
pools=[]
for path in sorted(set(s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in s and '/' in s)):
    lib=ctypes.CDLL(path)
    for symbol in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads']:
        try:fn=getattr(lib,symbol)
        except AttributeError:continue
        fn.restype=ctypes.c_int;count=fn();assert count<=2
        pools.append(dict(path=path,threads=count));break
    else:raise RuntimeError(path)
budget=json.loads((out/'EXT_budget.json').read_text())
elapsed=time.time()-budget['start_unix']+budget['planning_seconds_upper']
assert elapsed<3600
summary=json.loads((out/'EXT_summary.json').read_text())
for group in summary['groups']:
    rs=[r for r in rows if (r['d'],r['gamma'],r['method'],r['tau'])==(group['d'],group['gamma'],group['method'],group['tau'])]
    assert len(rs)==16
    for metric,v in group['statistics'].items():
        values=np.array([r[metric] for r in rs])
        assert abs(values.mean()-v['mean'])<1e-12
        assert abs(values.std(ddof=1)/4-v['se'])<1e-12
result=dict(audit_seconds=time.monotonic()-st,total_elapsed_since_prepare_plus_planning_upper=elapsed,
            replay_configuration=dict(d=d,gamma=gamma,seed=seed),dense_checks=checks,
            max_dense_error=maxerr,records_verified=len(rows),groups_verified=len(summary['groups']),
            frozen_verified=True,cgroup=controls,threadpools=pools,
            rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            no_new_seeds=True)
with (out/'EXT_audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['dense_checks','threadpools']}))
