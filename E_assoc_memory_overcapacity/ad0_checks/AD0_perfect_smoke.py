"""Whole-dataset retrieval smoke for the additional all-query theorem."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import resource
import time
import numpy as np

resource.setrlimit(resource.RLIMIT_CPU,(900,920))
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(20261107004)
wall,cpu=time.monotonic(),time.process_time()
design={'seed':20261107004,'d':[64,128,256],'loads':[.08,.12,.16,.2,.3],
        'reps':16,'T':[1,4,'infinity'],'eta':.5,
        'method':'all n queries, all n candidates, explicit full training arrays'}
with (OUT/'AD0_perfect_design.json').open('x') as f:
    json.dump(design,f,indent=2)
rows=[]
with (OUT/'AD0_perfect_rows.jsonl').open('x') as stream:
    for d in design['d']:
        for load in design['loads']:
            candidates=np.arange(d+2,d*d)
            n=int(candidates[np.argmin(np.abs(candidates*np.log(candidates)/d**2-load))])
            for rep in range(design['reps']):
                X=rng.standard_normal((d,n))/np.sqrt(d)
                Y=rng.standard_normal((d,n))/np.sqrt(d)
                S=d/n*(X@X.T);Z=d/np.sqrt(n)*(Y@X.T)
                eig,V=np.linalg.eigh(S);ZV=Z@V;VX=V.T@X
                for T in design['T']:
                    p=1/eig if T=='infinity' else (1-(1-.5*eig)**T)/eig
                    scores=Y.T@((ZV*p)@VX)
                    correct=np.diag(scores).copy()
                    ranks=1+np.sum(scores>correct[None,:],axis=0)
                    row={'d':d,'load':load,'n':n,'actual_load':float(n*np.log(n)/d**2),
                         'T':T,'rep':rep,'perfect':int(np.all(ranks==1)),
                         'recall':float(np.mean(ranks==1)),
                         'step_stable':bool(.5*eig[-1]<2)}
                    rows.append(row);stream.write(json.dumps(row)+'\n')
            stream.flush()
        print('perfect sweep complete',d,flush=True)
summary=[]
for d in design['d']:
    for load in design['loads']:
        for T in design['T']:
            rs=[r for r in rows if r['d']==d and r['load']==load and r['T']==T]
            vals=np.array([r['perfect'] for r in rs])
            summary.append({'d':d,'load':load,'n':rs[0]['n'],'T':T,
                            'perfect_probability':float(vals.mean()),
                            'se':float(vals.std(ddof=1)/np.sqrt(len(rs))),
                            'mean_recall':float(np.mean([r['recall'] for r in rs])),
                            'unstable_reps':sum(not r['step_stable'] for r in rs)})
result={'design':design,'summary':summary,'cpu_seconds':time.process_time()-cpu,
        'wall_seconds':time.monotonic()-wall,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'Threads':[x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')]}
with (OUT/'AD0_perfect_results.json').open('x') as f:
    json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
