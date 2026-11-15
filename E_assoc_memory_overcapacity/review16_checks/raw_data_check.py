"""Fresh Gaussian training data and literal all-candidate ranks; r=0 only.
Single-dataset coverage is a diagnostic, not a consequence of the p_C theorem.
"""
import csv
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from independent_audit import moment_data, endpoint, bands, threadpool_info

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
params=json.loads((ROOT/'experiments/x1/X1_population.json').read_text())['rows']
coeff=list(csv.DictReader((ROOT/'y3_checks/Y3_coefficients.csv').open()))
rng=np.random.default_rng(16061108); rows=[]; start=time.monotonic()
for d in (128,256,512):
    n=d*d//4; L=math.ceil(.1*n); zlo,zhi=bands(n)
    # Exactly the original iid Gaussian data definition, with n actual candidates.
    data=rng.standard_normal((3*d,n)); data/=math.sqrt(d)
    X,Y0,YE=np.split(data,3,axis=0)
    C=data@data.T; C*=d/n
    F=np.linalg.cholesky(C); lx,l0,le=np.split(F,3,axis=0)
    S=(d/n)*(X@X.T); Z0=(d/math.sqrt(n))*(Y0@X.T)
    H=(d/math.sqrt(n))*(YE@X.T)-Z0@S
    qi=rng.choice(n,256,replace=False)
    # Recover genuine Stiefel columns only to test representation identities.
    u=solve_triangular(F,data[:,qi]*math.sqrt(d/n),lower=True)
    for method in ('GD','Adam1','Sign'):
        param=next(v for v in params if v['r']==0 and v['method']==method)
        response=H if method=='GD' else np.sign(H) if method=='Sign' else H/(np.abs(H)+math.sqrt(2))
        U=Z0+param['amplitude']*response
        for task,Y,lk in [('old',Y0,l0),('new',YE,le)]:
            cf=next(v for v in coeff if float(v['r'])==0 and v['method']==method and v['task']==task)
            q=float(cf['q']); A=lk.T@U@lx; M=(A+A.T)/2; N=A.T@A
            info=moment_data(M,N,n,d,q)
            low=endpoint(M,N,n,d,q,zhi,info)[0]; high=endpoint(M,N,n,d,q,zlo,info)[0]
            lower=max(0.,low['lower']-.002); upper=min(1.,high['upper']+.002)
            successes=0; maxD=0.; maxV=0.
            for begin in range(0,len(qi),32):
                ix=qi[begin:begin+32]; uu=u[:,begin:begin+32]
                scores=Y.T@(U@X[:,ix]); scores/=math.sqrt(q)
                direct=scores[ix,np.arange(len(ix))]
                Vdirect=(np.sum(scores*scores,axis=0)-direct*direct)/(n-1)
                ranks=1+np.sum(scores>direct,axis=0)
                successes+=int(np.sum(ranks<=L))
                Du=n/d/math.sqrt(q)*np.sum(uu*(M@uu),axis=0)
                Vu=(n/d/math.sqrt(q))**2/(n-1)*(np.sum(uu*(N@uu),axis=0)-(Du*d*math.sqrt(q)/n)**2)
                maxD=max(maxD,float(np.max(np.abs(Du-direct))))
                maxV=max(maxV,float(np.max(np.abs(Vu-Vdirect))))
            obs=successes/len(qi)
            rows.append(dict(d=d,r=0,method=method,task=task,queries=len(qi),
                             lower=lower,upper=upper,observed=obs,covered=lower<=obs<=upper,
                             max_D_identity_error=maxD,max_V_identity_error=maxV))
    print(json.dumps(dict(d=d,rows=len(rows),wall_seconds=time.monotonic()-start)),flush=True)
    del X,Y0,YE,data,C,F,lx,l0,le
cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
cgp=Path('/sys/fs/cgroup')/cg.lstrip('/')
with (OUT/'raw_data_check.json').open('x') as f:
    json.dump(dict(seed=16061108,rows=rows,wall_seconds=time.monotonic()-start,
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   memory_max=(cgp/'memory.max').read_text().strip(),
                   swap_max=(cgp/'memory.swap.max').read_text().strip(),
                   threadpools=threadpool_info(),
                   note='One fresh training dataset per d; no single-dataset concentration claim'),f,indent=2)
