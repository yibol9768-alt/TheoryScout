"""Independent full-column energy checks and finite-d geometric centers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.special import ndtri

resource.setrlimit(resource.RLIMIT_CPU,(300,320))
OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(20261107005)
wall,cpu=time.monotonic(),time.process_time()
errors=[]
for alg in ('GD4','GDinf','Sign65'):
    d,n=8,90
    X=rng.standard_normal((d,n))/np.sqrt(d)
    Y=rng.standard_normal((d,n))/np.sqrt(d)
    S=d/n*(X@X.T);Z=d/np.sqrt(n)*(Y@X.T)
    C=d/n*(np.vstack((X,Y))@np.vstack((X,Y)).T)
    if alg=='GD4':
        U=np.zeros((d,d))
        for _ in range(4): U+=.5*(Z-U@S)
    elif alg=='GDinf': U=np.linalg.solve(S,Z.T).T
    else:
        U=np.zeros((d,d))
        for _ in range(65): U+=.6*np.sign(Z-U@S)
    F=np.linalg.cholesky(C)
    Q=np.sqrt(d/n)*np.linalg.solve(F,np.vstack((X,Y)))
    B=F[d:,:].T@U@F[:d,:]/np.linalg.norm(U)
    K=(B+B.T)/2;N=B.T@B
    scores=d/np.linalg.norm(U)*(Y.T@U@X)
    i=3;u=Q[:,i]
    D=n*u@K@u
    V=n*n/(n-1)*(u@N@u-(u@K@u)**2)
    directV=(np.sum(scores[:,i]**2)-scores[i,i]**2)/(n-1)
    errors.append({'alg':alg,'correct_error':float(abs(D-scores[i,i])),
                   'column_energy_error':float(abs(V-directV)),
                   'center_trace_error':float(abs(np.trace(K)-np.sum(U*Z)/(np.sqrt(n)*np.linalg.norm(U))))})
if max(max(r[k] for k in ('correct_error','column_energy_error','center_trace_error')) for r in errors)>1e-10:
    raise AssertionError(errors)

rows=[]
d,n=128,1577
for rep in range(32):
    X=rng.standard_normal((d,n))/np.sqrt(d);Y=rng.standard_normal((d,n))/np.sqrt(d)
    S=d/n*(X@X.T);Sy=d/n*(Y@Y.T);Z=d/np.sqrt(n)*(Y@X.T)
    eig,V=np.linalg.eigh(S);ZV=Z@V
    for T in (1,2,4,8,'infinity'):
        p=1/eig if T=='infinity' else (1-(1-.5*eig)**T)/eig
        U=(ZV*p)@V.T;normsq=np.sum(U*U)
        mu=np.sum(U*Z)/(np.sqrt(n)*np.sqrt(normsq))
        energy=np.trace(U@S@U.T@Sy)/normsq
        A=U@Z.T/np.sqrt(n)
        sigma2=energy+np.trace(A@A)/normsq-2*mu*mu/n
        v=np.sqrt(n/(n-1)*energy)
        rows.append({'T':T,'rep':rep,'mu':float(mu),'v':float(v),'sigma':float(np.sqrt(sigma2)),
                     'effective_center_b':float((mu-v*ndtri(1-1/(n-1)))/np.sqrt(sigma2))})
summary=[]
for T in (1,2,4,8,'infinity'):
    rs=[r for r in rows if r['T']==T]
    summary.append({'T':T,**{k:float(np.mean([r[k] for r in rs])) for k in ('mu','v','sigma','effective_center_b')}})
result={'seed':20261107005,'identity_errors':errors,'finite_geometry_d':d,'finite_geometry_n':n,
        'summary':summary,'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-wall,
        'Threads':[x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')]}
with (OUT/'AD0_geometry_results.json').open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
