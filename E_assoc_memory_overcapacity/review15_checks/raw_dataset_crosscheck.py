"""Original-array Monte Carlo; independent of the conditional sampler.

Computes diagonal scores directly and all-candidate squared scores through
low-rank matrix multiplication. No asymptotic term enters either statistic.
"""
import json
import math
from pathlib import Path
import time

import numpy as np

from independent_checks import GaussianIntegration, predicted, resources, scalar_response

HERE=Path(__file__).resolve().parent
assert Path.cwd()==HERE.parent
out=HERE/'raw_datasets.json'
assert not out.exists()
rng=np.random.default_rng(1511062027)
gi=GaussianIntegration()
reps=1500
rows=[]
for d in (16,32,64,96):
    tic=time.monotonic()
    n=d*d//4
    specs=[('GD',1.),('Sign',1.),('Adam',.1),('Adam',1.),('Adam',2.)]
    totals=np.zeros((5,12)); squares=np.zeros_like(totals)
    preds=[predicted(m,0.,.6,e,.25) for m,e in specs]
    max_identity=0.
    for rep in range(reps):
        X,Y0,Y1=rng.normal(size=(3,d,n))/math.sqrt(d)
        S=d/n*(X@X.T)
        Z0=d/math.sqrt(n)*(Y0@X.T)
        Z1=d/math.sqrt(n)*(Y1@X.T)
        H=Z1-Z0@S
        C0,C1=[d/n*(Y@Y.T) for Y in (Y0,Y1)]
        L=np.linalg.cholesky(S).T
        for j,((method,e),p) in enumerate(zip(specs,preds)):
            U=Z0+.6*scalar_response(H,method,e*math.sqrt(2))
            UX=U@X
            diag0=np.sum(Y0*UX,axis=0)/math.sqrt(p['q'])
            diag1=np.sum(Y1*UX,axis=0)/math.sqrt(p['q'])
            vc=np.array([np.sum((U.T@C@U)*S)/(d*d*p['q']) for C in (C0,C1)])
            means=np.array([diag0.mean(),diag1.mean()])
            second=np.array([np.mean(diag0**2),np.mean(diag1**2)])
            wrong=(n*vc-second)/(n-1)
            condm,condv,_=gi.integrate(L,n,0.,.6,method,e,p['q'])
            val=np.r_[means,wrong,means-condm,vc-condv,second,vc]
            totals[j]+=val; squares[j]+=val*val
            if rep<2 and d<=32:
                score=Y0.T@UX/math.sqrt(p['q'])
                np.fill_diagonal(score,0.)
                direct=np.sum(score*score)/(n*(n-1))
                max_identity=max(max_identity,abs(direct-wrong[0]))
    for j,((method,e),p) in enumerate(zip(specs,preds)):
        mean=totals[j]/reps
        se=np.sqrt(np.maximum(squares[j]-totals[j]*totals[j]/reps,0)/(reps-1)/reps)
        for k in (0,1):
            rows.append(dict(d=d,n=n,method=method,e=e,task=k,reps=reps,
                             raw_mean=float(mean[k]),raw_mean_se=float(se[k]),
                             raw_wrong_square=float(mean[k+2]),raw_wrong_square_se=float(se[k+2]),
                             conditional_mean_difference=float(mean[k+4]),conditional_mean_difference_se=float(se[k+4]),
                             conditional_all_energy_difference=float(mean[k+6]),conditional_all_energy_difference_se=float(se[k+6]),
                             second_moment=float(mean[k+8]),second_moment_se=float(se[k+8]),
                             coefficient_mean=p['m'][k],coefficient_wrong=p['w'][k],
                             direct_score_identity_max_error=max_identity))
    print(json.dumps(dict(d=d,seconds=time.monotonic()-tic,max_identity=max_identity)),flush=True)
res=dict(resources=resources(),seed=1511062027,rows=rows)
with out.open('x') as f:
    json.dump(res,f,indent=2)
    f.write('\n')
