"""Full training arrays / ranks, independent of the reviewed implementation."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert int(os.environ[key]) <= 2
import math
import time
import numpy as np
from scipy.linalg import solve
from independent_mc import OUT, coefficients, integrated_probability, resources, write_json

start = time.monotonic()
rng = np.random.default_rng(1811062091)
cfgs = coefficients()
keys = list(cfgs)
all_results = []
max_score_identity = 0.
min_j = np.inf
for d,n,repetitions,queries in [(20,257,4096,8), (64,1024,1024,8)]:
    collected = np.empty((repetitions,len(keys),3))
    for b in range(repetitions):
        X,Y0,YE = rng.normal(size=(3,d,n))/math.sqrt(d)
        S = (d/n)*(X@X.T)
        Z0 = (d/math.sqrt(n))*(Y0@X.T)
        ZE = (d/math.sqrt(n))*(YE@X.T)
        ids = rng.choice(n,queries,replace=False)
        x = X[:,ids]
        invx = solve(S,x,assume_a='pos')
        leverage = (d/n)*np.sum(x*invx,axis=0)
        threshold_normals = rng.normal(size=(queries,n-1))
        T = np.partition(threshold_normals,n-1-math.ceil(.1*n),axis=1)[:,n-1-math.ceil(.1*n)]
        T /= np.linalg.norm(threshold_normals,axis=1)/math.sqrt(n-1)
        chi = rng.chisquare(n-d-1,size=queries)
        i = 0
        for r in (0.,-.5):
            Y1 = r*Y0+math.sqrt(1-r*r)*YE
            Z1 = r*Z0+math.sqrt(1-r*r)*ZE
            H = Z1-Z0@S
            for method in ('GD','Sign','Adam.1','Adam1'):
                c = cfgs[(r,method,'old')]
                if method == 'GD':
                    delta = H
                elif method == 'Sign':
                    delta = np.sign(H)
                else:
                    eps = (.1 if method == 'Adam.1' else 1.)*math.sqrt(2*(1-r))
                    delta = H/(np.abs(H)+eps)
                U = Z0+c['alpha']*delta
                t = U@x
                tau = np.linalg.norm(t,axis=0)/math.sqrt(d)
                for task,Y,Z in (('old',Y0,Z0),('new',Y1,Z1)):
                    scores = Y.T@t
                    correct = scores[ids,np.arange(queries)]
                    ranks = 1+np.sum(scores>correct,axis=0)
                    indicator = ranks <= math.ceil(.1*n)
                    zt = Z.T@t
                    invzt = solve(S,zt,assume_a='pos')
                    lam = np.sum((Z@invx)*t,axis=0)/(math.sqrt(n)*tau)
                    P = np.sum(zt*invzt,axis=0)/(d*tau*tau)
                    # Exact sample identity (6), using the actual residual
                    # rather than generating a claimed conditional law.
                    mean_scores = X.T@invzt/math.sqrt(n)
                    residual_scores = (scores-mean_scores)/tau
                    G = residual_scores[ids,np.arange(queries)]/np.sqrt(1-leverage)
                    actual_J = np.sum(residual_scores**2,axis=0)-G*G
                    min_j = min(min_j,float(np.min(actual_J)))
                    V_direct = (np.sum(scores**2,axis=0)-correct**2)/((n-1)*tau*tau)
                    Q = lam+np.sqrt(1-leverage)*G
                    V_repr = (P+G*G+actual_J-Q*Q)/(n-1)
                    max_score_identity = max(max_score_identity,float(np.max(np.abs(V_direct-V_repr))),
                                             float(np.max(np.abs(correct/tau-Q))))
                    lo,hi = integrated_probability(lam,leverage,P,chi,T,n)
                    collected[b,i] = [indicator.mean(),lo.mean(),hi.mean()]
                    i += 1
        if (b+1)%512 == 0:
            print(dict(d=d,completed=b+1,seconds=time.monotonic()-start),flush=True)
    with (OUT/f'full_rank_d{d}.npz').open('xb') as f:
        np.savez_compressed(f,groups=collected,keys=np.array(keys,dtype=str))
    for i,(r,method,task) in enumerate(keys):
        delta = collected[:,i,0]-(collected[:,i,1]+collected[:,i,2])/2
        all_results.append(dict(d=d,n=n,repetitions=repetitions,queries=queries,
                                r=r,method=method,task=task,
                                direct=float(collected[:,i,0].mean()),
                                residual_lower=float(collected[:,i,1].mean()),
                                residual_upper=float(collected[:,i,2].mean()),
                                difference=float(delta.mean()),
                                se=float(delta.std(ddof=1)/math.sqrt(repetitions))))

# Conditional event equivalence over a broad physically feasible parameter set.
N = 400000
n = 203
h = rng.uniform(.001,.8,size=N)
lam = rng.uniform(-6,6,size=N)
P = lam*lam/h+rng.exponential(100,size=N)
J = rng.chisquare(n-10-1,size=N)
T = rng.uniform(.001,5,size=N)
G = rng.normal(size=N)
Q = lam+np.sqrt(1-h)*G
V = (P+G*G+J-Q*Q)/(n-1)
assert np.min(V)>=0
a = (n-1+T*T)*(1-h)-T*T
linear = 2*(n-1+T*T)*lam*np.sqrt(1-h)
constant = (n-1+T*T)*lam**2-T*T*(P+J)
root = (-linear+np.sqrt(linear**2-4*a*constant))/(2*a)
assert np.all(a>0)
mismatch = int(np.count_nonzero((Q>=T*np.sqrt(V)) != (G>=root)))
assert mismatch == 0 and max_score_identity < 1e-10
exceptional_bounds=[]
for hval,threshold in [(.999,2.),(.1,-1.)]:
    lower,upper=integrated_probability(np.array([0.]),np.array([hval]),np.array([1.]),
                                      np.array([100.]),threshold,203)
    assert float(lower[0])==0. and float(upper[0])==1.
    exceptional_bounds.append([float(lower[0]),float(upper[0])])
write_json('rank_checks.json',dict(results=all_results,max_score_identity=max_score_identity,
                                 min_residual_J=min_j,event_checks=N,event_mismatch=mismatch,
                                 exceptional_bounds=exceptional_bounds,
                                 wall_seconds=time.monotonic()-start,resources=resources()))
