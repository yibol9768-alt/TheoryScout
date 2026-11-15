"""Independent exact GD1 mean-rank reduction via score DIFFERENCE.
No sphere-tail approximation and no import of audited source.
"""
import os
assert all(os.environ.get(s)=='1' for s in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
import json
import math
import time
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import stdtr, ndtr, ndtri
from scipy.integrate import quad

START=time.monotonic()
OUT=Path(__file__).resolve().parent

def chi_nodes(df, order):
    # Normalized chi-square orthogonal-polynomial recurrence.
    j=np.arange(order,dtype=float)
    k=np.arange(1,order,dtype=float)
    nodes,vectors=eigh_tridiagonal(1+4*j/df,np.sqrt(2*k/df*(1+2*(k-1)/df)))
    return nodes,vectors[0]**2

def exact_rank(d, order):
    n=d*d
    ax,wa=chi_nodes(d,order)
    cn,wc=chi_nodes(n-2,order)
    gz,wg=np.polynomial.hermite_e.hermegauss(order)
    wg/=math.sqrt(2*math.pi)
    A=ax[:,None,None]
    C=(n-2)*cn[None,:,None]
    G=gz[None,None,:]
    L=d/n*A*A+A/n*(G*G+C)
    delta=A-np.sqrt(A/d)*G
    argument=-delta/np.sqrt(2*L-delta*delta/d)
    prob=stdtr(d,argument)
    return float(np.einsum('i,j,k,ijk',wa,wc,wg,prob))

def direct_conditional_check():
    rng=np.random.default_rng(907701)
    d,n=5,25
    X=rng.normal(size=(d,n))/math.sqrt(d)
    v=X.T@X[:,0]*d/math.sqrt(n)
    L=np.sum(v*v)/d
    delta=v[0]-v[1]
    pred=float(stdtr(d,-delta/math.sqrt(2*L-delta*delta/d)))
    observations=[]
    # Each output row uses the actual correlated candidates.
    for _ in range(200):
        Y=rng.normal(size=(1000,d,n))/math.sqrt(d)
        out=Y@v
        diff=np.sum((Y[:,:,0]-Y[:,:,1])*out,axis=-1)
        observations.extend(diff<0)
    obs=np.mean(observations)
    se=math.sqrt(obs*(1-obs)/len(observations))
    return {'d':d,'n':n,'replicates':len(observations),'prediction':pred,
            'observed':float(obs),'SE':se,'residual_SE':float((obs-pred)/se)}

rows=[]
base=float(ndtr(-1/math.sqrt(2)))
beta=math.exp(-.25)/math.sqrt(4*math.pi)*1.25
for d in (32,64,128,256,512,1024,4096):
    lo=exact_rank(d,32)
    hi=exact_rank(d,56)
    rows.append({'d':d,'exact_rank_quadrature':hi,'effective_beta':d*(hi-base),
                 'quadrature_32_56_difference':abs(hi-lo),
                 'remainder_times_d2':d*d*(hi-base-beta/d)})
result={'limiting_rank':base,'beta':beta,'rows':rows,
        'conditional_probability_check':direct_conditional_check(),
        'seconds':time.monotonic()-START,'python':sys.executable,
        'threads':len(list(Path('/proc/self/task').iterdir()))}
with (OUT/'student_rank_results.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
