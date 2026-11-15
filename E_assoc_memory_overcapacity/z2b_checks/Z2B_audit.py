"""Independent full-data rank check and symbolic coefficient audit."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(k)=='2'
import json
import math
from pathlib import Path
import numpy as np
import sympy as sp
from Z2B_residual import root_probability,sphere_order

OUT=Path(__file__).resolve().parent
rng=np.random.default_rng(20261106072)
samples=12000;d=12;n=144;rho=.1
direct=[];smooth=[];invalid=0;max_identity=0.
for rep in range(samples):
    X=rng.normal(size=(d,n))/math.sqrt(d)
    Y=rng.normal(size=(d,n))/math.sqrt(d)
    YE=rng.normal(size=(d,n))/math.sqrt(d)
    S=d/n*X@X.T;Z=d/math.sqrt(n)*Y@X.T;ZE=d/math.sqrt(n)*YE@X.T
    U=Z+.9*np.sign(ZE-Z@S)
    x=X[:,0];t=U@x;norm=np.linalg.norm(t)
    zt=Z.T@t
    h=float(d/n*x@np.linalg.solve(S,x))
    lam=float(math.sqrt(d/n)*(Z@np.linalg.solve(S,x))@t/norm)
    projected=float(zt@np.linalg.solve(S,zt)/(norm*norm))
    max_identity=max(max_identity,lam*lam-h*projected)
    scores=Y.T@t
    direct.append(1+np.count_nonzero(scores[1:]>scores[0])<=math.ceil(rho*n))
    T=sphere_order(rng,n,1,rho)
    J=rng.chisquare(n-d-1,size=1)
    lo,hi,bad=root_probability(lam,h,projected,J,T,n)
    invalid+=bad
    smooth.append(float(lo[0]))
direct=np.asarray(direct,dtype=float);smooth=np.asarray(smooth)
diff=direct-smooth
se=float(diff.std(ddof=1)/math.sqrt(samples))
assert invalid==0 and abs(diff.mean())<5*se

m,mu,ell,kappa,z=sp.symbols('m mu ell kappa z',nonzero=True,real=True)
t=z-mu
from_residual=m-mu*ell/2-mu/2+t/(2*kappa)-t/(2*kappa)-z*mu**2/2
candidate=m-mu*(ell+mu**2)/2-mu/2-mu**2*t/2
assert sp.simplify(from_residual-candidate)==0
lambda_variance=1/kappa+3*mu**2+mu**2-4*mu**2
assert sp.simplify(lambda_variance-1/kappa)==0
result=dict(samples=samples,d=d,n=n,method='Sign',amplitude=.9,r=0,
            seed=20261106072,direct_full_rank_mean=float(direct.mean()),
            exact_residual_mean=float(smooth.mean()),paired_difference=float(diff.mean()),
            paired_standard_error=se,invalid=invalid,
            max_positive_cauchy_violation=max_identity,
            BR_symbolic_difference=str(sp.simplify(from_residual-candidate)),
            lambda_variance_symbolic=str(sp.simplify(lambda_variance)),
            note='Numerical identity check, not a proof of the asymptotic remainder.')
with (OUT/'Z2B_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,ensure_ascii=False))
