"""Exact Gaussian-residual integration; writes only exclusive-created Z2B outputs.

No fitted coefficients, no LLM APIs. Run with the prescribed systemd scope.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '2', key
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'z2b_checks'
CONFIG = dict(seed=20261106071, d=512, outer=2048, inner=16,
              correlations=[0.,-.5], task='old', rho=.1,
              delta_joint=.01, methods=['GD','Sign'])

def dump(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def thread_info():
    result=[]
    paths={s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
           if 'openblas' in s.lower() and s.split()[-1].startswith('/')}
    for p in sorted(paths):
        lib=ctypes.CDLL(p)
        for s in ('scipy_openblas_get_num_threads64_','openblas_get_num_threads64_',
                  'openblas_get_num_threads','scipy_openblas_get_num_threads_'):
            try:
                f=getattr(lib,s);f.restype=ctypes.c_int
                count=f();assert count<=2
                result.append(dict(path=p,threads=count));break
            except AttributeError:
                pass
    return result

def root_probability(lam, h, projected, J, threshold, n):
    """Exact on the usual branch; globally valid [lower,upper] otherwise."""
    m=n-1
    K=m+threshold**2
    H=K*(1-h)-threshold**2
    valid=(H>0)&(threshold>0)&(h<1)&(projected>=0)&(J>=0)
    safe=np.where(valid,H,1.)
    root=(-K*np.sqrt(np.maximum(0,1-h))*lam+
          threshold*np.sqrt(np.maximum(0,K*lam**2+safe*(projected+J))))/safe
    value=ndtr(-root)
    return np.where(valid,value,0.),np.where(valid,value,1.),int(np.sum(~valid))

def sphere_order(rng,n,inner,rho):
    m=n-1;L=math.ceil(rho*n)
    g=rng.normal(size=(inner,m))
    radius=np.sqrt(np.mean(g*g,axis=1))
    g.partition(m-L,axis=1)
    return g[:,m-L]/radius

def sample_base(rng,d,n,inner):
    L=np.tril(rng.normal(size=(d,d)),-1)
    L[np.diag_indices(d)]=np.sqrt(rng.chisquare(n-np.arange(d)))
    L/=math.sqrt(n)
    S=L@L.T
    G0=rng.normal(size=(d,d));GE=rng.normal(size=(d,d))
    Z0=G0@L.T;ZE=GE@L.T
    g=rng.normal(size=(d,inner))
    W=np.sum(g*g,axis=0)+rng.chisquare(n-d,size=inner)
    v=math.sqrt(n/d)*g/np.sqrt(W)
    h=d/n*np.sum(v*v,axis=0)
    return L,S,G0,GE,Z0,ZE,v,h

def exact_row(L,S,G0,GE,Z0,ZE,v,h,r,coeff,J,T,n):
    Z1=r*Z0+math.sqrt(1-r*r)*ZE
    H=Z1-Z0@S
    b=G0@v
    values={};invalid=0
    for method in ('GD','Sign'):
        c=coeff[(r,method)]
        U=Z0+float(c['amplitude'])*(H if method=='GD' else np.sign(H))
        t=(U@L)@v
        tnorm=np.sum(t*t,axis=0)
        lam=math.sqrt(len(S)/n)*np.sum(b*t,axis=0)/np.sqrt(tnorm)
        projected=np.sum((G0.T@t)**2,axis=0)/tnorm
        lo,hi,bad=root_probability(lam,h,projected,J,T,n)
        values[method]=(float(np.mean(lo)),float(np.mean(hi)))
        invalid+=bad
    return values['GD'][0]-values['Sign'][1],values['GD'][1]-values['Sign'][0],invalid

def conditional_checks():
    """Check the exact root against direct Gaussian residual simulation."""
    rng=np.random.default_rng(20261106070)
    cases=[]
    for lam,h,projected,z in [(1.3,.1,25.,1.28),(-.4,.2,12.,1.5),(2.,.05,90.,.8)]:
        n=100;J=rng.chisquare(n-5-1,size=300000)
        lo,hi,bad=root_probability(lam,h,projected,J,z,n)
        G=rng.normal(size=len(J));D=lam+math.sqrt(1-h)*G
        V=(projected+G*G+J-D*D)/(n-1)
        assert np.min(V)>=0 and bad==0
        direct=D>=z*np.sqrt(V)
        err=float(np.mean(direct-lo))
        se=float(np.std(direct-lo,ddof=1)/math.sqrt(len(J)))
        assert abs(err)<6*se
        cases.append(dict(lam=lam,h=h,projected=projected,threshold=z,
                          root_mean=float(np.mean(lo)),direct_mean=float(np.mean(direct)),
                          difference=err,standard_error=se))
    # Check residual representation against one complete training matrix.
    d,n=6,91
    X=rng.normal(size=(d,n))/math.sqrt(d)
    Y0=rng.normal(size=(d,n))/math.sqrt(d)
    YE=rng.normal(size=(d,n))/math.sqrt(d)
    S=d/n*X@X.T;Z0=d/math.sqrt(n)*Y0@X.T
    ZE=d/math.sqrt(n)*YE@X.T
    U=Z0+.7*np.sign(ZE-Z0@S)
    P=X.T@np.linalg.solve(X@X.T,X)
    residual=Y0-Z0@np.linalg.solve(S,X)/math.sqrt(n)
    projection_error=float(np.max(np.abs(residual@X.T)))
    assert projection_error<1e-12
    # Conditional score mean, residual norm and leverage identities.
    i=3;t=U@X[:,i];tau=np.linalg.norm(t)/math.sqrt(d)
    mean=X.T@np.linalg.solve(S,Z0.T@t)/math.sqrt(n)
    projected=float(np.linalg.norm(mean)**2/tau**2)
    lam=float(mean[i]/tau);h=float(P[i,i])
    assert lam*lam<=h*projected+1e-12
    cases.append(dict(projection_identity_error=projection_error,
                      cauchy_slack=h*projected-lam*lam))
    return cases

def main():
    start=time.monotonic()
    files=['Z1_G1_PROOF.md','Z2_RECALL_BRIDGE.md','Y3_FINITE_D_NONLINEAR.md',
           'paper/appendix/M_nonlinear_finite_d.tex','y3_checks/Y3_coefficients.csv']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in files}
    dump('Z2B_design.json',dict(**CONFIG,input_hashes=hashes))
    dump('Z2B_conditional_checks.json',conditional_checks())
    coeff={}
    for c in csv.DictReader((ROOT/'y3_checks/Y3_coefficients.csv').open()):
        if c['task']=='old':coeff[(float(c['r']),c['method'])]=c
    rng=np.random.default_rng(CONFIG['seed'])
    d=CONFIG['d'];n=d*d//4;inner=CONFIG['inner']
    rows=[]
    for rep in range(CONFIG['outer']):
        base=sample_base(rng,d,n,inner)
        J=rng.chisquare(n-d-1,size=inner)
        T=sphere_order(rng,n,inner,CONFIG['rho'])
        for r in CONFIG['correlations']:
            lo,hi,bad=exact_row(*base,r,coeff,J,T,n)
            rows.append(dict(rep=rep,r=r,lower=lo,upper=hi,invalid=bad))
        if (rep+1)%64==0:
            print(json.dumps(dict(completed=rep+1,outer=CONFIG['outer'],
                                  wall_seconds=time.monotonic()-start)),flush=True)
    with (OUT/'Z2B_outer_rows.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    results=[]
    # Four one-sided confidence bounds: 2 correlations x lower/upper.
    failure=CONFIG['delta_joint']/(2*len(CONFIG['correlations']))
    logterm=math.log(2/failure)
    B=CONFIG['outer']
    for r in CONFIG['correlations']:
        sub=[s for s in rows if s['r']==r]
        lo=np.array([s['lower'] for s in sub]);hi=np.array([s['upper'] for s in sub])
        radius_lo=math.sqrt(2*np.var(lo,ddof=1)*logterm/B)+14*logterm/(3*(B-1))
        radius_hi=math.sqrt(2*np.var(hi,ddof=1)*logterm/B)+14*logterm/(3*(B-1))
        limit=float(coeff[(r,'GD')]['recall_limit'])-float(coeff[(r,'Sign')]['recall_limit'])
        correction=(float(coeff[(r,'GD')]['B_recall'])-float(coeff[(r,'Sign')]['B_recall']))/d
        results.append(dict(r=r,mean_lower=float(lo.mean()),mean_upper=float(hi.mean()),
                            sd_lower=float(lo.std(ddof=1)),radius_lower=radius_lo,
                            radius_upper=radius_hi,interval_lower=float(lo.mean()-radius_lo),
                            interval_upper=float(hi.mean()+radius_hi),
                            interval_width=float(hi.mean()-lo.mean()+radius_lo+radius_hi),
                            total_invalid=sum(s['invalid'] for s in sub),
                            limit=limit,BR_prediction=limit+correction))
    assert hashes=={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in files}
    cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
    dump('Z2B_outer_result.json',dict(**CONFIG,results=results,
         wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),
         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         memory_max=(cp/'memory.max').read_text().strip(),
         memory_swap_max=(cp/'memory.swap.max').read_text().strip(),threadpools=thread_info(),
         input_hashes_unchanged=True,
         arithmetic='ordinary float64, not an outward-rounded numerical certificate'))

if __name__=='__main__':main()
