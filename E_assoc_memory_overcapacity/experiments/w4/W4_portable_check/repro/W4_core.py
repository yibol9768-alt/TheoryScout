"""Portable CPU numerical core. No imports from the parent project; no APIs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
import json, math, resource, sys, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import brentq
from scipy.integrate import quad
import ctypes

HERE = Path(__file__).resolve().parent
NAMES = ['GD','Sign1','Sign2','Adam16']

def threadpool_info():
    paths=sorted(set(s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
                     if 'openblas' in s and '/' in s))
    pools=[]
    for path in paths:
        lib=ctypes.CDLL(path)
        for sym in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads',
                    'openblas_get_num_threads','openblas_get_num_threads64_']:
            try: fn=getattr(lib,sym)
            except AttributeError: continue
            fn.restype=ctypes.c_int
            pools.append(dict(library=path,num_threads=fn())); break
        else: raise RuntimeError(('Cannot verify BLAS thread count',path))
    assert pools, 'No observable BLAS pool'
    return pools

def resource_info():
    line = next(s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    cg = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
    controls = {k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.peak']}
    pools = threadpool_info()
    assert controls['memory.max'] == '2147483648' and controls['memory.swap.max'] == '0', controls
    assert all(p['num_threads'] <= 2 for p in pools), pools
    return dict(controls=controls,threadpools=pools,python=sys.version,executable=sys.executable,
                numpy=np.__version__,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)

def rng(kind,d,seed,stream):
    return np.random.default_rng(np.random.SeedSequence([20261106,440004,kind,d,seed,stream]))

def mdata(d,seed,kind=1):
    n=d*d
    x=rng(kind,d,seed,0).normal(size=(d,n))/math.sqrt(d)
    y=rng(kind,d,seed,1).normal(size=(d,n))/math.sqrt(d)
    return x,y,d/n*(x@x.T),d/math.sqrt(n)*(y@x.T)

def adam(z,s):
    u=np.zeros_like(z); m=u.copy(); v=u.copy()
    for t in range(1,17):
        r=u-z if s is None else u@s-z
        m=.9*m+.1*r; v=.999*v+.001*r*r
        u-=.2*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+.1)
    return u

def cases(z,s):
    yield 'GD',z,1.,1.,z
    yield 'Sign1',.6*np.sign(z),.36,math.sqrt(2/math.pi),.6*np.sign(z)
    h=.6*np.sign(z); hp=h+.6*np.sign(z-h)
    q=8*.6**2*ndtr(-.6); c=math.sqrt(2)*math.exp(-.6**2/2)/math.sqrt(2*math.pi*ndtr(-.6))
    yield 'Sign2',h+.6*np.sign(z-h@s),q,c,hp
    p=next(r for r in json.loads((HERE/'moments.json').read_text()) if r['b1']==.9 and r['e']==.1 and r['T']==16)
    yield 'Adam16',adam(z,s),p['q'],p['c'],adam(z,None)

def diag_scores(u,x,y,q=1.):
    n=x.shape[1]; out=np.empty(n)
    for lo in range(0,n,2048):
        hi=min(n,lo+2048)
        out[lo:hi]=np.sum(y[:,lo:hi]*(u@x[:,lo:hi]),axis=0)/math.sqrt(q)
    return out

def ks_uniform(pit):
    v=np.sort(pit); n=len(v)
    return float(max(np.max(np.arange(1,n+1)/n-v),np.max(v-np.arange(n)/n)))

def retrieve(u,x,y,ids,q,c,truth_all=None):
    n=x.shape[1]
    truth=(diag_scores(u,x,y,q) if truth_all is None else truth_all)[ids]
    greater=np.zeros(len(ids),dtype=np.int64); wrongks=[]
    for lo in range(0,len(ids),32):
        ii=ids[lo:lo+32]
        scores=(u@x[:,ii]).T@y/math.sqrt(q)
        scores[np.arange(len(ii)),ii]=-np.inf
        greater[lo:lo+len(ii)]=np.sum(scores>truth[lo:lo+len(ii),None],axis=1)
        if lo==0:
            for j in range(min(8,len(ii))):
                wrongks.append(ks_uniform(ndtr(np.delete(scores[j],ii[j]))))
    rank=greater/(n-1)
    sp=ndtr(truth-c); rp=ndtr(ndtri(rank)+c)
    grid=np.arange(1,100)/100
    return dict(ids=ids.tolist(),scores=truth.tolist(),ranks=rank.tolist(),
        score_pit_cdf=np.mean(sp[:,None]<=grid,axis=0).tolist(),
        rank_pit_cdf=np.mean(rp[:,None]<=grid,axis=0).tolist(),
        score_ks=ks_uniform(sp),rank_ks=ks_uniform(rp),wrong_ks=wrongks,
        score_mean=float(truth.mean()),score_variance=float(truth.var(ddof=1)),
        variance128=float(truth[:128].var(ddof=1)),mean_rank=float(rank.mean()),
        top10=float(np.mean(greater+1<=math.ceil(.1*n))))

def full_moments(u,x,y):
    d,n=x.shape; diag=diag_scores(u,x,y)
    gx=x@x.T; gy=y@y.T
    wm=(y.sum(axis=1)@u@x-diag)/(n-1)
    total=float(np.sum((u@gx@u.T)*gy))
    variance=(total-float(diag@diag))/(n*(n-1))-float(np.mean(wm**2))
    margin=float(np.mean(diag-wm))
    return dict(msr=margin/math.sqrt(variance),variance=variance,margin=margin,
                score_mean=float(diag.mean()),score_variance=float(diag.var(ddof=1)))

def ft(d,seed,kind=2):
    x0,y0,s0,z0=mdata(d,seed,kind)
    n=d*d
    x1=rng(kind,d,seed,2).normal(size=(d,n))/math.sqrt(d)
    y1=rng(kind,d,seed,3).normal(size=(d,n))/math.sqrt(d)
    s=d/n*(x1@x1.T); z1=d/math.sqrt(n)*(y1@x1.T)
    lam,v=np.linalg.eigh(s); opt=np.linalg.solve(s,z1.T).T
    res=(z0-opt)@v; weights=np.sum(res**2,axis=0)*lam
    loss=lambda a:float(np.sum(weights*(1-a*lam)**8)/weights.sum())
    fixed=1-.75**(1/8); matched=brentq(lambda a:loss(a)-.75,0,.15,xtol=1e-13)
    k=1-math.sqrt(.75); prediction=(1-k)/math.sqrt((1-k)**2+k*k)
    rows=[]
    for mode,a in [('fixed',fixed),('matched',matched)]:
        u=z0.copy()
        for _ in range(4): u-=a*(u@s-z1)
        exact=opt+(res*(1-a*lam)**4)@v.T
        assert np.max(np.abs(exact-u))<1e-10
        rows.append(dict(kind='FT',d=d,seed=seed,mode=mode,step=a,loss=loss(a),
                         prediction=prediction,old=full_moments(u,x0,y0),new=full_moments(u,x1,y1)))
    return rows

def efficiency(e):
    if e==0: return math.sqrt(2/math.pi),1.,math.sqrt(2/math.pi)
    f=lambda z:math.sqrt(2/math.pi)*math.exp(-z*z/2)
    a=quad(lambda z:f(z)*z*z/(z+e),0,np.inf,epsabs=2e-13)[0]
    q=quad(lambda z:f(z)*z*z/(z+e)**2,0,np.inf,epsabs=2e-13)[0]
    return a,q,a/math.sqrt(q)

def ext_point(d,gamma,seed=0):
    n=d*gamma
    x=rng(9,d,seed,20+gamma).normal(size=(d,n)); y=rng(9,d,seed,30+gamma).normal(size=(d,n))
    z=(y@x.T-np.outer(y.sum(axis=1),x.sum(axis=1))/n)/math.sqrt(n)
    noise=rng(9,d,seed,40+gamma).normal(size=(d,d))
    aa,ss,bb=np.linalg.svd(z,full_matrices=False); mu=math.sqrt(d)*(aa@bb)
    a,q,c=efficiency(.1); u=z/(abs(z)+.1)/math.sqrt(q)
    rows=[]
    for tau in [0.,2.]:
        for name,mat in [('Adam_e0.1',u),('Muon',mu),('Sign',np.sign(z))]:
            mm=full_moments(mat+tau*noise,x/math.sqrt(d),y/math.sqrt(d))
            rows.append(dict(d=d,gamma=gamma,method=name,tau=tau,**mm,
                msr_scaled=mm['msr']/math.sqrt(d/gamma),variance_prediction=1+2*c*c/gamma+tau*tau))
    return rows
