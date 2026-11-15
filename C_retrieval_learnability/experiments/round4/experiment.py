"""Round 4 CPU float64 analytic gradients. No network or external project imports."""
import os
for key,value in [('OMP_NUM_THREADS','3'),('OPENBLAS_NUM_THREADS','1'),('MKL_NUM_THREADS','1'),('NUMEXPR_NUM_THREADS','1')]:
    os.environ[key]=value
import sys, time, json, hashlib, resource
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.special import expit
import ctypes
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
_lib=ctypes.CDLL(str(OUT/'bias_solver.so'))
_ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
_lib.solve_rows.argtypes=[_ptr,_ptr,ctypes.c_int,ctypes.c_int,_ptr,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_long),ctypes.POINTER(ctypes.c_long)]
_lib.solve_rows.restype=ctypes.c_int
T=100000
MODES=('zero','b','theta','solve','b_legacy')

def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
def data(d,seed):
    pos=np.array(list(combinations(range(16),3))); A=np.zeros((560,16),bool)
    A[np.arange(560)[:,None],pos]=True
    rng=np.random.Generator(np.random.PCG64(seed))
    return A,dict(U=norm(rng.normal(size=(560,d))),V=norm(rng.normal(size=(16,d))))

def root_bias_reference(base,A,warm=None):
    Y=2.*A-1
    lo=-base.max(1)-40; hi=-base.min(1)+40
    ordered=np.partition(base, (12,13), axis=1)
    middle=-(ordered[:,12]+ordered[:,13])/2
    b=middle.copy() if warm is None else np.clip(warm,lo,hi).copy()
    if warm is not None:
        old_res=np.abs((-Y*expit(-Y*(base+b[:,None]))).sum(1))
        mid_res=np.abs((-Y*expit(-Y*(base+middle[:,None]))).sum(1))
        b=np.where(mid_res<old_res,middle,b)
    fallback=0
    for it in range(80):
        Z=base+b[:,None]
        g=(-Y*expit(-Y*Z)).sum(1)
        if np.max(np.abs(g))<=5e-11: break
        active=np.abs(g)>5e-11
        lo=np.where(active & (g<0),b,lo); hi=np.where(active & (g>0),b,hi)
        h=(expit(Z)*expit(-Z)).sum(1)
        q=b-g/np.maximum(h,1e-300)
        bad=(q<=lo)|(q>=hi)|~np.isfinite(q)
        if it>=12: bad[:]=True
        fallback+=int(np.any(active & bad))
        b=np.where(active,np.where(bad,(lo+hi)/2,q),b)
    else: raise RuntimeError('bias solver did not converge')
    return b,dict(iterations=it+1,fallbacks=fallback,residual=float(np.max(np.abs(g))))

def root_bias(base,A,warm=None):
    assert np.all(A.sum(1)==3)
    base=np.ascontiguousarray(base,dtype=np.float64)
    out=np.empty(len(base)); previous=np.zeros(len(base)) if warm is None else np.ascontiguousarray(warm)
    residual=ctypes.c_double(); iterations=ctypes.c_long(); fallbacks=ctypes.c_long()
    code=_lib.solve_rows(base,previous,int(warm is not None),len(base),out,ctypes.byref(residual),ctypes.byref(iterations),ctypes.byref(fallbacks))
    if code: raise RuntimeError(f'native bias root failed on row {code-1}')
    return out,dict(iterations=iterations.value,fallbacks=fallbacks.value,residual=residual.value)


def effective(p,mode,beta,b=None):
    S=p['U']@p['V'].T
    if mode=='theta': shift=-p['theta']
    elif mode in ('b','b_legacy'): shift=p['b']/beta
    elif mode=='solve': shift=b/beta
    else: shift=np.zeros(len(S))
    return S,shift

def obj(p,A,mode,beta,b=None,need_loss=True):
    S,shift=effective(p,mode,beta,b)
    Z=beta*(S+shift[:,None]); Y=2.*A-1
    Q=-Y*expit(-Y*Z)/A.size
    g=dict(U=beta*(Q@p['V']),V=beta*(Q.T@p['U']))
    if 'a' in p:
        # b and eliminated b held constant in partial derivative; theta held constant for theta.
        base=Z if mode=='theta' else beta*S
        g['a']=np.array([(Q*base).sum()])
    if mode in ('b','b_legacy'): g['b']=Q.sum(1)
    if mode=='theta': g['theta']=-beta*Q.sum(1)
    return (float(np.logaddexp(0,-Y*Z).mean()) if need_loss else 0.),g

class Adam:
    def __init__(self,p,rates):
        self.rates=rates; self.m={k:np.zeros_like(v) for k,v in p.items()}; self.v={k:np.zeros_like(v) for k,v in p.items()}; self.t=0
    def update(self,p,g,factor=1.):
        self.t+=1
        for k in p:
            self.m[k]*=.9; self.m[k]+=.1*g[k]
            self.v[k]*=.999; self.v[k]+=.001*g[k]**2
            step=(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
            p[k]-=factor*self.rates[k]*step
            if k in ('U','V'): p[k][:]=norm(p[k])

def calibrate(p,A,mode,beta,b):
    _,g=obj(p,A,mode,beta,b); rates={}
    for k in ('U','V'):
        direction=g[k]/(np.abs(g[k])+1e-8)
        def rms(rate): return np.sqrt(np.mean((norm(p[k]-rate*direction)-p[k])**2))
        lo,hi=0.,.02
        while rms(hi)<.001: hi*=2
        for _ in range(60):
            mid=(hi+lo)/2
            if rms(mid)<.001: lo=mid
            else: hi=mid
        rates[k]=(lo+hi)/2
    if 'a' in p: rates['a']=.1*rates['U']
    fallback=False; reference_move=None
    if mode in ('b','theta'):
        key='b' if mode=='b' else 'theta'
        ref={k:v.copy() for k,v in p.items()}; ref[key][:]=0
        _,rg=obj(ref,A,mode,beta)
        direction=rg[key]/(np.abs(rg[key])+1e-8)
        scale=beta if key=='b' else 1.
        unit=float(np.sqrt(np.mean(direction**2))/scale)
        fallback=unit<1e-14
        rates[key]=scale*rates['U'] if fallback else .001/unit
        reference_move=rates[key]*unit
    if mode=='b_legacy': rates['b']=rates['U']
    q={k:v.copy() for k,v in p.items()}; Adam(q,rates).update(q,g)
    moves={k:float(np.sqrt(np.mean((q[k]-p[k])**2))) for k in p}
    assert all(abs(moves[k]-.001)<1e-10 for k in ('U','V'))
    _,shift=effective(p,mode,beta,b); _,shift1=effective(q,mode,beta,b)
    return rates,dict(moves=moves,bias_reference_effective_move=reference_move,fallback=fallback,
                     initial_actual_bias_effective_move=float(np.sqrt(np.mean((shift1-shift)**2))))

def metrics(p,A,mode,beta,b,arrays=False):
    S,shift=effective(p,mode,beta,b)
    low=np.where(A,S,np.inf).min(1); high=np.where(~A,S,-np.inf).max(1)
    gamma=low-high; z=np.minimum(low+shift,-high-shift)
    k=3; cutoff=np.partition(S,13,axis=1)[:,13]
    above=S>cutoff[:,None]; tied=S==cutoff[:,None]
    recall=((above*A).sum(1)+(k-above.sum(1))*(tied*A).sum(1)/tied.sum(1))/k
    r=dict(rank_fraction=float((gamma>1e-10).mean()),sign_fraction=float((z>1e-10).mean()),
           all_rank=bool(gamma.min()>1e-10),all_sign=bool(z.min()>1e-10),
           all_rank_strict=bool(gamma.min()>0),all_sign_strict=bool(z.min()>0),
           robust_sign=bool(z.min()>=1e-4),min_gamma=float(gamma.min()),min_zeta=float(z.min()),
           beta=float(beta),recall=float(recall.mean()))
    if arrays: r.update(gamma=gamma.tolist(),zeta=z.tolist(),shift=shift.tolist())
    return r

def train(d,seed,mode,schedule,steps,deadline):
    start=time.time(); cpu=time.process_time(); A,p=data(d,seed)
    fingerprint=hashlib.sha256(p['U'].tobytes()+p['V'].tobytes()).hexdigest()
    beta=16.; b=None
    if mode in ('b','theta','solve'):
        b,_=root_bias(beta*(p['U']@p['V'].T),A)
        if mode=='b': p['b']=b.copy()
        if mode=='theta': p['theta']=-b/beta
    if mode=='b_legacy': p['b']=np.zeros(len(A))
    if schedule=='train': p['a']=np.array([np.log(beta)])
    rates,cal=calibrate(p,A,mode,beta,b)
    opt=Adam(p,rates); history=[]; detections=[]; status='completed'
    stats=dict(calls=0,iterations=0,fallbacks=0,max_residual=0.)
    initial=None
    for t in range(steps+1):
        beta=float(np.exp(p['a'][0])) if schedule=='train' else 16.*256.**(t/steps)
        if not 1e-8<=beta<=1e4 or not all(np.isfinite(v).all() for v in p.values()):
            status='numerical_or_temperature_failure'; break
        if t%1000==0 and time.time()>deadline: status='budget_truncated'; break
        if mode=='solve':
            b,info=root_bias(beta*(p['U']@p['V'].T),A,b)
            stats['calls']+=1; stats['iterations']+=info['iterations']; stats['fallbacks']+=info['fallbacks']; stats['max_residual']=max(stats['max_residual'],info['residual'])
        if t%1000==0 or t==steps:
            met=metrics(p,A,mode,beta,b,arrays=(t==0))
            if t==0: initial=met
            detections.append(dict(step=t,**{k:met[k] for k in ('all_rank','all_sign','robust_sign')}))
            history.append(dict(step=t,loss=obj(p,A,mode,beta,b)[0],**met))
        if t==steps: break
        loss,g=obj(p,A,mode,beta,b,need_loss=False)
        if not np.isfinite(loss) or not all(np.isfinite(v).all() for v in g.values()): status='nonfinite_gradient'; break
        opt.update(p,g,.5*(1+np.cos(np.pi*t/steps)))
    hit={}; sustained={}
    for key in ('all_rank','all_sign','robust_sign'):
        yes=[x['step'] for x in detections if x[key]]; hit[key]=min(yes) if yes else None
        sustained[key]=None
        if detections and detections[-1][key] and status=='completed':
            lastfail=max([x['step'] for x in detections if not x[key]],default=-1)
            sustained[key]=min(x['step'] for x in detections if x['step']>lastfail)
    final=metrics(p,A,mode,beta,b,True)
    result=dict(config=dict(d=d,seed=seed,mode=mode,schedule=schedule),status=status,steps=t,requested=steps,
                fingerprint=fingerprint,rates=rates,calibration=cal,initial=initial,final=final,
                first=hit,sustained=sustained,history=history,solver=stats,
                seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    weights={**p,'beta':np.array([beta])}
    if mode=='solve': weights['solved_b']=b
    return result,weights
