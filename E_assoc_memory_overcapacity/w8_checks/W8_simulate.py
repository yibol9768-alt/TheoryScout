"""W8: independent-seed actual Gaussian training, no APIs or project imports."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '2'
import sys
sys.dont_write_bytecode = True
import json, math, time, signal, resource, ctypes, hashlib
from pathlib import Path
import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'w8_checks'
C = math.sqrt(2/math.pi)
ETA, ALPHA = .6, .6
Q = ETA**2 + 2*ETA*ALPHA*C + ALPHA**2
MU = (ETA+ALPHA*C)/math.sqrt(Q)
COEF = -ALPHA*C/math.sqrt(Q)*ETA**2/(2*(1-ETA)**2)

def resources():
    cgline = next(s for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    cg = Path('/sys/fs/cgroup') / cgline[3:].lstrip('/')
    controls = {k:(cg/k).read_text().strip() for k in ('memory.max','memory.swap.max','memory.peak')}
    assert controls['memory.max']=='2147483648' and controls['memory.swap.max']=='0'
    pools=[]
    paths=sorted(set(s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in s and '/' in s))
    for path in paths:
        lib=ctypes.CDLL(path)
        for sym in ('scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads','openblas_get_num_threads64_'):
            try: fn=getattr(lib,sym)
            except AttributeError: continue
            fn.restype=ctypes.c_int
            pools.append({'library':path,'threads':fn()})
            break
        else: raise RuntimeError(path)
    assert pools and all(p['threads']<=2 for p in pools)
    return dict(controls=controls,pools=pools,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)

def scores(u,x,y):
    out=np.empty(x.shape[1])
    for lo in range(0,x.shape[1],2048):
        hi=min(lo+2048,x.shape[1])
        out[lo:hi]=np.sum(y[:,lo:hi]*(u@x[:,lo:hi]),axis=0)
    return out

def mean_se(v):
    v=np.asarray(v)
    return dict(mean=float(v.mean()),se=float(v.std(ddof=1)/math.sqrt(len(v))),datasets=len(v))

def main():
    start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('W8 1100 second limit')))
    signal.alarm(1100)
    before=resources()
    rows=[]
    # Fixed in source before observing results; no seed selection or expansion.
    with (OUT/'W8_raw.jsonl').open('x',buffering=1) as f:
        for d,reps in ((64,64),(128,64),(256,32)):
            n=d*d
            for seed in range(reps):
                rng=np.random.default_rng(np.random.SeedSequence([20261106,880008,d,seed]))
                x=rng.normal(size=(d,n))/math.sqrt(d)
                y=rng.normal(size=(d,n))/math.sqrt(d)
                s=d/n*(x@x.T); z=d/math.sqrt(n)*(y@x.T)
                sign=np.sign(z)
                sign1=scores(sign,x,y)
                sign2=scores(ALPHA*sign+ALPHA*np.sign(z-ALPHA*sign@s),x,y)
                from scipy.special import ndtr
                q2=8*ALPHA**2*ndtr(-ALPHA)
                sign2/=math.sqrt(q2)
                hybrid=scores(ETA*z+ALPHA*np.sign(z-ETA*z@s),x,y)/math.sqrt(Q)
                s2=s@s; s3=s2@s
                vv=np.diag(s-2*ETA*s2+ETA**2*s3)
                rb= d/math.sqrt(n*Q)*(ETA*np.trace(s)/d+ALPHA*C*np.mean(np.diag(s-ETA*s2)/np.sqrt(vv)))
                row=dict(d=d,n=n,seed=seed,sign1_mean=float(sign1.mean()),sign1_var=float(sign1.var(ddof=1)),
                         sign2_mean=float(sign2.mean()),sign2_var=float(sign2.var(ddof=1)),
                         hybrid_mean=float(hybrid.mean()),hybrid_rb_mean=float(rb))
                rows.append(row); f.write(json.dumps(row)+'\n')
                del x,y,s,z,sign,sign1,sign2,hybrid,s2,s3
            print(json.dumps(dict(d=d,completed=reps,seconds=time.monotonic()-start)),flush=True)
    summary=[]
    for d in (64,128,256):
        rr=[r for r in rows if r['d']==d]; n=d*d
        # Stable gamma ratio: the asymptotic is already < 1e-14 here.
        mn=1-1/(4*n)+1/(32*n*n)+5/(128*n**3)
        samplevar=1+4*C*C*(d-1)/(n+2)
        summary.append(dict(d=d,sign1_mean_prediction=C*mn,sign1_var_prediction=samplevar,
            sign1_var_first_order=1+4*C*C/d,hybrid_mean_limit=MU,hybrid_mean_first_order=MU+COEF/d,
            hybrid_mean_second_taylor=MU+COEF*(d+1)/n-ALPHA*C/math.sqrt(Q)*(1-3*ETA)*(1+ETA)/(4*(1-ETA)**2*n),
            **{k:mean_se([r[k] for r in rr]) for k in ('sign1_mean','sign1_var','sign2_mean','sign2_var','hybrid_mean','hybrid_rb_mean')}))
    # W4 is read only; use its 16 independent data seeds as originally reported.
    w4=[]
    with (ROOT/'experiments/w4/W4_raw.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            if r.get('kind')=='M' and r.get('method')=='Sign1': w4.append(r)
    w4summary=[]
    for d in (64,128,256):
        vals=[r['full_variance'] for r in w4 if r['d']==d]
        assert len(vals)==16
        pred=1+4*C*C*(d-1)/(d*d+2)
        ss=mean_se(vals)
        w4summary.append(dict(d=d,prediction=pred,**ss,residual_se=(ss['mean']-pred)/ss['se']))
    meta=dict(seconds=time.monotonic()-start,before=before,after=resources(),python=sys.executable,
              version=sys.version,numpy=np.__version__,hybrid=dict(eta=ETA,alpha=ALPHA,q=Q,mu=MU,coefficient=COEF),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              root_seed=[20261106,880008],summary=summary,w4_sign1=w4summary)
    with (OUT/'W8_summary.json').open('x') as f: json.dump(meta,f,indent=2)
    signal.alarm(0)
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__': main()
