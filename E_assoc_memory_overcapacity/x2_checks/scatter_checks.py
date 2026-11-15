"""Exact Wishart sampling and Haar-integrated ALL-query moments.
No n-column data arrays; sufficient statistics have the exact data distribution.
"""
import os
for s in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ[s]=='1'
import sys,time,json,math,resource,signal
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from checks import gd,ft_independent

HERE=Path(os.environ.get('X2_OUTPUT',str(Path(__file__).resolve().parent)))
start=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('2400 seconds')))
signal.alarm(2400)

def wishart(p,n,rng):
    b=np.tril(rng.standard_normal((p,p)),-1)
    b[np.diag_indices(p)]=np.sqrt(rng.chisquare(n-np.arange(p)))
    return b@b.T/n

def measure(u,sx,sy,z,n,q=1.):
    d=len(u)
    # Normalized scatter of [X,Y] equals [[sx,z.T/sqrt(n)],[z/sqrt(n),sy]].
    mean=float(np.sum(u*z)/(d*math.sqrt(n)))
    total=n*n/d**2*float(np.sum((u@sx@u.T)*sy))
    mm=u@z.T
    trace2=n/d**2*float(np.trace(mm@mm))
    trace=n*mean
    e_diag2=(trace*trace+total+trace2)/(n*(n+2))
    var=n/(n-1)*(e_diag2-mean*mean)
    wrong=(total-n*e_diag2)*(n-2)/(n*(n-1)**2)
    return dict(mean=mean/math.sqrt(q),sample_variance=var/q,wrong_variance=wrong/q,
                msr=mean/math.sqrt(wrong))

def main():
    out=(HERE/'scatter_raw.jsonl').open('x',buffering=1)
    specs=[(64,96),(128,96),(256,64),(512,32)]
    for d,reps in specs:
      n=d*d; eta0=1-.75**(1/8)
      for seed in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([20261106,220099,d,seed]))
        # Row block ordering X0,Y0,X1,Y1; four independent tasks/roles.
        J=wishart(4*d,n,rng)
        blocks=[slice(j*d,(j+1)*d) for j in range(4)]
        s0,sy0,s1,sy1=[J[b,b] for b in blocks]
        z0=math.sqrt(n)*J[blocks[1],blocks[0]]
        z1=math.sqrt(n)*J[blocks[3],blocks[2]]
        def emit(method,mode,u,q=1.,extra=None):
            out.write(json.dumps(dict(d=d,seed=seed,method=method,mode=mode,
              **measure(u,s0,sy0,z0,n,q),**(extra or {})))+'\n')
        emit('GD1','fixed',z0)
        # GD4(.3) starts from zero, separately from the GD1 endpoint above.
        u=np.zeros_like(z0)
        for _ in range(4): u+=.3*(z0-u@s0)
        emit('GD4','fixed',u,(1-.7**4)**2)
        emit('Sign1','fixed',np.sign(z0))
        emit('Adam1_e.1','fixed',z0/(abs(z0)+.1),.6774135028538578)
        vals,vec=np.linalg.eigh(s1)
        opt=np.linalg.solve(s1,z1.T).T
        res=(z0-opt)@vec
        weights=np.sum(res*res,axis=0)*vals
        for T in [1,4]:
            fixed=1-.75**(1/(2*T))
            loss=lambda a:float(weights@(1-a*vals)**(2*T)/weights.sum())
            matched=brentq(lambda a:loss(a)-.75,0,.3,xtol=1e-14)
            for mode,a in [('fixed',fixed),('matched',matched)]:
                endpoint=opt+(res*(1-a*vals)**T)@vec.T
                emit('FT'+str(T),mode,endpoint,extra=dict(eta=a,loss=loss(a)))
        if seed%16==0: print(json.dumps(dict(d=d,seed=seed,seconds=time.monotonic()-start)),flush=True)
      print(json.dumps(dict(d=d,completed=reps,seconds=time.monotonic()-start)),flush=True)
    out.close()
    rows=[json.loads(s) for s in (HERE/'scatter_raw.jsonl').read_text().splitlines()]
    summary=[]
    for d,reps in specs:
      for method,mode in sorted({(r['method'],r['mode']) for r in rows}):
        rr=[r for r in rows if r['d']==d and r['method']==method and r['mode']==mode]
        s=dict(d=d,reps=len(rr),method=method,mode=mode)
        for key in ['mean','sample_variance','wrong_variance','msr','eta','loss']:
            if key not in rr[0]: continue
            v=np.array([r[key] for r in rr]); s[key]=float(v.mean()); s[key+'_se']=float(v.std(ddof=1)/math.sqrt(len(v)))
        summary.append(s)
    with (HERE/'scatter_summary.json').open('x') as f: json.dump(summary,f,indent=2)
    cg=Path('/sys/fs/cgroup')/next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')).lstrip('/')
    with (HERE/'scatter_metadata.json').open('x') as f:
        json.dump(dict(seconds=time.monotonic()-start,python=sys.executable,
          rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          controls={s:(cg/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak']},
          process_threads=len(list(Path('/proc/self/task').iterdir()))),f,indent=2)

if __name__=='__main__': main()
