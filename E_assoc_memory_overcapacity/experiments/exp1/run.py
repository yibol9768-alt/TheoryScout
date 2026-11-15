"""EXP1: offline NumPy CPU experiment; imports no other project code."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/exp1'
QS = (.01, .1, .5)
GRID = np.linspace(-4, 4, 33)
NORMAL = statistics.NormalDist()

def cdf(x):
    return .5 * math.erfc(-x/math.sqrt(2))

def phi(x):
    return math.exp(-x*x/2)/math.sqrt(2*math.pi)

def adam(z, gram, alpha, b1, b2, e, T):
    u = np.zeros_like(z); m = np.zeros_like(z); v = np.zeros_like(z)
    for t in range(1, T+1):
        r = u-z if gram is None else u@gram-z
        m = b1*m+(1-b1)*r; v = b2*v+(1-b2)*r*r
        u -= alpha*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+e)
    return u

def moments():
    result = []
    for b1,b2 in ((0.,0.),(.9,.999)):
        for e in (.1,1.):
            for T in (1,2,8,16):
                checks=[]
                for size in (120001,240001):
                    z=np.linspace(-9,9,size); w=np.exp(-z*z/2)/math.sqrt(2*math.pi)*(z[1]-z[0])
                    h=adam(z,None,.2,b1,b2,e,T)
                    q=float(np.sum(w*h*h)); a=float(np.sum(w*z*h))
                    checks.append((q,a,a/math.sqrt(q)))
                q,a,c=checks[-1]
                result.append(dict(b1=b1,b2=b2,e=e,T=T,alpha=.2,q=q,a=a,c=c,
                                   quadrature_difference=max(abs(x-y) for x,y in zip(*checks))))
    return result

def predictions(c,k):
    return dict(mean_rank=cdf(-c/math.sqrt(2*k)),
                **{f'top{q:g}':cdf(c/math.sqrt(k)-NORMAL.inv_cdf(1-q)) for q in QS})

class Budget:
    def __init__(self, seconds):
        self.start=time.monotonic(); self.deadline=self.start+seconds; self.max_op=0.
    def check(self):
        if time.monotonic()+max(2.,2*self.max_op)>=self.deadline:
            raise TimeoutError('wall-clock reserve reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>=1800:
            raise TimeoutError('RSS reserve: 1800 MiB (2 GiB cgroup)')
    def dot(self,a,b):
        self.check(); t=time.monotonic(); out=a@b
        self.max_op=max(self.max_op,time.monotonic()-t); return out

def retrieve(u,x,y,indices,q,budget):
    n=x.shape[1]; m=len(indices)
    vectors=budget.dot(u,x[:,indices])/math.sqrt(q)
    truth=np.sum(y[:,indices]*vectors,axis=0)
    greater=np.zeros(m,dtype=np.int64); counts=np.zeros((m,len(GRID)),dtype=np.int64)
    sums=np.zeros(m); squares=np.zeros(m)
    # Query blocks 32, candidate blocks 1024: never an n x n score matrix.
    for lo in range(0,m,32):
        hi=min(lo+32,m); ids=indices[lo:hi]; vv=vectors[:,lo:hi]
        for start in range(0,n,1024):
            stop=min(start+1024,n); scores=budget.dot(vv.T,y[:,start:stop])
            mask=(ids>=start)&(ids<stop); row=np.flatnonzero(mask); col=ids[mask]-start
            scores[row,col]=-np.inf
            greater[lo:hi]+=np.sum(scores>truth[lo:hi,None],axis=1)
            bins=np.searchsorted(GRID,scores,side='left')
            offsets=np.arange(hi-lo)[:,None]*(len(GRID)+1)
            hist=np.bincount((bins+offsets).ravel(),minlength=(hi-lo)*(len(GRID)+1)).reshape(hi-lo,-1)
            block=np.cumsum(hist,axis=1)[:,:-1]; block[row,:]-=1
            counts[lo:hi]+=block
            scores[row,col]=0
            sums[lo:hi]+=np.sum(scores,axis=1); squares[lo:hi]+=np.sum(scores*scores,axis=1)
    rank=greater/(n-1); fc=counts/(n-1)
    normcdf=np.array([cdf(float(g)) for g in GRID])
    ks_grid=np.max(np.abs(fc-normcdf),axis=1)
    result=dict(mean_rank=float(np.mean(rank)),top1=float(np.mean(greater==0)),
        true_mean=float(np.mean(truth)),true_variance=float(np.var(truth,ddof=1)),
        wrong_mean=float(np.mean(sums/(n-1))),wrong_second=float(np.mean(squares/(n-1))),
        cdf_grid=GRID.tolist(),wrong_cdf_mean=np.mean(fc,axis=0).tolist(),
        wrong_cdf_per_query=fc.tolist(),wrong_cdf_grid_sup_per_query=ks_grid.tolist(),
        wrong_cdf_grid_sup_mean=float(np.mean(ks_grid)),
        wrong_cdf_aggregate_sup=float(np.max(np.abs(np.mean(fc,axis=0)-normcdf))),
        query_indices=indices.tolist(),true_scores=truth.tolist(),ranks=(greater+1).tolist())
    result.update({f'top{q:g}':float(np.mean(greater+1<=math.ceil(q*n))) for q in QS})
    return result

def cases(z,s,phase,mom):
    yield dict(name='GD',u=z,q=1.,c=1.,T=1,alpha=1.,proxy=None)
    a,_,b=np.linalg.svd(z,full_matrices=False)
    yield dict(name='polar',u=math.sqrt(len(z))*(a@b),q=1.,c=8/(3*math.pi),T=1,alpha=None,proxy=None)
    yield dict(name='Sign1',u=.6*np.sign(z),q=.36,c=math.sqrt(2/math.pi),T=1,alpha=.6,proxy=None)
    for alpha in ((.6,) if phase=='A' else (.2,.6,1.2,1.5)):
        h1=alpha*np.sign(z); hp=h1+alpha*np.sign(z-h1)
        u=h1+alpha*np.sign(z-h1@s)
        q=8*alpha*alpha*cdf(-alpha); c=math.sqrt(2)*phi(alpha)/math.sqrt(cdf(-alpha))
        for kind,mat in (('proxy',hp),('actual',u)):
            yield dict(name=f'Sign2_a{alpha:g}_{kind}',u=mat,q=q,c=c,T=2,alpha=alpha,proxy=hp if kind=='actual' else None)
    for p in mom:
        if phase=='A' and (p['b1']!=.9 or p['e']!=.1 or p['T'] not in (2,16)):
            continue
        hp=adam(z,None,.2,p['b1'],p['b2'],p['e'],p['T'])
        u=adam(z,s,.2,p['b1'],p['b2'],p['e'],p['T'])
        for kind,mat in (('proxy',hp),('actual',u)):
            yield dict(name=f"Adam_b{p['b1']:g}_e{p['e']:g}_T{p['T']}_{kind}",u=mat,q=p['q'],c=p['c'],
                       T=p['T'],alpha=.2,b1=p['b1'],b2=p['b2'],e=p['e'],proxy=hp if kind=='actual' else None)

def validate(budget):
    rng=np.random.default_rng(99991); d=8;n=24
    x=rng.normal(size=(d,n))/math.sqrt(d);y=rng.normal(size=(d,n))/math.sqrt(d)
    z=d/math.sqrt(n)*(y@x.T); ids=np.array([0,7,23]); q=.36;u=.6*np.sign(z)
    v=u@x[:,ids]/math.sqrt(q); dense=v.T@y;truth=np.sum(v*y[:,ids],axis=0)
    for k,i in enumerate(ids): dense[k,i]=-np.inf
    brute=1+np.sum(dense>truth[:,None],axis=1)
    out=retrieve(u,x,y,ids,q,budget)
    assert out['ranks']==brute.tolist()
    for k,i in enumerate(ids):
        scores=np.delete(dense[k],i)
        target=[float(np.mean(scores<=g)) for g in GRID]
        assert np.allclose(out['wrong_cdf_per_query'][k],target,atol=1e-15)
    for scale in (.25,3.):
        assert retrieve(scale*u,x,y,ids,q*scale*scale,budget)['ranks']==brute.tolist()
    s=d/n*(x@x.T)
    for b1,b2 in ((0.,0.),(.9,.999)):
        e=.1;alpha=.2;T=8
        actual=adam(z,s,alpha,b1,b2,e,T)
        # Independent implementation in original W units, raw gradient/state.
        W=np.zeros_like(z);m=np.zeros_like(z);v=np.zeros_like(z);B=z/math.sqrt(n)
        for t in range(1,T+1):
            g=W@s-B;m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g
            W-=alpha/math.sqrt(n)*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+e/math.sqrt(n))
        assert np.allclose(W*math.sqrt(n),actual,atol=3e-14,rtol=1e-13)
    h1=.6*np.sign(z); hp=h1+.6*np.sign(z-h1)
    assert np.allclose(hp,1.2*np.sign(z)*(np.abs(z)>.6))
    return dict(dense_rank_and_cdf=True,positive_scaling=True,adam_raw_units=True,sign2_threshold=True)

def solve_n(d,lam):
    lo,hi=2,d*d
    while lo<hi:
        mid=(lo+hi)//2
        if mid*math.log(mid)<lam*d*d: lo=mid+1
        else: hi=mid
    return lo

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['moments','smoke','formal']);args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(3_500_000_000,3_500_000_000))
    if args.mode=='moments':
        data=moments();(OUT/'moments.json').write_text(json.dumps(data,indent=2)+'\n'); print(json.dumps(data));return
    frozen=(ROOT/'PREREG_EXP1.md').read_bytes(); digest=hashlib.sha256(frozen).hexdigest()
    assert digest==(ROOT/'PREREG_EXP1.sha256').read_text().split()[0]
    budget=Budget(300 if args.mode=='smoke' else 3300)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('hard alarm')))
    signal.alarm(300 if args.mode=='smoke' else 3300)
    mom=json.loads((OUT/'moments.json').read_text());checks=validate(budget)
    rows=OUT/(args.mode+'_rows.jsonl');events=OUT/(args.mode+'_events.jsonl')
    assert not rows.exists(), 'No overwriting an existing run; preserve all seeds.'
    if args.mode=='smoke': tasks=[('smoke',32,1.,0,None),('smoke',64,.5,1,None)]
    else:
        tasks=[('A',d,k,s,None) for d in (32,64,128) for k in (.5,1.,2.) for s in range(5)]
        tasks += [('B',d,k,s,None) for d in (128,256) for k in (.5,1.,2.) for s in range(5)]
        tasks += [('C',d,None,s,l) for d in (64,128) for l in (.20,.28,.34,.40,.46,.60) for s in range(5)]
    completed=0;timed_out=False
    with rows.open('w',buffering=1) as output, events.open('w',buffering=1) as log:
        for phase,d,k,seed,lam in tasks:
            if args.mode=='formal' and ((phase=='A' and time.monotonic()-budget.start>600) or
                (phase=='B' and time.monotonic()-budget.start>1800) or
                (phase=='C' and time.monotonic()-budget.start>2700)):
                log.write(json.dumps(dict(event='omitted',phase=phase,d=d,k=k,seed=seed,lam=lam,reason='preregistered phase budget'))+'\n');continue
            try:
                budget.check();t=time.monotonic()
                n=round(k*d*d) if lam is None else solve_n(d,lam)
                k=n/(d*d)
                # Fresh streams, disjoint from old smoke_v2.py. Same d,k,seed in A/B reuse data intentionally.
                entropy=[20261106,d,n,seed,1 if lam is None else 2]
                rng=np.random.default_rng(np.random.SeedSequence(entropy))
                x=rng.normal(size=(d,n));x/=math.sqrt(d)
                y=rng.normal(size=(d,n));y/=math.sqrt(d)
                s=(d/n)*budget.dot(x,x.T);z=(d/math.sqrt(n))*budget.dot(y,x.T)
                indices=rng.choice(n,size=128,replace=False)
                method_phase='A' if phase in ('A','C') else 'B'
                for case in cases(z,s,method_phase,mom):
                    budget.check()
                    if phase=='C' and case['name'] not in ('GD','Sign1','Sign2_a0.6_actual'):continue
                    u=case.pop('u'); proxy=case.pop('proxy');q=case['q'];c=case['c']
                    measured=retrieve(u,x,y,indices,q,budget)
                    extra={}
                    if proxy is not None:
                        err=u-proxy
                        extra=dict(weight_mse=float(np.mean(err*err)),relative_weight_mse=float(np.mean(err*err)/q),
                                   diagonal_mse=float(np.mean(np.sum(y[:,indices]*budget.dot(err,x[:,indices]),axis=0)**2)),
                                   normalized_diagonal_mse=float(np.mean(np.sum(y[:,indices]*budget.dot(err,x[:,indices]),axis=0)**2)/q))
                    if case['name'] in ('Sign1','Sign2_a0.6_proxy'):
                        sig=np.sqrt(np.diag(s));alpha=.6
                        gain=alpha*math.sqrt(2/math.pi)/sig if case['name']=='Sign1' else 4*alpha*np.exp(-alpha*alpha/(2*sig*sig))/(math.sqrt(2*math.pi)*sig)
                        means=d/math.sqrt(n)*np.sum(x[:,indices]**2*gain[:,None],axis=0)/math.sqrt(q)
                        extra['conditional_mean_per_query']=means.tolist()
                        extra['conditional_mean']=float(np.mean(means))
                    result=dict(phase=phase,d=d,n=n,kappa=k,seed=seed,seed_entropy=entropy,lambda_target=lam,
                                lambda_actual=n*math.log(n)/(d*d),raw_lr=None if case['alpha'] is None else case['alpha']/math.sqrt(n),
                                raw_epsilon=case.get('e',0)/math.sqrt(n),predictions=predictions(c,k),
                                score_mean_prediction=c/math.sqrt(k),**case,**measured,**extra)
                    output.write(json.dumps(result,separators=(',',':'))+'\n')
                completed+=1
                log.write(json.dumps(dict(event='completed',phase=phase,d=d,n=n,seed=seed,seconds=time.monotonic()-t,
                    elapsed=time.monotonic()-budget.start,rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))+'\n')
                print(f'{phase} d={d} n={n} seed={seed} elapsed={time.monotonic()-budget.start:.1f}s',flush=True)
                del x,y,s,z,u,proxy
            except TimeoutError as ex:
                log.write(json.dumps(dict(event='budget_stop',phase=phase,d=d,seed=seed,reason=str(ex)))+'\n');timed_out=True;break
    signal.alarm(0)
    metadata=dict(mode=args.mode,completed=completed,planned=len(tasks),budget_stop=timed_out,
        elapsed_seconds=time.monotonic()-budget.start,max_single_matmul_seconds=budget.max_op,
        max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        RLIMIT_AS=3_500_000_000,prereg_sha256=digest,checks=checks,numpy=np.__version__,
        threads={v:os.environ.get(v) for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        cgroup=Path('/proc/self/cgroup').read_text(),device='CPU')
    (OUT/(args.mode+'_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
