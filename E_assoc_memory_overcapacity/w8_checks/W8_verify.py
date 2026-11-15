"""Independent small-n checks of exact moments; complete-dataset recall check."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
import json, math, time, signal
from pathlib import Path
import numpy as np
from scipy.special import gammaln, ndtr, ndtri
from W8_simulate import resources, scores

OUT=Path(__file__).resolve().parent
C=math.sqrt(2/math.pi)

def check(v,target):
    v=np.asarray(v); mean=float(v.mean()); se=float(v.std(ddof=1)/math.sqrt(len(v)))
    z=(mean-target)/se
    assert abs(z)<6,(mean,target,se,z)
    return dict(observed=mean,prediction=target,se=se,z=z)

def main():
    start=time.monotonic(); before=resources()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('W8 verification 300 seconds')))
    signal.alarm(300)
    d,n,reps=3,7,100000
    mn=math.sqrt(2/n)*math.exp(gammaln((n+1)/2)-gammaln(n/2))
    mu=C*d*mn/math.sqrt(n)
    var=1+6*C*C*(d-1)/(n+2)-(2*d-1)*C*C*mn*mn/n
    cov=2*C*C*(d-1)/(n+2)-(2*d-1)*C*C*mn*mn/n
    sv=1+4*C*C*(d-1)/(n+2)
    wrong=1+2*C*C*(d-1)/(n+2)
    rr=np.random.default_rng(880081)
    allvals=[]
    for _ in range(reps//1000):
        x=rr.normal(size=(1000,d,n))/math.sqrt(d)
        y=rr.normal(size=(1000,d,n))/math.sqrt(d)
        u=np.sign(y@x.transpose(0,2,1))
        ss=np.sum(y*(u@x),axis=1)
        w=np.sum(y[:,:,1]*(u@x[:,:,0:1])[:,:,0],axis=1)
        allvals.append(np.column_stack((ss[:,0],(ss[:,0]-mu)**2,(ss[:,0]-mu)*(ss[:,1]-mu),ss.var(axis=1,ddof=1),w*w)))
    vals=np.concatenate(allvals)
    exact={k:check(vals[:,i],target) for i,(k,target) in enumerate(zip(('mean','variance','covariance','sample_variance','wrong_second_moment'),(mu,var,cov,sv,wrong)))}
    # Fixed X and independent Y replications validate the actual GD -> Sign conditional mean.
    d,n,eta,alpha=5,25,.6,.6
    q=eta*eta+2*eta*alpha*C+alpha*alpha
    x=rr.normal(size=(d,n))/math.sqrt(d); s=d/n*(x@x.T)
    s2=s@s; s3=s2@s
    pred=d/math.sqrt(n*q)*(eta*np.trace(s)/d+alpha*C*np.mean(np.diag(s-eta*s2)/np.sqrt(np.diag(s-2*eta*s2+eta*eta*s3))))
    vals=[]
    for _ in range(20):
        y=rr.normal(size=(1000,d,n))/math.sqrt(d)
        z=d/math.sqrt(n)*(y@x.T)
        u=eta*z+alpha*np.sign(z-eta*z@s)
        vals.extend(np.sum(y*(u@x),axis=(1,2))/(n*math.sqrt(q)))
    hybrid=check(vals,pred)
    recalls=[]
    with (OUT/'W8_recall_raw.jsonl').open('x') as f:
        for d in (32,64,96):
            n=d*d
            for seed in range(16):
                rr=np.random.default_rng(np.random.SeedSequence([880082,d,seed]))
                x=rr.normal(size=(d,n))/math.sqrt(d); y=rr.normal(size=(d,n))/math.sqrt(d)
                u=np.sign(y@x.T)
                truth=scores(u,x,y); success=0
                for lo in range(0,n,128):
                    hi=min(n,lo+128)
                    sc=(u@x[:,lo:hi]).T@y
                    sc[np.arange(hi-lo),np.arange(lo,hi)]=-np.inf
                    cnt=np.sum(sc>truth[lo:hi,None],axis=1)
                    success+=np.count_nonzero(cnt+1<=math.ceil(.1*n))
                r=dict(d=d,seed=seed,recall=success/n)
                recalls.append(r); f.write(json.dumps(r)+'\n')
            print(json.dumps(dict(d=d,seconds=time.monotonic()-start)),flush=True)
    recall_summary=[]
    for d in (32,64,96):
        v=np.array([r['recall'] for r in recalls if r['d']==d])
        recall_summary.append(dict(d=d,datasets=len(v),theory=float(ndtr(C-ndtri(.9))),mean=float(v.mean()),dataset_sd=float(v.std(ddof=1))))
    out=dict(seconds=time.monotonic()-start,before=before,after=resources(),small_n=dict(d=3,n=7,reps=reps,checks=exact),
             hybrid_conditional_mean=hybrid,recall_summary=recall_summary,
             note='Six-SE smoke checks are diagnostic; no calibrated hypothesis tests or concentration rates.')
    with (OUT/'W8_verification.json').open('x') as f: json.dump(out,f,indent=2)
    signal.alarm(0)
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
