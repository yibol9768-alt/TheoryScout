"""Fresh-batch regression and thin-SVD Muon, fixed gain one."""
import math
import time

import numpy as np

ALGORITHMS=("sgd","oracle","muon0","muon09")
STEPS={1,2,4,8,16,32,64,128,256,512,1024}


class Task:
    def __init__(self):
        rng=np.random.default_rng(20261105)
        self.v=rng.normal(size=(128,4))
        self.v/=np.linalg.norm(self.v,axis=1,keepdims=True)
        q,_=np.linalg.qr(rng.normal(size=(128,128)))
        self.rot={n:np.linalg.qr(q[:n,:n])[0] for n in (16,32,64)}

    def problem(self,n,alpha,basis="rotated"):
        i=np.arange(1,129,dtype=float)
        lam=i[:n]**-alpha
        teacher=i[:n,None]**((alpha-2)/2)*self.v[:n]
        weights=i[n:]**-2
        cov=(self.v[n:].T*weights)@self.v[n:]
        ev,vec=np.linalg.eigh(cov)
        root=vec*np.sqrt(np.maximum(ev,0))
        U=self.rot[n] if basis=="rotated" else np.eye(n)
        return dict(n=n,alpha=alpha,lam=lam,sqrtlam=np.sqrt(lam),teacher=teacher,
                    tail=float(np.sum(weights)),tail_root=root,U=U,basis=basis)


def polar(G):
    u,s,vt=np.linalg.svd(G,full_matrices=False)
    active=s>np.finfo(float).eps*max(G.shape)*s[0] if s[0]>0 else np.zeros_like(s,dtype=bool)
    return (u*active)@vt


def updates(G,p,b,alg,momentum):
    svd_time=0.
    if alg=="sgd":
        update=G
    elif alg=="oracle":
        U=p["U"]
        update=U@((U.T@G)/np.sqrt(p["lam"][:,None]+1/b))
    else:
        mu=.9 if alg=="muon09" else 0.
        momentum=mu*momentum+(1-mu)*G
        t=time.perf_counter()
        update=polar(momentum)
        svd_time=time.perf_counter()-t
    return update,momentum,svd_time


def data(rng,p,b):
    xe=rng.normal(size=(b,p["n"]))*p["sqrtlam"]
    y=xe@p["teacher"] + rng.normal(size=(b,4))@p["tail_root"].T + rng.normal(size=(b,4))*.5
    return xe@p["U"].T,y


def risk(W,p):
    E=p["U"].T@W-p["teacher"]
    return float(np.sum(p["lam"][:,None]*E*E))+p["tail"]


def train(task,seed,n,alpha,b,alg,eta,protocol,steps,guard):
    p=task.problem(n,alpha)
    rng=np.random.default_rng([20261105,seed,n,int(alpha*2),b])
    W=np.zeros((n,4)); M=np.zeros_like(W)
    lr=eta if protocol=="fixed" else eta*(n/32)**-1
    rows=[]; sample_time=gradient_time=svd_time=update_time=0.
    started=time.perf_counter()
    for t in range(1,steps+1):
        if t%128==1:
            guard()
        tic=time.perf_counter();X,Y=data(rng,p,b)
        sample_time+=time.perf_counter()-tic
        tic=time.perf_counter();G=X.T@(X@W-Y)/b
        gradient_time+=time.perf_counter()-tic
        tic=time.perf_counter();delta,M,st=updates(G,p,b,alg,M)
        W-=lr*delta
        update_time+=time.perf_counter()-tic
        svd_time+=st
        if t in STEPS or t==steps:
            R=risk(W,p)
            if not math.isfinite(R):
                raise FloatingPointError("nonfinite training risk")
            rows.append(dict(T=t,D=b*t,nD=n*b*t,risk=R,capacity_tail=p["tail"],
                             represented_risk=R-p["tail"],wall_seconds=time.perf_counter()-started,
                             sampling_seconds=sample_time,gradient_seconds=gradient_time,
                             update_seconds=update_time,svd_seconds=svd_time))
    return dict(seed=seed,n=n,alpha=alpha,b=b,algorithm=alg,eta_calibrated=eta,
                eta_used=lr,protocol=protocol,basis="rotated",steps=steps,rows=rows,
                elapsed_seconds=time.perf_counter()-started,
                sampling_seconds=sample_time,gradient_seconds=gradient_time,
                update_seconds=update_time,svd_seconds=svd_time,finite=True)


def local(task,ctx,smoke=False):
    results=[];fits=[]
    alphas=[1.5] if smoke else [1.5,3.]
    bases=["oracle","rotated"]
    batches=[4] if smoke else [4,16,64]
    n=32
    for alpha in alphas:
        for basis in bases:
            p=task.problem(n,alpha,basis)
            E0=-p["teacher"]
            concentrated=np.zeros_like(E0)
            concentrated[-1]=np.linalg.norm(E0)*task.v[n-1]
            snapshots=[("fit",E0),("holdout",.25*E0),("concentrated",concentrated)]
            for snapshot,E in snapshots:
                for b in batches:
                    ctx.guard()
                    rng=np.random.default_rng([20261105,11,int(alpha*2),b,
                                               0 if snapshot=="fit" else 1 if snapshot=="holdout" else 2])
                    U=p["U"];Er=U@E
                    W=U@(p["teacher"]+E)
                    samples={alg:[] for alg in ("sgd","oracle","muon0")}
                    elapsed={alg:0. for alg in samples}
                    # The same G is transformed by each algorithm for paired comparison.
                    for _ in range(64):
                        X,Y=data(rng,p,b)
                        G=X.T@(X@W-Y)/b
                        for alg in samples:
                            tic=time.perf_counter()
                            delta,_,_=updates(G,p,b,alg,np.zeros_like(G))
                            elapsed[alg]+=time.perf_counter()-tic
                            samples[alg].append(U.T@delta)
                    arrays={}
                    for alg,values in samples.items():
                        v=np.asarray(values)
                        mean=v.mean(axis=0)
                        energy=np.sum(E*E,axis=1)
                        valid=energy>0
                        projected=np.sum(v*E[None,:,:],axis=2)[:,valid]/energy[valid]
                        rates=projected.mean(axis=0)
                        se=projected.std(axis=0,ddof=1)/8
                        flat=v.reshape(64,-1)
                        cov=np.cov(flat,rowvar=False,ddof=1)
                        fitted=np.zeros_like(E)
                        fitted[valid]=rates[:,None]*E[valid]
                        denom=float(np.sum(mean*mean))
                        orth=float(np.sum((mean-fitted)**2)/max(denom,1e-300))
                        cov4=cov.reshape(n,4,n,4)
                        block_energies=np.array([[np.sum(cov4[:,j,:,k]**2) for k in range(4)] for j in range(4)])
                        coupling=float((block_energies.sum()-np.trace(block_energies))/max(block_energies.sum(),1e-300))
                        arrays[alg+"_mean"]=mean
                        arrays[alg+"_covariance"]=cov
                        arrays[alg+"_rates"]=rates
                        arrays[alg+"_rate_se"]=se
                        results.append(dict(alpha=alpha,basis=basis,snapshot=snapshot,b=b,algorithm=alg,
                                            repeats=64,indices=(np.flatnonzero(valid)+1).tolist(),
                                            rates=rates.tolist(),se=se.tolist(),
                                            orthogonal_fraction=orth,noise_covariance_trace=float(np.trace(cov)),
                                            cross_column_covariance_fraction=coupling,
                                            covariance_file=f"local_a{alpha}_{basis}_{snapshot}_b{b}.npz",
                                            elapsed_seconds=elapsed[alg]))
                    np.savez_compressed(ctx.out/f"local_a{alpha}_{basis}_{snapshot}_b{b}.npz",**arrays)
                    ctx.save("local",dict(records=results,fits=fits))
            for alg in ("sgd","oracle","muon0"):
                train_rows=[r for r in results if r["alpha"]==alpha and r["basis"]==basis and
                            r["algorithm"]==alg and r["snapshot"]=="fit"]
                held=[r for r in results if r["alpha"]==alpha and r["basis"]==basis and
                      r["algorithm"]==alg and r["snapshot"]=="holdout"]
                y=np.concatenate([r["rates"] for r in train_rows])
                options=[]
                for nu in (.25,.5,1.,2.,4.,8.):
                    x=np.concatenate([p["lam"]/np.sqrt(p["lam"]+r["b"]**-nu) for r in train_rows])
                    gain=float(np.dot(x,y)/np.dot(x,x))
                    options.append((float(np.mean((gain*x-y)**2)),nu,gain))
                mse,nu,gain=min(options)
                yh=np.concatenate([r["rates"] for r in held])
                pred=np.concatenate([gain*p["lam"]/np.sqrt(p["lam"]+r["b"]**-nu) for r in held])
                seh=np.concatenate([r["se"] for r in held])
                errors=yh-pred
                fits.append(dict(alpha=alpha,basis=basis,algorithm=alg,nu=nu,gain=gain,
                                 fit_rmse=math.sqrt(mse),holdout_rmse=float(np.sqrt(np.mean(errors**2))),
                                 holdout_relative_rmse=float(np.linalg.norm(errors)/max(np.linalg.norm(yh),1e-300)),
                                 holdout_outside_descriptive_95_count=int(np.sum(np.abs(errors)>1.96*seh)),
                                 holdout_coordinate_count=len(yh),
                                 note="95% is descriptive; fit uncertainty and multiple comparisons preclude a new pass threshold"))
    out=dict(records=results,fits=fits,chi_fixed=0,chi_identifiable=False,
             momentum09_local_stationarity_tested=False)
    ctx.save("local",out)
    ctx.event("local_complete",fits=fits)
    return out


def frontiers(records):
    out=[]
    for seed in (0,1,2):
        for alpha in (1.5,3.):
            for alg in ALGORITHMS:
                for protocol in ("fixed","decay"):
                    configs=[r for r in records if (r["seed"],r["alpha"],r["algorithm"],r["protocol"])==
                             (seed,alpha,alg,protocol)]
                    points=[dict(p,n=r["n"],b=r["b"]) for r in configs for p in r["rows"]]
                    for cost in ("nD","wall_seconds"):
                        best=math.inf;pareto=[]
                        for p in sorted(points,key=lambda p:p[cost]):
                            if p["risk"]<best:
                                best=p["risk"]
                                pareto.append(p)
                        out.append(dict(seed=seed,alpha=alpha,algorithm=alg,protocol=protocol,
                                        cost=cost,rows=pareto,finite_grid_only=True))
    return out


def run(ctx,smoke=False):
    task=Task()
    if smoke:
        r=train(task,0,16,1.5,4,"muon0",.01,"fixed",32,ctx.guard)
        # Rank-deficient polar: zero singular directions must remain zero.
        G=np.zeros((16,4));G[0,0]=2.
        Q=polar(G)
        assert np.allclose(Q,G/2,atol=1e-14)
        # Pair rotated/eigenbasis updates using identical samples in local().
        diag=local(task,ctx,True)
        return dict(training=r,local_fits=diag["fits"],rank_deficient_polar_pass=True)
    started=time.monotonic()
    deadline=started+2100
    guard=lambda:ctx.guard(deadline)
    # Diagnose the oracle spectrum first, then rotation, before training.
    local(task,ctx)
    calibration=[];locked={}
    for alpha in (1.5,3.):
        for b in (4,16,64):
            for alg in ALGORITHMS:
                candidates=[]
                for eta in (.003,.01,.03,.1):
                    guard()
                    r=train(task,0,32,alpha,b,alg,eta,"fixed",256,guard)
                    candidates.append(r);calibration.append(r)
                chosen=min(candidates,key=lambda r:(r["rows"][-1]["risk"],r["eta_used"]))
                locked[(alpha,b,alg)]=chosen["eta_used"]
                ctx.save("calibration",dict(records=calibration,
                    locked=[dict(alpha=a,b=bb,algorithm=aa,eta=e) for (a,bb,aa),e in locked.items()],
                    total_samples=sum(r["b"]*r["steps"] for r in calibration),
                    total_svd_seconds=sum(r["svd_seconds"] for r in calibration)))
    records=[]
    for seed in (0,1,2):
        for n in (16,32,64):
            for alpha in (1.5,3.):
                for b in (4,16,64):
                    for alg in ALGORITHMS:
                        for protocol in ("fixed","decay"):
                            guard()
                            r=train(task,seed,n,alpha,b,alg,locked[(alpha,b,alg)],protocol,1024,guard)
                            records.append(r)
                            ctx.save("matrix",dict(records=records,expected_count=432,
                                complete=len(records)==432,elapsed_seconds=time.monotonic()-started))
                            if len(records)%24==0:
                                ctx.event("training_progress",completed=len(records),expected=432,
                                          elapsed_seconds=time.monotonic()-started)
    out=dict(records=records,expected_count=432,complete=len(records)==432,
             elapsed_seconds=time.monotonic()-started,
             total_samples=sum(r["b"]*r["steps"] for r in records),
             total_svd_seconds=sum(r["svd_seconds"] for r in records))
    ctx.save("matrix",out)
    ctx.save("matrix_frontiers",frontiers(records))
    return out
