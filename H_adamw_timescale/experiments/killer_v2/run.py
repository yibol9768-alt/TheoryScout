"""V2 frozen experiments. Only stdlib/installed numpy and this directory imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
B, ETA, EPS = 16, .01, 1e-8
DELTAS = np.r_[0., np.geomspace(1e-6, .1, 41)]
HS = np.geomspace(1e-4, .2, 25)
START = time.monotonic()


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frozen():
    expected = (ROOT / 'PREREG_KILLER_V2.sha256').read_text().split()[0]
    assert digest(ROOT / 'PREREG_KILLER_V2.md') == expected
    return expected


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def direction(w):
    return np.sqrt(w.shape[-1])*w/np.linalg.norm(w, axis=-1, keepdims=True)


def tangent(x, q):
    return x-q*np.sum(x*q, axis=-1, keepdims=True)


def layout(problem, calibration=False):
    groups, types, ds, hs, warm, learned, amps = [], [], [], [], [], [], []
    for amp in (['fixed'] if problem == 1 else ['learned', 'oracle']):
        arms = ['adam_cold', 'proj_cold'] if calibration else [
            'adam_cold', 'proj_cold', 'adam_warm', 'proj_warm', 'angle']
        for arm in arms:
            grid = HS if arm == 'angle' else DELTAS
            kind = 2 if arm == 'angle' else int(arm.startswith('proj'))
            groups.extend([amp+'_'+arm]*len(grid))
            types.extend([kind]*len(grid)); warm.extend(['warm' in arm]*len(grid))
            learned.extend([amp == 'learned']*len(grid)); amps.extend([amp]*len(grid))
            ds.extend(np.zeros(len(grid)) if kind == 2 else grid)
            hs.extend(grid if kind == 2 else np.zeros(len(grid)))
    return dict(groups=np.array(groups), kind=np.array(types), delta=np.array(ds),
        h=np.array(hs), warm=np.array(warm), learned=np.array(learned), amplitude=np.array(amps))


def model(n, problem):
    if problem == 1:
        c = np.arange(1,n+1,dtype=float)**-1.5; c /= c.sum()
        H = np.diag(c); v = np.where(np.arange(n)%2 == 0,1.,-1.)
        eig, Q = c, np.eye(n)
        return dict(H=H, L=np.diag(np.sqrt(c)), v=v, a0=1., s2=0., eig=eig, Q=Q)
    rng = np.random.default_rng(np.random.SeedSequence([3201,n,4096]))
    c = np.arange(1,4097,dtype=float)**-1.5; c /= c.sum()
    theta = np.where(np.arange(4096)%2 == 0,1.,-1.)
    A = rng.standard_normal((n,4096))/np.sqrt(n)
    H = (A*c)@A.T; b=A@(c*theta); v=np.linalg.solve(H,b)
    s2 = 1-float(b@v)
    assert s2 >= -1e-12
    eig,Q=np.linalg.eigh(H); order=np.arange(n-1,-1,-1); eig,Q=eig[order],Q[:,order]
    return dict(H=H,L=np.linalg.cholesky(H),v=v,a0=np.linalg.norm(v)/np.sqrt(n),
        s2=max(s2,0.),eig=eig,Q=Q,A=A,c=c,theta=theta,b=b)


def samples(rng, mod, size=B):
    x=rng.standard_normal((size,len(mod['v'])))@mod['L'].T
    y=x@mod['v']+np.sqrt(.25+mod['s2'])*rng.standard_normal(size)
    return x,y


def grad(w, amp, x, y):
    u=direction(w); pred=u@x.T
    residual=amp[:,None]*pred-y
    gu=amp[:,None]*(residual@x/len(y))
    q=u/np.sqrt(w.shape[-1])
    g=tangent(gu,q)*np.sqrt(w.shape[-1])/np.linalg.norm(w,axis=-1,keepdims=True)
    ga=np.mean(residual*pred,axis=-1)
    return g,ga


def advance(w,m,v,amp,g,ga,t,lay):
    m=.9*m+.1*g; v=.95*v+.05*g*g
    raw=(m/(1-.9**t))/(np.sqrt(v/(1-.95**t))+EPS)
    q=w/np.linalg.norm(w,axis=-1,keepdims=True)
    perp=tangent(raw,q)
    applied=np.where((lay['kind'] == 0)[:,None],raw,perp)
    new=(1-lay['delta'])[:,None]*w-ETA*applied
    pn=np.linalg.norm(perp,axis=-1,keepdims=True)
    ang=direction(w)-lay['h'][:,None]*np.divide(perp,pn,out=np.zeros_like(perp),where=pn>0)
    new=np.where((lay['kind'] == 2)[:,None],direction(ang),new)
    amp=amp-.01*lay['learned']*ga
    return new,m,v,amp,raw,applied


def coeff(w,amp):
    return amp[:,None]*direction(w)


def modes(w,amp,mod):
    z=(coeff(w,amp)-mod['v'])@mod['Q']
    return z*z*mod['eig']


def popG(w,amp,mod):
    q=direction(w)/np.sqrt(w.shape[-1])
    return np.sqrt(w.shape[-1])*amp[:,None]*tangent((coeff(w,amp)-mod['v'])@mod['H'],q)


def calibrate(n,seed,problem,mod,smoke=False):
    lay=layout(problem,True); k=len(lay['kind'])
    rng=np.random.default_rng(np.random.SeedSequence([seed,n,problem,9]))
    target=np.sqrt(n)*mod['v']/np.linalg.norm(mod['v'])
    start=direction((target+.01*rng.standard_normal(n))[None,:])[0]
    w=np.tile(start,(k,1)); m=np.zeros_like(w); v=np.zeros_like(w)
    amp=np.full(k,mod['a0']); records=[]; steps=32 if smoke else 2048
    minimum=float('inf'); begin=time.monotonic()
    for t in range(1,steps+1):
        x,y=samples(rng,mod); g,ga=grad(w,amp,x,y)
        w,m,v,amp,_,_=advance(w,m,v,amp,g,ga,t,lay)
        norms=np.linalg.norm(w,axis=-1)
        minimum=min(minimum,float(norms.min()))
        assert np.isfinite(w).all() and np.isfinite(amp).all() and norms.min()>1e-12
        if t>steps-(8 if smoke else 512): records.append(norms.copy())
    records=np.array(records); radii=np.median(records,axis=0)
    # Stability is descriptive; no assertion that the median is stationary.
    half=len(records)//2
    drift=np.median(records[half:],axis=0)/np.median(records[:half],axis=0)-1
    return radii,dict(N=n,seed=seed,problem=problem,steps=steps,candidates=k,
        samples=steps*B,sample_candidate_products=steps*B*k,
        sample_parameter_products=steps*B*k*n,min_norm=minimum,
        wall_seconds=time.monotonic()-begin,median_radii=radii.tolist(),
        late_radius_relative_change=drift.tolist(),groups=lay['groups'].tolist())


def probes(w,m,v,amp,t,lay,mod,rng,count=256):
    n=w.shape[-1]; k=len(w); q=direction(w)/np.sqrt(n)
    G=popG(w,amp,mod); gn=np.linalg.norm(G,axis=-1)
    unit=np.divide(G,gn[:,None],out=np.zeros_like(G),where=gn[:,None]>0)
    sumd=np.zeros_like(w); sumsq=np.zeros(k); sigsum=np.zeros(k); sigsq=np.zeros(k)
    angles=np.zeros(k); radial=np.zeros(k); rawrad=np.zeros(k)
    for st in range(0,count,16):
        p=min(16,count-st)
        xx,yy=samples(rng,mod,p*B); xx=xx.reshape(p,B,n); yy=yy.reshape(p,B)
        u=direction(w)
        pred=np.einsum('kn,pbn->pkb',u,xx,optimize=True)
        res=amp[None,:,None]*pred-yy[:,None,:]
        gu=amp[None,:,None]*np.einsum('pkb,pbn->pkn',res,xx,optimize=True)/B
        g=tangent(gu,q)*np.sqrt(n)/np.linalg.norm(w,axis=-1,keepdims=True)
        ga=np.mean(res*pred,axis=-1)
        nw,_,_,_,raw,applied=advance(w,m,v,amp,g,ga,t+1,lay)
        nq=direction(nw)/np.sqrt(n); d=tangent(nq-q,q)
        sig=-np.sum(d*unit,axis=-1)
        sumd+=d.sum(axis=0); sumsq+=np.sum(d*d,axis=-1).sum(axis=0)
        sigsum+=sig.sum(axis=0); sigsq+=(sig*sig).sum(axis=0)
        # atan2 is more accurate than arccos for tiny rotations.
        dot=np.sum(nq*q,axis=-1)
        angles+=np.arctan2(np.linalg.norm(tangent(nq,q),axis=-1),dot).sum(axis=0)
        rawrad+=(-2*ETA*(1-lay['delta'])*np.sum(w*raw,axis=-1)).sum(axis=0)
        radial+=(-2*ETA*(1-lay['delta'])*np.sum(w*applied,axis=-1)).sum(axis=0)
    mean=sumd/count
    V=np.maximum(0.,(sumsq-count*np.sum(mean*mean,axis=-1))/(count-1))
    D=sigsum/count
    se=np.sqrt(np.maximum(0.,(sigsq-count*D*D)/(count-1))/count)
    # F undefined -> NaN in npz only, preserved and failed by bridge gate.
    F=np.divide(V,2*D,out=np.full(k,np.nan),where=(D>0)&(gn>0))
    metrics=np.stack([D,V,F,se,angles/count,rawrad/count,radial/count,gn],axis=-1)
    assert np.isfinite(metrics[:,[0,1,3,4,5,6,7]]).all() and np.isfinite(mean).all()
    return metrics,mean


def train(n,seed,problem,mode,ratio,mod,radii,smoke=False):
    lay=layout(problem); k=len(lay['kind'])
    streams=np.random.SeedSequence([seed,n,problem,mode,ratio]).spawn(4)
    data,_,shuffle,probe=[np.random.default_rng(s) for s in streams]
    # Common initial direction across online/replay, separate from calibration.
    init=np.random.default_rng(np.random.SeedSequence([seed,n,problem,0,0]).spawn(4)[1])
    w0=direction(init.standard_normal((1,n)))[0]
    w=np.tile(w0,(k,1)); cal=layout(problem,True)
    for group in dict.fromkeys(lay['groups']):
        if '_warm' in group:
            idx=np.flatnonzero(lay['groups']==group)
            ci=np.flatnonzero(cal['groups']==group.replace('_warm','_cold'))
            w[idx]*=radii[ci,None]/np.sqrt(n)
    m=np.zeros_like(w); v=np.zeros_like(w); amp=np.full(k,mod['a0'])
    initial=modes(w,amp,mod)
    checks=([64,256,1024] if problem==1 else [16,64,256,1024])
    checks=[r*n//B for r in checks]
    if smoke: checks=[64]
    if mode:
        size=32 if smoke else ratio*n
        xx,yy=samples(data,mod,size); steps=(2 if smoke else 8)*size//B
        checks=[steps]
    else: steps=checks[-1]
    risks=[]; modal=[]; diags=[]; ps=[]; mus=[]; checksteps=[]; monitor=[]
    minimum=float('inf'); begin=time.monotonic()
    for t in range(1,steps+1):
        if mode:
            j=(t-1)%(size//B)
            if j==0: order=shuffle.permutation(size)
            ii=order[j*B:(j+1)*B]; x,y=xx[ii],yy[ii]
        else: x,y=samples(data,mod)
        g,ga=grad(w,amp,x,y)
        old=w; oldamp=amp
        w,m,v,amp,raw,applied=advance(w,m,v,amp,g,ga,t,lay)
        norms=np.linalg.norm(w,axis=-1)
        minimum=min(minimum,float(norms.min()))
        if not np.isfinite(w).all() or not np.isfinite(amp).all() or norms.min()<=1e-12:
            raise FloatingPointError(f'invalid {problem=} {n=} {seed=} {t=}')
        if t%64==0 or t in checks:
            oq=direction(old)/np.sqrt(n); nq=direction(w)/np.sqrt(n)
            d=tangent(nq-oq,oq); G=popG(old,oldamp,mod); gn=np.linalg.norm(G,axis=-1)
            signal=np.divide(-np.sum(d*G,axis=-1),gn,out=np.zeros(k),where=gn>0)
            angle=np.arctan2(np.linalg.norm(tangent(nq,oq),axis=-1),np.sum(nq*oq,axis=-1))
            diagnostic=np.stack([norms,amp,gn,signal,np.maximum(0.,np.sum(d*d,axis=-1)-signal**2),
                -2*ETA*(1-lay['delta'])*np.sum(old*raw,axis=-1),
                -2*ETA*(1-lay['delta'])*np.sum(old*applied,axis=-1),angle],axis=-1)
            monitor.append((t,diagnostic))
        if t in checks:
            mo=modes(w,amp,mod); risks.append(mo.sum(axis=-1)); modal.append(mo)
            diags.append(diagnostic); checksteps.append(t)
            if not mode and (problem==1 or t>=64*n//B):
                pr,mu=probes(w,m,v,amp,t,lay,mod,probe,16 if smoke else 256)
                ps.append(pr); mus.append(mu)
    result=dict(risks=np.array(risks),modes=np.array(modal),initial_modes=initial,
        diagnostics=np.array(diags),monitor_steps=np.array([s for s,_ in monitor]),
        monitor=np.array([d for _,d in monitor]),probes=np.array(ps),probe_mu=np.array(mus),
        check_steps=np.array(checksteps),**lay)
    meta=dict(problem=problem,N=n,seed=seed,mode=mode,ratio=ratio,steps=steps,candidates=k,
        samples=steps*B,unique_samples=(size if mode else steps*B),
        sample_candidate_products=steps*B*k,sample_parameter_products=steps*B*k*n,
        probe_batches=len(ps)*(16 if smoke else 256),probe_samples=len(ps)*(16 if smoke else 256)*B,
        probe_sample_parameter_products=len(ps)*(16 if smoke else 256)*B*k*n,
        min_norm=minimum,wall_seconds=time.monotonic()-begin)
    return result,meta


def smoke():
    frozen(); rng=np.random.default_rng(2991); mod=model(16,1); lay=layout(1)
    w=rng.standard_normal((193,16)); amp=np.ones(193)
    x,y=samples(rng,mod); g,ga=grad(w,amp,x,y)
    def loss(ww,aa): return .5*np.mean((aa[:,None]*(direction(ww)@x.T)-y)**2,axis=-1)
    h=1e-5; fd=[]
    for j in range(16):
        dw=np.zeros_like(w); dw[0,j]=h
        fd.append((loss(w+dw,amp)[0]-loss(w-dw,amp)[0])/(2*h))
    ge=float(np.linalg.norm(np.array(fd)-g[0])/np.linalg.norm(g[0]))
    ae=float(abs((loss(w,amp+h)[0]-loss(w,amp-h)[0])/(2*h)-ga[0])/max(abs(ga[0]),1e-15))
    se=float(np.max(abs(loss(7.3*w,amp)-loss(w,amp)))/max(np.max(abs(loss(w,amp))),1e-15))
    te=float(np.max(abs(np.sum(w*g,axis=-1))/(np.linalg.norm(w,axis=-1)*np.linalg.norm(g,axis=-1))))
    nw,_,_,_,raw,app=advance(w,np.zeros_like(w),np.zeros_like(w),amp,g,ga,1,lay)
    ii=np.flatnonzero(lay['kind']==0)
    manual=(1-lay['delta'][ii,None])*w[ii]-ETA*g[ii]/(abs(g[ii])+EPS)
    adam=float(np.max(abs(nw[ii]-manual)))
    jj=np.flatnonzero(lay['kind']!=2)
    exact=(1-lay['delta'][jj])**2*np.sum(w[jj]**2,axis=-1)-2*ETA*(1-lay['delta'][jj])*np.sum(w[jj]*app[jj],axis=-1)+ETA**2*np.sum(app[jj]**2,axis=-1)
    identity=float(np.max(abs(np.sum(nw[jj]**2,axis=-1)-exact)/exact))
    jj=np.flatnonzero(lay['kind']==1)
    pe=float(np.max(abs(np.sum(w[jj]*app[jj],axis=-1))/(np.linalg.norm(w[jj],axis=-1)*np.linalg.norm(app[jj],axis=-1))))
    jj=np.flatnonzero(lay['kind']==2); oq=w[jj]/np.linalg.norm(w[jj],axis=-1,keepdims=True)
    nq=nw[jj]/np.linalg.norm(nw[jj],axis=-1,keepdims=True)
    angle=np.arctan2(np.linalg.norm(tangent(nq,oq),axis=-1),np.sum(nq*oq,axis=-1))
    he=float(np.max(abs(angle-np.arctan(lay['h'][jj]/4))))
    pm=model(16,2); schur=float(abs(1-float(pm['v']@pm['H']@pm['v'])-pm['s2']))
    reconstruction=float(np.max(abs(modes(w,amp,pm).sum(-1)-np.einsum('ki,ij,kj->k',coeff(w,amp)-pm['v'],pm['H'],coeff(w,amp)-pm['v'])))/np.max(modes(w,amp,pm).sum(-1)))
    assert ge<=1e-6 and ae<=1e-6 and se<=1e-12 and te<=1e-12
    assert adam<=1e-12 and identity<=1e-12 and pe<=1e-12 and he<=1e-12
    assert schur<=1e-10 and reconstruction<=1e-10
    tasks=[]
    for problem,seed in [(1,2991),(2,3991)]:
        mo=model(16,problem); rad,cm=calibrate(16,seed,problem,mo,True); tasks.append(cm)
        for mode in ([0] if problem==1 else [0,1]):
            result,tm=train(16,seed,problem,mode,2 if mode else 0,mo,rad,True)
            assert np.isfinite(result['risks']).all(); tasks.append(tm)
    output=dict(passed=True,prereg_sha256=frozen(),utc=utc(),grad_error=ge,amplitude_grad_error=ae,
        scale_error=se,tangent_error=te,adam_error=adam,norm_identity_error=identity,
        projection_error=pe,angle_error=he,schur_error=schur,modal_reconstruction_error=reconstruction,
        tasks=tasks,wall_seconds=time.monotonic()-START)
    dump(OUT/'smoke.json',output); print(json.dumps(output),flush=True)


def formal():
    prereg=frozen(); assert json.loads((OUT/'smoke.json').read_text())['passed']
    folder=OUT/'formal'; assert not folder.exists(),'Refuse overwrite'; folder.mkdir()
    meta=dict(start_utc=utc(),prereg_sha256=prereg,code_sha256=digest(__file__),complete=False,
        python=os.sys.version,numpy=np.__version__,threads={k:os.getenv(k) for k in [
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE']},
        calibration=[],tasks=[],deltas=DELTAS.tolist(),h=HS.tolist(),model_files=[])
    dump(folder/'metadata.json',meta)
    try:
        models={(1,64):model(64,1)}
        for n in [32,64,128]: models[(2,n)]=model(n,2)
        for (problem,n),mo in models.items():
            fn=f'model_p{problem}_n{n}.npz'; np.savez_compressed(folder/fn,**mo); meta['model_files'].append(fn)
        radii={}
        for problem,n,seed in [(1,64,s) for s in range(2101,2109)]+[(2,n,s) for n in [32,64,128] for s in range(3101,3109)]:
            if time.monotonic()-START>2700: raise TimeoutError('new-task deadline')
            rad,cm=calibrate(n,seed,problem,models[(problem,n)])
            radii[(problem,n,seed)]=rad; meta['calibration'].append(cm)
            dump(folder/'metadata.json',meta); print(json.dumps(dict(calibration=cm)),flush=True)
        tasks=[(1,64,s,0,0) for s in range(2101,2109)]
        tasks += [(2,n,s,0,0) for n in [32,64,128] for s in range(3101,3109)]
        tasks += [(2,n,s,1,r) for n in [32,64,128] for r in [16,64,256] for s in range(3101,3109)]
        for problem,n,seed,mode,ratio in tasks:
            if time.monotonic()-START>2700: raise TimeoutError('new-task deadline')
            result,tm=train(n,seed,problem,mode,ratio,models[(problem,n)],radii[(problem,n,seed)])
            fn=f'p{problem}_n{n}_s{seed}_m{mode}_r{ratio}.npz'
            np.savez_compressed(folder/fn,**result); tm['file']=fn; meta['tasks'].append(tm)
            meta['wall_seconds']=time.monotonic()-START; dump(folder/'metadata.json',meta)
            print(json.dumps(tm),flush=True)
        meta['complete']=True
    except Exception as e:
        meta['error']=repr(e); raise
    finally:
        meta['end_utc']=utc(); meta['wall_seconds']=time.monotonic()-START
        meta['peak_rss_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        dump(folder/'metadata.json',meta)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['smoke','formal'])
    args=parser.parse_args(); (smoke if args.mode=='smoke' else formal)()
