import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key)=='2', (key,os.environ.get(key))
import json, math, time, sys, hashlib, resource
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import t as student, chi2, norm
from threadpoolctl import threadpool_info

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/x1'
METHODS=['GD','Adam2','Adam1','Adam.1','Sign']
PAIRS=[('GD','Sign'),('Adam2','Adam.1'),('Adam.1','Sign')]
RS=[.5,0.,-.5]
TARGET=float(ndtr(1.36+ndtri(.1)))
START=time.monotonic()

def dump(path,obj):
    with Path(path).open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def frozen():
    for line in (ROOT/'X1_PREREG.sha256').read_text().splitlines():
        h,p=line.split('  ',1)
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p

def meta(mode,extra=None):
    cg=Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
    cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
    return dict(mode=mode,elapsed=time.monotonic()-START,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        python=sys.executable,numpy=np.__version__,threads=threadpool_info(),
        cgroup=cg,memory_max=(cp/'memory.max').read_text().strip(),
        memory_swap_max=(cp/'memory.swap.max').read_text().strip(),**(extra or {}))

def population():
    rows=[]
    for ir,r in enumerate(RS):
        sig=math.sqrt(2*(1-r))
        for method in METHODS:
            if method=='GD': a=q=1.
            elif method=='Sign': a=math.sqrt(2/math.pi); q=1.
            else:
                e={'Adam2':2.,'Adam1':1.,'Adam.1':.1}[method]
                def weight(z): return math.sqrt(2/math.pi)*math.exp(-z*z/2)
                a=quad(lambda z:weight(z)*z*z/(z+e),0,np.inf,epsabs=1e-12,epsrel=1e-12)[0]
                q=quad(lambda z:weight(z)*(z/(z+e))**2,0,np.inf,epsabs=1e-12,epsrel=1e-12)[0]
            c=a/math.sqrt(q); k=sig*c/2; B=(1+r)*k/math.sqrt(1-k*k); M2=r*r+B*B
            sup=math.sqrt(M2) if r+k*k>0 else k
            assert r<.68<sup
            c0=(r*.68+B*math.sqrt(M2-.68**2))/M2
            theta=math.acos(c0)
            mag=math.sin(theta)/(math.sqrt(1-k*k)*c0+k*math.sin(theta))
            def c1(t): return (r+k*t)/math.sqrt(1-2*k*t+t*t)
            hi=k*(1+r)/(r+k*k) if r+k*k>0 else max(1.,mag*2)
            check=brentq(lambda t:c1(t)-.68,0,hi,xtol=1e-13)
            assert abs(check-mag)<1e-10 and abs(c1(mag)-.68)<1e-12
            amp=mag/sig if method=='GD' else mag/math.sqrt(q)
            rows.append(dict(ir=ir,r=r,method=method,A=a,Q=q,c=c,k=k,c0=c0,t=mag,
                amplitude=amp,old_recall=float(ndtr(2*c0+ndtri(.1))),new_recall=TARGET,
                first_branch_t_check=check,sign_sup=sup))
    dump(OUT/'X1_population.json',dict(target=TARGET,rows=rows,meta=meta('population')))

def forecast(ir,method):
    return next(p for p in POP if p['ir']==ir and p['method']==method)

def dataset(d,ir,seed,phase):
    n=d*d//4
    def rng(s): return np.random.default_rng(np.random.SeedSequence([20261106,913752,phase,d,ir,seed,s]))
    x=rng(0).standard_normal((d,n))/math.sqrt(d)
    y0=rng(1).standard_normal((d,n))/math.sqrt(d)
    r=RS[ir]
    y1=r*y0+math.sqrt(1-r*r)*rng(2).standard_normal((d,n))/math.sqrt(d)
    ids=rng(3).choice(n,min(2048,n),replace=False)
    s=d/n*(x@x.T); z0=d/math.sqrt(n)*(y0@x.T); z1=d/math.sqrt(n)*(y1@x.T)
    h=z1-z0@s
    return x,y0,y1,ids,z0,h

def direction(h,method,r):
    if method=='GD': return h
    if method=='Sign': return np.sign(h)
    e={'Adam2':2.,'Adam1':1.,'Adam.1':.1}[method]
    return h/(abs(h)+e*math.sqrt(2*(1-r)))

def ranks(u,x,y,ids):
    uv=u@x[:,ids]
    truth=np.einsum('ij,ij->j',y[:,ids],uv)
    greater=np.zeros(len(ids),dtype=np.int64)
    for lo in range(0,y.shape[1],2048):
        hi=min(lo+2048,y.shape[1]); scores=y[:,lo:hi].T@uv
        mask=(ids>=lo)&(ids<hi)
        scores[ids[mask]-lo,np.flatnonzero(mask)]=-np.inf
        greater+=np.sum((scores>truth)|((scores==truth)&(np.arange(lo,hi)[:,None]<ids)),axis=0)
    rank=greater+1
    return rank,rank<=math.ceil(.1*x.shape[1])

def config(d,ir,seed,phase,smoke=False):
    begin=time.monotonic()
    x,y0,y1,ids,z0,h=dataset(d,ir,seed,phase)
    rows=[]; arrays={'ids':ids}
    for method in METHODS:
        p=forecast(ir,method); u=z0+p['amplitude']*direction(h,method,RS[ir])
        row=dict(d=d,r=RS[ir],ir=ir,seed=seed,method=method,phase=phase,amplitude=p['amplitude'])
        for task,y in [('old',y0),('new',y1)]:
            rk,ok=ranks(u,x,y,ids)
            arrays[method+'_'+task+'_ranks']=rk
            arrays[method+'_'+task+'_success']=ok
            row[task]=float(ok.mean())
            if smoke:
                scores=y.T@(u@x[:,ids])
                exact=[]
                for j,idx in enumerate(ids):
                    order=np.lexsort((np.arange(x.shape[1]),-scores[:,j]))
                    exact.append(int(np.flatnonzero(order==idx)[0])+1)
                assert np.array_equal(exact,rk),(task,method)
        rows.append(row)
    if smoke:
        n=x.shape[1]; s=d/n*(x@x.T)
        # Independent loss gradient by sample sum; real gradient must include S.
        z1=(z0@s+h)
        grad=d/n*((z0@x-math.sqrt(n)*y1)@x.T)
        assert np.max(abs(grad+h))<1e-11
        assert np.max(abs(z0+forecast(ir,'GD')['amplitude']*h-(z0-forecast(ir,'GD')['amplitude']*grad)))<1e-11
    return rows,arrays,time.monotonic()-begin

def run(mode):
    frozen()
    phase=11 if mode=='pilot' else 22
    if mode=='smoke':
        for ir in range(3): config(16,ir,0,33,True)
        dump(OUT/'X1_smoke.json',meta(mode,dict(passed=True)))
        return
    design=json.loads((OUT/'X1_design.json').read_text()) if mode=='formal' else None
    dims=design['dimensions'] if design else [128,256]
    counts=design['counts'] if design else {'128':8,'256':8}
    jobs=[(d,ir,se) for se in range(max(counts.values())) for d in dims for ir in range(3) if se<counts[str(d)]]
    completed=[]
    path=OUT/f'X1_{mode}_rows.jsonl'
    with path.open('x') as f:
        for d,ir,se in jobs:
            if mode=='formal' and time.monotonic()-START>6200: break
            rows,arrays,seconds=config(d,ir,se,phase)
            ap=OUT/f'X1_{mode}_d{d}_r{ir}_s{se}.npz'
            with ap.open('xb') as z: np.savez_compressed(z,**arrays)
            for row in rows:
                row['seconds_config']=seconds
                f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush()
            completed.append([d,ir,se])
            print(json.dumps(dict(mode=mode,d=d,ir=ir,seed=se,seconds=round(seconds,3),elapsed=round(time.monotonic()-START,2))),flush=True)
    frozen()
    dump(OUT/f'X1_{mode}_meta.json',meta(mode,dict(expected=len(jobs),completed=completed,complete=len(completed)==len(jobs))))

def plan():
    frozen()
    rows=[json.loads(l) for l in (OUT/'X1_pilot_rows.jsonl').read_text().splitlines()]
    info=[]; counts={}; times={}; intended={}
    for d in [128,256]:
        times[d]=float(np.mean([r['seconds_config'] for r in rows if r['d']==d]))
        req=48
        for ir in range(3):
            rr={m:[r for r in rows if r['d']==d and r['ir']==ir and r['method']==m] for m in METHODS}
            for a,b in PAIRS:
                vals=np.array([ra['old']-rb['old'] for ra,rb in zip(rr[a],rr[b])])
                var=float(vals.var(ddof=1)); vu=max(7*var/chi2.ppf(.1,7),1/2048**2)
                delta=forecast(ir,a)['old_recall']-forecast(ir,b)['old_recall']; half=min(.002,.4*delta)
                n=48
                while n<100000:
                    crit=float(student.ppf(1-.05/36,n-1))
                    if crit*math.sqrt(vu/n)<=half and math.sqrt(n)*delta/math.sqrt(vu)>=crit+norm.ppf(.9): break
                    n+=1
                req=max(req,n)
                info.append(dict(d=d,ir=ir,a=a,b=b,pilot_variance=var,upper_variance=vu,
                    delta=delta,target_halfwidth=half,required=n))
        intended[str(d)]=req; counts[str(d)]=min(192,req)
    # Reserve secondary cost at 8 full configurations per selected seed.
    secondary_est=3*16*times[256]*8*1.5
    estimated=lambda cc:sum(3*cc[str(d)]*times[d]*1.5 for d in [128,256])+secondary_est
    if estimated(counts)>6500:
        scale=max(0.,(6500-secondary_est)/sum(3*counts[str(d)]*times[d]*1.5 for d in [128,256]))
        counts={d:max(24,int(n*scale)) for d,n in counts.items()}
    dims=[128,256]
    mismatch=max(abs(np.mean([r['new'] for r in rows if r['d']==256 and r['ir']==ir and r['method']==m])-TARGET) for ir in range(3) for m in METHODS)
    extra=3*48*times[256]*16*1.5
    include=mismatch>.025 and estimated(counts)+extra<=6500
    if include: dims.append(512); counts['512']=48
    for item in info:
        n=counts[str(item['d'])]; crit=float(student.ppf(1-.05/36,n-1))
        se=math.sqrt(item['upper_variance']/n)
        item.update(planned_n=n,expected_halfwidth=crit*se,
                    approximate_power=float(norm.cdf(item['delta']/se-crit)))
    design=dict(dimensions=dims,counts=counts,intended=intended,pilot_times=times,
        secondary_estimated_seconds=secondary_est,estimated_seconds=estimated(counts)+(extra if include else 0),
        pilot_max_new_mismatch=mismatch,d512_added=include,d512_extra_seconds=extra,variance_plan=info)
    dump(OUT/'X1_design.json',design)
    text='# X1 试算后设计锁定\n\n此文件执行冻结预注册中的方差/资源规则，不更改预注册。pilot不计入正式结果。\n\n'
    text+=f'维数：{dims}；每个r的正式seed数：{counts}。未截断需求：{intended}。预计正式及次要耗时{design["estimated_seconds"]:.1f}秒。\n\n'
    text+=f'd256试算最大新任务匹配偏差{mismatch:.6f}；d512附加成本估计{extra:.1f}秒，加入={include}。\n\n'
    text+='|d|r|差|试算SD|SD的90%上界|所需N|计划N|预计校正半宽|近似功效|\n|---|---|---|---|---|---|---|---|---|\n'
    for z in info:
        text+=f'|{z["d"]}|{RS[z["ir"]]}|{z["a"]}−{z["b"]}|{math.sqrt(z["pilot_variance"]):.6f}|{math.sqrt(z["upper_variance"]):.6f}|{z["required"]}|{z["planned_n"]}|{z["expected_halfwidth"]:.6f}|{z["approximate_power"]:.4f}|\n'
    with (ROOT/'X1_DESIGN.md').open('x') as f:f.write(text)
    print(json.dumps({k:v for k,v in design.items() if k!='variance_plan'}),flush=True)

def secondary():
    frozen()
    formal_meta=json.loads((OUT/'X1_formal_meta.json').read_text())
    available={(d,ir,se) for d,ir,se in formal_meta['completed']}
    rows=[]; used=0
    with (OUT/'X1_secondary_rows.jsonl').open('x') as f:
        for se in range(16):
            for ir in range(3):
                if (256,ir,se) not in available:continue
                if formal_meta['elapsed']+time.monotonic()-START>7100:break
                x,y0,y1,ids,z0,h=dataset(256,ir,se,22); ids=ids[:256]
                base=y1.T@(z0@x[:,ids]); n=x.shape[1]
                for method in METHODS:
                    p=forecast(ir,method); v=p['amplitude']*direction(h,method,RS[ir])
                    ds=y1.T@(v@x[:,ids]); cache={}
                    def evaluate(lam):
                        if lam not in cache:
                            scores=base+lam*ds
                            truth=scores[ids,np.arange(len(ids))].copy()
                            scores[ids,np.arange(len(ids))]=-np.inf
                            cnt=np.sum(scores>truth,axis=0)
                            cache[lam]=float(np.mean(cnt+1<=math.ceil(.1*n)))
                        return cache[lam]
                    bracket=None; prev=0.
                    evaluate(0.)
                    for cur in np.linspace(0,2,33)[1:]:
                        if evaluate(float(cur))>=TARGET:bracket=(prev,float(cur));break
                        prev=float(cur)
                    if bracket:
                        lo,hi=bracket
                        for _ in range(10):
                            mid=(lo+hi)/2
                            if evaluate(mid)>=TARGET:hi=mid
                            else:lo=mid
                    best=min(cache,key=lambda t:(abs(cache[t]-TARGET),t))
                    u=z0+best*v
                    rk,ok=ranks(u,x,y0,ids)
                    row=dict(d=256,ir=ir,r=RS[ir],seed=se,method=method,lambda_selected=best,
                        amplitude=best*p['amplitude'],new=evaluate(best),old=float(ok.mean()),
                        bracketed=bracket is not None,visits=sorted(cache.items()))
                    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();rows.append(row)
                used+=1
                print(json.dumps(dict(mode='secondary',ir=ir,seed=se,elapsed=round(time.monotonic()-START,2))),flush=True)
    dump(OUT/'X1_secondary_meta.json',meta('secondary',dict(datasets=used,formal_plus_secondary_seconds=formal_meta['elapsed']+time.monotonic()-START)))

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='population':population()
    else:
        POP=json.loads((OUT/'X1_population.json').read_text())['rows']
        if mode in ['smoke','pilot','formal']:run(mode)
        elif mode=='plan':plan()
        elif mode=='secondary':secondary()
        else:raise ValueError(mode)
