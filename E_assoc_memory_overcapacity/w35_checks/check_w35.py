"""Independent W35 diagnostics. No imports of project experiment code; exclusive outputs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ[key] == '2'
import sys, math, json, time, resource, ctypes, hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'w35_checks'
START=time.monotonic()

def put(name, obj):
    with (OUT/name).open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.write('\n')

def meta():
    cg=next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    base=Path('/sys/fs/cgroup')/cg.lstrip('/')
    controls={k:(base/k).read_text().strip() for k in ('memory.max','memory.swap.max','memory.peak','memory.events')}
    assert controls['memory.max']=='2147483648' and controls['memory.swap.max']=='0'
    pools=[]
    for path in sorted({s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in s and '/' in s}):
        lib=ctypes.CDLL(path)
        for symbol in ('scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads','openblas_get_num_threads64_'):
            try: fn=getattr(lib,symbol)
            except AttributeError: continue
            fn.restype=ctypes.c_int
            pools.append(dict(library=path,threads=fn()))
            break
    assert pools and all(p['threads']<=2 for p in pools)
    return dict(python=sys.executable,controls=controls,pools=pools,seconds=time.monotonic()-START,
                rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)

def scalar(method,T,a,z,e=.1,b1=.9,b2=.99,derivative=False):
    z=np.asarray(z); u=np.zeros_like(z); m=u.copy(); v=u.copy()
    du=u.copy(); dm=u.copy(); dv=u.copy()
    for t in range(1,T+1):
        r=u-z
        if method=='Sign': direction=np.sign(r)
        else:
            m=b1*m+(1-b1)*r; v=b2*v+(1-b2)*r*r
            mh=m/(1-b1**t); vh=v/(1-b2**t)
            root=np.sqrt(vh); direction=mh/(root+e)
            if derivative:
                dm=b1*dm+(1-b1)*du; dv=b2*dv+2*(1-b2)*r*du
                dr=np.divide(dv/(1-b2**t),2*root,out=np.zeros_like(root),where=root>0)
                dd=(dm/(1-b1**t)-direction*dr)/(root+e)
                du-=direction+a*dd
        u-=a*direction
    return (u,du) if derivative else u

def pop(method,T,a,e=.1,b1=.9,b2=.99):
    # Dimensionless R/sigma. Sign integrals are exact Gaussian interval formulas.
    if method=='Sign':
        if a==0: b=v=0.
        else:
            edges=np.r_[np.arange(T+1)*a,np.inf]
            mids=np.r_[(edges[:-2]+edges[1:-1])/2,edges[-2]+1]
            ds=scalar(method,T,a,mids)
            phi=lambda x:np.exp(-x*x/2)/math.sqrt(2*math.pi)
            b=float(2*np.sum(ds*(phi(edges[:-1])-phi(edges[1:]))))
            v=float(2*np.sum(ds*ds*(ndtr(edges[1:])-ndtr(edges[:-1]))))
    else:
        den=lambda z:math.sqrt(2/math.pi)*math.exp(-z*z/2)
        b=quad(lambda z:den(z)*z*float(scalar(method,T,a,z,e,b1,b2)),0,12,epsabs=2e-11)[0]
        v=quad(lambda z:den(z)*float(scalar(method,T,a,z,e,b1,b2))**2,0,12,epsabs=2e-11)[0]
    # sigma^2=2, r=0 for the numerical matching study.
    return np.array([1-2*b+v, 1-2*b+2*v, 1-b, b, -b, 2*v])

def real(method,T,a,z0,z1,s,e=.1,b1=.9,b2=.99):
    sig=math.sqrt(2); u=z0.copy(); m=np.zeros_like(u); v=m.copy()
    for t in range(1,T+1):
        r=u@s-z1
        if method=='Sign': direction=np.sign(r)
        else:
            m=b1*m+(1-b1)*r; v=b2*v+(1-b2)*r*r
            direction=(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+e*sig)
        u-=a*sig*direction
    return u

def search(fn,T,steps):
    cache={}
    def f(a):
        if a not in cache: cache[a]=float(fn(a))
        return cache[a]-.75
    prev=0.; fp=f(prev); bracket=None
    for cur in np.linspace(0,.6/T,33)[1:]:
        cur=float(cur); fc=f(cur)
        if fp>=0 and fc<=0:
            bracket=(prev,cur); break
        prev,fp=cur,fc
    if bracket is None: return None
    lo,hi=bracket
    for _ in range(steps):
        mid=(lo+hi)/2
        if f(mid)>0: lo=mid
        else: hi=mid
    best=min(cache,key=lambda a:abs(cache[a]-.75))
    residuals=sorted(abs(v-.75) for v in cache.values())
    return dict(step=best,loss=cache[best],bracket=bracket,
                minimum_comparison_gap=min(abs(v-.75) for v in cache.values()),
                best_gap=residuals[1]-residuals[0])

def matching():
    configs=[('Sign',.1,.9,.99),('Adam',.1,.9,.99),('Adam',1.,.9,.99),
             ('Adam',.1,.9,0.),('Adam',.1,0.,0.)]
    forecasts=[]; pred={}
    for T in (1,4):
        grid=np.linspace(0,.6/T,49)
        for method,e,b1,b2 in configs:
            key=(method,T,e,b1,b2)
            pp=np.array([pop(method,T,float(a),e,b1,b2) for a in grid]); pred[key]=pp
            root=brentq(lambda a:pop(method,T,a,e,b1,b2)[0]-.75,0,.6/T,xtol=1e-13)
            fixed=search(lambda a:pop(method,T,a,e,b1,b2)[0],T,24)
            # Derivatives by differentiated recursion, independent of finite differences.
            if method=='Adam':
                xx,ww=np.polynomial.legendre.leggauss(256)
                z=(xx+1)*6; w=ww*6*math.sqrt(2/math.pi)*np.exp(-z*z/2)
                slopes=[]
                for a in grid:
                    u,du=scalar(method,T,a,z,e,b1,b2,True)
                    slopes.append(float(np.sum(w*2*(u-z)*du)))
            else: slopes=[]
            forecasts.append(dict(method=method,T=T,e=e,beta1=b1,beta2=b2,root=root,
                fixed24=fixed,grid_largest_loss_increment=float(np.max(np.diff(pp[:,0]))),
                derivative_grid_max=max(slopes) if slopes else None,
                root_loss_residual=float(pop(method,T,root,e,b1,b2)[0]-.75)))
    put('matching_population.json',forecasts)
    rows=[]
    for d in (16,32,64,128):
      n=d*d
      for seed in range(4):
        rng=np.random.default_rng(np.random.SeedSequence([350611,10,d,seed]))
        x0=rng.standard_normal((d,n))/math.sqrt(d); y0=rng.standard_normal((d,n))/math.sqrt(d)
        x1=rng.standard_normal((d,n))/math.sqrt(d); y1=rng.standard_normal((d,n))/math.sqrt(d)
        z0=d/math.sqrt(n)*(y0@x0.T); z1=d/math.sqrt(n)*(y1@x1.T)
        s=d/n*(x1@x1.T); opt=np.linalg.solve(s,z1.T).T
        def loss(u):
            diff=u-opt
            return float(np.mean((diff@s)*diff))
        baseline=loss(z0)
        for T in (1,4):
          grid=np.linspace(0,.6/T,49)
          for method,e,b1,b2 in configs:
            vals=[]; max_proxy=0.
            for a in grid:
                u=real(method,T,float(a),z0,z1,s,e,b1,b2); diff=u-z0
                proxy=z0+math.sqrt(2)*scalar(method,T,a,(z1-z0)/math.sqrt(2),e,b1,b2)
                max_proxy=max(max_proxy,float(np.mean((u-proxy)**2)))
                vals.append([loss(u)/baseline,float(np.mean(u*u)),float(np.mean(u*z0)),
                             float(np.mean(u*z1)),float(np.mean(z0*diff)),float(np.mean(diff*diff))])
            errors=np.max(np.abs(np.array(vals)-pred[(method,T,e,b1,b2)]),axis=0)
            chosen=search(lambda a:loss(real(method,T,a,z0,z1,s,e,b1,b2))/baseline,T,36)
            assert chosen is not None
            target=next(f['root'] for f in forecasts if (f['method'],f['T'],f['e'],f['beta1'],f['beta2'])==(method,T,e,b1,b2))
            rows.append(dict(d=d,seed=seed,T=T,method=method,e=e,beta1=b1,beta2=b2,
                grid_sup_errors=errors.tolist(),grid_proxy_mse_max=max_proxy,
                selected=chosen,root_error=chosen['step']-target))
        print('matching',d,seed,flush=True)
    put('matching_rows.json',rows)
    summary=[]
    for d in (16,32,64,128):
      for method in ('Sign','Adam'):
        rr=[r for r in rows if r['d']==d and r['method']==method]
        summary.append(dict(d=d,method=method,records=len(rr),
            mean_grid_sup_errors=np.mean([r['grid_sup_errors'] for r in rr],axis=0).tolist(),
            mean_max_proxy_mse=float(np.mean([r['grid_proxy_mse_max'] for r in rr])),
            mean_abs_root_error=float(np.mean([abs(r['root_error']) for r in rr]))))
    put('matching_summary.json',summary)
    put('matching_meta.json',meta())

def mu(g):
    return quad(lambda u:2/math.pi*(math.sqrt(1-u*u/g)*(1+u*u)+
        u*u*(1-u*u)/(g*math.sqrt(1-u*u/g))),0,1,epsabs=2e-13)[0]

def eff(e):
    if e==0:return math.sqrt(2/math.pi)
    den=lambda z:math.sqrt(2/math.pi)*math.exp(-z*z/2)
    a=quad(lambda z:den(z)*z*z/(z+e),0,12,epsabs=2e-13)[0]
    q=quad(lambda z:den(z)*z*z/(z+e)**2,0,12,epsabs=2e-13)[0]
    return a/math.sqrt(q)

def stabilizer():
    forecasts=[]
    for g in (16,64,256):
        m=mu(g); lo=1-m*m/g**2; hi=lo+8*(1-m*m)/(g*g*(1-1/g)**2)
        bands=[]
        for tau in (0,1,2):
            xx=tau*tau
            cm=m*math.sqrt((1+xx)/(hi+xx-2*m*m/g))
            cp=m*math.sqrt((1+xx)/(lo+xx-2*m*m/g))
            inverse=lambda c:brentq(lambda e:eff(e)-c,1e-7,100) if math.sqrt(2/math.pi)<c<1 else None
            bands.append(dict(tau=tau,c_lower=cm,c_upper=cp,e_lower=inverse(cm),e_upper=inverse(cp)))
        c=eff(.1)
        noise_lo=(m*m*(1+2*c*c/g)-c*c*hi)/(c*c-m*m)
        noise_hi=(m*m*(1+2*c*c/g)-c*c*lo)/(c*c-m*m)
        forecasts.append(dict(gamma=g,mu=m,variance_lower=lo,variance_upper=hi,
            old_delta=52/g**1.5,bands=bands,noise_squared_bracket=[noise_lo,noise_hi],
            tau_cross_bracket=[math.sqrt(max(0,noise_lo)),math.sqrt(max(0,noise_hi))]))
    put('stabilizer_forecasts.json',forecasts)
    rows=[]
    for d in (32,64,128,256):
      for g in (16,64):
       n=d*g
       for seed in range(5):
        rng=np.random.default_rng(np.random.SeedSequence([350611,20,d,g,seed]))
        x=rng.standard_normal((d,n)); y=rng.standard_normal((d,n))
        xc=x-x.mean(axis=1,keepdims=True); yc=y-y.mean(axis=1,keepdims=True)
        s=xc@xc.T/n; raw=yc@xc.T
        ua,sv,vh=np.linalg.svd(raw,full_matrices=False); o=ua@vh
        sn=(np.linalg.eigh(s)); sq=(sn[1]*np.sqrt(sn[0]))@sn[1].T
        inv=(sn[1]*(1/sn[0]))@sn[1].T
        # M=raw/sqrt(nd), T=|M|, including exact n rather than m=n-1.
        t=(vh.T*(sv/math.sqrt(n*d)))@vh
        q=float(np.trace(inv@t@s@t)/d); t2=float(np.trace(t@t)/d)
        k=q-t2
        rotated=sn[1].T@t@sn[1]
        ev=sn[0]
        kk=float(np.sum(((ev[:,None]-ev[None,:])**2/(ev[:,None]*ev[None,:]))*rotated**2)/(2*d))
        bound=float(((ev[-1]-ev[0])**2/(2*ev[-1]*ev[0]))*(t2-(np.trace(t)/d)**2))
        assert abs(k-kk)<2e-12 and k>=-1e-12 and k<=bound+1e-12
        gx=x@x.T;gy=y@y.T
        total=float(np.trace(o@gx@o.T@gy)/(d*n*n))
        centered=float(np.trace(o@(xc@xc.T)@o.T@(yc@yc.T))/(d*n*n))
        diag=np.sum(y*(o@x),axis=0)/math.sqrt(d)
        diagc=np.sum(yc*(o@xc),axis=0)/math.sqrt(d)
        wrong=(y.sum(axis=1)@(o@x)/math.sqrt(d)-diag)/(n-1)
        diagonal=float(diag@diag/(n*(n-1)))
        variance=n/(n-1)*total-diagonal-float(np.mean(wrong*wrong))
        # Projected complement: identity without explicitly forming an n by n basis.
        projected=(raw@np.linalg.solve(xc@xc.T,raw.T))
        complement=yc@yc.T-projected
        residual=float(np.trace(o@s@o.T@complement)/(n*d))
        approx=(1-(d+1)/n)*float(np.trace(s)/d)+q/g
        assert abs(centered-(residual+q/g))<3e-12
        rows.append(dict(d=d,gamma=g,seed=seed,total=total,centered_total=centered,
            trace_q=q,t2=t2,k=k,k_entrywise=kk,k_finite_bound=bound,
            mu_sample=float(np.trace(t)/d),variance=variance,diagonal=diagonal,
            predicted_diagonal=mu(g)**2/g**2,mean_squares=float(np.mean(wrong*wrong)),
            residual_trace_error=centered-approx,
            diagonal_fluctuation=float(np.mean((diagc-np.mean(diagc))**2)),
            min_eigen_s=float(ev[0])))
        if d==32:
            dense=y.T@o@x/math.sqrt(d)
            assert abs(np.mean(dense*dense)-total)<2e-12
            np.fill_diagonal(dense,0)
            ww=dense.sum(axis=0)/(n-1)
            vv=np.sum(dense*dense)/(n*(n-1))-np.mean(ww*ww)
            assert abs(vv-variance)<2e-12
       print('polar',d,g,flush=True)
    put('polar_rows.json',rows)
    put('polar_summary.json',[dict(d=d,gamma=g,n=5,
        **{key:float(np.mean([r[key] for r in rows if r['d']==d and r['gamma']==g]))
           for key in ('variance','k','diagonal','predicted_diagonal','mean_squares','residual_trace_error')})
        for d in (32,64,128,256) for g in (16,64)])
    put('stabilizer_meta.json',meta())

if __name__=='__main__':
    {'matching':matching,'stabilizer':stabilizer}[sys.argv[1]]()
