"""Review 16: independent implementation; never imports reviewed/project code.

Build full 3d matrices, use trace powers instead of reviewed 2d compression,
derive moments by a bivariate cumulant polynomial recursion, and independently
sample the spherical candidate order statistic. All output uses exclusive create.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert int(os.environ[name]) <= 2
import csv
import ctypes
import hashlib
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import beta, chi2

def threadpool_info():
    found={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        path=line.split()[-1]
        if 'openblas' not in path.lower() or path in found: continue
        lib=ctypes.CDLL(path)
        symbols=['scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads_',
                 'openblas_get_num_threads64_', 'openblas_get_num_threads']
        for symbol in symbols:
            try: fn=getattr(lib,symbol)
            except AttributeError: continue
            fn.restype=ctypes.c_int
            found[path]=dict(library=path,num_threads=fn()); break
    assert len(found)>=1
    return list(found.values())

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
START = time.monotonic()
DESIGN = dict(seed=16061107, dimensions=[128, 256, 512], r=[0., .5, -.5],
              methods=['GD', 'Adam1', 'Sign'], scatters=2, mc_queries=4096,
              eta=.002, rho=.1, dtype='float64', rounded_certificate=False)
INPUTS = ['Z2_RECALL_BRIDGE.md', 'z2_checks/Z2_bridge.py', 'z2_checks/Z2_outer.py',
          'z2_checks/Z2_outer_rows.csv', 'z2_checks/Z2_outer_result.json',
          'z2_checks/Z2_conditional_intervals.csv', 'experiments/x1/X1_population.json',
          'y3_checks/Y3_coefficients.csv', 'y3d/formal/analysis/diagnostics72.csv',
          'y3d/formal/analysis/secondary_old45.csv']
HASHES = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}

def dump(name, data):
    with (OUT/name).open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

def save_csv(name, rows):
    with (OUT/name).open('x', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def moment_data(M, N, n, d, q):
    # log E exp(t Q_M + w Q_N), coefficients needed through (4,0),(2,1),(0,2).
    # Exponentiate as a formal bivariate series with total degree <=4.
    M2=M@M; N2=N@N
    powers=[float(np.trace(M)), float(np.sum(M*M)),
            float(np.sum(M2*M)), float(np.sum(M2*M2))]
    logpoly={(j,0): 2**(j-1)*powers[j-1]/j for j in range(1,5)}
    logpoly.update({(0,1):float(np.trace(N)), (0,2):float(np.sum(N*N)),
                    (1,1):2*float(np.sum(M*N)), (2,1):4*float(np.sum(M2*N))})
    def mul(a,b):
        out={}
        for (i,j),v in a.items():
            for (k,l),w in b.items():
                if i+k<=4 and j+l<=2 and i+j+k+l<=4:
                    out[i+k,j+l]=out.get((i+k,j+l),0.)+v*w
        return out
    poly={(0,0):1.}; term={(0,0):1.}
    for order in range(1,5):
        term={key:v/order for key,v in mul(term,logpoly).items()}
        for key,vv in term.items(): poly[key]=poly.get(key,0.)+vv
    def spherical(i,j):
        return poly[i,j]*math.factorial(i)*math.factorial(j)/math.prod(n+2*k for k in range(i+j))
    c=n/d/math.sqrt(q); m=n-1
    D2=c*c*spherical(2,0)
    v=c*c/m*spherical(0,1)
    EV=c*c/m*(spherical(0,1)-spherical(2,0))
    EV2=c**4/m**2*(spherical(0,2)-2*spherical(2,1)+spherical(4,0))
    err=EV2-2*v*EV+v*v
    assert err>0
    return dict(mu=c*spherical(1,0), v=v, D2=D2, EV=EV, EV2=EV2, V_error2=err)

def bands(n):
    m=n-1; L=math.ceil(DESIGN['rho']*n); a=DESIGN['eta']/4
    blo,bhi=beta.ppf([a,1-a],L,n-L)
    rlo,rhi=np.sqrt(chi2.ppf([a,1-a],m)/m)
    return float(ndtri(1-bhi)/rhi),float(ndtri(1-blo)/rlo)

def normal_stats(B, scalar, n):
    p=len(B); m0=n-p
    # B is leading p block; n-p entries equal scalar (not zero).
    square=B@B
    t2=float(np.sum(B*B))+m0*scalar**2
    t4=float(np.sum(square*square))+m0*scalar**4
    mean=float(np.trace(B))+m0*scalar
    return dict(normal=float(ndtr(mean/math.sqrt(2*t2))), sd=math.sqrt(2*t2),
                gamma=min(1.,math.sqrt(2*t4)/t2))

def endpoint(M,N,n,d,q,z,info):
    v=info['v']; p=len(M)
    a=M/(d*math.sqrt(q))-z*n*N/(2*(n-1)*d*d*q*math.sqrt(v))
    t=z*math.sqrt(v)/2/n
    B=a-t*np.eye(p)
    base=normal_stats(B,-t,n)
    e=z/(2*math.sqrt(v))*(info['D2']/(n-1)+info['V_error2']/v)
    epsilon=math.sqrt(e*base['sd']*math.sqrt(2*math.pi))
    up=normal_stats(B+epsilon/n*np.eye(p),-t+epsilon/n,n)
    return dict(lower=base['normal']-base['gamma'],
                upper=up['normal']+up['gamma']+e/epsilon,
                e=e, epsilon=epsilon, **base), B, -t

def path_min_sd(B, scalar, direction, direction0, n, left, right):
    # Exact minimum of 2||B+t direction||_F^2 on a real interval.
    m0=n-len(B)
    aa=float(np.sum(direction*direction))+m0*direction0**2
    bb=float(np.sum(B*direction))+m0*scalar*direction0
    x=min(right,max(left,-bb/aa)) if aa else left
    return math.sqrt(2*(float(np.sum((B+x*direction)**2))+m0*(scalar+x*direction0)**2))

def envelope(M,N,n,d,q,info,zlo,zhi,low,high,center,B0,b0):
    z=float(ndtri(.9)); p=len(M); eye=np.eye(p)
    T=n*N/(2*(n-1)*d*d*q*math.sqrt(info['v']))+math.sqrt(info['v'])/(2*n)*eye
    t0=math.sqrt(info['v'])/(2*n)
    E=d*high['e']; eps=math.sqrt(E/d)
    Bminus=B0-(zlo-z)*T; scalar=b0-(zlo-z)*t0
    sstar=min(path_min_sd(B0,b0,-T,-t0,n,zlo-z,zhi-z),
              path_min_sd(Bminus,scalar,eye/n,1/n,n,0,eps))
    # The only gammas needed in (20) are the two actual endpoints.
    gp=normal_stats(Bminus+eps/n*eye,scalar+eps/n,n)['gamma']
    G=math.sqrt(d)*max(low['gamma'],gp)
    tnorm=math.sqrt(float(np.sum(T*T))+(n-p)*t0*t0)
    AC=math.sqrt(info['v'])/(sstar*math.sqrt(2*math.pi))+math.sqrt(2)*tnorm/(sstar*math.sqrt(2*math.pi*math.e))
    BC=1/(sstar*math.sqrt(2*math.pi))+math.sqrt(2/n)/(sstar*math.sqrt(2*math.pi*math.e))
    error=DESIGN['eta']+AC*max(z-zlo,zhi-z)+(G+(1+BC)*math.sqrt(E))/math.sqrt(d)
    return dict(E_envelope=E,G_envelope=G,s_star=sstar,A_C=AC,B_C=BC,
                cor4_error=error, cor4_center=center['normal'])

def sample_thresholds(rng,n,count):
    m=n-1; L=math.ceil(DESIGN['rho']*n); out=[]
    for begin in range(0,count,32):
        g=rng.standard_normal((min(32,count-begin),m))
        radius=np.sqrt(np.mean(g*g,axis=1))
        g.partition(m-L,axis=1)
        out.extend((g[:,m-L]/radius).tolist())
    return np.array(out)

def main():
    dump('design.json',dict(**DESIGN,input_hashes=HASHES))
    params=json.loads((ROOT/'experiments/x1/X1_population.json').read_text())['rows']
    coeff=list(csv.DictReader((ROOT/'y3_checks/Y3_coefficients.csv').open()))
    rng=np.random.default_rng(DESIGN['seed']); rows=[]; threshold_rows=[]
    for d in DESIGN['dimensions']:
        n=d*d//4; p=3*d; count=DESIGN['mc_queries']; zlo,zhi=bands(n)
        ts=sample_thresholds(rng,n,count)
        threshold_rows.append(dict(d=d,zlo=zlo,zhi=zhi,trials=count,
                                   failures=int(np.sum((ts<zlo)|(ts>zhi))),
                                   mean=float(ts.mean()),sd=float(ts.std(ddof=1))))
        for rep in range(DESIGN['scatters']):
            # Generate independent 3d-dimensional Wishart factor, retain full p representation.
            F=np.zeros((p,p))
            ii,jj=np.tril_indices(p,-1); F[ii,jj]=rng.standard_normal(len(ii))
            F[np.diag_indices(p)]=np.sqrt(rng.chisquare(n-np.arange(p)))
            F/=math.sqrt(n)
            lx,l0,le=np.split(F,3,axis=0)
            S=lx@lx.T; Z0=math.sqrt(n)*(l0@lx.T)
            g=rng.standard_normal((p,count))
            W=np.sum(g*g,axis=0)+rng.chisquare(n-p,count)
            sphere=g/np.sqrt(W)
            x=lx@sphere
            for r in DESIGN['r']:
                l1=r*l0+math.sqrt(1-r*r)*le
                H=math.sqrt(n)*(l1@lx.T)-Z0@S
                for method in DESIGN['methods']:
                    param=next(v for v in params if v['r']==r and v['method']==method)
                    response=H if method=='GD' else np.sign(H) if method=='Sign' else H/(np.abs(H)+math.sqrt(2*(1-r)))
                    U=Z0+param['amplitude']*response
                    ux=U@x
                    for task,lk in [('old',l0),('new',l1)]:
                        cf=next(v for v in coeff if float(v['r'])==r and v['method']==method and v['task']==task)
                        q=float(cf['q']); c=n/d/math.sqrt(q)
                        A=lk.T@U@lx; M=(A+A.T)/2; N=A.T@A
                        info=moment_data(M,N,n,d,q)
                        low,_,_=endpoint(M,N,n,d,q,zhi,info)
                        high,_,_=endpoint(M,N,n,d,q,zlo,info)
                        center,B0,b0=endpoint(M,N,n,d,q,float(ndtri(.9)),info)
                        lower=max(0.,low['lower']-DESIGN['eta']); upper=min(1.,high['upper']+DESIGN['eta'])
                        # Conditional column directions; spherical T independently generated above.
                        y=lk@sphere
                        D=c*np.sum(y*ux,axis=0)
                        Au=lk.T@ux
                        V=c*c/(n-1)*(np.sum(Au*Au,axis=0)-(D/c)**2)
                        assert np.all(V>0)
                        observed=float(np.mean(D>=ts*np.sqrt(V)))
                        R=D-float(ndtri(.9))*np.sqrt(V)-(D-float(ndtri(.9))*math.sqrt(info['v'])-float(ndtri(.9))/(2*math.sqrt(info['v']))*(V+D*D/(n-1)-info['v']))
                        Rformula=float(ndtri(.9))/(2*math.sqrt(info['v']))*(D*D/(n-1)+(V-info['v'])**2/(np.sqrt(V)+math.sqrt(info['v']))**2)
                        env=envelope(M,N,n,d,q,info,zlo,zhi,low,high,center,B0,b0)
                        row=dict(d=d,rep=rep,r=r,method=method,task=task,lower=lower,upper=upper,
                                 observed=observed,mc_se=math.sqrt(observed*(1-observed)/count),
                                 covered=lower<=observed<=upper,width=upper-lower,
                                 gamma=center['gamma'],e=center['e'],epsilon=center['epsilon'],
                                 moment_D2_mc=float(np.mean(D*D)),moment_EV_mc=float(np.mean(V)),
                                 moment_EV2_mc=float(np.mean(V*V)),min_R=float(R.min()),
                                 R_identity_maxerr=float(np.max(np.abs(R-Rformula))),
                                 R_mean=float(R.mean()),R_bound=center['e'],**info,**env)
                        rows.append(row)
            print(json.dumps(dict(d=d,rep=rep,completed_rows=len(rows),wall=time.monotonic()-START)),flush=True)
            save_csv(f'conditional_d{d}_rep{rep}.csv',rows[-18:])
    save_csv('conditional_all.csv',rows); save_csv('thresholds.csv',threshold_rows)
    # Aggregate existing outer rows without invoking the authors' code.
    outer=list(csv.DictReader((ROOT/'z2_checks/Z2_outer_rows.csv').open()))
    ell=np.array([float(v['GD_lower'])-float(v['Sign_upper']) for v in outer])
    up=np.array([float(v['GD_upper'])-float(v['Sign_lower']) for v in outer])
    radius=math.sqrt(2*math.log(200)/len(outer))
    outercheck=dict(B=len(outer),mean_lower=float(ell.mean()),mean_upper=float(up.mean()),
                    radius=radius,interval=[float(ell.mean()-radius),float(up.mean()+radius)],
                    total_width=float(up.mean()-ell.mean()+2*radius),
                    analytic_mean_width=float((up-ell).mean()),outer_added_width=2*radius,
                    sd_lower=float(ell.std(ddof=1)),sd_upper=float(up.std(ddof=1)),
                    zero_radius_minimum_B=math.ceil(2*math.log(200)/ell.mean()**2))
    dump('outer_reaggregate.json',outercheck)
    comparisons=[]
    old=list(csv.DictReader((ROOT/'y3d/formal/analysis/secondary_old45.csv').open()))
    diag=list(csv.DictReader((ROOT/'y3d/formal/analysis/diagnostics72.csv').open()))
    for d in DESIGN['dimensions']:
        for r in DESIGN['r']:
            ir={.5:0,0.:1,-.5:2}[r]
            for method in DESIGN['methods']:
                for task in ('old','new'):
                    candidates=old if task=='old' else diag
                    found=next(v for v in candidates if int(v['d'])==d and int(v['ir'])==ir and (v.get('method')==method if task=='old' else v['task']=='new' and v['label']==method))
                    obs=float(found['mean'] if task=='old' else found['observed'])
                    subset=[v for v in rows if v['d']==d and v['r']==r and v['method']==method and v['task']==task]
                    comparisons.append(dict(d=d,r=r,method=method,task=task,formal_observed=obs,
                        mean_conditional_lower=float(np.mean([v['lower'] for v in subset])),
                        mean_conditional_upper=float(np.mean([v['upper'] for v in subset])),
                        mean_mc=float(np.mean([v['observed'] for v in subset])),
                        note='descriptive cross-C comparison only; NOT a confidence guarantee'))
    save_csv('formal_comparison.csv',comparisons)
    cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    cgp=Path('/sys/fs/cgroup')/cg.lstrip('/')
    pools=threadpool_info(); assert all(v['num_threads']<=2 for v in pools)
    final={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}
    assert final==HASHES
    summary=dict(rows=len(rows),all_covered=all(v['covered'] for v in rows),
                 max_identity_error=max(v['R_identity_maxerr'] for v in rows),
                 min_remainder=min(v['min_R'] for v in rows),
                 wall_seconds=time.monotonic()-START,cpu_seconds=time.process_time(),
                 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 cgroup=cg,memory_max=(cgp/'memory.max').read_text().strip(),
                 swap_max=(cgp/'memory.swap.max').read_text().strip(),threadpools=pools,
                 reviewed_inputs_unchanged=HASHES==final)
    dump('summary.json',summary); print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
