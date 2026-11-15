"""Finite-dimensional conditional recall bounds. No existing file is written.

Run only under the systemd scope specified in Z2_RECALL_BRIDGE.md.
Outputs are exclusive-created. These are floating-point checks, not interval proofs.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '2', key
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import beta, chi2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'z2_checks'
START = time.monotonic()
INPUTS = ['ICML_REVIEW_SIM5.md', 'Y3_FINITE_D_NONLINEAR.md',
          'paper/appendix/M_nonlinear_finite_d.tex', 'X1_PARAMETERS.md',
          'experiments/x1/X1_population.json', 'y3_checks/Y3_coefficients.csv']
HASHES = {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in INPUTS}
SETTINGS = dict(seed=2026110602, dimensions=[128, 512], methods=['GD', 'Adam1', 'Sign'],
                eta=.002, rho=.1, kappa=.25, scatters_per_dimension=1)

def threadpool_info():
    paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
           if 'openblas' in line.lower() and line.split()[-1].startswith('/')}
    result=[]
    for path in sorted(paths):
        lib=ctypes.CDLL(path)
        for symbol in ('openblas_get_num_threads64_', 'openblas_get_num_threads',
                       'scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads_'):
            try:
                fun=getattr(lib,symbol);fun.restype=ctypes.c_int
                count=fun();assert count<=2
                result.append(dict(path=path,symbol=symbol,num_threads=count));break
            except AttributeError:continue
    return result

def dump(name, obj):
    with (OUT/name).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)

def sphere_thresholds(n, rho, eta):
    m = n-1
    ell = math.ceil(rho*n)
    # 1-Phi(G_(ell)) ~ Beta(ell,m-ell+1).
    pminus, pplus = beta.ppf([eta/4, 1-eta/4], ell, m-ell+1)
    gminus, gplus = ndtri(1-pplus), ndtri(1-pminus)
    rminus, rplus = np.sqrt(chi2.ppf([eta/4, 1-eta/4], m)/m)
    assert gminus > 0 and rminus > 0
    return float(gminus/rplus), float(gplus/rminus)

def moments(M, N, n, d, q):
    c = n/(d*math.sqrt(q))
    m = n-1
    m1, n1 = np.trace(M), np.trace(N)
    M2, N2 = M@M, N@N
    tm2, tn2 = np.trace(M2), np.trace(N2)
    tm3 = np.sum(M2*M.T)
    tm4 = np.sum(M2*M2.T)
    tmn = np.sum(M*N.T)
    tm2n = np.sum(M2*N.T)
    T20 = m1*m1+2*tm2
    T02 = n1*n1+2*tn2
    T21 = n1*T20+4*m1*tmn+8*tm2n
    T40 = m1**4+12*m1*m1*tm2+12*tm2**2+32*m1*tm3+48*tm4
    den2 = n*(n+2)
    ED2 = c*c*T20/den2
    v = c*c*n1/(m*n)
    EV = v-ED2/m
    EV2 = c**4/m**2*(T02/den2-2*T21/(den2*(n+4))+T40/(den2*(n+4)*(n+6)))
    evar = EV2-2*v*EV+v*v
    assert evar >= -1e-11
    return dict(mu=float(c*m1/n), v=float(v), ED2=float(ED2), EV=float(EV),
                EV2=float(EV2), V_error2=float(max(0,evar)))

def quadratic_tail(eigen_K_over_n, n, threshold, shift=0):
    lam = eigen_K_over_n-(threshold-shift)/n
    extra = n-len(lam)
    lam0 = -(threshold-shift)/n
    mean = float(lam.sum()+extra*lam0)
    t2 = float(np.sum(lam**2)+extra*lam0**2)
    t4 = float(np.sum(lam**4)+extra*lam0**4)
    sd = math.sqrt(2*t2)
    gamma = min(1., math.sqrt(2*t4)/t2)
    return float(ndtr(mean/sd)), gamma, sd, mean

def fixed_threshold_bound(M, N, n, d, q, z, mom):
    v = mom['v']
    m = n-1
    K_over_n = M/(d*math.sqrt(q))-z*n*N/(2*m*d*d*q*math.sqrt(v))
    eig = np.linalg.eigvalsh(K_over_n)
    threshold = z*math.sqrt(v)/2
    e = z/(2*math.sqrt(v))*(mom['ED2']/m+mom['V_error2']/v)
    normal, gamma, sd, mean = quadratic_tail(eig,n,threshold)
    epsilon = math.sqrt(max(e, 1e-30)*sd*math.sqrt(2*math.pi))
    normal_up, gamma_up, sd_up, _ = quadratic_tail(eig,n,threshold,epsilon)
    lower = max(0.,normal-gamma)
    upper = min(1.,normal_up+gamma_up+e/epsilon)
    return dict(lower=lower,upper=upper,normal=normal,gamma=gamma,e=e,epsilon=epsilon,
                sd=sd,mean=mean,normal_up=normal_up,gamma_up=gamma_up,sd_up=sd_up)

def compressed_matrices(S, Ckk, Ckx, U):
    d = len(S)
    gram = np.block([[Ckk,Ckx],[Ckx.T,S]])
    R = np.linalg.cholesky(gram)
    zero = np.zeros((d,d))
    RM = np.block([[zero,U/2],[U.T/2,zero]])
    RN = np.block([[zero,zero],[zero,U.T@Ckk@U]])
    M, N = R.T@RM@R, R.T@RN@R
    return (M+M.T)/2, (N+N.T)/2

def checks():
    rng = np.random.default_rng(2026110601)
    p,n,d,q = 4,43,2,.7
    A = rng.normal(size=(p,p))
    M = (A+A.T)/2
    N = A.T@A
    mom = moments(M,N,n,d,q)
    samples = 160000
    totals = np.zeros(5)
    total2 = np.zeros(5)
    min_remainder = 1.
    z = 1.2815515655446004
    fixed = fixed_threshold_bound(M,N,n,d,q,z,mom)
    max_identity = 0.
    for _ in range(samples//2000):
        g = rng.normal(size=(2000,p))
        W = np.sum(g*g,axis=1)+rng.chisquare(n-p,size=2000)
        a = np.einsum('bi,ij,bj->b',g,M,g)/W
        b = np.einsum('bi,ij,bj->b',g,N,g)/W
        c=n/(d*math.sqrt(q))
        D=c*a
        V=c*c/(n-1)*(b-a*a)
        Vtilde=c*c/(n-1)*b
        v=mom['v']
        linear=D-z*math.sqrt(v)-z/(2*math.sqrt(v))*(Vtilde-v)
        remainder=D-z*np.sqrt(V)-linear
        exact=z/(2*math.sqrt(v))*(D*D/(n-1)+(V-v)**2/(np.sqrt(V)+math.sqrt(v))**2)
        min_remainder=min(min_remainder,float(np.min(remainder)))
        max_identity=max(max_identity,float(np.max(np.abs(remainder-exact))))
        vals=np.array([D*D,V,V*V,remainder,(D>=z*np.sqrt(V)).astype(float)]).T
        totals+=vals.sum(axis=0)
        total2+=(vals*vals).sum(axis=0)
    means=totals/samples
    se=np.sqrt(np.maximum(0,total2/samples-means**2)/samples)
    targets=np.array([mom['ED2'],mom['EV'],mom['EV2']])
    standardized=np.abs(means[:3]-targets)/se[:3]
    assert standardized.max()<6
    assert min_remainder > -1e-11 and max_identity < 1e-10
    assert means[3] < fixed['e']+6*se[3]
    assert fixed['lower'] <= means[4]+6*se[4] <= fixed['upper']+12*se[4]
    # Matrix compression agrees in all nonzero eigenvalues.
    lx=rng.normal(size=(d,3*d)); lk=rng.normal(size=(d,3*d)); U=rng.normal(size=(d,d))
    Afull=lk.T@U@lx
    Mc,Nc=compressed_matrices(lx@lx.T,lk@lk.T,lk@lx.T,U)
    Mfull=(Afull+Afull.T)/2
    Nfull=Afull.T@Afull
    compression_error=max(abs(np.trace(Mfull@Mfull)-np.trace(Mc@Mc)),
                          abs(np.trace(Nfull@Nfull)-np.trace(Nc@Nc)))
    assert compression_error<1e-7
    # Direct spherical order statistic check, independent of all other checks.
    nn=401; eta=.02
    lo,hi=sphere_thresholds(nn,.1,eta)
    count=0; trials=16000
    for _ in range(trials//400):
        g=rng.normal(size=(400,nn-1))
        kth=np.partition(g,-math.ceil(.1*nn),axis=1)[:,-math.ceil(.1*nn)]
        T=kth/np.sqrt(np.mean(g*g,axis=1))
        count+=int(np.sum((T<lo)|(T>hi)))
    assert count/trials < eta+6*math.sqrt(eta/trials)
    return dict(moment_errors_in_MC_SE=standardized.tolist(),identity_error=max_identity,
                minimum_remainder=min_remainder,mean_remainder=float(means[3]),
                remainder_bound=fixed['e'],compression_error=float(compression_error),
                spherical_band=[lo,hi],spherical_failure=count/trials,
                note='Monte Carlo checks are diagnostics, not mathematical proofs.')

def main():
    dump('Z2_settings.json',dict(**SETTINGS,input_hashes=HASHES,
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    verification=checks()
    dump('Z2_verification.json',verification)
    coeff=list(csv.DictReader((ROOT/'y3_checks/Y3_coefficients.csv').open()))
    population=json.loads((ROOT/'experiments/x1/X1_population.json').read_text())['rows']
    rng=np.random.default_rng(SETTINGS['seed'])
    rows=[]
    for d in SETTINGS['dimensions']:
        n=d*d//4;p=3*d
        # Bartlett factor: C=B B^T/n ~ Wishart_p(n,I)/n. No n-column data generated.
        B=np.tril(rng.normal(size=(p,p)),-1)
        B[np.diag_indices(p)]=np.sqrt(rng.chisquare(n-np.arange(p)))
        F=B/math.sqrt(n)
        C=F@F.T
        S=C[:d,:d]
        C00=C[d:2*d,d:2*d]; CEE=C[2*d:,2*d:]
        C0E=C[d:2*d,2*d:]
        C0x=C[d:2*d,:d]; CEx=C[2*d:,:d]
        Z0=math.sqrt(n)*C0x
        zlo,zhi=sphere_thresholds(n,SETTINGS['rho'],SETTINGS['eta'])
        for pp in population:
            if pp['method'] not in SETTINGS['methods']:continue
            r=pp['r'];a=pp['amplitude'];method=pp['method']; rr=math.sqrt(1-r*r)
            C1x=r*C0x+rr*CEx
            C11=r*r*C00+rr*rr*CEE+r*rr*(C0E+C0E.T)
            H=math.sqrt(n)*C1x-Z0@S
            if method=='Sign':h=np.sign(H)
            elif method=='GD':h=H
            else:h=H/(np.abs(H)+math.sqrt(2*(1-r))) # e=1, fixed physical epsilon
            U=Z0+a*h
            for task,Ckk,Ckx in [('old',C00,C0x),('new',C11,C1x)]:
                frozen=next(v for v in coeff if float(v['r'])==r and v['method']==method and v['task']==task)
                q=float(frozen['q'])
                M,N=compressed_matrices(S,Ckk,Ckx,U)
                mom=moments(M,N,n,d,q)
                lower=fixed_threshold_bound(M,N,n,d,q,zhi,mom)
                upper=fixed_threshold_bound(M,N,n,d,q,zlo,mom)
                fixed=fixed_threshold_bound(M,N,n,d,q,float(ndtri(.9)),mom)
                row=dict(d=d,n=n,r=r,method=method,task=task,
                         lower=max(0,lower['lower']-SETTINGS['eta']),
                         upper=min(1,upper['upper']+SETTINGS['eta']),
                         population_limit=float(frozen['recall_limit']),
                         conditional_normal=fixed['normal'],zlo=zlo,zhi=zhi,
                         gamma=fixed['gamma'],remainder_bound=fixed['e'],
                         epsilon=fixed['epsilon'],sd=fixed['sd'],**mom)
                rows.append(row)
                print(json.dumps({k:row[k] for k in ['d','r','method','task','lower','upper'] }),flush=True)
    with (OUT/'Z2_conditional_intervals.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    final_hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in INPUTS}
    assert HASHES==final_hashes
    cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    cgp=Path('/sys/fs/cgroup')/cg.lstrip('/')
    dump('Z2_resources.json',dict(wall_seconds=time.monotonic()-START,
         cpu_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         cgroup=cg,memory_max=(cgp/'memory.max').read_text().strip(),
         memory_swap_max=(cgp/'memory.swap.max').read_text().strip(),
         threadpools=threadpool_info(),input_hashes_unchanged=True))

if __name__=='__main__':main()
