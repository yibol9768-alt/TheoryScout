"""AA1 M-model quantitative checks. New outputs only; no network or LLM API.

Float64 diagnostics, NOT directed-rounding certificates. All independent
outer draws are retained. Exact probability identities are proved in AA1.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ.get(key) == '1', key
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'aa1_checks' / (sys.argv[1] if len(sys.argv)>1 else 'AA1_replay')
CONFIG = dict(seed=202611060101, d=64, kappa=.25, rho=.1, outer=8192,
              joint_failure=.01, threads=1,
              methods=['GD2', 'Sign1', 'Sign2_06', 'Sign2_15', 'Adam2_e1'])

def dump(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)

def scalar(z, method):
    z = np.asarray(z)
    u = np.zeros_like(z, dtype=float)
    if method == 'GD2':
        return .75 * z
    if method.startswith('Sign'):
        alpha = 1.5 if method.endswith('15') else .6
        for _ in range(1 if method == 'Sign1' else 2):
            u += alpha * np.sign(z-u)
        return u
    m = np.zeros_like(z); v = np.zeros_like(z)
    for t in (1, 2):
        residual = u-z
        m = .9*m + .1*residual
        v = .99*v + .01*residual**2
        u -= .6*(m/(1-.9**t))/(np.sqrt(v/(1-.99**t))+1.)
    return u

def endpoint(S, Z, method):
    u = np.zeros_like(Z)
    if method.startswith('Sign'):
        alpha = 1.5 if method.endswith('15') else .6
        for _ in range(1 if method == 'Sign1' else 2):
            u -= alpha*np.sign(u@S-Z)
        return u
    if method == 'GD2':
        for _ in range(2):
            u -= .5*(u@S-Z)
        return u
    m = np.zeros_like(Z); v = np.zeros_like(Z)
    for t in (1, 2):
        residual = u@S-Z
        m = .9*m+.1*residual
        v = .99*v+.01*residual**2
        u -= .6*(m/(1-.9**t))/(np.sqrt(v/(1-.99**t))+1.)
    return u

def root_prob(lam, h, projected, J, threshold, n):
    K = n-1+threshold**2
    H = K*(1-h)-threshold**2
    if H <= 0 or threshold <= 0 or h >= 1:
        return 0., 1.
    # Larger root, retaining the joint correct-score / wrong-energy dependence.
    root = (-K*math.sqrt(1-h)*lam + threshold*math.sqrt(
        K*lam**2+H*(projected+J)))/H
    value = float(ndtr(-root))
    return value, value

def draw(rng, d, n, rho):
    chol = np.tril(rng.normal(size=(d, d)), -1)
    chol[np.diag_indices(d)] = np.sqrt(rng.chisquare(n-np.arange(d)))
    chol /= math.sqrt(n)
    S = chol@chol.T
    G = rng.normal(size=(d, d))
    Z = G@chol.T
    g = rng.normal(size=d)
    u = g/math.sqrt(float(g@g)+rng.chisquare(n-d))
    x = math.sqrt(n/d)*(chol@u)
    h = float(u@u)
    J = rng.chisquare(n-d-1)
    wrong = rng.normal(size=n-1)
    radius = math.sqrt(float(wrong@wrong)/(n-1))
    index = n-1-math.ceil(rho*n)
    wrong.partition(index)
    threshold = float(wrong[index]/radius)
    results = {}
    for method in CONFIG['methods']:
        U = endpoint(S, Z, method)
        t = U@x
        norm = np.linalg.norm(t)
        if norm == 0:
            results[method] = (0., 1.)
            continue
        b = G.T@t/norm
        lam = float(u@b)
        projected = float(b@b)
        assert lam*lam <= h*projected+1e-10
        results[method] = root_prob(lam, h, projected, J, threshold, n)
    return results

def constants(method):
    # q,a are diagnostic quadratures; exact q is the integral in the theorem.
    breaks = [0., .6, 1.5, 12.]
    density = lambda z: math.sqrt(2/math.pi)*math.exp(-z*z/2)
    a = sum(quad(lambda z: z*float(scalar(z, method))*density(z), x, y,
                 epsabs=1e-12)[0] for x, y in zip(breaks[:-1], breaks[1:]))
    q = sum(quad(lambda z: float(scalar(z, method))**2*density(z), x, y,
                 epsabs=1e-12)[0] for x, y in zip(breaks[:-1], breaks[1:]))
    return a, q

def dimension_bound(method, gap):
    kappa = CONFIG['kappa']; rho = CONFIG['rho']
    xi = eta = gap/40
    a, q = constants(method)
    b = math.sqrt(q/2)
    if method == 'GD2':
        K = .75; va = 2*K*K; vq = 2*K**4
        Cgd = .5**2
        Fbase = 2*(2*Cgd+K); rate = .5
    elif method.startswith('Sign'):
        alpha = 1.5 if method.endswith('15') else .6
        T = 1 if method == 'Sign1' else 2
        K = T*alpha; va = K*K; vq = K**4
        A = 2*T*T/math.sqrt(2*math.pi)
        B = sum((t*alpha+2)**2 for t in range(T))
        Fbase = 2*T*alpha*math.sqrt(A*(2*B/A)**(1/3)+B*(A/(2*B))**(2/3))
        rate = 1/6
    else:
        alpha = .6; e = 1.; beta2 = .99; T = 2
        lam = 1+2*alpha/e; c0 = math.sqrt(1-beta2)
        H = [e*(lam**t-1)/(2*c0) for t in range(T+1)]
        At = [0.]
        for t in range(1,T+1):
            Lt = (1+lam**(t-1)/c0)/e
            At.append(At[-1]+alpha*Lt*math.sqrt(sum(
                (2*At[k-1]+H[k-1]+2)**2 for k in range(1,t+1))))
        K = H[-1]; va = K*K; vq = K**4
        Fbase = At[-1]; rate = .5
    ax = 2*math.log(9)+math.log(2/xi)
    D = 4*math.sqrt(ax/kappa)+4*ax/kappa
    V = 2*K*math.sqrt(4*math.log(9)+2*math.log(2/xi))
    F = Fbase*D**(2*rate)
    z = float(ndtri(1-rho)); Z = z+1
    M = 3/math.sqrt(kappa); D4 = 8*M**4+480
    rstar = min(1/(64*Z), 1/(4*math.sqrt(52*Z)))
    KR = 12*(1+Z)+3*math.sqrt(52*Z)
    Kn = 12*(1+Z)+3*math.sqrt(2*Z*(M*M+2))
    srho = min(rho/2, (.5-rho)/2)
    Drho = math.sqrt(2*math.pi)*math.exp(float(ndtri(1-rho+srho))**2/2)
    Hrho = 5*math.sqrt(2)*Drho+4*math.sqrt(2)*z
    xx = math.log(4/eta)
    N0 = max(128,(32*Z)**2,8*Z*(M*M+2),8*math.sqrt(Z*D4),
             25*xx/srho**2,16*xx,xx*Hrho**2/min(z/2,1)**2)
    d0 = max(1, 3/kappa,(8*D)**2,(4*V/(b*rstar))**2,
             2*math.sqrt(vq)/(q*math.sqrt(xi)),math.sqrt(N0/kappa))
    C = (4*F/(b*rstar)+4*F/(b*math.sqrt(kappa))+2*D/math.sqrt(kappa)
         +math.sqrt(va)/(b*math.sqrt(kappa*xi))
         +math.sqrt(vq)/(b*b*math.sqrt(kappa*xi))
         +12*(1+Z)*D+2*KR*(V+F)/b+Kn/math.sqrt(kappa)
         +12*math.sqrt(Z*D4)/kappa+Hrho*math.sqrt(xx/kappa)
         +2/9+1/(2*kappa**1.5))
    return dict(method=method, q=q, a=a, c=a/math.sqrt(q), rate=rate,
                constant_C=C, admissibility_d0=d0, xi=xi, eta=eta)

def checks():
    rng = np.random.default_rng(202611060100)
    root_checks = []
    for lam,h,P,z in [(1.3,.1,25.,1.28),(-.4,.2,12.,1.5),(2.,.05,90.,.8)]:
        n = 100; size = 100000
        J = rng.chisquare(n-6,size=size); g = rng.normal(size=size)
        direct = lam+math.sqrt(1-h)*g
        V = (P+g*g+J-direct**2)/(n-1)
        assert V.min() > 0
        K=n-1+z*z; H=K*(1-h)-z*z
        roots=(-K*math.sqrt(1-h)*lam+z*np.sqrt(K*lam*lam+H*(P+J)))/H
        p=ndtr(-roots); observed=direct>=z*np.sqrt(V)
        difference=float(np.mean(observed-p)); se=float(np.std(observed-p,ddof=1)/math.sqrt(size))
        assert abs(difference)<6*se
        root_checks.append(dict(difference=difference,standard_error=se))
    # Full-data rank vs a separately generated residual integral, same M law.
    d=8; n=64; B=4000; vals=[]
    for _ in range(B):
        X=rng.normal(size=(d,n))/math.sqrt(d)
        Y=rng.normal(size=(d,n))/math.sqrt(d)
        S=d/n*X@X.T; Z=d/math.sqrt(n)*Y@X.T
        U=endpoint(S,Z,'Sign2_15')
        scores=Y.T@(U@X[:,0])
        priorities=rng.random(n)
        ahead=np.sum(scores[1:]>scores[0])+np.sum((scores[1:]==scores[0])&(priorities[1:]<priorities[0]))
        direct=float(ahead<math.ceil(.1*n))
        integrated=draw(rng,d,n,.1)['Sign2_15']
        vals.append(direct-sum(integrated)/2)
    se=float(np.std(vals,ddof=1)/math.sqrt(B)); difference=float(np.mean(vals))
    assert abs(difference)<6*se
    return dict(root_checks=root_checks,full_data_check=dict(d=d,n=n,outer=B,
               difference=difference,standard_error=se))

def main():
    start=time.monotonic()
    OUT.mkdir(exist_ok=False)
    cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
    assert (cp/'memory.max').read_text().strip()=='3221225472'
    assert (cp/'memory.swap.max').read_text().strip()=='0'
    files=['ICML_REVIEW_SIM6.md','Z2_RECALL_BRIDGE.md','Z2B_BR_PROOF.md',
           'paper/appendix/C.tex','paper/sections/main_results.tex',
           'paper/sections/setup.tex']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    dump('AA1_design.json',dict(**CONFIG,input_sha256=hashes,
         arithmetic='float64 diagnostics; no outward rounding'))
    dump('AA1_identity_checks.json',checks())
    moments={m:constants(m) for m in CONFIG['methods']}
    limits={m:float(ndtr(a/math.sqrt(q*CONFIG['kappa'])-ndtri(1-CONFIG['rho'])))
            for m,(a,q) in moments.items()}
    rng=np.random.default_rng(CONFIG['seed'])
    n=round(CONFIG['kappa']*CONFIG['d']**2)
    rows=[]
    for rep in range(CONFIG['outer']):
        result=draw(rng,CONFIG['d'],n,CONFIG['rho'])
        row={'rep':rep}
        for m,(lo,hi) in result.items(): row[m+'_lower']=lo; row[m+'_upper']=hi
        rows.append(row)
        if (rep+1)%1024==0:
            print(json.dumps(dict(completed=rep+1,wall_seconds=time.monotonic()-start)),flush=True)
    with (OUT/'AA1_outer_rows.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pairs=[('GD2','Sign2_15'),('Sign2_06','Sign1'),('Sign1','Sign2_15'),('Adam2_e1','Sign1')]
    radius=math.sqrt(2*math.log(2*len(pairs)/CONFIG['joint_failure'])/CONFIG['outer'])
    comparisons=[]; dimensions=[]
    for A,B in pairs:
        lo=np.array([r[A+'_lower']-r[B+'_upper'] for r in rows])
        hi=np.array([r[A+'_upper']-r[B+'_lower'] for r in rows])
        gap=limits[A]-limits[B]
        comparisons.append(dict(A=A,B=B,mean_lower=float(lo.mean()),mean_upper=float(hi.mean()),
            interval_lower=float(lo.mean()-radius),interval_upper=float(hi.mean()+radius),
            hoeffding_radius=radius,limit_gap=gap,invalid_outer=int(np.sum(hi>lo))))
        ca=dimension_bound(A,gap); cb=dimension_bound(B,gap)
        r=min(ca['rate'],cb['rate'])
        threshold=max(ca['admissibility_d0'],cb['admissibility_d0'],
                      (2*(ca['constant_C']+cb['constant_C'])/gap)**(1/r))
        dimensions.append(dict(A=ca,B=cb,limit_recall_gap=gap,
                               sufficient_d_strictly_above=threshold,
                               log10_sufficient_d=math.log10(threshold)))
    assert hashes=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    assert len(os.listdir('/proc/self/task'))<=2
    dump('AA1_results.json',dict(**CONFIG,n=n,limits=limits,comparisons=comparisons,
         dimension_bounds=dimensions,input_hashes_unchanged=True,
         wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),
         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         memory_max=(cp/'memory.max').read_text().strip(),
         memory_swap_max=(cp/'memory.swap.max').read_text().strip(),
         memory_peak=(cp/'memory.peak').read_text().strip(),
         actual_threads=len(os.listdir('/proc/self/task')),
         arithmetic='ordinary float64; numerical endpoints are not interval certificates'))

if __name__=='__main__': main()
