"""Independent review calculations. Never imports audited project code.

Run from the project root with the scope and environment in README.md.
Only output files under review9_checks are created; existing outputs are refused.
"""
import os
assert all(os.environ.get(s) == '1' for s in
           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
import itertools
import json
import math
import resource
import sys
import time
import hashlib
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
import sympy as sp
from scipy import integrate, special, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
START = time.monotonic()
INPUTS = [ROOT / s for s in ('X2_FINITE_D.md', 'W8_FINITE_D.md',
                            'PROOF_REVIEW7.md', 'W4_RESULTS.md')]
INPUTS += sorted((ROOT / 'x2_checks').rglob('*'))
HASHES = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in INPUTS if p.is_file()}
report = {}

def save(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

@lru_cache(None)
def pairings(length):
    def generate(seq):
        if not seq:
            yield ()
        else:
            for j in range(1, len(seq)):
                for tail in generate(seq[1:j] + seq[j+1:]):
                    yield ((seq[0], seq[j]),) + tail
    return tuple(generate(tuple(range(length))))

def components(vertices, edges, fixed=False):
    """Connected component count by graph traversal, including fixed labels."""
    neighbors = {v: set() for v in vertices}
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    remaining = set(vertices)
    free = 0
    while remaining:
        todo = [remaining.pop()]
        group = set(todo)
        while todo:
            for w in neighbors[todo.pop()]:
                if w in remaining:
                    remaining.remove(w)
                    group.add(w)
                    todo.append(w)
        forced = group & {'QUERY_I', 'QUERY_J'} if fixed else set()
        if len(forced) > 1:
            return None
        free += not bool(forced)
    return free

def score_terms(task, wrong, copy):
    # Tokens are (Gaussian species, task, coordinate index, sample index).
    a, b, c, l, m = [f'{s}{copy}' for s in 'abclm']
    i, j = 'QUERY_I', 'QUERY_J' if wrong else 'QUERY_I'
    outer = [('Y', task, a, j), ('X', task, b, i)]
    return [
        (1, 0, 0, outer + [('Y', 0, a, l), ('X', 0, b, l)]),
        (-1, 1, 1, outer + [('Y', 0, a, l), ('X', 0, c, l),
                           ('X', 1, c, m), ('X', 1, b, m)]),
        (1, 1, 0, outer + [('Y', 1, a, l), ('X', 1, b, l)])]

def exact_ft(task, power, wrong):
    # Integer Laurent polynomial indexed by (d,n,alpha,rx,ry) powers.
    total = Counter()
    choices = [score_terms(task, wrong, z) for z in range(power)]
    for terms in itertools.product(*choices):
        tokens = sum((t[3] for t in terms), [])
        species = [[t for t in tokens if t[0] == s] for s in ('X', 'Y')]
        coords, samples = {t[2] for t in tokens}, {t[3] for t in tokens}
        pref = math.prod(t[0] for t in terms)
        ap = sum(t[1] for t in terms)
        npower = -sum(t[2] for t in terms)
        for px, py in itertools.product(pairings(len(species[0])), pairings(len(species[1]))):
            ce, se, cross = [], [], []
            for ts, pairing in zip(species, (px, py)):
                cross.append(sum(ts[u][1] != ts[v][1] for u, v in pairing))
                ce.extend((ts[u][2], ts[v][2]) for u, v in pairing)
                se.extend((ts[u][3], ts[v][3]) for u, v in pairing)
            nn = components(samples, se, True)
            if nn is not None:
                dd = components(coords, ce)
                total[dd - power, nn + npower, ap, *cross] += pref
    return {k: v for k, v in total.items() if v}

alpha, x, y = sp.symbols('alpha x y', real=True)
kap = sp.symbols('kappa', positive=True)
q = (1-alpha)**2 + alpha**2 + 2*x*y*(1-alpha)*alpha

def asymptotic(poly, power, degree):
    return sp.expand(sum(c * kap**sp.Rational(2*nn-power, 2) * alpha**aa * x**xx * y**yy
                         for (dd, nn, aa, xx, yy), c in poly.items()
                         if dd + 2*nn - power == degree))

def evaluate(poly, power, d, n, a, rx, ry):
    return math.fsum(c * d**dd * n**(nn-power/2) * a**aa * rx**xx * ry**yy
                     for (dd, nn, aa, xx, yy), c in poly.items())

def check_wick():
    polynomials, details = {}, []
    u, r = 1-alpha, x*y
    ws = [2+alpha*(-4+4*r-4*x*x)+alpha**2*(3+y*y-6*r+5*x*x-2*x**3*y),
          x*x+y*y+alpha*(-2-2*y*y+4*r-4*x*x)+alpha**2*(5+y*y-8*r+3*x*x)]
    hs = [u*u+y*y*alpha*alpha+2*r*u*alpha,
          y*y*u*u+alpha*alpha+2*r*u*alpha]
    for task in (0, 1):
        mean = exact_ft(task, 1, False)
        second = exact_ft(task, 2, False)
        wrong = exact_ft(task, 2, True)
        polynomials[task] = (mean, second, wrong)
        aa = [u+r*alpha, r*u+alpha][task]
        m = -alpha * [x*x, r][task] / kap**sp.Rational(3, 2)
        chi_raw = y*y*[alpha**2, u*u][task]*(1-x*x)/kap
        residuals = [
            sp.simplify(asymptotic(mean, 1, 0)-aa/sp.sqrt(kap)),
            sp.simplify(asymptotic(mean, 1, -1)-m),
            sp.simplify(asymptotic(wrong, 2, 0)-q),
            sp.simplify(asymptotic(wrong, 2, -1)-ws[task]/kap),
            sp.simplify(asymptotic(second, 2, 0)-q-aa*aa/kap),
            sp.simplify(asymptotic(second, 2, -1)-2*aa/sp.sqrt(kap)*m
                        -(ws[task]-hs[task]+3*aa*aa)/kap-chi_raw)]
        assert all(v == 0 for v in residuals), residuals
        details.append({'task': task, 'symbolic_residuals': list(map(str, residuals)),
                        'monomials': list(map(len, (mean, second, wrong)))})
    # Compare data, never executing audited code.
    target = json.loads((ROOT/'x2_checks/verified/ft1_exact.json').read_text())
    errors = []
    for row in target:
        d, n, a, rx, ry = row['d'], row['d']**2, row['alpha'], row['rx'], row['ry']
        qq = float(q.subs({alpha:a, x:rx, y:ry}))
        pm, ps, pw = polynomials[row['k']]
        mm = evaluate(pm, 1, d, n, a, rx, ry)/math.sqrt(qq)
        vv = evaluate(ps, 2, d, n, a, rx, ry)/qq-mm*mm
        ww = evaluate(pw, 2, d, n, a, rx, ry)/qq
        errors += [abs(mm-row['mean']), abs(vv-row['variance']), abs(ww-row['wrong'])]
    report['wick'] = {'symbolic':details, 'maximum_table_error':max(errors)}
    save('independent_ft_polynomials.json', {str(k):{
        s:[{'powers':list(kk), 'integer':v} for kk,v in p.items()]
        for s,p in zip(('mean','second','wrong'), pp)} for k,pp in polynomials.items()})
    return polynomials

def nonlinearity(e):
    if e == 0:
        return lambda z: np.sign(z)
    return lambda z: z/(np.abs(z)+e)

def nonlinear_integral(d, n, e):
    # Adaptive integration in standardized chi-square coordinates, independent
    # of the audited Gamma Jacobi matrix rule. Tails beyond 13 SD are negligible
    # at the reported d >= 64; no interval-certificate claim is made.
    h = nonlinearity(e)
    norm = math.sqrt(2/math.pi)
    def inner(v):
        av = integrate.quad(lambda z:norm*math.exp(-z*z/2)*z*h(math.sqrt(v)*z),
                            0, np.inf, epsabs=2e-12)[0]
        qv = integrate.quad(lambda z:norm*math.exp(-z*z/2)*h(math.sqrt(v)*z)**2,
                            0, np.inf, epsabs=2e-12)[0]
        jv = integrate.quad(lambda z:norm*math.exp(-z*z/2)*z*z*h(math.sqrt(v)*z)**2,
                            0, np.inf, epsabs=2e-12)[0]
        return av, qv, jv
    av, qq, _ = inner(1.)
    sigma = math.sqrt(2/n)
    def radial(t):
        v = 1+sigma*t
        a, b, c = inner(v)
        return np.array([math.sqrt(v)*a/math.sqrt(n), v*a*a/(n+2),
                         v*(3*c+(n-1)*b)/(n+2),
                         v*(c+(n+1)*b)/(n+2), v*(c-b)/(n+2)]) * stats.chi2.pdf(n*v, n)*n*sigma
    vals, err = integrate.quad_vec(radial, max(-13., -1/sigma+1e-9), 13., epsabs=2e-11)
    L, B, C, Cw, Ccross = vals
    return {'d':d,'e':e,'a':av,'q':qq,'mean':d*L/math.sqrt(qq),
            'variance':(C+6*(d-1)*B-(2*d-1)*L*L)/qq,
            'wrong_variance':(Cw+2*(d-1)*B)/qq,
            'covariance':(Ccross+2*(d-1)*B-(2*d-1)*L*L)/qq,
            'sample_variance':(C-Ccross+4*(d-1)*B)/qq,
            'quad_vec_error':float(err)}

def check_nonlinear():
    rows = [nonlinear_integral(d,d*d,e) for d in (64,128) for e in (0.,.1)]
    targets = json.loads((ROOT/'x2_checks/verified/nonlinear_exact.json').read_text())
    errors = []
    for row in rows:
        target = next(r for r in targets if r['d']==row['d'] and r['e']==row['e'])
        errors.extend(abs(row[s]-target[s]) for s in ('mean','variance','sample_variance','wrong_variance'))
    report['nonlinear']={'rows':rows,'maximum_table_error':max(errors)}

def check_delta():
    z, mu, g, c, chi = sp.symbols('z mu g c chi', real=True)
    tau = mu*mu/2+chi
    b = g+3*mu/4+z/2-mu*mu*z/2-(mu+z)*(z*z-1+c+2*mu*z)/2
    v = tau+2*(mu+z/2)**2+(mu+z)**2/2
    rec = sp.expand(b-(sp.diff(v,z)-z*v)/2)
    residual_rec = sp.simplify(rec-(g-mu/4-mu*c/2+z*(chi-c)/2))
    poly = sp.Poly(sp.expand(-b+(mu+z)*v/2),z)
    # Product of normal densities: conditional z has N(-mu/2,1/2).
    def moment(k):
        return sum(sp.binomial(k,2*j)*(-mu/2)**(k-2*j)*sp.factorial2(2*j-1)/2**j
                   for j in range(k//2+1))
    rank = sp.expand(sum(coeff*moment(powers[0]) for powers,coeff in poly.terms()))
    residual_rank = sp.simplify(rank-(-g+mu*(1+c+chi)/4))
    assert residual_rec == 0 and residual_rank == 0
    # Independently derive the second-order coefficients of D/sqrt(V).
    eps, ax, by, nz = sp.symbols('eps ax by nz', real=True)
    # A=1+eps ax, C/d=1+eps by-eps^2; B=C/d+z^2/d.
    D=(mu+eps*nz+eps**2*g)*(1+eps*ax)+sp.sqrt(1-(mu+eps*nz)**2*eps**2)*sp.sqrt(1+eps*ax)*z
    V=1+eps*by+eps**2*(z*z-1+2*mu*z+c)
    series=sp.series(D/sp.sqrt(V),eps,0,3).removeO().expand()
    drift=sp.expand(series.coeff(eps,2)).subs({ax**2:2,by**2:2,nz**2:tau})
    drift=sp.Poly(sp.expand(drift),ax,by,nz).coeff_monomial(1)
    diffusion=sp.expand(series.coeff(eps,1)**2).subs({ax**2:2,by**2:2,nz**2:tau})
    diffusion=sp.Poly(sp.expand(diffusion),ax,by,nz).coeff_monomial(1)
    assert sp.simplify(drift-b)==0 and sp.simplify(diffusion-v)==0
    report['delta']={'recall_algebra_residual':str(residual_rec),
                     'rank_algebra_residual':str(residual_rank),
                     'independently_expanded_drift_residual':str(sp.simplify(drift-b)),
                     'independently_expanded_diffusion_residual':str(sp.simplify(diffusion-v))}

def check_ft4_stats():
    # Different route: Taylor coefficients of scalar spectral functions and
    # independent-Wishart identity, with automatic symbolic derivatives.
    s, eta = sp.symbols('s eta',real=True)
    aa=(1-eta*s)**4
    bb=sp.cancel((1-aa)/s)
    u=aa.subs(s,1); v=bb.subs(s,1); qq=u*u+v*v
    coeff=lambda f:sp.diff(f,s,2).subs(s,1)/2
    Lo=coeff(aa*aa+s*bb*bb)+u*u
    wo=Lo/qq+u*u/qq
    mu=u/sp.sqrt(qq)
    M=coeff(aa)/sp.sqrt(qq)-mu*wo/2
    # Weight expectation in loss numerator: (s+1) a(s)^2; denominator 2.
    loss1=coeff((s+1)*aa*aa)/2
    shift=-loss1/sp.diff(u*u,eta)
    matched=M+sp.diff(mu,eta)*shift
    e=1-.75**(1/8)
    numeric={k:float(f.subs(eta,e)) for k,f in
             [('mu',mu),('fixed',M),('matched',matched),('wrong',wo),('loss',loss1),('shift',shift)]}
    raw=[json.loads(t) for t in (ROOT/'x2_checks/scatter_raw.jsonl').read_text().splitlines()]
    rows=[]
    for d in (64,128,256,512):
        paired=[]
        for mode in ('fixed','matched'):
            rr=sorted((r for r in raw if r['d']==d and r['method']=='FT4' and r['mode']==mode),key=lambda r:r['seed'])
            vv=np.array([r['msr'] for r in rr]); paired.append(vv)
            pred=numeric['mu']+numeric[mode]/d
            se=vv.std(ddof=1)/math.sqrt(len(vv)); tt=(vv.mean()-pred)/se
            rows.append({'d':d,'mode':mode,'n':len(vv),'bias':vv.mean()-numeric['mu'],
                         'se':se,'residual_t':tt,'nominal_two_sided_p':2*stats.t.sf(abs(tt),len(vv)-1),
                         'bonferroni_8_halfwidth':stats.t.ppf(1-.05/16,len(vv)-1)*se})
        difference=paired[1]-paired[0]
        expected=(numeric['matched']-numeric['fixed'])/d
        se=difference.std(ddof=1)/math.sqrt(len(difference))
        report.setdefault('paired_protocol_difference',[]).append({'d':d,'difference':float(difference.mean()),
                      'prediction':expected,'se':float(se),'residual_t':float((difference.mean()-expected)/se),
                      'correlation':float(np.corrcoef(paired)[0,1])})
    # Approximate family tests; the two protocols are strongly correlated.
    family={}
    for mode in ('fixed','matched'):
        test=sum(r['residual_t']**2 for r in rows if r['mode']==mode)
        family[mode]={'sum_squared_t':test,'chi2_4_reference_p':stats.chi2.sf(test,4)}
    report['FT4']={'independently_derived':numeric,'recomputed_rows':rows,'approximate_family_tests':family}

def direct_raw_checks(polynomials):
    # Direct actual training, including previously untested negative ry and
    # singular cross-task correlation rx=1. No scatter/Wick generated samples.
    rng=np.random.default_rng(910733)
    rows=[]
    for d,n,a,rx,ry in ((3,9,.14,.6,.7),(2,7,.8,1.,-.7),(3,12,.4,.3,-.8)):
        observations=[]
        reps=60000
        for start in range(0,reps,600):
            gs=rng.standard_normal((4,600,d,n))/math.sqrt(d)
            x0,y0,xi,yi=gs
            x1=rx*x0+math.sqrt(1-rx*rx)*xi
            y1=ry*y0+math.sqrt(1-ry*ry)*yi
            transpose=lambda t:np.swapaxes(t,-1,-2)
            z0=d/math.sqrt(n)*(y0@transpose(x0))
            z1=d/math.sqrt(n)*(y1@transpose(x1))
            s1=d/n*(x1@transpose(x1))
            U=z0-a*(z0@s1-z1)
            scores=[transpose(yy)@U@xx for xx,yy in ((x0,y0),(x1,y1))]
            vals=[]
            for score in scores:
                dd=np.diagonal(score,axis1=-2,axis2=-1)
                vals.extend((dd.mean(axis=-1),(dd*dd).mean(axis=-1),
                             ((score*score).sum(axis=(-2,-1))-(dd*dd).sum(axis=-1))/(n*(n-1))))
            Ua=z0/(abs(z0)+.1)
            A=transpose(y0)@Ua@x0
            da=np.diagonal(A,axis1=-2,axis2=-1)
            vals.extend((da.mean(axis=-1),da.var(axis=-1,ddof=1),
                         ((A*A).sum(axis=(-2,-1))-(da*da).sum(axis=-1))/(n*(n-1))))
            observations.append(np.stack(vals,axis=-1))
        data=np.concatenate(observations)
        pred=[]
        for k in (0,1):
            pred += [evaluate(p,power,d,n,a,rx,ry) for p,power in zip(polynomials[k],(1,2,2))]
        nl=nonlinear_integral(d,n,.1)
        pred += [nl['mean']*math.sqrt(nl['q']),nl['sample_variance']*nl['q'],nl['wrong_variance']*nl['q']]
        mm=data.mean(axis=0); se=data.std(axis=0,ddof=1)/math.sqrt(reps)
        rows.append({'d':d,'n':n,'alpha':a,'rx':rx,'ry':ry,'reps':reps,
                     'labels':['old_mean','old_second','old_wrong','new_mean','new_second','new_wrong',
                               'adam_mean','adam_sample_variance','adam_wrong'],
                     'means':mm.tolist(),'predictions':pred,'SE':se.tolist(),'residual_t':((mm-pred)/se).tolist()})
    report['raw_training']=rows

def main():
    polys=check_wick()
    print('Independent symbolic FT1 coefficients verified',flush=True)
    check_delta()
    check_ft4_stats()
    print('Delta algebra and FT4 coefficients/statistics finished',flush=True)
    check_nonlinear()
    print('Independent adaptive nonlinear quadrature finished',flush=True)
    direct_raw_checks(polys)
    report['input_hashes_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in HASHES.items())
    assert report['input_hashes_unchanged']
    save('results.json',report)
    save('input_hashes.json',HASHES)
    cg=Path('/sys/fs/cgroup')/next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')).lstrip('/')
    save('resources.json',{'python':sys.executable,'seconds':time.monotonic()-START,
                         'threads':len(list(Path('/proc/self/task').iterdir())),
                         'ru_maxrss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                         'cgroup':{k:(cg/k).read_text().strip() for k in ('memory.max','memory.swap.max','memory.peak')}})
    print(json.dumps({'FT4':report['FT4'],'raw_training':report['raw_training'],
                      'nonlinear_table_error':report['nonlinear']['maximum_table_error']},indent=2),flush=True)

if __name__=='__main__':
    main()
