"""Independent floating/high precision diagnostics. No imports from w6_checks.

Uses raw displacement (not D/z) and raw residual moment differentiation.
The high precision quadrature is a diagnostic, not an interval proof.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from fractions import Fraction
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
import mpmath as mp

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
assert Path.cwd() == ROOT
INPUTS = [ROOT/p for p in ['W6_ADAM_ROOT.md','w6_checks/adam_certificate.py',
          'w6_checks/verify_certificate.py','w6_checks/certificate.json',
          'w6_checks/verification_final.json','paper/sections/fine_tuning.tex',
          'paper/appendix/G.tex','paper/sections/setup.tex',
          'experiments/ft/FT_run.py','experiments/ft/FT_forecasts.json',
          'paper/figures/make_figures.py']]
BEFORE = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}

def raw(z, a, e, ctx=None):
    if ctx is None:
        c = float
        sqrt = np.sqrt
    else:
        c, sqrt = ctx.mpf, ctx.sqrt
    b, g = c('0.9'), c('0.99')
    d = m = v = da = ma = va = c(0)
    for t in range(1, 5):
        s = d-z
        m, ma = b*m+(1-b)*s, b*ma+(1-b)*da
        v, va = g*v+(1-g)*s*s, g*va+2*(1-g)*s*da
        mh, mah = m/(1-b**t), ma/(1-b**t)
        vh, vah = v/(1-g**t), va/(1-g**t)
        root = sqrt(vh)
        if z == 0:
            return [c(0)]*6
        denom = root+e
        direction = mh/denom
        direction_a = mah/denom - mh*(vah/(2*root))/(denom*denom)
        d, da = d-a*direction, da-direction-a*direction_a
    return [(d-z)**2, 2*(d-z)*da, z*d, d*d, z*da, 2*d*da]

def integrals(a, e):
    # Adaptive vector quadrature over [0,infinity), different from the certificate mesh.
    pref = np.sqrt(2/np.pi)
    val, err = quad_vec(lambda z: np.array(raw(z,a,e))*pref*np.exp(-z*z/2),
                        0, np.inf, epsabs=2e-13, epsrel=2e-13, limit=300)
    return val, float(err)

def high_F(a, e):
    pref = mp.sqrt(2/mp.pi)
    return mp.quad(lambda z: raw(z,a,e,mp)[0]*pref*mp.exp(-z*z/2),
                   [0, mp.mpf('.025'), mp.mpf('.1'), mp.mpf('.4'), 1, 3, 7, 12, mp.inf])

results=[]
for estr in ['.1','1']:
    e=float(estr)
    root=brentq(lambda a: integrals(a,e)[0][0]-.75, 0,.15,xtol=5e-16)
    points=[]
    for a in [0., .025, root, .075, .10, .15]:
        vals,err=integrals(a,e)
        step=1e-5
        fd=(-integrals(a+2*step,e)[0][0]+8*integrals(a+step,e)[0][0]
            -8*integrals(a-step,e)[0][0]+integrals(a-2*step,e)[0][0])/(12*step)
        assert abs(fd-vals[1])<3e-9
        points.append(dict(a=a,F=vals[0],Fprime=vals[1],B=vals[2],V=vals[3],
                           Bprime=vals[4],Vprime=vals[5],finite_difference=fd,quad_error_diagnostic=err))
    # Literal original scan + 24 bisections, evaluating every point independently.
    cache={}
    def evaluate(a):
        if a not in cache: cache[a]=integrals(a,e)[0][0]-.75
        return cache[a]
    prev=0.; fp=evaluate(prev)
    for cur in np.linspace(0,.6/4,33)[1:]:
        fc=evaluate(float(cur))
        if fp>=0 and fc<=0:
            lo,hi=prev,float(cur)
            break
        prev,fp=float(cur),fc
    initial=[lo,hi]
    path=[]
    for k in range(24):
        mid=(lo+hi)/2
        fm=evaluate(mid)
        path.append(dict(k=k+1,a=mid,residual=fm))
        if fm>0: lo=mid
        else: hi=mid
    sorted_cache=sorted(cache,key=lambda a:abs(cache[a]))
    chosen=sorted_cache[0]
    search=dict(initial=initial,final=[lo,hi],path=path,
                chosen=chosen,chosen_fraction=str(Fraction.from_float(chosen)),
                min_abs_residual=min(abs(x) for x in cache.values()),
                cache_gap=abs(cache[sorted_cache[1]])-abs(cache[chosen]),
                width_fraction=str(Fraction.from_float(hi)-Fraction.from_float(lo)))
    mp.mp.dps=45
    ee=mp.mpf(estr)
    # Two Newton steps started from an ordinary double result.
    ar=mp.mpf(root)
    for _ in range(2):
        der=mp.quad(lambda z:raw(z,ar,ee,mp)[1]*mp.sqrt(2/mp.pi)*mp.exp(-z*z/2),
                    [0,mp.mpf('.025'),mp.mpf('.1'),mp.mpf('.4'),1,3,7,12,mp.inf])
        ar -= (high_F(ar,ee)-mp.mpf('.75'))/der
    mp.mp.dps=60
    ee=mp.mpf(estr)
    # Precision change and a fresh independent derivative integral.
    froot=high_F(ar,ee)
    droot=mp.quad(lambda z:raw(z,ar,ee,mp)[1]*mp.sqrt(2/mp.pi)*mp.exp(-z*z/2),
                  [0,mp.mpf('.025'),mp.mpf('.1'),mp.mpf('.4'),1,3,7,12,mp.inf])
    endpoints=['.04819026078294','.04819026078295'] if estr=='.1' else ['.08287875576148','.08287875576149']
    endpoints=[dict(a=x,residual=mp.nstr(high_F(mp.mpf(x),ee)-mp.mpf('.75'),55)) for x in endpoints]
    moment_root=integrals(float(ar),e)[0]
    moment_chosen=integrals(chosen,e)[0]
    errors=[]
    for r in [0.,.5,-.5]:
        def msr(vals):
            b=(2-2*r)*vals[2]; v=(2-2*r)*vals[3]
            return np.array([1-b/2,r+b/2])/np.sqrt(1-b+v)
        errors.append(dict(r=r,MSR_at_root=msr(moment_root).tolist(),
                           actual_MSR_error=(msr(moment_chosen)-msr(moment_root)).tolist()))
    row=dict(e=estr,root=mp.nstr(ar,55),F_at_root_minus_h=mp.nstr(froot-mp.mpf('.75'),30),
             Fprime_at_root=mp.nstr(droot,55),root_endpoints=endpoints,points=points,search=search,msr=errors)
    results.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)

after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS}
assert BEFORE==after
cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('::',1)[1].strip().lstrip('/')
resources={k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.peak','memory.events']}
resources.update(executable=os.sys.executable,threads=next(int(l.split()[1]) for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('Threads:')),
                 maxrss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seconds=time.monotonic()-START)
assert resources['memory.max']=='2147483648' and resources['memory.swap.max']=='0' and resources['threads']<=2
with open('review5_checks/numeric_results.json','x') as f:
    json.dump(dict(results=results,resources=resources,baseline_hashes=BEFORE,existing_files_unchanged=True),f,indent=2)
print('DONE',resources,flush=True)
