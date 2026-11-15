"""Independent review: raw Gaussian Wick contractions and exact constructions.

Never imports or executes reviewed project code. Outputs are exclusive creations.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    assert 1 <= int(os.environ[key]) <= 2
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import sympy as sp
from scipy.optimize import brentq

START = time.monotonic()
ROOT = Path.cwd().resolve()
assert ROOT == Path("/home/liuyibo/auto-llm-driving/TheoryScout/E_assoc_memory_overcapacity").resolve()
OUT = ROOT / "review11_checks"
INPUTS = [ROOT / "Y1_D3_PROOFS.md", *sorted((ROOT / "y1_checks").glob("*")),
          ROOT / "paper/appendix/D3_finite_d.tex", ROOT / "paper/appendix/A.tex",
          ROOT / "paper/sections/fine_tuning.tex", ROOT / "paper/sections/setup.tex",
          ROOT / "paper/sections/experiments.tex", ROOT / "X2_FINITE_D.md",
          ROOT / "PROOF_REVIEW9.md", ROOT / "W8_FINITE_D.md", ROOT / "ICML_REVIEW_SIM4.md"]
def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in INPUTS if p.is_file()}
BEFORE = hashes()
RESULTS = {}
def zero(name, value):
    ans = sp.factor(value)
    assert ans == 0, (name, ans)
    RESULTS[name] = str(ans)

d, n, kap = sp.symbols("d n kappa", positive=True)
x, y, al = sp.symbols("x y alpha", real=True)
u = 1-al
q = u*u+al*al+2*x*y*u*al
rho = ((sp.S.One, x), (x, sp.S.One))

def pairings(ids):
    if not ids:
        yield ()
        return
    first = ids[0]
    for k in range(1, len(ids)):
        for tail in pairings(ids[1:k]+ids[k+1:]):
            yield ((first, ids[k]),)+tail

def raw_wick(tokens, row_ids, col_ids):
    """Sum raw Isserlis pairings; free coordinate/sample loops give d/n powers.

    A token is (task, coordinate label, sample label). FIX is sample index i,
    not summed. This does not use the two-centered-block trace rule in Y1.
    """
    total = 0
    for matching in pairings(tuple(range(len(tokens)))):
        parents = {("r", r): ("r", r) for r in row_ids}
        parents.update({("c", c): ("c", c) for c in col_ids})
        parents[("c", "FIX")] = ("c", "FIX")
        def find(a):
            while parents[a] != a:
                a = parents[a]
            return a
        def join(a, b):
            parents[find(a)] = find(b)
        weight = 1
        for a, b in matching:
            ta, ra, ca = tokens[a]
            tb, rb, cb = tokens[b]
            weight *= rho[ta][tb]
            join(("r", ra), ("r", rb))
            join(("c", ca), ("c", cb))
        nr = len({find(("r", r)) for r in row_ids})
        nc = len({find(("c", c)) for c in col_ids} - {find(("c", "FIX"))})
        total += weight * d**nr * n**nc
    return sp.expand(total)

def trace(word):
    m = len(word)
    toks = []
    for j, (a, b) in enumerate(word):
        toks += [(a, j, j), (b, (j+1) % m, j)]
    return sp.expand(raw_wick(toks, range(m), range(m)) / (d*n**m))

zero("raw_block_covariance_0011", trace([(0,0),(1,1)])-(1+x*x*(d+1)/n))
zero("raw_block_covariance_1110", trace([(1,1),(1,0)])-x*(1+(d+1)/n))

def query_term(task, other, with_s, tag):
    a, b, c = (tag, "a"), (tag, "b"), (tag, "sample")
    if with_s:
        return [(task,a,"FIX"),(1,a,c),(1,b,c),(other,b,"FIX")], [a,b], [c], 1/(d*n)
    return [(task,a,"FIX"),(other,a,"FIX")], [a], [], 1/d

def query_expect(task, specs):
    total = 0
    for index, (coef, other, with_s) in enumerate(specs):
        toks, rr, cc, norm = query_term(task, other, with_s, index)
        total += coef * norm * raw_wick(toks, rr, cc)
    return sp.expand(total)

def query_square(task, specs):
    total = 0
    for ca, oa, sa in specs:
        ta, ra, ia, na = query_term(task, oa, sa, 0)
        for cb, ob, sb in specs:
            tb, rb, ib, nb = query_term(task, ob, sb, 1)
            total += ca*cb*na*nb*raw_wick(ta+tb, ra+rb, ia+ib)
    return sp.expand(total)

energy_terms = [(1,[(0,0)]), (-al,[(1,1),(0,0)]), (-al,[(0,0),(1,1)]),
                (al*al,[(1,1),(0,0),(1,1)]), (al*al,[(1,1)]),
                (al*y,[(0,1)]), (al*y,[(1,0)]),
                (-al*al*y,[(1,1),(0,1)]), (-al*al*y,[(1,0),(1,1)])]
W = [2+al*(-4+4*x*y-4*x*x)+al*al*(3+y*y-6*x*y+5*x*x-2*x**3*y),
     x*x+y*y+al*(-2-2*y*y+4*x*y-4*x*x)+al*al*(5+y*y-8*x*y+3*x*x)]
Hpop = [u*u+y*y*al*al+2*x*y*u*al, y*y*u*u+al*al+2*x*y*u*al]
apop = [u+x*y*al, x*y*u+al]
chi_num = [y*y*al*al*(1-x*x), y*y*u*u*(1-x*x)]
exact = {}
for k in (0,1):
    specs = [(1,0,False),(-al,0,True),(y*al,1,False)] if k == 0 else [
             (y,0,False),(-y*al,0,True),(al,1,False)]
    et = query_expect(k,specs)
    et2 = query_square(k,specs)
    en = sp.expand(sum(coef*trace([(k,k)]+word) for coef,word in energy_terms))
    hp = sp.expand(en-(1-y*y)*(al*al*trace([(k,k),(1,1)]) if k == 0 else
                    sum(coef*trace([(k,k)]+word) for coef,word in energy_terms[:4])))
    # theta_j^2 = (d E H - E theta_i^2)/(n-1) follows from summing
    # regression coefficients; here both quantities were computed from raw arrays.
    var = sp.expand(en/q + d*et2/(n*q)+d*d*(et2-et*et)/(n*q))
    wrong = sp.expand(en/q+(1+1/d)*(d*hp/q-d*d*et2/(n*q))/(n-1))
    zero(f"FT{k}_exact_mean", et-(apop[k]-(al*x*x if k == 0 else al*x*y)*(d+1)/n))
    zero(f"FT{k}_energy_leading", sp.limit(d*(en.subs(n,kap*d*d)-q), d,sp.oo)-(W[k]-Hpop[k])/kap)
    zero(f"FT{k}_correct_variance_coefficient", sp.limit(d*(var.subs(n,kap*d*d)-1),d,sp.oo)
         -(W[k]-Hpop[k]+3*apop[k]**2+chi_num[k])/(kap*q))
    zero(f"FT{k}_wrong_second_moment_coefficient", sp.limit(d*(wrong.subs(n,kap*d*d)-1),d,sp.oo)-W[k]/(kap*q))
    exact[str(k)] = {"query_mean":str(sp.factor(et)), "query_second_moment":str(sp.factor(et2)),
                     "energy":str(sp.factor(en)), "correct_variance":str(sp.factor(var))}

# Re-expand the exact F,N construction, introducing eps=d^(-1/2).
eps = sp.symbols("eps", positive=True)
mu, aa, bb, ee, gg, hh, ttau, zz = sp.symbols("mu a b e g h tau z", real=True)
nu = mu+eps*ee+eps**2*gg
Ay = 1+eps*aa
By = 1+eps*bb+eps**2*(zz*zz-1)
td = sp.sqrt(1-nu*nu*eps**2)
F = nu*Ay+td*sp.sqrt(Ay)*zz
N = nu*nu*eps**2*Ay+2*nu*td*eps**2*sp.sqrt(Ay)*zz+td*td*By
radius = sp.series(N+hh/kap*eps**2,eps,0,3).removeO()
ratio = sp.series(sp.series(F,eps,0,3).removeO()/sp.sqrt(radius),eps,0,3).removeO().expand()
s1, s2 = ratio.coeff(eps,1), ratio.coeff(eps,2)
def centered_mean(poly):
    ans = 0
    moments = [{0:1,1:0,2:2},{0:1,1:0,2:2},{0:1,1:0,2:ttau}]
    for powers,coef in sp.Poly(sp.expand(poly),aa,bb,ee).terms():
        for i,power in enumerate(powers):
            coef *= moments[i][power]
        ans += coef
    return sp.expand(ans)
drift, diffusion = centered_mean(s2), centered_mean(s1*s1)
root_mean = centered_mean(-s2+s1*sp.diff(s1,zz))
rec = sp.expand(-root_mean+zz*diffusion/2).subs(ttau,mu*mu/2+sp.Symbol("chi"))
zero("exact_expansion_recall", rec-(gg-mu/4-mu*hh/(2*kap)+zz*(sp.Symbol("chi")-hh/kap)/2))
poly = sp.Poly(sp.expand(-drift+(mu+zz)*diffusion/2),zz)
def gauss_moment(power):
    return sum(sp.binomial(power,2*j)*sp.factorial2(2*j-1)*(sp.Rational(1,2))**j
               *(-mu/2)**(power-2*j) for j in range(power//2+1))
rank = sum(coef*gauss_moment(power[0]) for power,coef in poly.terms())
zero("exact_expansion_rank", rank.subs(ttau,mu*mu/2+sp.Symbol("chi"))
     -(-gg+mu*(1+hh/kap+sp.Symbol("chi"))/4))

# A pathwise check of the joint representation, rather than moment matching.
rng = np.random.default_rng(11062026)
max_errors = np.zeros(4)
for dim, sample_n in ((3,9),(7,19),(19,71)):
    for hval, nuval in ((0.,0.),(.35,.4),(1.,.9),(.7,-.8)):
        for rep in range(25):
            gv,wv,ev = rng.normal(size=(3,dim))
            Ayv = gv@gv/dim
            zv = gv@wv/np.linalg.norm(gv)
            Cyv = wv@wv-zv*zv
            tdv = np.sqrt(1-nuval*nuval/dim)
            sv = np.sqrt((dim*hval-nuval*nuval)/(dim-nuval*nuval))
            vv = nuval/np.sqrt(dim)*gv+tdv*wv
            yc = gv/np.sqrt(dim)
            ys = (sv*wv+np.sqrt(1-sv*sv)*ev)/np.sqrt(dim)
            rest = rng.normal(size=(sample_n-2,dim))/np.sqrt(dim)
            ds = ys@vv
            dr = rest@vv
            vraw = (ds*ds+dr@dr)/(sample_n-1)
            Nv = vv@vv/dim
            zstar = ev@vv/np.linalg.norm(vv)
            cnv = (dr@dr)/Nv
            byv = (Cyv+zv*zv)/dim
            ff = nuval*Ayv+tdv*np.sqrt(Ayv)*zv
            nn = nuval*nuval/dim*Ayv+2*nuval*tdv/dim*np.sqrt(Ayv)*zv+tdv*tdv*byv
            ww = sv*(nuval*np.sqrt(Ayv)*zv/np.sqrt(dim)+np.sqrt(dim-nuval*nuval)*byv)+np.sqrt(1-sv*sv)*np.sqrt(nn)*zstar
            max_errors = np.maximum(max_errors,np.abs([ff-yc@vv,nn-Nv,ww-ds,(ww*ww+nn*cnv)/(sample_n-1)-vraw]))
assert max(max_errors) < 1e-12

# The random matching root is not defined on all Gaussian data without fallback.
# Exact excess-loss ratio for eigenvalues (1,100), weights (.9,.1), T=4.
power = 8
a = (.9/(.1*100))**(1/(power-1))
eta_min = (1+a)/(100+a)
loss_min = .9*(1-eta_min)**power+.1*(1-100*eta_min)**power
assert loss_min > .75
root_obstruction = {"T":4,"eigenvalues":[1,100],"weights":[.9,.1],
                    "global_minimizer":eta_min,"global_minimum":loss_min,
                    "target":.75,"matching_root_exists":False,
                    "reason":"Strictly convex sum of even powers; the unique critical point is its global minimum. A neighborhood has positive Gaussian-model probability."}

# Deterministic root expansion at an exact conditional configuration.
root_rows = []
for dim in (64,256,1024,4096):
    kappa, hval, muval = 1., .6, .7
    a0, b0, e0, zstar = .4/np.sqrt(dim), -.3/np.sqrt(dim), .2/np.sqrt(dim), .5
    nuval = muval+e0
    Ayv, Cyv = 1+a0, dim-1+dim*b0
    tdv = np.sqrt(1-nuval*nuval/dim)
    sv = np.sqrt((dim*hval-nuval*nuval)/(dim-nuval*nuval))
    nn_samples = dim*dim
    def rexact(z):
        byv = (Cyv+z*z)/dim
        fv = nuval*Ayv+tdv*np.sqrt(Ayv)*z
        nv = nuval*nuval/dim*Ayv+2*nuval*tdv/dim*np.sqrt(Ayv)*z+tdv*tdv*byv
        wv = sv*(nuval*np.sqrt(Ayv)*z/np.sqrt(dim)+np.sqrt(dim-nuval*nuval)*byv)+np.sqrt(1-sv*sv)*np.sqrt(nv)*zstar
        return fv/np.sqrt((wv*wv+nv*(nn_samples-2))/(nn_samples-1))
    z0 = 1.1
    t0 = z0-muval
    def S(z):
        return e0+(muval+z/2)*a0-(muval+z)*b0/2
    def J(z):
        return (e0*a0-z*a0*a0/8-muval*muval*z/(2*dim)
                -(e0+(muval+z/2)*a0)*b0/2+3*(muval+z)*b0*b0/8
                -(muval+z)*(z*z-1+2*muval*z+hval/kappa)/(2*dim))
    predicted = t0-S(t0)-J(t0)+S(t0)*(a0-b0)/2
    actual = brentq(lambda z:rexact(z)-z0,-5,5,xtol=1e-14)
    root_rows.append({"d":dim,"exact_root":actual,"second_order_root":predicted,
                      "error_times_d_3_2":(actual-predicted)*dim**1.5})

AFTER = hashes()
assert BEFORE == AFTER
cg_rel = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
cg_dir = Path("/sys/fs/cgroup")/cg_rel.lstrip("/")
controls = {name:(cg_dir/name).read_text().strip() for name in ("memory.max","memory.swap.max","memory.peak")}
assert controls["memory.max"] == str(3*1024**3)
assert controls["memory.swap.max"] == "0"
threads = len(list(Path("/proc/self/task").iterdir()))
assert threads <= 2
results = {"symbolic_residuals":RESULTS,"exact_raw_Wick_formulas":exact,
           "joint_representation_max_abs_errors_F_N_W_V":max_errors.tolist(),
           "pathwise_joint_constructions":300,"root_obstruction":root_obstruction,
           "root_remainder_diagnostic":root_rows,
           "inputs_unchanged":BEFORE==AFTER,
           "limits":"Finite algebra, exact identities, and root diagnostic; these do not certify probability remainder bounds.",
           "resources":{"python":sys.executable,"seconds":time.monotonic()-START,
                        "threads":threads,"cgroup":controls,"rss_peak_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
with (OUT/"results.json").open("x") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)
with (OUT/"input_hashes.json").open("x") as f:
    json.dump(BEFORE,f,ensure_ascii=False,indent=2)
print(json.dumps({key:value for key,value in results.items() if key != "exact_raw_Wick_formulas"},ensure_ascii=False,indent=2))
