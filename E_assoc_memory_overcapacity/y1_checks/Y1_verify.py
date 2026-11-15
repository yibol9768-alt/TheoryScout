"""Finite algebra checks only; probability remainder proofs are in Y1_D3_PROOFS.md."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    assert 1 <= int(os.environ[name]) <= 2, name

import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import sympy as sp

started = time.monotonic()
root = Path.cwd().resolve()
assert root == Path('/home/liuyibo/auto-llm-driving/TheoryScout/E_assoc_memory_overcapacity').resolve()
output = root / 'y1_checks'
inputs = [
    'ICML_REVIEW_SIM4.md', 'X2_FINITE_D.md', 'PROOF_REVIEW9.md',
    'W8_FINITE_D.md', 'paper/appendix/D3_finite_d.tex',
    'paper/appendix/A.tex', 'paper/sections/fine_tuning.tex',
]

def hashes():
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in inputs}

before = hashes()
residuals = {}

def check(name, expression):
    value = sp.factor(sp.simplify(expression))
    assert value == 0, (name, value)
    residuals[name] = str(value)

x, y, alpha = sp.symbols('x y alpha', real=True)
rho = [[sp.Integer(1), x], [x, sp.Integer(1)]]

def trace_delta(word):
    """Lemma 1.4: two centered blocks, not a raw-array Wick enumeration."""
    ans = 0
    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            ai, bi = word[i]
            aj, bj = word[j]
            val = rho[ai][bj] * rho[bi][aj]
            for s, (aa, bb) in enumerate(word):
                if s != i and s != j:
                    val *= rho[aa][bb]
            ans += val
    return sp.expand(ans)

u, r = 1-alpha, x*y
Hs = [u*u+y*y*alpha*alpha+2*r*u*alpha,
      y*y*u*u+alpha*alpha+2*r*u*alpha]
Ws = [2+alpha*(-4+4*r-4*x*x)
      +alpha*alpha*(3+y*y-6*r+5*x*x-2*x**3*y),
      x*x+y*y+alpha*(-2-2*y*y+4*r-4*x*x)
      +alpha*alpha*(5+y*y-8*r+3*x*x)]
Js = [1+alpha*(-2-4*x*x+2*r)
      +alpha*alpha*(2+5*x*x-4*r-2*x**3*y),
      x*x+alpha*(-2-4*x*x+2*r)
      +alpha*alpha*(4+3*x*x-6*r)]
for k in [0, 1]:
    terms = [
        (1, [(0, 0)]),
        (-alpha, [(1, 1), (0, 0)]),
        (-alpha, [(0, 0), (1, 1)]),
        (alpha**2, [(1, 1), (0, 0), (1, 1)]),
        (alpha**2, [(1, 1)]),
        (alpha*y, [(0, 1)]),
        (alpha*y, [(1, 0)]),
        (-alpha**2*y, [(1, 1), (0, 1)]),
        (-alpha**2*y, [(1, 0), (1, 1)]),
    ]
    got = sum(coef*trace_delta([(k, k)]+word) for coef, word in terms)
    check(f'FT{k}_trace_energy', got-Js[k])
    check(f'FT{k}_wrong_W_equals_J_plus_H', Ws[k]-Js[k]-Hs[k])

mu, g, tau, chi, c, z = sp.symbols('mu g tau chi c z', real=True)
v = tau+2*(mu+z/2)**2+(mu+z)**2/2
b = g+3*mu/4+z/2-mu*mu*z/2-(mu+z)*(z*z-1+c+2*mu*z)/2
recall = b-sp.diff(v, z)/2+z*v/2
check('recall_coefficient', recall.subs(tau, mu*mu/2+chi)
      -(g-mu/4-mu*c/2+z*(chi-c)/2))

def normal_moment(j, mean, variance):
    return sp.expand(sum(sp.binomial(j, 2*k)*sp.factorial2(2*k-1)
                         *variance**k*mean**(j-2*k)
                         for k in range(j//2+1)))

integrand = sp.Poly(sp.expand(-b+v*(mu+z)/2), z)
rank = sum(coef*normal_moment(power[0], -mu/2, sp.Rational(1, 2))
           for power, coef in integrand.terms())
check('rank_gaussian_integral', rank.subs(tau, mu*mu/2+chi)
      -(-g+mu*(1+c+chi)/4))

A, B, kap = sp.symbols('A B kappa', real=True)
K = 1+4*A+A*A+2*B
gd_g = mu*(A+B)/kap-mu*K/(2*kap)-mu/4
check('GD_g_simplification', gd_g+mu*(sp.Rational(1, 4)+(1+A)**2/(2*kap)))
check('GD_rank_simplification', (-g+mu*(1+c)/4).subs({g: gd_g, c: 1/kap})
      -mu*(sp.Rational(1, 2)+((1+A)**2/2+sp.Rational(1, 4))/kap))
check('GD_recall_simplification', (g-mu/4-mu*c/2-z*c/2).subs({g: gd_g, c: 1/kap})
      +mu/2+(mu*((1+A)**2+1)+z)/(2*kap))

d, n, nu, hh = sp.symbols('d n nu h', positive=True)
ss = (d*hh-nu*nu)/(d-nu*nu)
ew2 = ss*(nu*nu/d+(d-nu*nu)*(1+2/d))+(1-ss)
check('exact_special_candidate_second_moment', ew2-(1+(d*hh-nu*nu)*(1+1/d)))
check('exact_radius_second_moment', (ew2+n-2)/(n-1)
      -(1+(d*hh-nu*nu)*(1+1/d)/(n-1)))
check('exact_correct_score_second_moment', nu*nu*(1+2/d)+(1-nu*nu/d)
      -(1+nu*nu*(1+1/d)))

# Verify the expectations of the displayed first and second perturbations.
ea, eb, ee = sp.symbols('radial_a radial_b key_e', real=True)
S = ee+(mu+z/2)*ea-(mu+z)*eb/2
J = (ee*ea-z*ea*ea/8-mu*mu*z/(2*d)
     -(ee+(mu+z/2)*ea)*eb/2+3*(mu+z)*eb*eb/8
     -(mu+z)*(z*z-1+2*mu*z+c)/(2*d))

def formal_mean(poly):
    moments = [{0: 1, 1: 0, 2: 2/d},
               {0: 1, 1: 0, 2: 2/d},
               {0: 1, 1: g/d, 2: tau/d}]
    ans = 0
    for powers, coef in sp.Poly(sp.expand(poly), ea, eb, ee).terms():
        for i, power in enumerate(powers):
            coef *= moments[i][power]
        ans += coef
    return sp.expand(ans)

check('drift_from_independent_primitive_moments', formal_mean(S+J)-b/d)
check('diffusion_from_independent_primitive_moments', formal_mean(S*S)-v/d)
check('root_cross_moment', formal_mean(S*sp.diff(S, z))-sp.diff(v, z)/(2*d))

after = hashes()
assert before == after
cg_relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                   if line.startswith('0::'))
cg = Path('/sys/fs/cgroup') / cg_relative.lstrip('/')
controls = {name: (cg / name).read_text().strip()
            for name in ['memory.max', 'memory.swap.max', 'memory.peak']}
assert controls['memory.max'] == str(3*1024**3), controls
assert controls['memory.swap.max'] == '0', controls
threads = len(list(Path('/proc/self/task').iterdir()))
assert threads <= 2, threads
results = {
    'purpose': 'algebra verification only; no probabilistic remainder certification',
    'checks_passed': len(residuals), 'symbolic_residuals': residuals,
    'input_hashes_unchanged': before == after,
    'resources': {'python': sys.executable, 'seconds': time.monotonic()-started,
                  'process_threads': threads, 'cgroup': controls,
                  'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
}
with (output / 'Y1_results.json').open('x') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
with (output / 'Y1_input_hashes.json').open('x') as f:
    json.dump(before, f, indent=2)
print(json.dumps(results, ensure_ascii=False, indent=2))
