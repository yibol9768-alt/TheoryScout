"""Only algebra/finite deterministic checks; no probability certification."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    assert os.environ.get(key) == '1', (key, os.environ.get(key))

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import brentq

START = time.monotonic()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'y1b_checks'
EXPECTED = {
    'PROOF_REVIEW11.md': '643574cb088e8e4277c00da5b5d74d19082ba02720646d5f1c676941cf7cdcbd',
    'Y1_D3_PROOFS.md': '6dd4401013ee8baa243bb6f64145603ea710c6daf9d43a3a3154cb5dd1f2f6b9',
    'paper/appendix/D3_finite_d.tex': 'aef2af2125270a5b9d8408af39a2422f827cb2c877721d7fc58c6f0da047db2f',
}

def hashes():
    ans = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in EXPECTED}
    assert ans == EXPECTED, ans
    return ans

initial_hashes = hashes()
checks = {}

def zero(name, value):
    residual = sp.factor(value)
    assert residual == 0, (name, residual)
    checks[name] = str(residual)

# Derive both trace coefficients independently from formal s=1+e series.
e, u, a1, a2 = sp.symbols('e u a1 a2', real=True)
v = 1 - u
a = u + a1 * e + a2 * e**2
b = sp.series((1 - a) / (1 + e), e, 0, 3).removeO()
b1, b2 = -a1 - v, -a2 + a1 + v
zero('b1', b.coeff(e, 1) - b1)
zero('b2', b.coeff(e, 2) - b2)
lo = a1**2 + 2*u*a2 + u**2 + 2*v*b1 + b1**2 + 2*v*b2
ll = a1**2 + 2*u*a2 + u*a1
zero('energy_trace_coefficient', sp.expand(a*a + (1+e)*b*b).coeff(e, 2) + u*u - lo)
zero('loss_trace_coefficient', sp.expand((2+e)*a*a).coeff(e, 2) - 2*ll)
eta = sp.symbols('eta', real=True)
ut = (1-eta)**4
qt = ut**2 + (1-ut)**2
zero('mu_derivative', sp.diff(ut / sp.sqrt(qt), eta)
     - (1-ut)*sp.diff(ut, eta) / qt**sp.Rational(3, 2))

# Spherical fourth moments: sum all raw indices, rather than assume formula.
n_sphere = 3
bb = sp.Matrix(n_sphere, n_sphere,
               lambda i, j: sp.Symbol(f'B{i}{j}', real=True))
raw = 0
for i in range(n_sphere):
    for j in range(n_sphere):
        for k in range(n_sphere):
            for l in range(n_sphere):
                pairing = int(i == j and k == l) + int(i == k and j == l) + int(i == l and j == k)
                raw += bb[i, j] * bb[k, l] * pairing
zero('haar_diagonal_fourth_pairings', raw - (sp.trace(bb)**2
     + sp.trace(bb*bb.T) + sp.trace(bb*bb)))
n, t2, ne2 = sp.symbols('n t2 ne2')
zero('haar_centered_variance', (t2-ne2)/(n*(n-1))
     - (t2-ne2)/(n*(n-1)**2) - (t2-ne2)*(n-2)/(n*(n-1)**2))

# Direct score identities and the block geometric lower bound, including a
# U that explicitly depends on all four data arrays. This is not a tail test.
rng = np.random.default_rng(202611061124)
max_margin_error = 0.
max_variance_error = 0.
min_lower_bound_slack = float('inf')
for repeat in range(40):
    d, nn = 4, 80
    xx0, yy0, xx1, yy1 = [rng.normal(size=(d, nn))/np.sqrt(d) for _ in range(4)]
    sx1 = d/nn * xx1@xx1.T
    z0, z1 = [d/np.sqrt(nn)*yy@xx.T for xx, yy in [(xx0, yy0), (xx1, yy1)]]
    amat = np.linalg.matrix_power(np.eye(d)-.04*sx1, 4)
    bmat = sum(np.linalg.matrix_power(np.eye(d)-.04*sx1, j)*.04 for j in range(4))
    uu = z0@amat + z1@bmat
    score = yy0.T@uu@xx0
    diag = np.diag(score)
    wrong_means = (score.sum(axis=0)-diag)/(nn-1)
    mask = ~np.eye(nn, dtype=bool)
    margin = np.mean(diag-wrong_means)
    variance = np.sum(((score-wrong_means[None, :])**2)[mask])/(nn*(nn-1))
    centered_cross = (yy0-yy0.mean(axis=1, keepdims=True))@xx0.T
    margin2 = np.sum(uu*centered_cross)/(nn-1)
    variance2 = (np.sum(score**2)-np.sum(diag**2))/(nn*(nn-1))-np.mean(wrong_means**2)
    ii, jj = np.arange(nn//2), np.arange(nn//2, nn)
    xblock = xx0[:, ii]@xx0[:, ii].T
    ycent = yy0[:, jj]-yy0[:, jj].mean(axis=1, keepdims=True)
    yblock = ycent@ycent.T
    block_energy = np.trace(uu@xblock@uu.T@yblock)/(nn*(nn-1))
    lower = np.linalg.eigvalsh(xblock)[0]*np.linalg.eigvalsh(yblock)[0]*np.sum(uu**2)/(nn*(nn-1))
    assert variance + 1e-12 >= block_energy >= lower - 1e-12
    max_margin_error = max(max_margin_error, abs(margin-margin2))
    max_variance_error = max(max_variance_error, abs(variance-variance2))
    min_lower_bound_slack = min(min_lower_bound_slack, variance-lower)
assert max_margin_error < 1e-12
assert max_variance_error < 1e-12

# The explicit root interval/spectral tolerance and the model's no-root
# example. Positive eigenvalues imply convex loss, so derivative maximum
# on the interval occurs at the upper endpoint for this check.
T, target = 4, .75
estar = 1-target**(1/(2*T))
eminus, eplus = estar/2, estar*1.5
tol = .5*min(.5, 1/eplus-1, estar/eminus-1, 1-estar/eplus)
lower_derivative = 2*T*(1-tol)*(1-eplus*(1+tol))**(2*T-1)

def select(lambdas, weights):
    loss = lambda rate: np.sum(weights*(1-rate*lambdas)**(2*T))/np.sum(weights)
    derivative = lambda rate: -2*T*np.sum(weights*lambdas*(1-rate*lambdas)**(2*T-1))/np.sum(weights)
    if loss(eminus) > target > loss(eplus) and derivative(eplus) < 0:
        return brentq(lambda rate: loss(rate)-target, eminus, eplus, xtol=1e-15), True
    return estar, False

root_errors = []
for _ in range(40):
    lambdas = rng.uniform(1-tol, 1+tol, 12)
    weights = rng.uniform(.01, 1, 12)
    chosen, good = select(lambdas, weights)
    assert good and eminus < chosen < eplus
    deriv = -2*T*np.sum(weights*lambdas*(1-eplus*lambdas)**(2*T-1))/sum(weights)
    assert -deriv >= lower_derivative
    root_errors.append(abs(sum(weights*(1-chosen*lambdas)**(2*T))/sum(weights)-target))
bad_choice, bad_good = select(np.array([1., 100.]), np.array([.9, .1]))
assert bad_choice == estar and not bad_good
aa = .09**(1/7)
minimizer = (1+aa)/(100+aa)
minimum = .9*(1-minimizer)**8 + .1*(1-100*minimizer)**8
assert minimum > target

# The coefficients reported in D3, recomputed from the polynomials.
ss = sp.symbols('ss')
ap = (1-eta*ss)**4
a1p = sp.diff(ap, ss).subs(ss, 1)
a2p = sp.diff(ap, ss, 2).subs(ss, 1)/2
lop = lo.subs({u: ut, a1: a1p, a2: a2p})
llp = ll.subs({u: ut, a1: a1p, a2: a2p})
mup = ut/sp.sqrt(qt)
wp = (lop+ut**2)/qt
mp = a2p/sp.sqrt(qt)-mup*wp/2
shiftp = -llp/(2*ut*sp.diff(ut, eta))
matchedp = mp + sp.diff(mup, eta)*shiftp
coefficients = {name: float(expr.subs(eta, estar)) for name, expr in
                [('mu', mup), ('w', wp), ('M', mp), ('s_eta', shiftp),
                 ('M_matched', matchedp), ('L_Lambda', llp)]}

cg_rel = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
              if line.startswith('0::'))
cg = Path('/sys/fs/cgroup')/cg_rel.lstrip('/')
controls = {key: (cg/key).read_text().strip() for key in
            ('memory.max', 'memory.swap.max', 'memory.peak')}
assert controls['memory.max'] == str(3*1024**3), controls
assert controls['memory.swap.max'] == '0', controls
threads = len(list(Path('/proc/self/task').iterdir()))
assert threads <= 2, threads
final_hashes = hashes()
result = {
    'scope': 'algebra and deterministic identities only; not probability proofs',
    'symbolic_residuals': checks,
    'deterministic_score_checks': {'repetitions': 40,
        'max_margin_error': max_margin_error, 'max_variance_error': max_variance_error,
        'minimum_variance_minus_lower_bound': min_lower_bound_slack},
    'root_checks': {'good_spectra_repetitions': 40, 'interval': [eminus, eplus],
        'spectral_tolerance': tol, 'derivative_lower_bound': lower_derivative,
        'max_root_residual': max(root_errors), 'no_root_example_minimum': minimum,
        'failure_branch_returns_eta_star': True},
    'D3_coefficients': coefficients,
    'input_hashes_before': initial_hashes, 'input_hashes_after': final_hashes,
    'resources': {'python': sys.executable, 'threads': threads, 'cgroup': str(cg),
        'controls': controls, 'rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'seconds': time.monotonic()-START},
}
with (OUT/'Y1B_results.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
