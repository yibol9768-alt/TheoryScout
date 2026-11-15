"""Independent raw-array and Haar-orbit checks; no project imports."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    assert os.environ.get(name) == '1'
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'review14_checks'
START = time.monotonic()
INPUTS = ['Y1B_D3_FIXES.md', 'PROOF_REVIEW11.md', 'Y1_D3_PROOFS.md',
          'paper/appendix/D3_finite_d.tex', 'paper/sections/setup.tex',
          'y1b_checks/Y1B_verify.py', 'y1b_checks/Y1B_results.json',
          'y1b_checks/Y1B_README.md', 'X2_FINITE_D.md']
def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}
before = hashes()
rng = np.random.default_rng(1406112026)
T = 4

def cp(k, count):
    return [0. if k == 0 else float(beta.ppf(.025, k, count-k+1)),
            1. if k == count else float(beta.ppf(.975, k+1, count-k))]

def singular_bound(d, n, tolerance):
    # |singular/sqrt(n)-1| <= tolerance is sufficient.
    t = np.sqrt(n)*tolerance-np.sqrt(d)
    return 1. if t <= 0 else float(min(1., 2*np.exp(-t*t/2)))

def model_batch(d, count):
    n = d*d
    g0, h0, g1, h1 = rng.normal(size=(4, count, d, n))
    s1 = g1 @ g1.swapaxes(-1, -2)/n
    z0 = h0 @ g0.swapaxes(-1, -2)/np.sqrt(n)
    z1 = h1 @ g1.swapaxes(-1, -2)/np.sqrt(n)
    r = z0-np.linalg.solve(s1, z1.swapaxes(-1, -2)).swapaxes(-1, -2)
    eigen, vectors = np.linalg.eigh(s1)
    weights = eigen*np.sum((r@vectors)**2, axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    return eigen, weights

root_results = []
for d, count in [(2, 60000), (3, 50000), (4, 40000), (6, 25000),
                 (8, 18000), (12, 10000), (16, 5000)]:
    records = {h: dict(no_root=0, default=0, bad_spectrum=0,
                       implication_violations=0, shifts=[], minima=[])
               for h in (.1, .75, .95)}
    for begin in range(0, count, 256):
        eigen, weights = model_batch(d, min(256, count-begin))
        size = eigen.shape[0]
        # Strict convexity: find the unique GLOBAL minimizer, not a
        # bracket failure or a local-optimizer failure.
        lo = np.zeros(size)
        hi = 1/eigen[:, 0]
        for _ in range(64):
            mid = (lo+hi)/2
            derivative = np.sum(weights*eigen*(1-mid[:, None]*eigen)**7, axis=1)
            lo = np.where(derivative > 0, mid, lo)
            hi = np.where(derivative > 0, hi, mid)
        minimizer = (lo+hi)/2
        minimum = np.sum(weights*(1-minimizer[:, None]*eigen)**8, axis=1)
        for h, rec in records.items():
            star = 1-h**(1/8)
            left, right = star/2, 1.5*star
            s = .5*min(.5, 1/right-1, star/left-1, 1-star/right)
            at_left = np.sum(weights*(1-left*eigen)**8, axis=1)
            at_right = np.sum(weights*(1-right*eigen)**8, axis=1)
            deriv_right = -8*np.sum(weights*eigen*(1-right*eigen)**7, axis=1)
            good = (at_left > h) & (at_right < h) & (deriv_right < 0)
            spectral = np.max(np.abs(eigen-1), axis=1) <= s
            no_root = minimum > h
            rec['no_root'] += int(np.sum(no_root))
            rec['default'] += int(np.sum(~good))
            rec['bad_spectrum'] += int(np.sum(~spectral))
            rec['implication_violations'] += int(np.sum(spectral & ~good))
            # A stronger, weight-free sufficient condition for a GLOBAL root.
            ratio = (eigen[:, -1]-eigen[:, 0])/(eigen[:, -1]+eigen[:, 0])
            assert not np.any(no_root & (ratio**8 <= h-1e-13))
            low = np.full(size, left)
            high = np.full(size, right)
            for _ in range(44):
                middle = (low+high)/2
                val = np.sum(weights*(1-middle[:, None]*eigen)**8, axis=1)
                low = np.where(val > h, middle, low)
                high = np.where(val > h, high, middle)
            chosen = np.where(good, (low+high)/2, star)
            rec['shifts'].extend((chosen-star).tolist())
    for h, rec in records.items():
        star = 1-h**(1/8)
        left, right = star/2, 1.5*star
        s = .5*min(.5, 1/right-1, star/left-1, 1-star/right)
        a_s = np.sqrt(1+s)-1
        kappa = (1+h**(1/8))/(1-h**(1/8))
        b = (np.sqrt(kappa)-1)/(np.sqrt(kappa)+1)
        shift = np.asarray(rec.pop('shifts'))
        rec.pop('minima')
        assert rec['implication_violations'] == 0
        root_results.append(dict(d=d, n=d*d, samples=count, h=h, interval=[left, right],
            spectral_tolerance=s, no_root_count=rec['no_root'],
            no_root_frequency=rec['no_root']/count, no_root_cp95=cp(rec['no_root'], count),
            default_count=rec['default'], default_frequency=rec['default']/count,
            default_cp95=cp(rec['default'], count), bad_spectrum_count=rec['bad_spectrum'],
            spectral_implies_selection_violations=rec['implication_violations'],
            Y1B_explicit_finite_d_bound=singular_bound(d, d*d, a_s),
            stronger_global_no_root_bound=singular_bound(d, d*d, b),
            d_threshold_for_simple_Y1B_exponent=int(np.ceil(4/a_s**2)),
            d_times_mean_root_shift=float(d*shift.mean()),
            root_shift_mean_se=float(shift.std(ddof=1)/np.sqrt(count))))
    print('root d=', d, 'finished', flush=True)

def endpoint(g0, h0, g1, h1, rate):
    d, n = g0.shape
    s = g1@g1.T/n
    z0, z1 = h0@g0.T/np.sqrt(n), h1@g1.T/np.sqrt(n)
    # Actual GD recurrence; do not encode the reviewed a,b polynomials.
    u = z0.copy()
    for _ in range(4):
        u = u-rate*(u@s-z1)
    return u

def actual(g, h, u):
    d, n = g.shape
    score = h.T@u@g/d
    diag = np.diag(score)
    wrong = (score.sum(axis=0)-diag)/(n-1)
    centered = score-wrong[None, :]
    np.fill_diagonal(centered, 0)
    a, v = np.mean(diag-wrong), np.sum(centered**2)/(n*(n-1))
    return a, v, a/np.sqrt(v), score

identity = []
haar = []
for case in range(4):
    d, n = 3, 15
    g0, h0, g1, h1 = rng.normal(size=(4, d, n))
    rate = [0., .035, .2, -.1][case]
    u = endpoint(g0, h0, g1, h1, rate)
    a, v, rho, score = actual(g0, h0, u)
    # Full training, no independence assumption between U and the blocks.
    ii, jj = np.arange(n//2), np.arange(n//2, n)
    xb = g0[:, ii]/np.sqrt(d)
    yb = h0[:, jj]/np.sqrt(d)
    yb -= yb.mean(axis=1, keepdims=True)
    lower = np.linalg.eigvalsh(xb@xb.T)[0]*np.linalg.eigvalsh(yb@yb.T)[0]
    lower *= np.sum(u*u)/(n*(n-1))
    margin_identity = np.sum(u*((h0-h0.mean(axis=1, keepdims=True))@g0.T/d))/(n-1)
    assert v >= lower-1e-12
    assert abs(a-margin_identity) < 1e-12
    identity.append(dict(case=case, variance=v, lower=lower,
                         margin_residual=float(a-margin_identity)))
    # Condition on all row scatters by rotating every array by the same Haar Q.
    m = np.trace(score)
    t2 = np.sum(score*score)
    p2 = np.trace(score@score)
    e2 = (m*m+t2+p2)/(n*(n+2))
    ac, vc = m/n, (t2-n*e2)*(n-2)/(n*(n-1)**2)
    values = []
    for _ in range(10000):
        q, r = np.linalg.qr(rng.normal(size=(n, n)))
        q *= np.sign(np.diag(r))[None, :]
        ar, vr, rr, _ = actual(g0@q, h0@q, u)
        values.append([ar, vr, rr])
    values = np.asarray(values)
    means, ses = values.mean(axis=0), values.std(axis=0, ddof=1)/100
    haar.append(dict(case=case, samples=len(values), analytic_Ac=ac, analytic_Vc=vc,
        empirical_means=means.tolist(), empirical_ses=ses.tolist(),
        z_for_Ac=float((means[0]-ac)/ses[0]), z_for_Vc=float((means[1]-vc)/ses[1]),
        ratio_of_conditional_moments=float(ac/np.sqrt(vc)),
        empirical_conditional_mean_of_ratio=float(means[2])))

cg_rel = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
              if line.startswith('0::'))
cg = Path('/sys/fs/cgroup')/cg_rel.lstrip('/')
controls = {key: (cg/key).read_text().strip() for key in
            ('memory.max', 'memory.swap.max', 'memory.peak')}
assert controls['memory.max'] == str(3*1024**3)
assert controls['memory.swap.max'] == '0'
threads = len(list(Path('/proc/self/task').iterdir()))
assert threads <= 2
after = hashes()
assert before == after
result = dict(scope='Independent raw Gaussian samples, GD recurrence, global convex minimization, Haar checks; not a proof of asymptotic remainders.',
    seed=1406112026, roots=root_results, geometric_identity_checks=identity,
    haar_conditional_checks=haar, hashes_before=before, hashes_after=after,
    resources=dict(python=sys.executable, threads=threads, controls=controls,
                   elapsed_seconds=time.monotonic()-START))
with (OUT/'results.json').open('x') as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
print(json.dumps(result['resources'], indent=2))
