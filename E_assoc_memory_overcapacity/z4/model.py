"""Actual one-step model, BR prediction, deterministic finite-d selector and ranks."""
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from common import METHODS

def moments(method):
    # Physical epsilon=e*sqrt(2) stays fixed under variance differentiation.
    v = 2.
    if method == 'GD': return 1., v, 0., 1.
    if method == 'Sign': return math.sqrt(2/math.pi/v), 1., -math.sqrt(2/math.pi/v)/(2*v), 0.
    eps = {'Adam2': 2., 'Adam.1': .1}[method] * math.sqrt(v)
    def integrate(f):
        return 2*quad(lambda z: f(z)*math.exp(-z*z/2)/math.sqrt(2*math.pi), 0, np.inf,
                      epsabs=2e-12, epsrel=2e-12)[0]
    def h(z): return math.sqrt(v)*z/(math.sqrt(v)*z+eps)
    b = integrate(lambda z: math.sqrt(v)*z*h(z))/v
    q = integrate(lambda z: h(z)**2)
    db = integrate(lambda z: math.sqrt(v)*z*h(z)*(z*z-3))/(2*v*v)
    dq = integrate(lambda z: h(z)**2*(z*z-1))/(2*v)
    return b, q, db, dq

MOMENTS = {m: moments(m) for m in METHODS}

def predict(method, alpha, d, task):
    b, Q, db, dQ = MOMENTS[method]
    # r=0, kappa=1/4, sigma^2=2; Z2B (2).
    k = .25
    r0, r1 = (1., 0.) if task == 'old' else (0., 1.)
    q = 1 - 2*alpha*b + alpha*alpha*Q
    a = r0 + alpha*(r1-r0)*b
    mu = a / math.sqrt(k*q)
    m = alpha*((r1-r0)*3*db-r0*b)/(k**1.5*math.sqrt(q))
    ell = (1+2*alpha*(-3*b-3*db)+alpha*alpha*(4*b*b+3*dQ))/(k*q)
    w = ell + mu*mu
    z = float(ndtri(.9))
    B = math.exp(-(z-mu)**2/2)/math.sqrt(2*math.pi)*(m-mu*w/2-mu/2-mu*mu*(z-mu)/2)
    lim = float(ndtr(mu-z))
    return dict(limit=lim, B=B, finite=lim+B/d, mu=mu, m=m, w=w, q=q)

def alpha_grid(method, count=33):
    # First ascending population branch; same fixed grid for every d.
    return np.linspace(0., 1/MOMENTS[method][0], count)

def analytic_alpha(method, d, target, finite=True):
    grid = np.linspace(0., 1/MOMENTS[method][0], 513)
    key = 'finite' if finite else 'limit'
    f = lambda a: predict(method, a, d, 'new')[key]-target
    for lo, hi in zip(grid[:-1], grid[1:]):
        if f(lo) <= 0 <= f(hi): return float(brentq(f, lo, hi, xtol=1e-13))
    return None

def pava(values):
    blocks = []
    for i, v in enumerate(values):
        blocks.append([float(v), 1, i, i+1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop(); a = blocks.pop(); n = a[1]+b[1]
            blocks.append([(a[0]*a[1]+b[0]*b[1])/n, n, a[2], b[3]])
    out = np.empty(len(values))
    for v, _, lo, hi in blocks: out[lo:hi] = v
    return out

def select(grid, raw, target, tolerance=.015):
    fitted = pava(raw)
    discrepancy = float(np.max(np.abs(fitted-raw)))
    if discrepancy > tolerance: return dict(alpha=None, reason='nonmonotonic', discrepancy=discrepancy)
    if target < fitted[0] or target > fitted[-1]:
        return dict(alpha=None, reason='unbracketed', discrepancy=discrepancy)
    j = int(np.searchsorted(fitted, target, side='left'))
    if j == 0 or fitted[j] == target: a = grid[j]
    else: a = grid[j-1]+(grid[j]-grid[j-1])*(target-fitted[j-1])/(fitted[j]-fitted[j-1])
    return dict(alpha=float(a), reason='selected', discrepancy=discrepancy)

def direction(h, method):
    if method == 'GD': return h
    if method == 'Sign': return np.sign(h)
    return h/(np.abs(h)+{'Adam2': 2., 'Adam.1': .1}[method]*math.sqrt(2))

def ranks_many(z0, hdir, alphas, x, y, ids, checkpoint=lambda: None, qc=128, cc=2048):
    """Affine scores shared across alphas; exact full-candidate ranking at each alpha."""
    n = x.shape[1]
    result = np.empty((len(alphas), len(ids)), dtype=np.int64)
    for lo in range(0, len(ids), qc):
        ix = ids[lo:lo+qc]
        b = z0 @ x[:, ix]; v = hdir @ x[:, ix]
        tb = np.einsum('ij,ij->j', y[:, ix], b)
        tv = np.einsum('ij,ij->j', y[:, ix], v)
        counts = np.zeros((len(alphas), len(ix)), dtype=np.int64)
        for c in range(0, n, cc):
            checkpoint()
            jb = y[:, c:c+cc].T @ b; jv = y[:, c:c+cc].T @ v
            js = np.arange(c, min(c+cc,n))[:, None]
            for k, a in enumerate(alphas):
                sc = jb + a*jv; truth = tb+a*tv
                gt = ((sc > truth) | ((sc == truth) & (js < ix))) & (js != ix)
                counts[k] += np.count_nonzero(gt, axis=0)
        result[:, lo:lo+len(ix)] = counts+1
    return result
