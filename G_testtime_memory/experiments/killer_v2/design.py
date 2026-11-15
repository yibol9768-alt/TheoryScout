"""V2 design: no fitted parameters; deterministic feedback predictor."""
import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr, roots_genlaguerre, gammaln
from numpy.polynomial.hermite import hermgauss

DIMS = (64, 128, 256)
LOADS = (2, 4, 8)
ALPHAS = (0., 1., 1.5)
FAMILIES = ('bounded', 'raw')
READOUTS = ('tau4', 'tau8', 'tau16', 'full')
BETAS = (.025, .05, .1, .2, .4, .8, 1.2)
METHODS = ('hebb',) + tuple(f'delta_{b:g}' for b in BETAS)
CAL_SEEDS = tuple(range(3001, 3009))
TEST_SEEDS = tuple(range(4001, 4017))
SMOKE_SEED = 93017
EPSILON = .05

def spectrum(d, a, family):
    x = np.arange(1, d+1, dtype=float)
    v = (x/d + EPSILON)**(-a) if family == 'bounded' else x**(-a)
    return v/v.mean()

def query_ids(n):
    m = min(n, 256)
    return np.floor((np.arange(m)+.5)*n/m).astype(int)

def value_dim(n, d, readout):
    return d if readout == 'full' else int(np.ceil(int(readout[3:])*np.log(n)))

def bins(n, ids):
    return np.minimum(3, 4*ids//n)

def feedback(d, load, a, family, method):
    """Exact matrix exponential of finite empirical-spectrum continuum ODE.

    q'=M q+b, r=<lambda,q>; no finite-d or learned correction.
    Query zero-based i corresponds to t=i/d, u=(n-1-i)/d.
    """
    n = d*load
    ids = query_ids(n)
    lam = spectrum(d, a, family)
    if method == 'hebb':
        m = np.ones(len(ids))
        q = np.full(len(ids), 1+load*np.mean(lam**2))
        return m, q, np.full(len(ids), load*np.mean(lam**2))
    beta = float(method.split('_')[1])
    M = np.diag(-2*beta*lam) + beta**2*np.outer(lam, lam)/d
    forcing = beta**2*lam
    # A nonsymmetric M is in fact symmetric for this diagonal Gaussian model.
    # Its eigenvalues are strictly negative for 0<beta<2 and mean(lambda)=1.
    w, vec = np.linalg.eigh(M)
    c = vec.T @ forcing
    def trajectory(times):
        times = np.atleast_1d(times)
        z = np.expm1(times[:, None]*w[None, :])/w[None, :]
        return (z*c[None, :]) @ vec.T
    qt = trajectory(ids/d)
    rk = float(trajectory([load])[0] @ lam/d)
    rt = qt @ lam/d
    u = (n-1-ids)/d
    decay = np.exp(-beta*u[:, None]*lam)
    f = decay @ lam/d
    g = (qt*decay) @ lam/d
    m = beta*f
    Q = rk - 2*beta*f*g + beta**2*(1+rt)*f*f
    assert np.all(Q > 0) and np.all(m*m <= Q+1e-12)
    return m, Q, np.full(len(ids), rk)

def recall(m, Q, n, p, order=64):
    """Finite-p chi target, Gaussian target residual, iid incorrect extremes.

    This is an explicitly falsifiable finite-size closure, not the exact
    correlated-score integral. There is no free SNR constant.
    """
    rho = np.asarray(m)/np.sqrt(Q)
    assert np.all((rho >= 0) & (rho <= 1+1e-12))
    rho = np.minimum(rho, 1.)
    z, wz = hermgauss(order)
    z, wz = z*np.sqrt(2), wz/np.sqrt(np.pi)
    x, wx = roots_genlaguerre(order, p/2-1)
    wx = np.exp(np.log(wx)-gammaln(p/2))
    r = np.sqrt(2*x)
    result = []
    for start in range(0, len(rho), 16):
        h = rho[start:start+16, None, None]
        arg = h*r[None, :, None] + np.sqrt(1-h*h)*z[None, None, :]
        result.extend((ndtr(arg)**(n-1) @ wz) @ wx)
    return np.array(result)

def seed_rng(seed, d, load, a, family, stream):
    return np.random.default_rng(np.random.SeedSequence(
        [seed, d, load, int(2*a), FAMILIES.index(family), stream]))
