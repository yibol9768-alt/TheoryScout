"""Frozen design and deterministic, one-parameter spectral-filter predictor."""
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.hermite import hermgauss

DIMS = (64, 128, 256)
LOADS = (2, 4, 8)
ALPHAS = (0.0, 0.5, 1.0, 1.5)
BETAS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2)
CAL_SEEDS = tuple(range(1001, 1009))
TEST_SEEDS = tuple(range(2001, 2013))
SMOKE_SEED = 9001
METHODS = ('hebb',) + tuple(f'delta_{b:g}' for b in BETAS)
Z, W = hermgauss(96)
Z, W = Z * np.sqrt(2), W / np.sqrt(np.pi)

def spectrum(d, alpha):
    lam = np.arange(1, d + 1, dtype=float) ** (-alpha)
    return d * lam / lam.sum()

def query_ids(n):
    m = min(n, 256)
    return np.floor((np.arange(m) + 0.5) * n / m).astype(int)

def snr(d, load, alpha, method):
    n = d * load
    lam = spectrum(d, alpha)
    age = n - 1 - query_ids(n)
    if method == 'hebb':
        m = np.ones(len(age))
        noise = np.full(len(age), (n - 1) * np.sum(lam**2) / d**2)
    else:
        beta = float(method.split('_')[1])
        h = 1 - beta * lam / d
        h_age = h[None, :] ** age[:, None]
        m = beta * (h_age @ lam) / d
        geom = -np.expm1(2 * n * np.log(h)) / (1 - h**2)
        noise = beta**2 / d**2 * ((geom[None, :] - h_age**2) @ (lam**2))
    return np.sqrt(d) * m / np.sqrt(m**2 + noise)

def predict(mu, n, c=1.0):
    f = ndtr(c * mu[:, None] + Z[None, :])
    return float((np.power(f, n - 1) @ W).mean())

def prediction(d, load, alpha, method, c=1.0):
    return predict(snr(d, load, alpha, method), d * load, c)
