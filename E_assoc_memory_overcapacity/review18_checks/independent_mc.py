"""Independent review 18. No imports from project code; exclusive output files.

Exact expectation Monte Carlo through Gaussian conditioning, with the final
one-dimensional Gaussian integrated analytically. Sphere thresholds are sampled
by actual Gaussian sorting, one independent threshold per outer group. All
uncertainties use the outer group as the independent sampling unit.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ[name] in ('1', '2')
import argparse
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
from scipy.integrate import quad
from scipy.linalg import cholesky
from scipy.special import ndtr, ndtri
from scipy.stats import wishart

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent/os.environ.get('REVIEW18_RUN_SUBDIR','')
METHODS = ('GD', 'Sign', 'Adam.1', 'Adam1')
RS = (0., -.5)
KAPPA = .25
RHO = .1

def write_json(name, obj):
    with (OUT/name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def resources():
    group = Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    base = Path('/sys/fs/cgroup')/group.lstrip('/')
    pools = []
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if 'openblas' in line.lower() and line.split()[-1].startswith('/')}
    for path in sorted(paths):
        lib = ctypes.CDLL(path)
        for symbol in ('scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads_',
                       'openblas_get_num_threads64_', 'openblas_get_num_threads'):
            if hasattr(lib, symbol):
                fun = getattr(lib, symbol)
                fun.restype = ctypes.c_int
                count = fun()
                assert count <= 2
                pools.append({'path': path, 'threads': count})
                break
    result = {'memory_max': (base/'memory.max').read_text().strip(),
              'swap_max': (base/'memory.swap.max').read_text().strip(),
              'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'cpu_seconds': time.process_time(), 'pools': pools}
    result['kernel_threads'] = int(next(s.split(':')[1] for s in
                                        Path('/proc/self/status').read_text().splitlines()
                                        if s.startswith('Threads:')))
    assert result['kernel_threads'] <= 2
    assert result['memory_max'] == '3221225472' and result['swap_max'] == '0'
    return result

def coefficients():
    # Read only the pre-existing amplitudes, recompute every prediction and
    # response integral. Physical epsilon stays fixed in variance derivatives.
    source = list(csv.DictReader((ROOT/'y3_checks/Y3_coefficients.csv').open()))
    ans = {}
    for r in RS:
        variance = 2*(1-r)
        sigma = math.sqrt(variance)
        for method in METHODS:
            row = next(c for c in source if float(c['r']) == r and
                       c['method'] == method and c['task'] == 'old')
            alpha = float(row['amplitude'])
            if method == 'GD':
                beta, dbeta, Q, dQ = 1., 0., variance, 1.
            elif method == 'Sign':
                beta = math.sqrt(2/math.pi/variance)
                dbeta, Q, dQ = -beta/(2*variance), 1., 0.
            else:
                epsilon = (.1 if method == 'Adam.1' else 1.)*sigma
                def response(g):
                    t = sigma*g
                    return t/(abs(t)+epsilon)
                def average(f):
                    return 2*quad(lambda g: f(g)*math.exp(-g*g/2)/math.sqrt(2*math.pi),
                                  0, np.inf, epsabs=2e-13, epsrel=2e-13)[0]
                beta = average(lambda g: g*response(g))/sigma
                dbeta = average(lambda g: (g**3-3*g)*response(g))/(2*sigma**3)
                Q = average(lambda g: response(g)**2)
                dQ = average(lambda g: (g*g-1)*response(g)**2)/(2*variance)
            q = 1-alpha*variance*beta+alpha*alpha*Q
            ell = (1+2*alpha*(beta*(r-3)+(r-1)*(3-2*r)*dbeta)
                   +alpha*alpha*(2*variance*beta*beta+(3-2*r)*dQ))/(KAPPA*q)
            for task, r0, r1 in (('old', 1., r), ('new', r, 1.)):
                a = r0+alpha*(r1-r0)*beta
                mu = a/math.sqrt(KAPPA*q)
                m = alpha*((r1-r0)*(3-2*r)*dbeta-r0*beta)/(KAPPA**1.5*math.sqrt(q))
                z = ndtri(1-RHO)
                br = math.exp(-(mu-z)**2/2)/math.sqrt(2*math.pi)*(m-mu*ell/2-mu/2-z*mu*mu/2)
                ref = next(c for c in source if float(c['r']) == r and
                           c['method'] == method and c['task'] == task)
                item = dict(r=r, method=method, task=task, alpha=alpha, q=q,
                            mu=mu, m=m, ell=ell, BR=br, Rinf=float(ndtr(mu-z)),
                            beta=beta, dbeta=dbeta, Q=Q, dQ=dQ)
                item['max_reference_error'] = max(abs(item[k]-float(ref[col])) for k, col in
                                                 [('q','q'), ('mu','mu'), ('m','m'),
                                                  ('ell','ell'), ('BR','B_recall')])
                ans[(r, method, task)] = item
    return ans

def integrated_probability(lam, leverage, energy, chi, threshold, n):
    # Derive the quadratic in G directly from
    # (n-1)(lam+sqrt(1-h)G)^2 >= T^2(P+G^2+J).
    # Use its discriminant, rather than the printed root formula (8).
    m = n-1
    scale = math.sqrt(1-leverage) if np.ndim(leverage) == 0 else np.sqrt(1-leverage)
    quadratic = (m+threshold**2)*scale**2-threshold**2
    linear = 2*(m+threshold**2)*lam*scale
    constant = (m+threshold**2)*lam**2-threshold**2*(energy+chi)
    good = (quadratic > 0) & (threshold > 0)
    discriminant = np.where(good, linear**2-4*quadratic*constant, 0.)
    assert np.min(discriminant) > -1e-7
    root = (-linear+np.sqrt(np.maximum(0, discriminant)))/(2*np.where(good,quadratic,1.))
    prob = ndtr(-root)
    return np.where(good, prob, 0.), np.where(good, prob, 1.)

def simulate(d, outer, inner, seed):
    start = time.monotonic()
    rng = np.random.default_rng(np.random.SeedSequence([seed, d]))
    params = coefficients()
    n = d*d//4
    keys = list(params)
    # columns: p lower, p upper, A, B, A^2, B^2, AB, R,
    # lambda, (lambda-mu)^2, |lambda-mu|^3, smooth z threshold.
    group = np.empty((outer, len(keys), 12))
    invalid = 0
    for b in range(outer):
        S = wishart.rvs(df=n, scale=np.eye(d)/n, random_state=rng)
        L = cholesky(S, lower=True, check_finite=False)
        N0, NE = rng.standard_normal((2, d, d))
        Z0, ZE = N0@L.T, NE@L.T
        g = rng.standard_normal((d, inner))
        sphere_den = np.sqrt(np.sum(g*g, axis=0)+rng.chisquare(n-d, size=inner))
        v = math.sqrt(n/d)*g/sphere_den
        x = L@v
        leverage = d/n*np.sum(v*v, axis=0)
        oldb = N0@v
        noiseb = NE@v
        # One exact order threshold shared within this outer group. This is
        # permissible: group averages are independent, inner draws need not be.
        normals = rng.standard_normal(n-1)
        radial = np.linalg.norm(normals)/math.sqrt(n-1)
        rank = n-1-math.ceil(RHO*n)
        threshold = np.partition(normals, rank)[rank]/radial
        chi = rng.chisquare(n-d-1, size=inner)
        out_index = 0
        for r in RS:
            N1 = r*N0+math.sqrt(1-r*r)*NE
            Z1 = r*Z0+math.sqrt(1-r*r)*ZE
            residual = Z1-Z0@S
            for method in METHODS:
                cfg = params[(r, method, 'old')]
                if method == 'GD':
                    response = residual
                elif method == 'Sign':
                    response = np.sign(residual)
                else:
                    epsilon = (.1 if method == 'Adam.1' else 1.)*math.sqrt(2*(1-r))
                    response = residual/(np.abs(residual)+epsilon)
                U = Z0+cfg['alpha']*response
                t = U@x
                norm2 = np.sum(t*t, axis=0)
                B = norm2/(d*cfg['q'])
                for task, N, query_projection in (
                        ('old', N0, oldb),
                        ('new', N1, r*oldb+math.sqrt(1-r*r)*noiseb)):
                    cfg = params[(r, method, task)]
                    A = np.sum(query_projection*t, axis=0)/math.sqrt(n*cfg['q'])
                    lam = A/np.sqrt(B)
                    energy = np.sum((N.T@t)**2, axis=0)/norm2
                    lo, hi = integrated_probability(lam, leverage, energy, chi, threshold, n)
                    loz, hiz = integrated_probability(lam, leverage, energy, chi, ndtri(.9), n)
                    invalid += np.count_nonzero(hi-lo)
                    R = energy*B/n
                    group[b, out_index] = [np.mean(s) for s in
                        (lo, hi, A, B, A*A, B*B, A*B, R, lam,
                         (lam-cfg['mu'])**2, np.abs(lam-cfg['mu'])**3, (loz+hiz)/2)]
                    out_index += 1
        if (b+1) % 256 == 0:
            print(json.dumps(dict(d=d, completed=b+1, outer=outer,
                                  wall_seconds=time.monotonic()-start)), flush=True)
    with (OUT/f'mc_d{d}.npz').open('xb') as f:
        np.savez_compressed(f, groups=group, keys=np.array(keys, dtype=str),
                            leverage_mean=d/n, d=d, n=n, seed=seed, inner=inner)
    summary = []
    for i, key in enumerate(keys):
        c = params[key]
        mean = group[:,i].mean(axis=0)
        se = group[:,i,0].std(ddof=1)/math.sqrt(outer)
        summary.append(dict(**c, d=d, outer=outer, inner=inner, mean=float(mean[0]),
                            se=float(se), scaled_bias=float(d*(mean[0]-c['Rinf'])),
                            scaled_se=float(d*se), remainder=float(mean[0]-c['Rinf']-c['BR']/d),
                            threshold_shift=float(np.mean(group[:,i,0]-group[:,i,11])),
                            threshold_shift_se=float(np.std(group[:,i,0]-group[:,i,11],ddof=1)/math.sqrt(outer)),
                            raw_moments=mean.tolist()))
    write_json(f'mc_d{d}.json', dict(d=d, outer=outer, inner=inner, seed=seed,
                                  wall_seconds=time.monotonic()-start, invalid=int(invalid),
                                  results=summary, resources=resources()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, required=True)
    parser.add_argument('--outer', type=int, default=8192)
    parser.add_argument('--inner', type=int, default=64)
    parser.add_argument('--seed', type=int, default=1811062026)
    args = parser.parse_args()
    simulate(args.d, args.outer, args.inner, args.seed)
