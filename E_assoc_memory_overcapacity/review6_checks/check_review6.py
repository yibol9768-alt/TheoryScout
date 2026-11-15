"""Independent W7 audit. No project code imports, no network, no LLM calls.

Run only under the systemd scope documented in README.md. Matrices use
samples as columns. Exact real gradients are recomputed at every step.
"""
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
INPUTS = ['W7_FT_RANK.md', 'THEOREM_M.md',
          'paper/sections/fine_tuning.tex', 'paper/appendix/G.tex']
SEED = 611063027
KAPPA = .5
CONFIGS = [(0., .8), (1., 0.), (1., .5),
           (math.sqrt(.5), math.sqrt(.5)), (1., -.5),
           (math.sqrt(.5), -math.sqrt(.5))]
REPEATS = 16
QUERIES = 160
FRACTIONS = np.array([.01, .05, .1, .25, .5, .75, .9])
PROBS = np.array([.1, .5, .9])


def dump_csv(name, rows):
    with (OUT / name).open('w') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in INPUTS}


def resources():
    # Inspect only the process's cgroup/resource metadata, not another project.
    cg = Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
    folder = Path('/sys/fs/cgroup') / cg.lstrip('/')
    pools = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        if 'openblas' not in line or '/' not in line:
            continue
        libpath = line.split()[-1]
        if libpath in [p['path'] for p in pools]:
            continue
        lib = ctypes.CDLL(libpath)
        count = None
        for sym in ['openblas_get_num_threads64_', 'openblas_get_num_threads_',
                    'openblas_get_num_threads', 'scipy_openblas_get_num_threads',
                    'scipy_openblas_get_num_threads64_']:
            try:
                fn = getattr(lib, sym)
                fn.restype = ctypes.c_int
                count = fn()
                break
            except AttributeError:
                pass
        pools.append({'path': libpath, 'threads': count})
    result = {'cgroup': cg, 'blas_pools': pools,
              'thread_env': {k: os.environ.get(k) for k in
                             ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                              'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']}}
    for key in ['memory.max', 'memory.swap.max']:
        result[key] = (folder / key).read_text().strip()
    assert result['memory.max'] == '2147483648'
    assert result['memory.swap.max'] == '0'
    assert all(p['threads'] is not None and p['threads'] <= 2 for p in pools)
    return result


def normal_integral(fn):
    return quad(lambda z: math.sqrt(2/math.pi)*math.exp(-z*z/2)*fn(z),
                0, np.inf, epsabs=2e-12, epsrel=2e-12)[0]


def direction(e):
    if e is None:
        return 1., 1.
    if e == 0:
        return math.sqrt(2/math.pi), 1.
    return (normal_integral(lambda z: z*z/(z+e)),
            normal_integral(lambda z: z*z/(z+e)**2))


def first_match(r, c, target):
    # Solve the amplitude curve itself, without W7's angle/root formula.
    k = c*math.sqrt((1-r)/2)
    def signal(t):
        return (r+k*t)/math.sqrt(1-2*k*t+t*t)
    den = r+k*k
    upper = k*(1+r)/den if den > 0 else 1.
    while signal(upper) < target:
        upper *= 2
    return brentq(lambda t: signal(t)-target, 0., upper, xtol=1e-14)


def one_step(r, name, target):
    e = {'GD': None, 'Sign': 0., 'Adam_.2': .2, 'Adam_2': 2.}[name]
    aa, qq = direction(e)
    c = aa/math.sqrt(qq)
    sigma = math.sqrt(2*(1-r))
    t = first_match(r, c, target)
    alpha = t/sigma if e is None else t/math.sqrt(qq)
    b = alpha*sigma*sigma if e is None else alpha*sigma*aa
    v = alpha*alpha*sigma*sigma if e is None else alpha*alpha*qq
    q = 1-b+v
    mu = np.array([1-b/2, r+b/2])/math.sqrt(KAPPA*q)
    return {'name': name, 'e': e, 'alpha': alpha, 'sigma': sigma,
            'b': b, 'v': v, 'q': q, 'mu': mu.tolist(), 'c': c}


def increment(z, p):
    if p['e'] is None:
        return p['alpha']*z
    if p['e'] == 0:
        return p['alpha']*np.sign(z)
    return p['alpha']*z/(np.abs(z)+p['e']*p['sigma'])


def ks_normal(values):
    z = np.sort(values)
    cdf = ndtr(z)
    count = len(z)
    return max(np.max(np.arange(1, count+1)/count-cdf),
               np.max(cdf-np.arange(count)/count))


def score_dataset(xs, ys, u, q, queries):
    correct, ranks, ks, wrong, candidate_hits = [], [], [], [], []
    for x, y in zip(xs, ys):
        # All training candidates are retained, including their contribution
        # to both the initial weights and real gradient.
        scores = y.T @ (u @ x[:, queries]) / math.sqrt(q)
        cs = scores[queries, np.arange(len(queries))]
        rk = (np.sum(scores > cs[None, :], axis=0))/(x.shape[1]-1)
        thisks, thiswrong = [], []
        for col in range(8):
            vals = np.delete(scores[:, col], queries[col])
            thisks.append(ks_normal(vals))
            thiswrong.append([np.mean(vals <= s) for s in [-1., 0., 1.]])
        # Distinct candidates also differ from the queried label.
        ids = np.array([j for j in range(3) if j != queries[0]])[:2]
        candidate_hits.append((scores[ids, 0] <= 0).astype(float))
        correct.append(cs)
        ranks.append(rk)
        ks.append(np.mean(thisks))
        wrong.append(np.mean(thiswrong, axis=0))
    return np.array(correct), np.array(ranks), ks, wrong, candidate_hits


def trajectory(z0, z1, s, kind, steps, alpha, epsilon, beta1, beta2):
    u = z0.copy()
    d = np.zeros_like(z0)
    m, v, pm, pv = [np.zeros_like(z0) for _ in range(4)]
    r = z1-z0
    for t in range(1, steps+1):
        g = u@s-z1
        pg = d-r
        if kind == 'Sign':
            u -= alpha*np.sign(g)
            d -= alpha*np.sign(pg)
        elif kind == 'GD':
            u -= alpha*g
            d -= alpha*pg
        else:
            m = beta1*m+(1-beta1)*g
            v = beta2*v+(1-beta2)*g*g
            pm = beta1*pm+(1-beta1)*pg
            pv = beta2*pv+(1-beta2)*pg*pg
            u -= alpha*(m/(1-beta1**t))/(np.sqrt(v/(1-beta2**t))+epsilon)
            d -= alpha*(pm/(1-beta1**t))/(np.sqrt(pv/(1-beta2**t))+epsilon)
    return u, z0+d


def main():
    start = time.perf_counter()
    initial = hashes()
    meta = resources()
    meta.update({'seed': SEED, 'repeats': REPEATS, 'queries': QUERIES,
                 'numpy': np.__version__, 'input_sha256': initial})
    rows, joint, raw, stability, preds = [], [], {}, [], []
    # This intentionally uses different stabilizers and a different target
    # from the author's reported experiment.
    for d in [64, 128]:
        n = round(KAPPA*d*d)
        for config, (rx, ry) in enumerate(CONFIGS):
            r = rx*ry
            target = r+.28*(1-r)
            parameters = [one_step(r, name, target) for name in
                          ['GD', 'Sign', 'Adam_.2', 'Adam_2']]
            for p in parameters:
                preds.append({'d': d, 'rx': rx, 'ry': ry, 'target': target, **p})
            for seed in range(REPEATS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, d, config, seed]))
                x0, xi, y0, zeta = [rng.normal(size=(d, n))/math.sqrt(d) for _ in range(4)]
                x1 = rx*x0+math.sqrt(1-rx*rx)*xi
                y1 = ry*y0+math.sqrt(1-ry*ry)*zeta
                z0 = d/math.sqrt(n)*(y0@x0.T)
                z1 = d/math.sqrt(n)*(y1@x1.T)
                s = d/n*(x1@x1.T)
                residual = z1-z0@s
                queries = rng.choice(n, size=QUERIES, replace=False)
                for p in parameters:
                    u = z0+increment(residual, p)
                    a = z0+increment(z1-z0, p)
                    correct, ranks, ks, wrong, hits = score_dataset(
                        [x0, x1], [y0, y1], u, p['q'], queries)
                    cov = np.cov(correct, ddof=1)
                    corr = np.corrcoef(correct)[0, 1]
                    joint.append({'d': d, 'config': config, 'seed': seed, 'name': p['name'],
                                  'r': r, 'cov': cov[0, 1], 'corr': corr,
                                  'joint_recall': np.mean((ranks[0] <= .1)&(ranks[1] <= .1))})
                    for task in [0, 1]:
                        mu = p['mu'][task]
                        quantiles = ndtr(-mu+ndtri(PROBS))
                        observed = np.quantile(ranks[task], PROBS)
                        row = {'d': d, 'config': config, 'rx': rx, 'ry': ry, 'seed': seed,
                               'name': p['name'], 'task': task, 'mu_theory': mu,
                               'mean': np.mean(correct[task]), 'variance': np.var(correct[task], ddof=1),
                               'wrong_ks': ks[task], 'wrong_cdf_minus1': wrong[task][0],
                               'wrong_cdf_zero': wrong[task][1], 'wrong_cdf_plus1': wrong[task][2],
                               'rank_cdf_maxerr': np.max(np.abs(
                                   np.mean(ranks[task, :, None] <= FRACTIONS, axis=0)-ndtr(mu+ndtri(FRACTIONS)))),
                               'recall10': np.mean(ranks[task] <= .1),
                               'recall10_theory': ndtr(mu+ndtri(.1)),
                               'quantile10': observed[0], 'quantile10_theory': quantiles[0],
                               'quantile50': observed[1], 'quantile50_theory': quantiles[1],
                               'quantile90': observed[2], 'quantile90_theory': quantiles[2],
                               'wrong_hit1': hits[task][0], 'wrong_hit2': hits[task][1]}
                        rows.append(row)
                    key = f'{d}_{config}_{p["name"]}'
                    raw.setdefault(key, {'correct': [], 'ranks': []})
                    raw[key]['correct'].append(correct)
                    raw[key]['ranks'].append(ranks)
                    if seed < 4:
                        err = u-a
                        stability.append({'d': d, 'config': config, 'seed': seed,
                                          'name': p['name'], 'steps': 1,
                                          'beta1': '', 'beta2': '',
                                          'fro_mse': np.mean(err*err),
                                          'correct_mse': np.mean(np.vstack([
                                              np.sum(y0*(err@x0), axis=0),
                                              np.sum(y1*(err@x1), axis=0)])**2)})
                if seed < 4 and config in [1, 4]:
                    for name, alpha, eps, b1, b2 in [
                            ('GD', .25, .2, 0., 0.), ('Sign', .17, .2, 0., 0.),
                            ('Adam_history', .15, .2, .9, .99),
                            ('Adam_beta2zero', .15, .2, .9, 0.)]:
                        u, a = trajectory(z0, z1, s, name, 4, alpha, eps, b1, b2)
                        err = u-a
                        stability.append({'d': d, 'config': config, 'seed': seed,
                                          'name': name, 'steps': 4, 'beta1': b1, 'beta2': b2,
                                          'fro_mse': np.mean(err*err),
                                          'correct_mse': np.mean(np.vstack([
                                              np.sum(y0*(err@x0), axis=0),
                                              np.sum(y1*(err@x1), axis=0)])**2)})
            print(f'completed d={d}, rx={rx:.4g}, ry={ry:.4g}', flush=True)
    dump_csv('per_dataset.csv', rows)
    dump_csv('joint_scores.csv', joint)
    dump_csv('stability.csv', stability)
    arrays = {}
    for key, item in raw.items():
        for field, val in item.items():
            arrays[key+'_'+field] = np.stack(val)
    np.savez_compressed(OUT/'query_samples.npz', **arrays)
    (OUT/'predictions.json').write_text(json.dumps(preds, indent=2))
    meta.update({'elapsed_seconds': time.perf_counter()-start,
                 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 'input_sha256_after': hashes()})
    assert initial == meta['input_sha256_after']
    (OUT/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ['elapsed_seconds', 'peak_rss_mib']}, indent=2))


if __name__ == '__main__':
    main()
