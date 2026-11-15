"""Offline CPU checks for FORMULATION_V2.md (no API or downloads).

Run with BLAS thread limits set to 1. The default grid is a smoke test,
not evidence of an asymptotic theorem. Never builds an N x N score array.
"""
import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np


def phi(x):
    return math.exp(-x*x/2) / math.sqrt(2*math.pi)


def cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def adam(target, steps, alpha=0.2, epsilon=0.1, gram=None):
    u = np.zeros_like(target)
    m = np.zeros_like(target)
    v = np.zeros_like(target)
    for t in range(1, steps + 1):
        g = u - target if gram is None else u @ gram - target
        m = 0.9*m + 0.1*g
        v = 0.999*v + 0.001*g*g
        u -= alpha * (m/(1-0.9**t)) / (np.sqrt(v/(1-0.999**t)) + epsilon)
    return u


def retrieve(u, x, y, indices, q, deadline):
    n = x.shape[1]
    vectors = u @ x[:, indices] / math.sqrt(q)
    truth = np.sum(y[:, indices] * vectors, axis=0)
    greater = np.zeros(len(indices), dtype=np.int64)
    cdf_grid = np.arange(-3.0, 3.01, 0.5)
    cdf_counts = np.zeros(len(cdf_grid))
    score_sum = score_squares = 0.0
    for start in range(0, n, 1024):
        if time.monotonic() >= deadline:
            raise TimeoutError('smoke test deadline')
        stop = min(n, start + 1024)
        scores = vectors.T @ y[:, start:stop]
        # Remove self explicitly: different BLAS accumulation can create a
        # spurious self-comparison even though the mathematical scores tie.
        mask = (indices >= start) & (indices < stop)
        scores[np.flatnonzero(mask), indices[mask] - start] = -np.inf
        greater += np.sum(scores > truth[:, None], axis=1)
        for k, point in enumerate(cdf_grid):
            cdf_counts[k] += np.sum(scores <= point)
        scores[np.flatnonzero(mask), indices[mask] - start] = 0.0
        score_sum += float(np.sum(scores))
        score_squares += float(np.sum(scores*scores))
    pairs = len(indices)*(n-1)
    cdf_values = (cdf_counts-len(indices))/pairs
    rank_fractions = greater/(n-1)
    return dict(mean_rank_fraction=float(np.mean(greater/(n-1))),
                top1percent=float(np.mean(greater + 1 <= math.ceil(0.01*n))),
                top10=float(np.mean(greater + 1 <= math.ceil(0.1*n))),
                top50=float(np.mean(greater + 1 <= math.ceil(0.5*n))),
                top1=float(np.mean(greater == 0)),
                true_score_mean=float(np.mean(truth)),
                true_score_sd=float(np.std(truth, ddof=1)),
                wrong_score_mean=score_sum/pairs,
                wrong_score_second_moment=score_squares/pairs,
                wrong_cdf_grid=cdf_grid.tolist(), wrong_cdf_values=cdf_values.tolist(),
                wrong_cdf_grid_max_error=float(np.max(np.abs(cdf_values-[cdf(g) for g in cdf_grid]))),
                query_indices=indices.tolist(), true_scores=truth.tolist(),
                query_rank_fractions=rank_fractions.tolist())


def run():
    # Enforce an address-space cap after imports; launch with one BLAS thread.
    cap = 3_500_000_000  # Strict decimal 3.5 GB, also below 3.5 GiB.
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    start = time.monotonic()
    deadline = start + 300  # This local subset gets five minutes, not an hour.
    rows = []
    alpha = 0.6
    q_sign = alpha**2
    q_two = 8*alpha**2*(1-cdf(alpha))
    c_sign = math.sqrt(2/math.pi)
    c_two = math.sqrt(2)*phi(alpha)/math.sqrt(1-cdf(alpha))
    grid = np.linspace(-9, 9, 120001)
    weight = np.exp(-grid*grid/2)/math.sqrt(2*math.pi)*(grid[1]-grid[0])
    adam_moments = {}
    for steps in (2, 16):
        h = adam(grid, steps)
        q = float(np.sum(weight*h*h))
        c = float(np.sum(weight*grid*h)/math.sqrt(q))
        adam_moments[steps] = (q, c)
    for d in (32, 64, 128):
        for kappa in (0.5, 1.0, 2.0):
            n = round(kappa*d*d)
            for seed in range(5):
                if time.monotonic() >= deadline:
                    raise TimeoutError('smoke test deadline')
                rng = np.random.default_rng(100000*d + 1000*int(2*kappa) + seed)
                x = rng.normal(size=(d, n))/math.sqrt(d)
                y = rng.normal(size=(d, n))/math.sqrt(d)
                gram = (d/n)*(x @ x.T)
                z = (d/math.sqrt(n))*(y @ x.T)
                indices = rng.choice(n, size=min(128, n), replace=False)
                h1 = alpha*np.sign(z)
                h2 = h1 + alpha*np.sign(z-h1)
                actual2 = h1 + alpha*np.sign(z-h1 @ gram)
                left, _, right = np.linalg.svd(z, full_matrices=False)
                cases = [('GD', z, 1.0, 1.0, None),
                         ('polar_baseline', math.sqrt(d)*(left @ right), 1.0,
                          8/(3*math.pi), None),
                         ('Sign_1', h1, q_sign, c_sign, None),
                         ('Sign_2_surrogate', h2, q_two, c_two, None),
                         ('Sign_2_actual', actual2, q_two, c_two, h2)]
                for steps in (2, 16):
                    q, c = adam_moments[steps]
                    proxy = adam(z, steps)
                    actual = adam(z, steps, gram=gram)
                    cases.extend([(f'Adam_{steps}_surrogate', proxy, q, c, None),
                                  (f'Adam_{steps}_actual', actual, q, c, proxy)])
                # A small, prespecified part of phase B checks the reversal.
                if d in (64, 128) and kappa == 1:
                    for step_size in (0.2, 1.2, 1.5):
                        first = step_size*np.sign(z)
                        proxy = first + step_size*np.sign(z-first)
                        actual = first + step_size*np.sign(z-first @ gram)
                        tail = 1-cdf(step_size)
                        q = 8*step_size**2*tail
                        c = math.sqrt(2)*phi(step_size)/math.sqrt(tail)
                        cases.append((f'Sign_2_actual_alpha{step_size}', actual, q, c, proxy))
                for name, u, q, c, proxy in cases:
                    measured = retrieve(u, x, y, indices, q, deadline)
                    rows.append(dict(d=d, n=n, kappa=kappa, seed=seed, method=name,
                                     c=c, q=q,
                                     rank_prediction=cdf(-c/math.sqrt(2*kappa)),
                                     top10_prediction=cdf(c/math.sqrt(kappa)-1.2815515655446004),
                                     score_mean_prediction=c/math.sqrt(kappa),
                                     surrogate_relative_mse=None if proxy is None else
                                     float(np.sum((u-proxy)**2)/(d*d*q)),
                                     sampled_diagonal_score_mse=None if proxy is None else
                                     float(np.mean(np.sum(y[:, indices] * ((u-proxy) @ x[:, indices]),
                                                          axis=0)**2)/q), **measured))
                # Exact mean identities for the two entrywise surrogates.
                variances = np.diag(gram)
                mean1 = mean2 = 0.0
                for idx in indices:
                    xx = x[:, idx]**2
                    mean1 += d/math.sqrt(n)*np.sum(xx * alpha*math.sqrt(2/math.pi)/np.sqrt(variances))
                    mean2 += d/math.sqrt(n)*np.sum(xx * 4*alpha*np.exp(-alpha**2/(2*variances))
                                                  /np.sqrt(2*math.pi*variances))
                for r in rows[-len(cases):]:
                    if r['method'] == 'Sign_1':
                        r['conditional_score_mean'] = mean1/len(indices)/math.sqrt(q_sign)
                    if r['method'] == 'Sign_2_surrogate':
                        r['conditional_score_mean'] = mean2/len(indices)/math.sqrt(q_two)
    summary = []
    for d in (32, 64, 128):
        for kappa in (0.5, 1.0, 2.0):
            for name in sorted({r['method'] for r in rows}):
                selected = [r for r in rows if r['d']==d and r['kappa']==kappa and r['method']==name]
                if not selected:
                    continue
                values = np.array([r['mean_rank_fraction'] for r in selected])
                summary.append(dict(d=d, kappa=kappa, method=name,
                                    rank_mean=float(np.mean(values)),
                                    rank_seed_se=float(np.std(values, ddof=1)/math.sqrt(len(values))),
                                    prediction=selected[0]['rank_prediction'],
                                    top10_mean=float(np.mean([r['top10'] for r in selected])),
                                    score_mean=float(np.mean([r['true_score_mean'] for r in selected])),
                                    wrong_cdf_grid_max_error_mean=float(np.mean([r['wrong_cdf_grid_max_error'] for r in selected])),
                                    sampled_diagonal_score_mse_mean=None if selected[0]['sampled_diagonal_score_mse'] is None else
                                    float(np.mean([r['sampled_diagonal_score_mse'] for r in selected])),
                                    proxy_mse_mean=None if selected[0]['surrogate_relative_mse'] is None else
                                    float(np.mean([r['surrogate_relative_mse'] for r in selected]))))
    output = dict(elapsed_seconds=time.monotonic()-start,
                  max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  address_space_limit_bytes=cap,
                  parameters=dict(sign_alpha=alpha, adam_alpha=0.2, adam_epsilon=0.1,
                                  beta1=0.9, beta2=0.999, queries=128, seeds=list(range(5)),
                                  extra_sign2_alphas=[0.2, 1.2, 1.5],
                                  blas_threads={k: os.environ.get(k) for k in
                                                ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
                  summary=summary, rows=rows)
    Path('smoke_v2_results.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','summary')}, indent=2))
    for row in summary:
        if row['d']==128 and row['kappa']==1:
            print(json.dumps(row))


if __name__ == '__main__':
    run()
