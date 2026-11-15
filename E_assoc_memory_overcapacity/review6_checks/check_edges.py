"""Independent branch, regression and complete-dataset checks."""
import json
import math
import resource
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from check_review6 import (OUT, SEED, KAPPA, resources, direction, first_match,
                           one_step, increment, dump_csv)


def full_ranks(x, y, u, block=128):
    n = x.shape[1]
    values = np.empty(n)
    # Never allocate the n by n matrix.
    for start in range(0, n, block):
        end = min(n, start+block)
        scores = y.T@(u@x[:, start:end])
        diagonal = scores[np.arange(start, end), np.arange(end-start)]
        values[start:end] = np.sum(scores > diagonal, axis=0)/(n-1)
    return values


def branches():
    rows = []
    cs = [direction(e)[0]/math.sqrt(direction(e)[1]) for e in [0., .2, 2.]]+[1.]
    max_formula_error = 0.
    min_gap = np.inf
    tested = 0
    for r in [-.95, -.8, -.5, 0., .5, .95]:
        ks = np.array(cs)*math.sqrt((1-r)/2)
        kk = ks[0]
        b = (1+r)*kk/math.sqrt(1-kk*kk)
        maximum = math.sqrt(r*r+b*b) if r+kk*kk > 0 else kk
        for fraction in [.01, .2, .5, .9, .999]:
            target = r+fraction*(maximum-r)
            olds = []
            for c, k in zip(cs, ks):
                t = first_match(r, c, target)
                q = 1-2*k*t+t*t
                old = (1-k*t)/math.sqrt(q)
                new = (r+k*t)/math.sqrt(q)
                b = (1+r)*k/math.sqrt(1-k*k)
                m2 = r*r+b*b
                closed = (r*target+b*math.sqrt(max(0., m2-target*target)))/m2
                max_formula_error = max(max_formula_error, abs(closed-old), abs(new-target))
                olds.append(old)
                rows.append({'r': r, 'target': target, 'c': c, 'amplitude': t,
                             'old': old, 'new': new})
            min_gap = min(min_gap, min(np.diff(olds)))
            assert np.all(np.diff(olds) > 0)
            tested += 1
    # A concrete counterexample if the first-branch restriction is dropped.
    r, target = .5, .7
    ca, cb = cs[0], cs[2]
    def curve(t, c):
        k = c*math.sqrt((1-r)/2)
        return (r+k*t)/math.sqrt(1-2*k*t+t*t)
    ta = first_match(r, ca, target)
    kb = cb*math.sqrt((1-r)/2)
    peak = kb*(1+r)/(r+kb*kb)
    upper = 2*peak
    while curve(upper, cb) > target:
        upper *= 2
    tb = brentq(lambda t: curve(t, cb)-target, peak, upper)
    def old(t, c):
        k = c*math.sqrt((1-r)/2)
        return (1-k*t)/math.sqrt(1-2*k*t+t*t)
    counterexample = {'r': r, 'u': target, 'Sign_first_old': old(ta, ca),
                      'Adam_e2_later_old': old(tb, cb), 'Adam_later_t': tb}
    # Tangency at the finite Sign maximum, and negative old-signal matching.
    r = -.2
    c = cs[0]
    k = c*math.sqrt((1-r)/2)
    t = k*(1+r)/(r+k*k)
    q = 1-2*k*t+t*t
    tangent = {'r': r, 't_peak': t, 'u_max': (r+k*t)/math.sqrt(q),
               'old_at_peak': (1-k*t)/math.sqrt(q)}
    dump_csv('branch_grid.csv', rows)
    return {'matched_targets': tested, 'max_formula_error': max_formula_error,
            'minimum_adjacent_old_signal_gap': min_gap,
            'later_branch_counterexample': counterexample, 'tangent_example': tangent}


def regression_checks():
    rng = np.random.default_rng(SEED+77)
    rows = []
    for d in [64, 128]:
        n = round(KAPPA*d*d)
        for rx, ry in [(1., .5), (1., -.5), (.6, -.7)]:
            x0, xi = [rng.normal(size=(d, n))/math.sqrt(d) for _ in range(2)]
            x1 = rx*x0+math.sqrt(1-rx*rx)*xi
            s0, s1, s01 = d/n*(x0@x0.T), d/n*(x1@x1.T), d/n*(x0@x1.T)
            sigma = np.block([[s0, ry*s01], [ry*s01.T, s1]])
            b = math.sqrt(d/n)*np.column_stack([
                np.r_[x0[:, 0], ry*x1[:, 0]],
                np.r_[ry*x0[:, 0], x1[:, 0]],
                np.r_[x0[:, 1], ry*x1[:, 1]],
                np.r_[ry*x0[:, 1], x1[:, 1]]])
            p = b.T@np.linalg.solve(sigma, b)
            c = np.kron(np.eye(2), np.array([[1., ry], [ry, 1.]]))
            assert np.linalg.eigvalsh(c-p)[0] > 0
            # Conditional covariance of linear GD projections by Wick's rule:
            # Cov(Tp Fp,Tl Fl) = Cpl E(Fp Fl)+E(Tp Fl)E(Tl Fp).
            eta = .23
            w0 = np.r_[(1-eta)*x0[:, 0], eta*x0[:, 0]]
            w1 = np.r_[(1-eta)*x1[:, 0], eta*x1[:, 0]]
            wf = np.column_stack([w0, w1, w0, w1])
            tf = b.T@wf
            correction = tf*tf.T
            rows.append({'d': d, 'rx': rx, 'ry': ry,
                         'min_sigma_eigenvalue': np.linalg.eigvalsh(sigma)[0],
                         'P_norm': np.linalg.norm(p, 2),
                         'd_times_P_norm': d*np.linalg.norm(p, 2),
                         'max_GD_Wick_cov_correction': np.max(np.abs(correction))})
    dump_csv('regression.csv', rows)


def full_dataset_checks():
    rows = []
    # Fully shared keys, including negative task correlation. These are full
    # training recalls, not recalls estimated on a subset of queries.
    for d in [64, 128]:
        n = round(KAPPA*d*d)
        for ry in [0., -.5]:
            target = ry+.28*(1-ry)
            p = one_step(ry, 'Sign', target)
            for seed in range(12):
                rng = np.random.default_rng(np.random.SeedSequence([SEED+991, d, int(2+2*ry), seed]))
                x, y0, zeta = [rng.normal(size=(d, n))/math.sqrt(d) for _ in range(3)]
                y1 = ry*y0+math.sqrt(1-ry*ry)*zeta
                z0 = d/math.sqrt(n)*(y0@x.T)
                z1 = d/math.sqrt(n)*(y1@x.T)
                s = d/n*(x@x.T)
                u = z0+increment(z1-z0@s, p)
                ranks = [full_ranks(x, y, u) for y in [y0, y1]]
                for task in [0, 1]:
                    rows.append({'d': d, 'ry': ry, 'seed': seed, 'task': task,
                                 'full_recall10': np.mean(ranks[task] <= .1),
                                 'theory': ndtr(p['mu'][task]+ndtri(.1)),
                                 'full_rank_mean': np.mean(ranks[task]),
                                 'rank_mean_theory': ndtr(-p['mu'][task]/math.sqrt(2)),
                                 'joint_full_recall10': np.mean((ranks[0] <= .1)&(ranks[1] <= .1))})
            print(f'full recall completed: d={d}, ry={ry}', flush=True)
    dump_csv('full_dataset.csv', rows)


if __name__ == '__main__':
    start = time.perf_counter()
    result = {'resources': resources(), 'branches': branches()}
    regression_checks()
    full_dataset_checks()
    result['elapsed_seconds'] = time.perf_counter()-start
    result['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (OUT/'edges_metadata.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['branches'], indent=2))
