"""Local numerical checks for FORMULATION.md; no network or model API.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 sanity_check.py
"""
import json
import resource
import time
from pathlib import Path
import numpy as np


def prediction(c, gamma):
    return float(0.5 + np.arcsin(np.clip(c / np.sqrt(gamma), -1, 1)) / np.pi)


def run():
    start = time.perf_counter()
    rows = []
    grid = (np.arange(200000) + 0.5) * 2 / 200000
    weight = np.sqrt(4 - grid**2) / np.pi * 2 / 200000
    times = [0.05, 0.25, 0.5, 1.0, 2.5]
    for d in [64, 128, 256]:
        for gamma in [1.25, 2.0, 4.0, 8.0]:
            for seed in range(5):
                rng = np.random.default_rng(seed)
                n = round(gamma * d)
                q, r = np.linalg.qr(rng.normal(size=(n, d)), mode='reduced')
                # Gaussian QR with positive diagonal produces Haar Stiefel columns.
                q = q * np.where(np.diag(r) >= 0, 1.0, -1.0)
                x = q.T
                y = rng.normal(size=(d, n)) / np.sqrt(d)
                g = y @ x.T
                u, s, vt = np.linalg.svd(g, full_matrices=False)
                cases = [('GD', g, 1.0),
                         ('polar', u @ vt, 8 / (3 * np.pi)),
                         ('Adam_first_eps0', np.sign(g) / np.sqrt(d), np.sqrt(2 / np.pi))]
                for t in times:
                    f = np.minimum(t, grid)
                    c = float(np.sum(weight * grid * f) / np.sqrt(np.sum(weight * f**2)))
                    cases.append((f'spectral_t{t}', (u * np.minimum(t, s)) @ vt, c))
                for name, w, c in cases:
                    score = w @ x
                    rho = np.sum(w * g, axis=1) / (np.linalg.norm(w, axis=1) * np.linalg.norm(y, axis=1))
                    finite_angle = float(np.mean(0.5 + np.arcsin(np.clip(rho, -1, 1)) / np.pi))
                    # The finite-angle formula is an expectation over a Haar rotation,
                    # not an exact identity for one realization's bit accuracy.
                    rows.append(dict(d=d, gamma=gamma, seed=seed, method=name,
                                     accuracy=float(np.mean(score * y > 0)),
                                     limit=prediction(c, gamma), finite_angle=finite_angle))
                # Verify the exact loss split and scalar discretization bound.
                w = cases[-2][1]
                lhs = np.linalg.norm(w @ x - y)**2
                rhs = np.linalg.norm(w - g)**2 + np.linalg.norm(y - g @ x)**2
                assert np.isclose(lhs, rhs, rtol=1e-10, atol=1e-10)
                eta = 0.02
                z = np.zeros_like(s)
                for k in range(1, 151):
                    z -= eta * np.sign(z - s)
                    assert np.max(np.abs(z - np.minimum(k * eta, s))) <= eta + 1e-10
    summary = []
    for gamma in [1.25, 2.0, 4.0, 8.0]:
        for name in ['GD', 'polar', 'Adam_first_eps0']:
            rr = [r for r in rows if r['d'] == 256 and r['gamma'] == gamma and r['method'] == name]
            aa = np.array([r['accuracy'] for r in rr])
            summary.append(dict(d=256, gamma=gamma, method=name, mean=float(aa.mean()),
                                seed_se=float(aa.std(ddof=1)/np.sqrt(len(aa))), limit=rr[0]['limit']))
    result = dict(elapsed_seconds=time.perf_counter()-start,
                  max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  checks='loss decomposition and discrete spectral bound passed',
                  summary=summary, rows=rows)
    Path('sanity_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    run()
