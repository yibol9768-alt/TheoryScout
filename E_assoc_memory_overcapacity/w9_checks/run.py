"""W9: actual same-data trajectories and training retrieval, <= 18 min wall time."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'
import argparse, json, time, resource
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

BASE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('W9_OUTPUT_DIR', str(BASE))).resolve()
assert OUT.is_relative_to(BASE), 'Outputs must stay within w9_checks/'
OUT.mkdir(parents=True, exist_ok=True)
CONFIG = dict(kappa=0.5, dims=[64, 128, 256], seeds=8, queries=128,
              max_seconds=1080, alpha=0.6, eta=0.5, e=0.2, beta1=0.9, beta2=0.99,
              methods=['gd1', 'gd4', 'sign1', 'sign4', 'adam4'],
              cases=['iso_g', 'diag_g', 'rot_g', 'rot_radkeys', 'rot_radboth'])

def block(case):
    if case.startswith('iso'):
        return np.eye(2)
    if case.startswith('diag'):
        return np.diag([0.25, 1.75])
    return np.array([[1., 0.75], [0.75, 1.]])

def rules(z, matmul):
    result = {'gd1': CONFIG['eta']*z, 'sign1': CONFIG['alpha']*np.sign(z)}
    u = np.zeros_like(z)
    for _ in range(4):
        u -= CONFIG['eta']*(matmul(u)-z)
    result['gd4'] = u.copy()
    u.fill(0)
    for _ in range(4):
        u -= CONFIG['alpha']*np.sign(matmul(u)-z)
    result['sign4'] = u.copy()
    u.fill(0)
    m, v = np.zeros_like(z), np.zeros_like(z)
    for t in range(1, 5):
        r = matmul(u)-z
        m = CONFIG['beta1']*m + (1-CONFIG['beta1'])*r
        v = CONFIG['beta2']*v + (1-CONFIG['beta2'])*r*r
        u -= CONFIG['alpha']*(m/(1-CONFIG['beta1']**t))/(np.sqrt(v/(1-CONFIG['beta2']**t))+CONFIG['e'])
    result['adam4'] = u.copy()
    return result

def predict(case):
    b = block(case)
    # Scrambled Sobol Gaussian integration; deterministic independent of data seeds.
    z = ndtri(qmc.Sobol(2, scramble=True, seed=9201).random_base2(19)) @ np.linalg.cholesky(b).T
    out = {}
    for method, f in rules(z, lambda u: u @ b).items():
        a = float(np.mean(np.sum(z*f, axis=1))/2)
        q = float(np.mean(np.sum((f @ b)*f, axis=1))/2)
        qflat = float(np.mean(f*f))
        cflat = a/np.sqrt(float(np.trace(b)/2)*qflat)
        out[method] = dict(A=a, Q=q, C=a/np.sqrt(q), naive_c=cflat,
                           pred_recall10=float(ndtr(a/np.sqrt(q*CONFIG['kappa'])-ndtri(.9))),
                           naive_recall10=float(ndtr(cflat/np.sqrt(CONFIG['kappa'])-ndtri(.9))))
    return out

def generate(rng, d, n, case):
    b = block(case)
    l = np.linalg.cholesky(b)
    if 'rad' in case:
        x = (rng.integers(0, 2, size=(n, d))*2.-1.)/np.sqrt(d)
    else:
        x = rng.normal(size=(n, d))/np.sqrt(d)
    x = (x.reshape(n, d//2, 2) @ l.T).reshape(n, d)
    if 'both' in case:
        y = (rng.integers(0, 2, size=(n, d))*2.-1.)/np.sqrt(d)
    else:
        y = rng.normal(size=(n, d))/np.sqrt(d)
    return x, y

def one(case, d, seed, prediction):
    start = time.monotonic()
    n = round(CONFIG['kappa']*d*d)
    # Covariance cases share raw data and query draws at each (d, seed).
    rng = np.random.default_rng(900000 + 10000*d + seed)
    x, y = generate(rng, d, n, case)
    s = (d/n)*(x.T @ x)
    z = (d/np.sqrt(n))*(y.T @ x)
    actual = rules(z, lambda u: u @ s)
    b = block(case)
    proxy = rules(z, lambda u: (u.reshape(d, d//2, 2) @ b).reshape(d, d))
    ids = rng.choice(n, size=min(n, CONFIG['queries']), replace=False)
    xx = x[ids].T
    output = []
    for method in CONFIG['methods']:
        p = prediction[method]
        scores = (y @ (actual[method] @ xx))/np.sqrt(p['Q'])
        diag = scores[ids, np.arange(len(ids))].copy()
        # Exact train-candidate scores; exclude the correct diagonal.
        greater = np.sum(scores > diag[None, :], axis=0)
        ties = np.sum(scores == diag[None, :], axis=0)-1
        rank = 1 + greater + 0.5*ties
        scores[ids, np.arange(len(ids))] = np.nan
        err_mean = float(np.nanmean(scores))
        err_var = float(np.nanmean(scores*scores)-err_mean**2)
        cdf_err = max(abs(float(np.nanmean(scores <= t)*n/(n-1))-ndtr(t)) for t in [-2., -1., 0., 1., 2.])
        dif = actual[method]-proxy[method]
        score_dif = (y @ (dif @ xx))/np.sqrt(p['Q'])
        diag_dif = score_dif[ids, np.arange(len(ids))].copy()
        score_dif[ids, np.arange(len(ids))] = 0
        row = dict(case=case, d=d, n=n, seed=seed, method=method,
                   correct_mean=float(np.mean(diag)), correct_var=float(np.var(diag, ddof=1)),
                   wrong_mean=err_mean, wrong_var=err_var, cdf_grid_error=cdf_err,
                   recall10=float(np.mean(rank <= np.ceil(.1*n))),
                   rank_mean=float(np.mean((rank-1)/(n-1))),
                   frob_error=float(np.mean(dif*dif)),
                   correct_error2=float(np.mean(diag_dif**2)),
                   wrong_error2=float(np.sum(score_dif**2)/((n-1)*len(ids))),
                   **p)
        output.append(row)
    return output, time.monotonic()-start

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = dict(CONFIG)
    if args.smoke:
        cfg.update(dims=[64], seeds=1, cases=['rot_g'])
    tag = 'smoke' if args.smoke else 'formal'
    predictions = {case: predict(case) for case in cfg['cases']}
    (OUT/f'{tag}_predictions.json').write_text(json.dumps(predictions, indent=2))
    start = time.monotonic()
    completed = []
    with (OUT/f'{tag}_rows.jsonl').open('x') as log:
        for d in cfg['dims']:
            for seed in range(cfg['seeds']):
                for case in cfg['cases']:
                    if time.monotonic()-start > cfg['max_seconds']:
                        break
                    rows, seconds = one(case, d, seed, predictions[case])
                    for row in rows:
                        log.write(json.dumps(row)+'\n')
                    log.flush()
                    completed.append([d, seed, case])
                    print(json.dumps(dict(d=d, seed=seed, case=case, seconds=round(seconds, 2))), flush=True)
                else:
                    continue
                break
            else:
                continue
            break
    metadata = dict(config=cfg, seconds=time.monotonic()-start, completed=completed,
                    maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    numpy=np.__version__, limitations='128 queries per dataset; seeds are independent units; nonlinear radboth exploratory')
    (OUT/f'{tag}_metadata.json').write_text(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
