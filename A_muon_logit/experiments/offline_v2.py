"""Read-only trajectory replay and ACF calibration; never calls an optimizer.

Only new outputs under results/v2_offline/. Imports model, not common (which
would change the thread limit to 3). Run with the theory venv and 2 BLAS threads.
This is retrospective exploration, not a fresh preregistered experiment.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
from model import make_bundle, forward

ROOT = Path(__file__).resolve().parent / 'results'
OUT = ROOT / 'v2_offline'
CONFIG = dict(freeze_step=150, top_pairs=4, eval_steps=list(range(160, 601, 10)),
              near_max_fraction=.9, coverage_gate=.8, explained_gate=.7,
              nondegenerate_delta_rms=.1, acf_n=100, acf_lags=20,
              acf_repeats=20000, seed=20261105,
              scope='18 E1 Muon beta=0 runs; all 72 heads; retrospective')


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   allow_nan=False)+'\n')


def check_budget(start):
    if time.monotonic()-start > 28*60:
        raise TimeoutError('Offline analysis budget')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 2_300_000_000:
        raise MemoryError('Offline RSS budget')


def maxacf(x):
    # Exactly the old n-lag denominator, not an asymptotic approximation.
    z = x-x.mean(axis=1, keepdims=True)
    den = (z*z).mean(axis=1)
    out = np.zeros(x.shape[0])
    for lag in range(1, 21):
        a = (z[:, lag:]*z[:, :-lag]).mean(axis=1)/(den+1e-30)
        out = np.maximum(out, np.abs(a).max(axis=1))
    return out


def calibration():
    rng = np.random.default_rng(CONFIG['seed'])
    def iid():
        return (2*rng.integers(0, 2, (20000, 100, 2))-1).astype(float)
    reference = maxacf(iid())
    threshold = float(np.quantile(reference, .95, method='higher'))
    validation = maxacf(iid())
    power = {}
    for rho in (.2, .4):
        x = np.empty((20000, 100, 2))
        x[:, 0] = 2*rng.integers(0, 2, (20000, 2))-1
        for t in range(1, 100):
            x[:, t] = x[:, t-1]*np.where(rng.random((20000, 2)) < (1+rho)/2, 1, -1)
        power[str(rho)] = float(np.mean(maxacf(x)>threshold))
    return dict(old_iid_pass_rate=float(np.mean(reference<=.2)),
                old_iid_false_rejection_rate=float(np.mean(reference>.2)),
                calibrated_95pct_threshold=threshold,
                independent_validation_rejection_rate=float(np.mean(validation>threshold)),
                validation_mc_standard_error=float(math.sqrt(.05*.95/20000)),
                markov_rademacher_power=power,
                warning='Only iid two-channel Rademacher null; not a universal threshold for state-dependent innovations.')


def replay(path, start):
    d = json.loads(path.read_text())
    bundle = make_bundle(d['seed'], d['condition'])
    assert all(np.array_equal(w, np.asarray(v)) for w, v in zip(bundle['initial'], d['initial_weights']))
    weights = np.asarray(d['initial_weights'], dtype=float)
    k0 = np.asarray(d['k0'])[:, None, None]
    sources = [(path, hashlib.sha256(path.read_bytes()).hexdigest())]
    direction_path = ROOT/d['directions_file']
    sources.append((direction_path, hashlib.sha256(direction_path.read_bytes()).hexdigest()))
    snapshots = []
    frozen = None
    measure_errors = []
    recorded = {row['step']:row for row in d['probe']}
    with gzip.open(direction_path, 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            t = row['step']
            direction = np.asarray([row['D_Q'], row['D_K']])
            weights = (1-d['alpha'])*weights-d['eta']*direction/k0
            if t<150 or t%10:
                continue
            full = forward(weights, bundle['probe'])
            ell, q, k = full['logits'], full['q'], full['k']
            measure_errors.append(abs(float(np.abs(ell).max())-recorded[t]['Mabs']))
            if t==150:
                frozen = []
                for h in range(4):
                    flat = np.argsort(-np.abs(ell[:,h,:]).ravel(), kind='stable')[:4]
                    pairs = [tuple(map(int, np.unravel_index(i, (128, 8)))) for i in flat]
                    frames = []
                    for i,j in pairs:
                        q0, k0v = q[i,h], k[i,j,h]
                        _, v = np.linalg.eigh(np.outer(q0,q0)+np.outer(k0v,k0v))
                        u = v[:,-1]
                        if u[np.argmax(np.abs(u))]<0:
                            u = -u
                        frames.append(u)
                    frozen.append(dict(pairs=pairs, frames=np.asarray(frames),
                                       base=np.array([ell[i,h,j] for i,j in pairs]),
                                       mode_base=np.array([(u@q[i,h])*(u@k[i,j,h])/math.sqrt(8)
                                                           for (i,j),u in zip(pairs,frames)])))
                continue
            point = []
            for h,f in enumerate(frozen):
                full_pair = np.array([ell[i,h,j] for i,j in f['pairs']])
                mode = np.array([(u@q[i,h])*(u@k[i,j,h])/math.sqrt(8)
                                 for (i,j),u in zip(f['pairs'],f['frames'])])
                actual_max = float(np.abs(ell[:,h,:]).max())
                winner = tuple(map(int,np.unravel_index(np.abs(ell[:,h,:]).argmax(), (128,8))))
                point.append(dict(full=full_pair, mode=mode, actual_max=actual_max,
                                  candidate_max=float(np.abs(full_pair).max()),
                                  winner_in_candidates=winner in f['pairs']))
            snapshots.append(point)
            check_budget(start)
    heads = []
    for h,f in enumerate(frozen):
        full = np.array([p[h]['full'] for p in snapshots])
        mode = np.array([p[h]['mode'] for p in snapshots])
        delta, delta_mode = full-f['base'], mode-f['mode_base']
        rms = float(np.sqrt(np.mean(delta**2)))
        explained = None if rms<.1 else float(1-np.sum((delta-delta_mode)**2)/np.sum(delta**2))
        coverage = float(np.mean([p[h]['candidate_max']>=.9*p[h]['actual_max'] for p in snapshots]))
        heads.append(dict(head=h, pairs=f['pairs'], frames=f['frames'].tolist(),
                          delta_rms=rms, explained_change=explained,
                          relative_level_rmse=float(np.linalg.norm(full-mode)/np.linalg.norm(full)),
                          near_max_coverage=coverage,
                          exact_winner_coverage=float(np.mean([p[h]['winner_in_candidates'] for p in snapshots])),
                          spatial_gate=bool(coverage>=.8 and explained is not None and explained>=.7)))
    return dict(run_id=d['run_id'], seed=d['seed'], eta=d['eta'], condition=d['condition'],
                heads=heads, max_probe_Mabs_replay_error=max(measure_errors),
                max_final_weight_error=float(np.max(np.abs(weights-np.asarray(d['final_weights'])))),
                source_sha256={str(p.relative_to(ROOT)):s for p,s in sources})


def main():
    started = time.monotonic()
    save('analysis_spec.json', CONFIG)  # Before computing; still retrospective.
    ac = calibration()
    runs = []
    for path in sorted((ROOT/'runs').glob('e1_*_muon0_*_base.json')):
        runs.append(replay(path, started))
    heads = [h for r in runs for h in r['heads']]
    def stats(key):
        values = [h[key] for h in heads if h[key] is not None]
        return dict(min=min(values), median=float(np.median(values)), max=max(values)) if values else None
    report = dict(config=CONFIG, acf=ac, runs=runs,
                  summary=dict(heads=len(heads), spatial_pass=sum(h['spatial_gate'] for h in heads),
                               nondegenerate=sum(h['explained_change'] is not None for h in heads),
                               near_max_coverage=stats('near_max_coverage'),
                               explained_change=stats('explained_change'),
                               relative_level_rmse=stats('relative_level_rmse'),
                               max_probe_Mabs_replay_error=max(r['max_probe_Mabs_replay_error'] for r in runs),
                               max_final_weight_error=max(r['max_final_weight_error'] for r in runs)),
                  elapsed_seconds=time.monotonic()-started,
                  peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  executable=os.path.realpath('/proc/self/exe'),
                  thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save('diagnostics.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('runs','config')}, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
