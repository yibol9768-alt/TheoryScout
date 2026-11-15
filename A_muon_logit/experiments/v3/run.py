"""Frozen A-prime analysis. Advance weights only with saved directions."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from model import make_bundle, forward

START = time.monotonic()
DATA = ROOT / 'experiments/results'
GAMMAS = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]
GAP_SCALES = [.25, .5, 1, 2, 4, 8, 16, 32, 64, 128]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    tmp.replace(path)


def budget():
    if time.monotonic() - START > 5000:
        raise TimeoutError('V3 active calculation limit')


def ranks(x):
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends)/2)[inv]


def spearman(x, y):
    x, y = ranks(x), ranks(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def auc(y, x):
    y, x = np.asarray(y, bool), np.asarray(x)
    n, m = int(y.sum()), int((~y).sum())
    if min(n, m) < 10 or np.std(x) == 0:
        return None
    return float((ranks(x)[y].sum() - n*(n+1)/2)/(n*m))


def prediction(s, qn, kn, xn, yn, candidate, c, uq, uk, horizon=25, gamma=1):
    """Only anchor state, public update cap and elapsed time enter this function."""
    mask = np.ones(len(s), bool)
    mask[candidate] = False
    for h in range(1, horizon+1):
        ch = c**h
        length = (1-ch)/(1-c) if c != 1 else h
        rq, rk = uq*length*xn, uk*length*yn
        radius = gamma*(ch*(qn*rk + kn*rq) + rq*rk)/np.sqrt(8)
        lower = np.maximum(ch*ch*s[candidate] - radius[candidate], 0).max()
        upper = (ch*ch*s[mask] + radius[mask]).max()
        if upper > lower/.9:
            return h
    return horizon+1


def analyze_run(path, out, smoke):
    started = time.monotonic()
    d = json.loads(path.read_text())
    dp = DATA / d['directions_file']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (path, dp)}
    bundle = make_bundle(d['seed'], d['condition'])
    assert np.array_equal(np.asarray(bundle['initial']), d['initial_weights'])
    with gzip.open(dp, 'rt') as f:
        rows = [json.loads(line) for line in f]
    assert [r['step'] for r in rows] == list(range(1, 601))
    dr = np.asarray([[r['D_Q'], r['D_K']] for r in rows])
    del rows
    direction_op = np.linalg.svd(dr.reshape(-1, 32, 32), compute_uv=False)[:, 0]
    cap_ok = bool(direction_op.max() <= 1+1e-6)
    w = np.asarray(d['initial_weights'], dtype=float)
    c = 1-d['alpha']
    uq, uk = d['eta']/np.asarray(d['k0'])*(1+1e-6)
    heads = 1 if smoke else 4
    # Include 100..149 solely for the preregistered historical 50-step feature.
    # Index = training step - 100. Scores signed for envelope audits.
    ell = np.empty((501, heads, 1024))
    qnorm = np.empty_like(ell)
    knorm = np.empty_like(ell)
    xn = np.repeat(np.linalg.norm(bundle['probe']['x'], axis=-1), 8)
    yn = np.linalg.norm(bundle['probe']['keys'], axis=-1).ravel()
    errors = []
    recorded = {r['step']: r for r in d['probe']}
    for t in range(601):
        if t:
            w = c*w - d['eta']*dr[t-1]/np.asarray(d['k0'])[:, None, None]
        if t >= 100:
            f = forward(w, bundle['probe'])
            ell[t-100] = f['logits'][:, :heads, :].transpose(1, 0, 2).reshape(heads, 1024)
            qnorm[t-100] = np.repeat(np.linalg.norm(f['q'][:, :heads], axis=-1).T, 8, axis=1)
            knorm[t-100] = np.linalg.norm(f['k'][:, :, :heads], axis=-1).transpose(2, 0, 1).reshape(heads, 1024)
            if t in recorded:
                errors.append(abs(np.abs(f['logits']).max() - recorded[t]['Mabs']))
        budget()
    final_error = float(np.max(np.abs(w-d['final_weights'])))
    del dr
    s = np.abs(ell)
    winners = s.argmax(axis=-1)
    changed = winners[1:] != winners[:-1]
    one_radius = (c*(qnorm[:-1]*uk*yn + knorm[:-1]*uq*xn) + uq*uk*xn*yn)/np.sqrt(8)
    one_error = np.abs(ell[1:]-c*c*ell[:-1])
    bound_violation = float(np.max(one_error-one_radius))
    nonzero = one_error > 1e-12
    ratios = one_radius[nonzero]/one_error[nonzero]
    fitrows, testrows = [], []
    multistep_violation = -np.inf
    certificate_violations = 0
    for origins, dest in [(range(150, 300, 25), fitrows), (range(300, 600, 25), testrows)]:
        for origin in origins:
            t = origin-100
            for head in range(heads):
                order = np.argsort(-s[t, head], kind='stable')
                candidates = order[:4]
                future = s[t+1:t+26, head]
                fullmax = future.max(axis=1)
                selected = future[:, candidates].max(axis=1)
                misses = np.flatnonzero(selected < .9*fullmax)
                exactmiss = np.flatnonzero(~np.isin(winners[t+1:t+26, head], candidates))
                tc = prediction(s[t, head], qnorm[t, head], knorm[t, head], xn, yn,
                                candidates, c, uq, uk)
                if dest is testrows:
                    certificate_violations += int(np.sum(selected[:tc-1]+1e-8 < .9*fullmax[:tc-1]))
                    for h in range(1, 26):
                        ch = c**h
                        length = (1-ch)/(1-c) if c != 1 else h
                        rq, rk = uq*length*xn, uk*length*yn
                        radius = (ch*(qnorm[t, head]*rk + knorm[t, head]*rq)+rq*rk)/np.sqrt(8)
                        multistep_violation = max(multistep_violation,
                            float(np.max(np.abs(ell[t+h, head]-ch*ch*ell[t, head])-radius)))
                past_rate = float(changed[t-50:t, head].mean())
                dest.append(dict(origin=origin, head=head, candidates=candidates,
                    T=int(misses[0]+1) if len(misses) else 26,
                    T_exact=int(exactmiss[0]+1) if len(exactmiss) else 26,
                    T_cert=tc, H=tc-1,
                    top1_top5_gap=float(s[t, head, order[0]]-s[t, head, order[4]]),
                    relative_slack=float(s[t, head, order[0]]/.9-s[t, head, order[4]]),
                    past_winner_rate=past_rate,
                    future_winner_rate=float(changed[t:t+25, head].mean()),
                    rate_prediction=int(np.clip(np.ceil(np.log(2)/(past_rate+1/50)), 1, 26)),
                    gamma_predictions=[prediction(s[t, head], qnorm[t, head], knorm[t, head],
                        xn, yn, candidates, c, uq, uk, gamma=g) for g in GAMMAS]))
    target = np.array([r['T'] for r in fitrows])
    losses = [np.mean(np.abs(np.array([r['gamma_predictions'][i] for r in fitrows])-target))
              for i in range(len(GAMMAS))]
    gi = max(i for i, v in enumerate(losses) if v == min(losses))
    constant = min(range(1, 27), key=lambda a: float(np.mean(np.abs(target-a))))
    gaploss = [np.mean(np.abs(np.clip(np.ceil(a*np.array([r['relative_slack'] for r in fitrows])), 1, 26)-target))
               for a in GAP_SCALES]
    ai = max(i for i, v in enumerate(gaploss) if v == min(gaploss))
    for r in fitrows+testrows:
        r['prediction'] = r['gamma_predictions'][gi]
        r['constant_prediction'] = constant
        r['gap_prediction'] = int(np.clip(np.ceil(GAP_SCALES[ai]*r['relative_slack']), 1, 26))
    truth = np.array([r['T'] for r in testrows])
    forecasts = {name: np.array([r[key] for r in testrows]) for name, key in
        [('geometry', 'prediction'), ('constant', 'constant_prediction'), ('rate', 'rate_prediction'),
         ('gap', 'gap_prediction'), ('certificate_expiry', 'T_cert')]}
    maes = {name: float(np.mean(np.abs(p-truth))) for name, p in forecasts.items()}
    improvements = {k: (1-maes['geometry']/maes[k] if maes[k] > 0 else None)
                    for k in ('constant', 'rate', 'gap')}
    event = truth <= 25
    event_mae = float(np.mean(np.abs(forecasts['geometry'][event]-truth[event]))) if event.any() else None
    falsealarm = float(np.mean(forecasts['geometry'][~event] <= 25)) if (~event).any() else None
    hs = np.array([r['H'] for r in testrows])
    # Online policy uses only the last full-scan state until the next full scan.
    policies = []
    for head in range(heads):
        anchor, scans, hits = 50, [], []
        candidate = np.argsort(-s[50, head], kind='stable')[:4]
        for t in range(51, 501):
            h = t-anchor
            # Checking all intermediate times is conservative; previous steps already checked.
            expires = prediction(s[anchor, head], qnorm[anchor, head], knorm[anchor, head],
                                 xn, yn, candidate, c, uq, uk, horizon=h)
            if expires <= h:
                anchor = t
                candidate = np.argsort(-s[t, head], kind='stable')[:4]
                scans.append(t+100)
            hits.append(bool(s[t, head, candidate].max()+1e-8 >= .9*s[t, head].max()))
        policies.append(dict(head=head, scans=scans, scan_fraction=len(scans)/450,
            coverage=float(np.mean(hits)), score_cost_ratio=(1024*len(scans)+4*450)/(1024*450)))
    scan_fraction = float(np.mean([r['scan_fraction'] for r in policies]))
    policy_coverage = float(np.mean([r['coverage'] for r in policies]))
    rho = spearman([r['past_winner_rate'] for r in testrows], [r['future_winner_rate'] for r in testrows])
    area = auc(event, [1/r['T_cert'] for r in testrows])
    g0 = final_error <= 1e-5 and max(errors) <= 1e-4 and cap_ok and max(bound_violation, multistep_violation) <= 1e-8 and certificate_violations == 0
    g1 = bool(np.median(hs) >= 5 and np.mean(hs >= 5) >= .5 and scan_fraction <= .2 and policy_coverage >= .99)
    identifiable = int(event.sum()) >= 10
    g2 = bool(identifiable and all(v is not None and v >= .2 for v in improvements.values())
              and event_mae is not None and event_mae <= 5 and falsealarm is not None and falsealarm <= .1)
    g3 = bool(rho is not None and rho >= .3 and area is not None and area >= .7)
    result = dict(run_id=d['run_id'], condition=d['condition'], eta=d['eta'], seed=d['seed'],
        heads=heads, source_sha256=hashes, fit_origins=fitrows, test_origins=testrows,
        fitted=dict(gamma=GAMMAS[gi], gamma_fit_mae=losses, constant=constant,
                    gap_scale=GAP_SCALES[ai], gap_fit_mae=gaploss),
        audit=dict(final_weight_error=final_error, probe_error=max(errors),
            max_direction_op=float(direction_op.max()), direction_cap_ok=cap_ok,
            one_step_max_violation=bound_violation, multistep_max_violation=multistep_violation,
            certificate_violations=certificate_violations,
            one_step_bound_to_actual_quantiles=np.quantile(ratios, [.1, .5, .9])),
        metrics=dict(winner_change_rate=float(changed[50:].mean()), H_median=float(np.median(hs)),
            H_ge5_fraction=float(np.mean(hs >= 5)), scan_fraction=scan_fraction,
            score_cost_ratio=float(np.mean([r['score_cost_ratio'] for r in policies])),
            policy_coverage=policy_coverage, event_windows=int(event.sum()),
            nonevent_windows=int((~event).sum()), exact_event_windows=sum(r['T_exact'] <= 25 for r in testrows),
            MAE=maes, improvements=improvements, event_MAE=event_mae, false_alarm_rate=falsealarm,
            winner_rate_spearman=rho, risk_AUROC=area, G2_identifiable=identifiable),
        gates=dict(G0=bool(g0), G1=g1, G2=g2, G3=g3), policies=policies,
        elapsed_seconds=time.monotonic()-started)
    for p, old in hashes.items():
        assert sha(ROOT/p) == old, 'Source mutated'
    np.savez_compressed(out / (d['run_id']+'.npz'), training_steps=np.arange(100, 601), signed_logits=ell,
                        qnorm=qnorm, knorm=knorm, xnorm=xn, ynorm=yn)
    save(out / (d['run_id']+'.json'), result)
    print(json.dumps(clean(dict(run_id=d['run_id'], gates=result['gates'],
                               metrics=result['metrics'], seconds=result['elapsed_seconds'])), ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    expected = (ROOT/'PREREG_V3_OFFLINE.sha256').read_text().split()[0]
    assert sha(ROOT/'PREREG_V3_OFFLINE.md') == expected
    out = Path(__file__).resolve().parent / ('smoke' if args.smoke else 'full')
    out.mkdir(exist_ok=True)
    runs = sorted((DATA/'runs').glob('e1_*_muon0_*_base.json'))
    assert len(runs) == 18
    if args.smoke:
        runs = runs[:1]
    results = [analyze_run(p, out, args.smoke) for p in runs]
    strata = []
    for condition in sorted({r['condition'] for r in results}):
        for eta in sorted({r['eta'] for r in results}):
            rr = [r for r in results if r['condition'] == condition and r['eta'] == eta]
            if not rr:
                continue
            strata.append(dict(condition=condition, eta=eta, seeds=[r['seed'] for r in rr],
                passes={g: sum(r['gates'][g] for r in rr) for g in ('G0', 'G1', 'G2', 'G3')},
                stratum_pass={g: sum(r['gates'][g] for r in rr) >= (len(rr) if g == 'G0' else 2)
                              for g in ('G0', 'G1', 'G2', 'G3')}))
    summary = dict(mode='smoke' if args.smoke else 'full', strata=strata,
        run_ids=[r['run_id'] for r in results],
        prereg_sha256=expected, script_sha256=sha(Path(__file__)), model_sha256=sha(ROOT/'experiments/model.py'),
        elapsed_seconds=time.monotonic()-START, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        executable=sys.executable, numpy=np.__version__,
        threads={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        scope='systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0',
        no_torch_import='torch' not in sys.modules, device='CPU', api_calls=0,
        all_gates={g: all(r['stratum_pass'][g] for r in strata) for g in ('G0', 'G1', 'G2', 'G3')})
    assert sha(ROOT/'PREREG_V3_OFFLINE.md') == expected
    save(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
