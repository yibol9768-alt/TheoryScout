"""Independent numerical checks and frozen-gate report, no training/import of run.py."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'experiments/v3'
START = time.monotonic()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def op(x):
    return np.linalg.norm(x, 2)


def math_checks():
    rng = np.random.default_rng(20261106)
    maximum_excess = -np.inf
    trials = 0
    for c in (0, .5, .99, 1):
        for _ in range(75):
            d, p = 3, 4
            q0, k0 = rng.normal(size=(2, d, p))
            x0, y0 = rng.normal(size=(2, p))
            q, k = q0.copy(), k0.copy()
            uq, uk = .07, .11
            base = (q0@x0)@(k0@y0)/np.sqrt(d)
            for h in range(1, 8):
                e, f = rng.normal(size=(2, d, p))
                q = c*q + uq*e/op(e)
                k = c*k + uk*f/op(f)
                dx, dy = rng.normal(size=(2, p))
                dx *= .05*h/np.linalg.norm(dx)
                dy *= .03*h/np.linalg.norm(dy)
                a = c**h
                length = sum(c**j for j in range(h))
                rq = a*op(q0)*np.linalg.norm(dx) + uq*length*(np.linalg.norm(x0)+np.linalg.norm(dx))
                rk = a*op(k0)*np.linalg.norm(dy) + uk*length*(np.linalg.norm(y0)+np.linalg.norm(dy))
                radius = (a*(np.linalg.norm(q0@x0)*rk + np.linalg.norm(k0@y0)*rq)+rq*rk)/np.sqrt(d)
                actual = (q@(x0+dx))@(k@(y0+dy))/np.sqrt(d)
                maximum_excess = max(maximum_excess, abs(actual-a*a*base)-radius)
                trials += 1
    assert maximum_excess <= 1e-10
    selection_trials = 0
    for _ in range(150):
        lower = rng.uniform(0, 5, 6)
        upper = lower+rng.uniform(0, 3, 6)
        rho = rng.choice([.5, .9, 1])
        exact = 6
        for bits in itertools.product([False, True], repeat=6):
            selected = np.array(bits)
            if not selected.any():
                continue
            omitted = upper[~selected].max() if (~selected).any() else 0
            if rho*omitted <= lower[selected].max():
                exact = min(exact, int(selected.sum()))
        formula = min(len({a} | set(np.flatnonzero(rho*upper > lower[a]))) for a in range(6))
        assert exact == formula
        selection_trials += 1
    clock_trials = 0
    for _ in range(300):
        scores = np.sort(rng.uniform(.5, 5, 5))[::-1]
        b = float(rng.uniform(.01, .2))
        rho = float(rng.choice([.5, .9, 1]))
        k = int(rng.integers(1, 5))
        gap = scores[0]-scores[k]
        clock = int(np.floor(((1-rho)*scores[0]+rho*gap)/((1+rho)*b)))
        assert rho*(scores[k]+b*clock) <= max(scores[0]-b*clock, 0)+1e-10
        assert rho*(scores[k]+b*(clock+1)) > max(scores[0]-b*(clock+1), 0)-1e-10
        for horizon in (0, clock, clock+1):
            lower = np.maximum(scores-b*horizon, 0)
            upper = scores+b*horizon
            exact = 5
            for bits in itertools.product([False, True], repeat=5):
                selected = np.array(bits)
                if selected.any() and (not (~selected).any() or rho*upper[~selected].max() <= lower[selected].max()+1e-12):
                    exact = min(exact, int(selected.sum()))
            formula = max(1, int(np.sum(rho*upper > lower[0]+1e-12)))
            assert exact == formula
        # Explicit fixed-input attention realization of the endpoint adversary.
        endpoint = np.maximum(scores-b*(clock+1), 0)
        endpoint[k] = scores[k]+b*(clock+1)
        Q, K = np.eye(5), np.sqrt(5)*np.diag(endpoint)
        logits = np.diag(Q.T@K)/np.sqrt(5)
        assert np.allclose(logits, endpoint)
        assert logits[:k].max() < rho*logits.max()+1e-10
        clock_trials += 1
    tex = []
    for path in sorted((ROOT/'lemmas_v3').glob('L*.tex')):
        content = path.read_text()
        assert content.count('\\begin{problem}') == content.count('\\end{problem}') == 1
        balance = 0
        for ch in content:
            balance += (ch == '{')-(ch == '}')
            assert balance >= 0
        assert balance == 0
        tex.append(path.name)
    return dict(spectral_drift_trials=trials, spectral_max_excess=float(maximum_excess),
                interval_selection_trials=selection_trials, sharp_clock_trials=clock_trials,
                tex_problem_and_brace_checks=tex,
                status='PASS; numerical checks do not replace proofs or formal verification')


def main():
    expected = (ROOT/'PREREG_V3_OFFLINE.sha256').read_text().split()[0]
    assert sha(ROOT/'PREREG_V3_OFFLINE.md') == expected
    summary = json.loads((HERE/'full/summary.json').read_text())
    assert summary['script_sha256'] == sha(HERE/'run.py')
    assert summary['model_sha256'] == sha(ROOT/'experiments/model.py')
    runs = [json.loads((HERE/'full'/f'{rid}.json').read_text()) for rid in summary['run_ids']]
    assert len(runs) == 18
    inputs, rows_checked, policy_steps = {}, 0, 0
    for run in runs:
        for p, digest in run['source_sha256'].items():
            assert sha(ROOT/p) == digest
            inputs[p] = digest
        raw = np.load(HERE/'full'/f"{run['run_id']}.npz", allow_pickle=False)
        signed = raw['signed_logits']
        s = np.abs(signed)
        assert signed.shape == (501, 4, 1024)
        d = json.loads((ROOT/next(p for p in run['source_sha256'] if '/runs/' in p)).read_text())
        c = 1-d['alpha']
        uq, uk = d['eta']/np.asarray(d['k0'])*(1+1e-6)
        h = np.arange(1, 26)
        a = c**h
        length = (1-a)/(1-c)
        for row in run['fit_origins']+run['test_origins']:
            t, head = row['origin']-100, row['head']
            candidate = row['candidates']
            mask = np.ones(1024, bool)
            mask[candidate] = False
            future = s[t+1:t+26, head]
            miss = np.flatnonzero(future[:, candidate].max(1) < .9*future.max(1))
            target = int(miss[0]+1) if len(miss) else 26
            assert target == row['T']
            rq = uq*length[:, None]*raw['xnorm']
            rk = uk*length[:, None]*raw['ynorm']
            radius = (a[:, None]*(raw['qnorm'][t, head]*rk+raw['knorm'][t, head]*rq)+rq*rk)/np.sqrt(8)
            center = a[:, None]**2*s[t, head]
            for gamma, name in [(1, 'T_cert'), (run['fitted']['gamma'], 'prediction')]:
                lower = np.maximum(center[:, candidate]-gamma*radius[:, candidate], 0).max(1)
                upper = (center[:, mask]+gamma*radius[:, mask]).max(1)
                expiration = np.flatnonzero(.9*upper > lower)
                pred = int(expiration[0]+1) if len(expiration) else 26
                assert pred == row[name]
            rows_checked += 1
        for policy in run['policies']:
            head = policy['head']
            candidate = np.argsort(-s[50, head], kind='stable')[:4]
            scans = set(policy['scans'])
            for t in range(51, 501):
                if t+100 in scans:
                    candidate = np.argsort(-s[t, head], kind='stable')[:4]
                assert s[t, head, candidate].max()+1e-8 >= .9*s[t, head].max()
                policy_steps += 1
    checks = math_checks()
    audit = dict(status='PASS', source_files=len(inputs), independent_origin_rows=rows_checked,
        independent_policy_steps=policy_steps, prereg_sha256=expected, mathematical_checks=checks,
        elapsed_seconds=time.monotonic()-START,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (HERE/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    # Compact, unrounded numbers retained separately from the narrative report.
    aggregate = []
    for st in summary['strata']:
        rr = [r for r in runs if r['condition'] == st['condition'] and r['eta'] == st['eta']]
        aggregate.append(dict(condition=st['condition'], eta=st['eta'],
            seeds=[r['seed'] for r in rr], gates=st['passes'],
            values={key: [r['metrics'][key] for r in rr] for key in rr[0]['metrics']},
            audit=[r['audit'] for r in rr], fitted=[r['fitted'] for r in rr]))
    (HERE/'aggregate.json').write_text(json.dumps(aggregate, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    for st in aggregate:
        print(json.dumps(st, ensure_ascii=False))


if __name__ == '__main__':
    main()
