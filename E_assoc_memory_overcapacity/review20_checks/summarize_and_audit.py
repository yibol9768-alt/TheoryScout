"""Independent score-contraction sanity check and read-only result audit."""
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def interval(a):
    a = np.asarray(a, dtype=float)
    se = float(np.std(a, ddof=1)/math.sqrt(len(a)))
    mean = float(np.mean(a))
    half = float(t.ppf(.975, len(a)-1))*se
    return dict(mean=mean, se=se, ci95=[mean-half, mean+half])


def sanity():
    rng = np.random.default_rng(42679)
    d, n = 13, 208
    x, y, u = rng.normal(size=(d,n)), rng.normal(size=(d,n)), rng.normal(size=(d,d))
    scores = y.T@u@x/d
    explicit_margins, explicit_vars = [], []
    for i in range(n):
        wrong = np.delete(scores[:, i], i)
        explicit_margins.append(scores[i,i]-wrong.mean())
        explicit_vars.append(np.var(wrong))
    vx = u@x
    diag = np.sum(y*vx, axis=0)/d
    means = ((y.sum(axis=1)@vx)/d-diag)/(n-1)
    total = np.sum((y@y.T)*(u@(x@x.T)@u.T))/d**2
    margin = float(np.mean(diag-means))
    variance = float((total-diag@diag)/(n*(n-1))-np.mean(means**2))
    assert abs(margin-np.mean(explicit_margins)) < 1e-12
    assert abs(variance-np.mean(explicit_vars)) < 1e-12
    # Integer ranking threshold must agree with fractional rank <=alpha.
    ranks = np.count_nonzero(scores>scores.diagonal(), axis=0)
    assert np.array_equal(ranks/(n-1) <= .1, ranks <= math.floor(.1*(n-1)))
    return dict(margin_difference=float(margin-np.mean(explicit_margins)),
                variance_difference=float(variance-np.mean(explicit_vars)),
                rank_threshold_equal=True)


def main():
    out = HERE/'audit_summary.json'
    if out.exists():
        raise FileExistsError(out)
    source = HERE/'independent_check.py'
    own = json.loads((HERE/'main_simulation.json').read_text())
    theory = json.loads((HERE/'theory.json').read_text())
    for data in (own, theory):
        assert data['inputs_unchanged']
        assert data['input_sha256_before'] == data['input_sha256_after']
        for p, h in data['input_sha256_before'].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h, p
        assert data['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert data['resource']['memory.max'] == str(3*1024**3)
        assert data['resource']['memory.swap.max'] == '0'
        assert all(int(value) <= 2 for value in data['threads'].values())
    methods = ['polar', 'adam_0.1', 'adam_0.12']
    assert len(own['records']) == len(own['parameters']['dims'])*own['parameters']['seeds']*len(methods)
    table = []
    for d in own['parameters']['dims']:
        for method in methods:
            rows = [r for r in own['records'] if r['d']==d and r['method']==method]
            table.append(dict(d=d, method=method,
                **{key:interval([r[key] for r in rows]) for key in (
                    'recall', 'normalized_msr', 'variance', 'margin_coefficient',
                    'leverage_mean', 'leverage_mse', 'correct_scaled_mse',
                    'correct_ks', 'wrong_cdf_sup', 'noise_only_cdf_sup', 'ordinary_noisy_recall')}))
    aa2 = []
    max_root_delta = 0.
    for filename in ('cue_results.json','cue_large_results.json'):
        saved = json.loads((ROOT/'aa2_checks'/filename).read_text())
        for r in saved['roots']:
            ref = next(x for x in theory['roots'] if x['gamma']==r['gamma'] and x['tau']==r['tau'])
            max_root_delta = max(max_root_delta,
                abs(r['cue_root']-ref['root_cue']), abs(r['msr_root']-ref['root_msr']))
        for d in saved['parameters']['dims']:
            adam = sorted([r for r in saved['records'] if r['d']==d and r['method']=='adam_0.1'],key=lambda r:r['seed'])
            polar = sorted([r for r in saved['records'] if r['d']==d and r['method']=='polar'],key=lambda r:r['seed'])
            aa2.append(dict(source=filename,d=d,paired_recall_difference=
                interval([a['recall']-p['recall'] for a,p in zip(adam,polar)])))
    syntax = []
    for p in HERE.glob('*.py'):
        ast.parse(p.read_text())
        syntax.append(p.name)
    result = dict(sanity=sanity(), inputs_unchanged=True, source_syntax_parsed=syntax,
        independent_root_max_absolute_difference=max_root_delta, independent_table=table,
        paired_summary=own['summary'], original_aa2_paired=aa2,
        comment='Dataset-level paired Student t intervals are Monte Carlo summaries, not mathematical guarantees.',
        artifacts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in HERE.iterdir() if p.is_file() and p.suffix in ('.py','.json')})
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(sanity=result['sanity'], root_difference=max_root_delta,
                          paired_summary=own['summary'],original_aa2_paired=aa2), indent=2))


if __name__ == '__main__':
    main()
