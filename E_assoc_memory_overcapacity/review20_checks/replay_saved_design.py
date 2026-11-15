"""Recreate selected saved random designs from definitions, not AA2 code.

This is an integrity check of saved outcomes, not extra independent samples.
Positive normalization of each direction is omitted because only ranks matter.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import polar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
out = HERE/'saved_design_replay.json'
if out.exists():
    raise FileExistsError(out)
rows = []
for d, seeds, filename in ((256, range(6), 'cue_results.json'),
                           (1024, (0, 5), 'cue_large_results.json')):
    saved = json.loads((ROOT/'aa2_checks'/filename).read_text())
    for seed in seeds:
        n = 16*d
        gen = np.random.default_rng(202611060001+10000*d+seed)
        keys = gen.standard_normal((d, n))
        values = gen.standard_normal((d, n))
        # Apply C through explicit column centering; use SciPy's polar API.
        z = (values-values.mean(axis=1, keepdims=True)) @ (
            keys-keys.mean(axis=1, keepdims=True)).T/math.sqrt(n)
        op = polar(z)[0]
        indices = gen.integers(n, size=2048)
        xi = gen.standard_normal((d, 2048))
        queries = keys[:, indices]/math.sqrt(d)+.2*xi
        for name, direction in (('polar', op), ('adam_0.1', z/(abs(z)+.1))):
            successes = 0
            for lo in range(0, 2048, 128):
                hi = min(2048, lo+128)
                scores = values.T@(direction@queries[:, lo:hi])
                target = scores[indices[lo:hi], np.arange(hi-lo)]
                ranks = np.sum(scores>target, axis=0)
                successes += int(np.sum(ranks <= math.floor(.1*(n-1))))
            computed = successes/2048
            ref = next(r['recall'] for r in saved['records']
                       if r['d']==d and r['seed']==seed and r['method']==name)
            rows.append(dict(d=d, seed=seed, method=name, computed=computed,
                             stored=ref, difference=computed-ref))
        print(json.dumps(dict(d=d, seed=seed, recalls=rows[-2:])), flush=True)
maximum = max(abs(r['difference']) for r in rows)
assert maximum == 0, rows
with out.open('x') as f:
    json.dump(dict(rows=rows, max_recall_difference=maximum,
                   note='Independent reimplementation of selected stored random designs; not pooled into new Monte Carlo estimates.'), f, indent=2)
print(json.dumps(dict(records=len(rows), max_recall_difference=maximum)))
