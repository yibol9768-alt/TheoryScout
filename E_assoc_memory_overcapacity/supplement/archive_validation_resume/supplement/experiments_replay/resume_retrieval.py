"""Resume a replay using intact query archives, without overwriting evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import replay_retrieval as replay


def measurements(arrays, design, d, ir, seed, observed):
    """Validate every stored array before accepting a completed dataset."""
    n = d*d//4
    count = 2048 if design == 'matched' else 8192 if (d, ir) == (512, 2) else 512
    ids = arrays['ids']
    assert ids.shape == (count,) and np.issubdtype(ids.dtype, np.integer)
    assert np.array_equal(ids, replay.rng(design, d, ir, seed, 3).choice(n, size=count, replace=False, shuffle=True))
    rows = []
    for method in replay.METHODS:
        for task in ['old', 'new']:
            prefix = method+'_'+task+'_'
            rank, ok, D, V = [arrays[prefix+k] for k in ['ranks', 'success', 'D', 'V']]
            assert all(x.shape == ids.shape for x in [rank, ok, D, V])
            assert np.issubdtype(rank.dtype, np.integer)
            assert rank.min() >= 1 and rank.max() <= n
            assert np.array_equal(ok, rank <= math.ceil(.1*n))
            assert np.isfinite(D).all() and np.isfinite(V).all() and (V >= 0).all()
            recall = float(ok.mean())
            assert recall == observed[d, ir, seed, method, task]
            rows.append(dict(d=d, ir=ir, seed=seed, method=method, task=task,
                recall=recall, D=float(D.mean()), V=float(V.mean()),
                var_D=float(D.var(ddof=1)), var_V=float(V.var(ddof=1)),
                cov_DV=float(np.cov(D, V, ddof=1)[0, 1])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--design', choices=['matched', 'confirmation'], required=True)
    p.add_argument('--existing', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.existing.is_absolute() and not a.output.is_absolute()
    assert '..' not in a.existing.parts and '..' not in a.output.parts
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    params = replay.population()
    pred = replay.predictions(params)
    frozen = json.loads((replay.HERE/'population_parameters.json').read_text())
    err = max(abs(row[k]-next(t for t in frozen if int(t['ir']) == row['ir'] and t['method'] == row['method'])[k]) for row in params for k in ['amplitude', 'q'])
    assert err < 1e-10
    (a.output/'population_predictions.json').write_text(json.dumps(dict(parameters=params, finite_dimension_predictions=pred), indent=2))
    lookup = {(row['ir'], row['method']): row for row in params}
    observed = replay.load_measurements(a.design)
    dims, seeds = ([128, 256], 192) if a.design == 'matched' else ([128, 256, 512], 32)
    rows, accepted, generated, rejected = [], [], [], []
    for seed in range(seeds):
        for d in dims:
            for ir in range(3):
                name = f'd{d}_r{ir}_s{seed}.npz'
                old = a.existing/name
                reused = False
                if old.exists():
                    try:
                        with np.load(old, allow_pickle=False) as arrays:
                            data = measurements(arrays, a.design, d, ir, seed, observed)
                        accepted.append(dict(file=name, sha256=hashlib.sha256(old.read_bytes()).hexdigest()))
                        reused = True
                    except (OSError, ValueError, EOFError, AssertionError, KeyError) as exc:
                        rejected.append(dict(file=name, reason=type(exc).__name__))
                if not reused:
                    count = 2048 if a.design == 'matched' else 8192 if (d, ir) == (512, 2) else 512
                    x, y0, y1, ids, z0, h = replay.dataset(a.design, d, ir, seed, count)
                    arrays = dict(ids=ids)
                    for method in replay.METHODS:
                        par = lookup[ir, method]
                        u = z0+par['amplitude']*replay.direction(h, method, replay.CORRELATIONS[ir])
                        for task, y in [('old', y0), ('new', y1)]:
                            rank, ok, D, V = replay.ranks(u, x, y, ids, par['q'])
                            for key, value in [('ranks', rank), ('success', ok), ('D', D), ('V', V)]:
                                arrays[method+'_'+task+'_'+key] = value
                    data = measurements(arrays, a.design, d, ir, seed, observed)
                    dest = a.output/name
                    with dest.open('xb') as handle:
                        np.savez_compressed(handle, **arrays)
                    generated.append(dict(file=name, sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
                    del arrays, x, y0, y1, z0, h, u
                rows.extend(data)
                print(json.dumps(dict(design=a.design, d=d, ir=ir, seed=seed, reused=reused, seconds=time.monotonic()-start)), flush=True)
    with (a.output/'measurements.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status='PASS', design=a.design, full=True, datasets=len(dims)*3*seeds,
        reused_datasets=len(accepted), generated_datasets=len(generated), rejected_archives=rejected,
        seed_level_recalls_equal_to_original=True, max_recall_error=0.,
        regenerated_population_max_error=err, seconds=time.monotonic()-start,
        small_replay_is_not_statistical_confirmation=False)
    (a.output/'archive_provenance.json').write_text(json.dumps(dict(accepted=accepted, generated=generated), indent=2))
    (a.output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
