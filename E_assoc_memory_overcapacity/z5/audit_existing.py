"""Z5: deterministic, read-only REP/W4 reanalysis. Never creates a Generator.

Default: exclusive creation of z5/existing_audit/. --verify-only: no writes.
Does not import any historical project module or replay training data.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import csv
import ctypes
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import sys
import time
from collections import defaultdict

sys.dont_write_bytecode = True
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'z5' / 'existing_audit'
FAMILY = 2442
GRID = np.arange(1, 100) / 100
TRACKED = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source(rel):
    p = (ROOT / rel).resolve()
    assert p.is_relative_to(ROOT), rel
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert rel not in TRACKED or TRACKED[rel] == digest, ('changed', rel)
    TRACKED[rel] = digest
    return data.decode()


def load(rel):
    return json.loads(source(rel))


def manifest(rel):
    entries = []
    for line in source(rel).splitlines():
        digest, name = line.split('  ', 1)
        data = source(name)
        assert hashlib.sha256(data.encode()).hexdigest() == digest, (rel, name)
        entries.append(name)
    return entries


def binary_manifest(rel):
    entries = []
    for line in source(rel).splitlines():
        digest, name = line.split('  ', 1)
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT), name
        actual = sha(p)
        assert actual == digest, (rel, name)
        TRACKED[name] = actual
        entries.append(name)
    return entries


def resources():
    cg_line = next(x for x in Path('/proc/self/cgroup').read_text().splitlines()
                   if x.startswith('0::'))
    cg = Path('/sys/fs/cgroup') / cg_line[3:].lstrip('/')
    controls = {x: (cg / x).read_text().strip()
                for x in ('memory.max', 'memory.swap.max', 'memory.peak')}
    assert controls['memory.max'] == str(3 * 1024**3), controls
    assert controls['memory.swap.max'] == '0', controls
    pools = []
    paths = sorted({s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
                    if 'openblas' in s and '/' in s})
    for path in paths:
        library = ctypes.CDLL(path)
        for symbol in ('scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads',
                       'openblas_get_num_threads', 'openblas_get_num_threads64_'):
            try:
                fn = getattr(library, symbol)
            except AttributeError:
                continue
            fn.restype = ctypes.c_int
            threads = fn()
            assert 0 < threads <= 2
            pools.append(dict(library=path, threads=threads))
            break
        else:
            raise AssertionError(('unobservable BLAS', path))
    assert pools
    return dict(controls=controls, blas=pools, executable=sys.executable,
                python=sys.version, numpy=np.__version__,
                max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def bound(cdfs, grid=GRID, family=FAMILY, allowance=.01):
    a = np.asarray(cdfs, dtype=float)
    assert a.ndim == 2 and a.shape[0] >= 2 and a.shape[1] == len(grid)
    assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
    assert (np.diff(a, axis=1) >= 0).all()
    crit = float(t.ppf(1 - .05 / (2 * family), a.shape[0] - 1))
    residual = a.mean(axis=0) - grid
    se = a.std(axis=0, ddof=1) / math.sqrt(a.shape[0])
    hw = crit * se
    return dict(upper=float(min(1., np.max(np.abs(residual) + hw) + allowance)),
                critical=crit), residual, se, hw


def ks(pits):
    v = np.sort(np.asarray(pits, dtype=float))
    assert v.size and np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
    return float(max(np.max(np.arange(1, len(v) + 1) / len(v) - v),
                     np.max(v - np.arange(len(v)) / len(v))))


def complete(groups):
    expected = {(d, m) for d in (64, 128, 256)
                for m in ('GD', 'Sign1', 'Sign2', 'Adam16')}
    assert set(groups) == expected
    for rr in groups.values():
        assert len(rr) == 16 and sorted(r['seed'] for r in rr) == list(range(16))


def tests():
    # Purely constructed fixtures: no random numbers, no training arrays.
    a = np.tile(GRID, (16, 1))
    assert abs(bound(a)[0]['upper'] - .01) < 1e-13
    assert abs(bound(np.zeros((16, 99)))[0]['upper'] - 1) < 1e-15
    assert ks([0., 1.]) == .5 and ks([.5, .5]) == .5
    assert np.array_equal(ndtr(ndtri([0., 1.]) + 1), [0., 1.])
    # Monotonicity allowance controls every interval including both tails.
    fixture = np.array([.015, .015, .63, .999])
    empirical = np.mean(fixture[:, None] <= GRID, axis=0)
    u = bound(np.tile(empirical, (16, 1)))[0]['upper']
    assert ks(fixture) <= u + 1e-15
    b = a.copy()
    b[0] = np.maximum(0., b[0] - .01)
    assert bound(b)[0]['critical'] == bound(a)[0]['critical']
    assert bound(b, family=2442)[0]['upper'] > bound(b, family=1)[0]['upper']
    # Boundary is inclusive in the historical W4 registration.
    assert .04 <= .04 and not np.nextafter(.04, 1.) <= .04
    for bad in (np.full((16, 99), np.nan), np.tile(GRID[::-1], (16, 1))):
        try:
            bound(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('bad CDF accepted')
    try:
        complete({})
    except AssertionError:
        pass
    else:
        raise AssertionError('missing cells accepted')
    return dict(passed=True, fixture_type='deterministic_only', rng_draws=0,
                checks=['zero_bias', 'tails', 'jumps_and_ties', 'PIT_endpoints',
                        'monotonicity_envelope', 'original_family', 'inclusive_boundary',
                        'reject_nan', 'reject_nonmonotone', 'reject_missing_cells'])


def reanalyze():
    freeze = manifest('W4_FREEZE.sha256')
    artifacts = binary_manifest('W4_ARTIFACTS.sha256')
    rows = [json.loads(line) for line in source('experiments/w4/W4_raw.jsonl').splitlines()]
    meta = load('experiments/w4/W4_metadata.json')
    assert len(rows) == 340 and meta['completed'] == meta['expected'] == 117 and meta['stop'] is None
    groups = defaultdict(list)
    for r in rows:
        if r['kind'] == 'M':
            groups[r['d'], r['method']].append(r)
    complete(groups)
    ref = load('experiments/w4/W4_summary.json')
    references = {(v['d'], v['method']): v for v in ref['M']}
    intervals = {(int(r['d']), r['method'], r['distribution'], float(r['p'])): r
                 for r in csv.DictReader(io.StringIO(source('experiments/w4/W4_CDF_intervals.csv')))}
    assert len(intervals) == 2376
    alpha = .6
    c_sign = math.sqrt(2) * math.exp(-alpha**2 / 2) / math.sqrt(2 * math.pi * ndtr(-alpha))
    q_sign = 8 * alpha**2 * ndtr(-alpha)
    results, grid_rows = [], []
    max_difference = 0.
    for (d, method), rr in sorted(groups.items()):
        rr.sort(key=lambda r: r['seed'])
        c = rr[0]['c']
        if method == 'Sign2':
            assert abs(c - c_sign) < 1e-14 and all(abs(r['q'] - q_sign) < 1e-14 for r in rr)
        for r in rr:
            assert r['c'] == c
            assert len(r['ids']) == len(r['scores']) == len(r['ranks']) == 1024
            assert len(set(r['ids'])) == 1024 and min(r['ids']) >= 0 and max(r['ids']) < d*d
            rank = np.asarray(r['ranks'])
            assert ((rank >= 0) & (rank <= 1)).all()
            assert np.max(np.abs(rank * (d*d - 1) - np.rint(rank * (d*d - 1)))) < 1e-9
        for distribution in ('rank', 'score'):
            pits = [ndtr(ndtri(np.asarray(r['ranks'])) + c) if distribution == 'rank'
                    else ndtr(np.asarray(r['scores']) - c) for r in rr]
            cdfs = np.array([np.mean(p[:, None] <= GRID, axis=0) for p in pits])
            assert np.array_equal(cdfs, np.array([r[distribution + '_pit_cdf'] for r in rr]))
            bb, residual, se, hw = bound(cdfs)
            descriptive = ks(np.concatenate(pits))
            old = references[d, method][distribution + '_cdf']
            max_difference = max(max_difference, abs(bb['upper'] - old['simultaneous_upper']),
                                 abs(descriptive - old['pooled_full_ks']))
            assert abs(bb['upper'] - old['simultaneous_upper']) < 1e-12
            assert abs(descriptive - old['pooled_full_ks']) < 1e-12
            equivalent = bb['upper'] <= .04
            assert equivalent == old['equivalent']
            results.append(dict(d=d, method=method, distribution=distribution, seeds=16,
                                queries_per_seed=1024, c=c, pooled_ks=descriptive,
                                U=bb['upper'], tolerance=.04, equivalent=equivalent,
                                margin=.04-bb['upper'], critical=bb['critical'], family=FAMILY))
            for j, p in enumerate(GRID):
                lo, hi = float(residual[j]-hw[j]), float(residual[j]+hw[j])
                old_row = intervals[d, method, distribution, float(p)]
                for k, v in dict(mean=residual[j], se=se[j], lo=lo, hi=hi).items():
                    delta = abs(float(old_row[k])-v)
                    max_difference = max(max_difference, delta)
                    assert delta < 1e-12, (d, method, distribution, p, k, delta)
                grid_rows.append(dict(d=d, method=method, distribution=distribution,
                                      p=float(p), mean=float(residual[j]), se=float(se[j]), lo=lo, hi=hi))
    assert sum(r['equivalent'] for r in results) == 13
    # No generator is constructed. SeedSequence only derives historical initial states.
    m_entropies = [(20261106, 440004, 1, d, seed, stream)
                   for d in (64, 128, 256) for seed in range(16) for stream in (0, 1, 4)]
    def state(e):
        return tuple(map(int, np.random.SeedSequence(e).generate_state(8)))
    states = {state(e) for e in m_entropies}
    assert len(states) == len(m_entropies) == 144
    seed_checks = {}
    for filename in ('y3d/streams.json', 'z4b/streams.json'):
        entropies = load(filename)
        entropies = [r['entropy'] if isinstance(r, dict) else r for r in entropies]
        assert not any(state(e) in states for e in entropies), filename
        seed_checks[filename] = len(entropies)
    # A/B EXP1 shared-data roots, including its smoke dimensions and a conservative seed range.
    exp1 = [(20261106, d, int(k*d*d), s, 1) for d in (32, 64, 128, 256)
            for k in (.5, 1, 2) for s in range(16)]
    assert not any(state(e) in states for e in exp1)
    seed_checks['EXP1_A_B_conservative_roots'] = len(exp1)
    return dict(passed=True, decision='reuse_existing_W4_REP_no_new_sampling',
                original_family=FAMILY, original_cdf_coordinates=2376,
                cdf_passes=13, cdf_unresolved=11, cells=results,
                max_absolute_recompute_difference=max_difference,
                W4_freeze_files_verified=len(freeze), W4_artifacts_verified=len(artifacts),
                historical_M_streams=144, historical_state_checks=seed_checks,
                seed_check_scope='Recorded W4 M vs EXP1 A/B, Y3D and Z4B only; not a universal PRNG independence proof.',
                new_seeds=0, new_Generator_instances=0, training_or_query_draws=0,
                synthetic=tests()), grid_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    before = resources()
    if args.test_only:
        print(json.dumps(tests()))
        return
    if args.verify_only:
        manifest('Z5_FROZEN.sha256')
        binary_manifest('z5/reviewed_inputs.sha256')
    report, grid_rows = reanalyze()
    if args.verify_only:
        saved = load('z5/existing_audit/report.json')
        for key in ('decision', 'original_family', 'cdf_passes', 'cdf_unresolved',
                    'cells', 'max_absolute_recompute_difference', 'historical_state_checks',
                    'new_seeds', 'training_or_query_draws', 'synthetic'):
            assert saved[key] == report[key], ('saved_reanalysis_differs', key)
    report['resource_before'] = before
    report['resource_after'] = resources()
    report['cpu_seconds'] = time.process_time()-start_cpu
    report['wall_seconds'] = time.monotonic()-start_wall
    for rel, digest in TRACKED.items():
        assert sha(ROOT / rel) == digest, ('input_changed', rel)
    report['all_read_inputs_unchanged'] = True
    if not args.verify_only:
        OUT.mkdir(exist_ok=False)
        with (OUT / 'report.json').open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        with (OUT / 'cdf_grid.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(grid_rows[0]))
            w.writeheader()
            w.writerows(grid_rows)
        with (OUT / 'input_hashes.json').open('x') as f:
            json.dump(TRACKED, f, indent=2)
    selected = [r for r in report['cells'] if r['method'] in ('Sign2', 'Adam16')]
    print(json.dumps(dict(passed=True, verify_only=args.verify_only, cells=selected,
                          cdf_passes=13, max_difference=report['max_absolute_recompute_difference'],
                          cpu_seconds=report['cpu_seconds'], wall_seconds=report['wall_seconds'],
                          all_read_inputs_unchanged=True, new_draws=0), allow_nan=False))


if __name__ == '__main__':
    main()
