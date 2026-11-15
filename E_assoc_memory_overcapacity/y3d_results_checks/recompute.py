"""独立审读：不导入项目分析器；仅从正式 NPZ 和冻结预测重算。

所有输出以 x 模式创建，拒绝覆盖。统计损失直接用两个平方之差实现。
"""
import os
import sys
import csv
import json
import math
import hashlib
import ctypes
import resource
import time
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CPU, (180, 190))
START = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'y3d_results_checks'
assert Path.cwd().resolve() == ROOT
assert sys.executable == '/home/liuyibo/venvs/theory/bin/python'
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert os.environ[name] == '2'
cg = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
cgpath = Path('/sys/fs/cgroup') / cg.lstrip('/')
assert (cgpath / 'memory.max').read_text().strip() == '3221225472'
assert (cgpath / 'memory.swap.max').read_text().strip() == '0'

import numpy as np
import scipy
from scipy.stats import t

libs = []
for path in sorted({x.split()[-1] for x in Path('/proc/self/maps').read_text().splitlines()
                    if 'openblas' in x and '.so' in x}):
    lib = ctypes.CDLL(path)
    for symbol in ('scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads',
                   'openblas_get_num_threads64_', 'openblas_get_num_threads'):
        if hasattr(lib, symbol):
            fn = getattr(lib, symbol)
            fn.restype = ctypes.c_int
            count = fn()
            assert 1 <= count <= 2
            libs.append(dict(library=path, threads=count))
            break
    else:
        raise RuntimeError('BLAS thread count unavailable')
assert libs

DIMS = (128, 256, 512)
METHODS = ('GD', 'Adam2', 'Adam1', 'Adam.1', 'Sign')
PAIRS = (('GD', 'Sign'), ('Adam2', 'Adam.1'), ('Adam.1', 'Sign'))
TASKS = ('old', 'new')
FORMAL = ROOT / 'y3d/formal'
ANALYSIS = FORMAL / 'analysis'
input_hashes = {}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def track(path):
    key = str(path.relative_to(ROOT))
    h = digest(path)
    assert key not in input_hashes or input_hashes[key] == h
    input_hashes[key] = h
    return h

def load_json(path):
    track(path)
    return json.loads(path.read_text())

def load_csv(path):
    track(path)
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def save_json(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def save_csv(name, rows):
    with (OUT / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def stats(values, family):
    a = np.asarray(values, dtype=np.float64)
    assert a.ndim == 1 and len(a) == 32 and np.isfinite(a).all()
    mean = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
    critical = float(t.ppf(1 - .05 / (2 * family), len(a) - 1))
    p = float(2 * t.sf(abs(mean / se), len(a) - 1)) if se else float(mean == 0)
    return dict(n=len(a), mean=mean, se=se, critical=critical,
                lower=mean-critical*se, upper=mean+critical*se,
                p_two_sided=p, p_bonferroni=min(1., family*p))

def loss(y, p0, p1):
    return (y-p0)**2 - (y-p1)**2

manifest_path = ROOT / 'Y3D_FROZEN.sha256'
manifest_sha = track(manifest_path)
frozen = dict(line.split('  ', 1)[::-1] for line in manifest_path.read_text().splitlines())
for rel in ('Y3D_PREREG.md', 'Y3D_READY.md', 'y3d/config.json', 'y3d/predictions.csv'):
    assert track(ROOT / rel) == frozen[rel]
accepted = load_json(ANALYSIS / 'FINAL_ACCEPTED.json')
assert accepted['publication_state'] == 'accepted'
assert accepted['analysis_complete'] is True and accepted['confirmation_valid'] is True
assert accepted['manifest_sha256'] == manifest_sha
cfg = load_json(ROOT / 'y3d/config.json')
assert cfg['blocks'] == 32 and cfg['min_blocks'] == 24 and cfg['confirm_family'] == 3
assert cfg['equivalence_band'] == .005 and cfg['query_target'] == 8192 and cfg['query_default'] == 512
status = load_json(FORMAL / 'runner_status.json')
assert status['reason'] == 'planned_complete' and status['error'] is None
for rel, expected in accepted['evidence_sha256'].items():
    assert track(FORMAL / rel) == expected, rel
predrows = load_csv(ROOT / 'y3d/predictions.csv')
pred = {}
for row in predrows:
    key = (int(row['d']), int(row['ir']), row['method'], row['task'])
    assert key not in pred
    pred[key] = (float(row['recall_limit']), float(row['recall_first_order']))
assert len(pred) == 90

observations = {}
recall_rows = []
expected_files = {f'd{d}_r{ir}_s{s}.npz' for d in DIMS for ir in range(3) for s in range(32)}
assert {p.name for p in FORMAL.glob('*.npz')} == expected_files
assert not list(FORMAL.glob('*.partial'))
for d in DIMS:
    for ir in range(3):
        for s in range(32):
            path = FORMAL / f'd{d}_r{ir}_s{s}.npz'
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z['metadata']))
                assert (meta['d'], meta['ir'], meta['seed']) == (d, ir, s)
                assert meta['manifest_sha256'] == manifest_sha
                q = 8192 if (d, ir) == (512, 2) else 512
                ids = z['ids']
                assert ids.shape == (q,) and len(np.unique(ids)) == q
                assert ids.min() >= 0 and ids.max() < d*d//4
                assert len(z.files) == 42
                for m in METHODS:
                    for task in TASKS:
                        ok = z[f'{m}_{task}_success']
                        rank = z[f'{m}_{task}_ranks']
                        assert ok.dtype == np.bool_ and ok.shape == rank.shape == (q,)
                        assert np.array_equal(ok, rank <= math.ceil(.1*(d*d//4)))
                        value = float(np.count_nonzero(ok) / q)
                        observations[d, ir, s, m, task] = value
                        recall_rows.append(dict(d=d, ir=ir, seed=s, method=m, task=task, recall=value))
assert len(observations) == 2880
save_csv('recall_from_npz.csv', recall_rows)

endpoint_values = {k: [] for k in ('three_dim_old', 'three_dim_new', 'd512_rminus_half_GD_minus_Sign_old')}
block_rows = []
cell_rows = []
cell_vectors = {}
for s in range(32):
    old_losses, new_losses = [], []
    for d in DIMS:
        for ir in range(3):
            for a, b in PAIRS:
                p0 = pred[d, ir, a, 'old'][0] - pred[d, ir, b, 'old'][0]
                p1 = pred[d, ir, a, 'old'][1] - pred[d, ir, b, 'old'][1]
                y = observations[d, ir, s, a, 'old'] - observations[d, ir, s, b, 'old']
                ell = loss(y, p0, p1)
                old_losses.append(ell)
                if (d, ir, a, b) == (512, 2, 'GD', 'Sign'):
                    endpoint_values['d512_rminus_half_GD_minus_Sign_old'].append(ell)
            for m in METHODS:
                new_losses.append(loss(observations[d, ir, s, m, 'new'], *pred[d, ir, m, 'new']))
    assert len(old_losses) == 27 and len(new_losses) == 45
    endpoint_values['three_dim_old'].append(float(np.mean(old_losses)))
    endpoint_values['three_dim_new'].append(float(np.mean(new_losses)))
    block_rows.append(dict(seed=s, **{k: v[-1] for k, v in endpoint_values.items()}))
endpoints = [dict(endpoint=k, **stats(v, 3), eligible=True) for k, v in endpoint_values.items()]
for row in endpoints:
    row['passed'] = row['lower'] > 0
save_csv('block_losses.csv', block_rows)
save_csv('endpoints_recomputed.csv', endpoints)

for d in DIMS:
    for ir in range(3):
        labels = [('old', a+'-'+b, a, b) for a, b in PAIRS]
        labels += [('new', m, m, None) for m in METHODS]
        for task, label, a, b in labels:
            y = np.array([observations[d, ir, s, a, task] -
                          (observations[d, ir, s, b, task] if b else 0.) for s in range(32)])
            p0, p1 = pred[d, ir, a, task]
            if b:
                p0 -= pred[d, ir, b, task][0]
                p1 -= pred[d, ir, b, task][1]
            st = stats(y - p1, 72)
            width = st['upper'] - st['lower']
            half = width / 2
            equiv = st['lower'] > -.005 and st['upper'] < .005
            row = dict(d=d, ir=ir, r=(.5, 0., -.5)[ir], task=task, label=label,
                       observed=float(y.mean()), uncorrected=p0, corrected=p1, **st,
                       width=width, halfwidth=half, equivalent=equiv,
                       width_at_least_band=width >= .01,
                       excludes_zero=st['lower'] > 0 or st['upper'] < 0,
                       beyond_band=st['lower'] > .005 or st['upper'] < -.005,
                       point_outside_band=abs(st['mean']) >= .005,
                       scaled_residual=d*st['mean'],
                       mean_loss_improvement=float(np.mean(loss(y, p0, p1))))
            cell_rows.append(row)
            cell_vectors[d, ir, task, label] = y
save_csv('diagnostics72_recomputed.csv', cell_rows)

# 旧任务绝对 recall：原次要族 45，区间减去冻结修正值后只作残差描述。
old45 = []
for d in DIMS:
    for ir in range(3):
        for m in METHODS:
            p0, p1 = pred[d, ir, m, 'old']
            y = np.array([observations[d, ir, s, m, 'old'] for s in range(32)])
            st = stats(y-p1, 45)
            old45.append(dict(d=d, ir=ir, r=(.5, 0., -.5)[ir], method=m,
                              observed=float(y.mean()), corrected=p1, **st,
                              excludes_zero=st['lower'] > 0 or st['upper'] < 0,
                              scaled_residual=d*st['mean']))
save_csv('old45_residuals.csv', old45)

# 完成全部独立估计后才读取正式统计表作数值比对。
comparisons = []
def compare(name, independent, official, keys, fields):
    off = {tuple(r[k] for k in keys): r for r in official}
    assert len(off) == len(independent) == len(official)
    maximum = 0.
    for row in independent:
        other = off[tuple(str(row[k]) for k in keys)]
        for field in fields:
            error = abs(float(row[field])-float(other[field]))
            maximum = max(maximum, error)
            assert math.isclose(float(row[field]), float(other[field]), rel_tol=1e-9, abs_tol=1e-12), (name, field, row, other)
    comparisons.append(dict(table=name, rows=len(independent), max_absolute_error=maximum, fields=fields))
compare('endpoints', endpoints, load_csv(ANALYSIS/'endpoints.csv'), ['endpoint'],
        ['n','mean','se','critical','lower','upper','p_two_sided','p_bonferroni'])
compare('diagnostics72', cell_rows, load_csv(ANALYSIS/'diagnostics72.csv'), ['d','ir','task','label'],
        ['n','observed','uncorrected','corrected','mean','se','critical','lower','upper','p_two_sided','p_bonferroni'])
compare('recall', recall_rows, load_csv(ANALYSIS/'all_completed.csv'), ['d','ir','seed','method','task'], ['recall'])

def summarize(rows):
    return dict(n=len(rows), equivalent=sum(r['equivalent'] for r in rows),
                width_at_least_band=sum(r['width_at_least_band'] for r in rows),
                mean_inside_band=sum(not r['point_outside_band'] for r in rows),
                excludes_zero=sum(r['excludes_zero'] for r in rows),
                beyond_band=sum(r['beyond_band'] for r in rows),
                positive=sum(r['mean'] > 0 for r in rows), negative=sum(r['mean'] < 0 for r in rows),
                width_min=min(r['width'] for r in rows), width_max=max(r['width'] for r in rows),
                width_median=float(np.median([r['width'] for r in rows])),
                mean_abs_residual=float(np.mean([abs(r['mean']) for r in rows])),
                mean_residual=float(np.mean([r['mean'] for r in rows])))

summary = dict(endpoints=endpoints, full72=summarize(cell_rows),
               by_dimension={str(d): summarize([r for r in cell_rows if r['d'] == d]) for d in DIMS},
               by_task={task: summarize([r for r in cell_rows if r['task'] == task]) for task in TASKS},
               by_dimension_task={str((d,task)): summarize([r for r in cell_rows if r['d'] == d and r['task'] == task])
                                  for d in DIMS for task in TASKS},
               significant_cells=[r for r in cell_rows if r['excludes_zero']],
               narrow_non_equivalent=[r for r in cell_rows if not r['equivalent'] and not r['width_at_least_band']],
               old45_significant=[r for r in old45 if r['excludes_zero']], comparisons=comparisons)
assert summary['full72']['equivalent'] == 10
assert all(r['passed'] for r in endpoints)
assert accepted['primary_both_passed'] and accepted['d512_passed'] and not accepted['full72_equivalent']
save_json('summary.json', summary)

# 记录所有实际读取的本项目输入，并复核读取期间未变；不遍历其他项目。
for rel, h in input_hashes.items():
    assert digest(ROOT / rel) == h, 'Input changed during review: '+rel
save_json('input_hashes.json', input_hashes)
u = resource.getrusage(resource.RUSAGE_SELF)
save_json('execution.json', dict(python=sys.version, executable=sys.executable,
    numpy=np.__version__, scipy=scipy.__version__, blas=libs,
    memory_max=3221225472, memory_swap_max=0, script_sha256=digest(Path(__file__).resolve()),
    cpu_seconds=u.ru_utime+u.ru_stime, wall_seconds=time.monotonic()-START,
    peak_rss_mib=u.ru_maxrss/1024, input_files=len(input_hashes), inputs_unchanged=True,
    imported_project_analyzer=False, new_random_streams=False, llm_api_calls=False))
print(json.dumps(dict(endpoints=endpoints, full72=summary['full72'], by_dimension=summary['by_dimension'],
                     by_task=summary['by_task'], significant_cells=summary['significant_cells'],
                     narrow_non_equivalent=summary['narrow_non_equivalent'],
                     old45_significant=summary['old45_significant'], comparisons=comparisons),
                 ensure_ascii=False, indent=2))
