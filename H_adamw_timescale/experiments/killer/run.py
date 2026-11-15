"""Frozen W protocol. All imports are stdlib or installed numpy, never sibling projects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ETA, B, EPS = .01, 16, 1e-8
DELTAS = np.r_[0., np.geomspace(1e-6, .1, 41)]
SEEDS = list(range(1101, 1109))


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen():
    expected = (ROOT / 'PREREG_KILLER.sha256').read_text().split()[0]
    assert digest(ROOT / 'PREREG_KILLER.md') == expected, 'prereg hash mismatch'
    return expected


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def setup(n, seed, mode=0, ratio=0, size=None):
    streams = np.random.SeedSequence([seed, n, mode, ratio]).spawn(3)
    data, init, shuffle = [np.random.default_rng(s) for s in streams]
    cov = np.arange(1, n + 1, dtype=float) ** -1.5
    cov /= cov.sum()
    teacher = np.where(np.arange(n) % 2 == 0, 1., -1.)
    w0 = init.standard_normal(n)
    w0 *= np.sqrt(n) / np.linalg.norm(w0)
    w = np.tile(w0, (len(DELTAS) if size is None else size, 1))
    return cov, teacher, w, data, shuffle


def direction(w):
    return np.sqrt(w.shape[1]) * w / np.linalg.norm(w, axis=1)[:, None]


def gradient(w, x, y):
    n = w.shape[1]
    u = direction(w)
    gu = (u @ x.T - y) @ x / len(y)
    return (gu - u * (u * gu).sum(axis=1)[:, None] / n) * (
        np.sqrt(n) / np.linalg.norm(w, axis=1)[:, None])


def adam_step(w, m, v, g, t, deltas=DELTAS):
    m = .9 * m + .1 * g
    v = .95 * v + .05 * g * g
    update = (m / (1 - .9 ** t)) / (np.sqrt(v / (1 - .95 ** t)) + EPS)
    new = (1 - np.asarray(deltas)[:, None]) * w - ETA * update
    return new, m, v, update


def risk(w, cov, teacher):
    return ((direction(w) - teacher) ** 2 * cov).sum(axis=1)


def train(n, seed, mode, ratio=0, smoke=False):
    cov, teacher, w, data, shuffle = setup(n, seed, mode, ratio)
    m, v = np.zeros_like(w), np.zeros_like(w)
    checks = ([32] if smoke else [r * n // B for r in [16, 64, 256, 1024]]) if mode == 0 else []
    if mode:
        samples = 32 if smoke else ratio * n
        x_all = data.standard_normal((samples, n)) * np.sqrt(cov)
        y_all = x_all @ teacher + .5 * data.standard_normal(samples)
        epochs = 2 if smoke else 8
        total = epochs * samples // B
        checks = [total]
    else:
        total = checks[-1]
    results, diagnostics = [], []
    min_norm = np.inf
    start = time.monotonic()
    for t in range(1, total + 1):
        if mode:
            j = (t - 1) % (samples // B)
            if j == 0:
                order = shuffle.permutation(samples)
            inds = order[j * B:(j + 1) * B]
            x, y = x_all[inds], y_all[inds]
        else:
            x = data.standard_normal((B, n)) * np.sqrt(cov)
            y = x @ teacher + .5 * data.standard_normal(B)
        g = gradient(w, x, y)
        old = w
        w, m, v, update = adam_step(w, m, v, g, t)
        norms = np.linalg.norm(w, axis=1)
        min_norm = min(min_norm, float(norms.min()))
        if not np.isfinite(w).all() or norms.min() <= 1e-12:
            raise FloatingPointError(f'invalid state n={n} seed={seed} step={t}')
        if t in checks:
            results.append(risk(w, cov, teacher))
            # This explicitly measures the radial term Adam preconditioning reintroduces.
            cos = (old * update).sum(axis=1) / np.maximum(
                np.linalg.norm(old, axis=1) * np.linalg.norm(update, axis=1), 1e-300)
            angular = np.arccos(np.clip((old * w).sum(axis=1) /
                (np.linalg.norm(old, axis=1) * norms), -1., 1.))
            diagnostics.append(np.stack([norms, cos, angular]))
    return np.array(results), np.array(diagnostics), {
        'N': n, 'seed': seed, 'mode': mode, 'ratio': ratio, 'steps': total,
        'D_seen': total * B, 'N_D_seen': n * total * B, 'min_norm': min_norm,
        'wall_seconds': time.monotonic() - start,
    }


def smoke():
    frozen()
    cov, teacher, w, rng, _ = setup(16, 991, size=1)
    x = rng.standard_normal((B, 16)) * np.sqrt(cov)
    y = x @ teacher + .5 * rng.standard_normal(B)
    g = gradient(w, x, y)
    h = 1e-5
    fd = np.empty(16)
    def loss(ww):
        return .5 * np.mean((direction(ww) @ x.T - y) ** 2)
    for j in range(16):
        step = np.zeros_like(w)
        step[0, j] = h
        fd[j] = (loss(w + step) - loss(w - step)) / (2 * h)
    scale_error = float(abs(loss(w * 7.3) - loss(w)) / max(abs(loss(w)), 1e-15))
    grad_error = float(np.linalg.norm(fd - g[0]) / max(np.linalg.norm(g), 1e-15))
    tangent_error = float(abs(np.sum(w * g)) / (np.linalg.norm(w) * np.linalg.norm(g)))
    new, _, _, _ = adam_step(w, np.zeros_like(w), np.zeros_like(w), g, 1, [.02])
    manual = .98 * w - ETA * g / (np.abs(g) + EPS)
    adam_error = float(np.max(np.abs(new - manual)))
    assert scale_error <= 1e-12 and grad_error <= 1e-6
    assert tangent_error <= 1e-12 and adam_error <= 1e-12
    trajectories = []
    for mode in [0, 1]:
        r, d, meta = train(16, 991, mode, 2 if mode else 0, True)
        assert np.isfinite(r).all()
        trajectories.append(meta)
    # Exact norm identity includes Adam's radial cross term.
    norm_identity_error = float(abs(np.sum(new ** 2) - (
        .98 ** 2 * np.sum(w ** 2) - 2 * .98 * ETA * np.sum(w * (g / (abs(g) + EPS)))
        + ETA ** 2 * np.sum((g / (abs(g) + EPS)) ** 2))))
    result = dict(prereg_sha256=frozen(), utc=utc(), scale_error=scale_error,
        grad_error=grad_error, tangent_error=tangent_error, adam_error=adam_error,
        norm_identity_error=norm_identity_error, trajectories=trajectories, passed=True)
    write_json(OUT / 'smoke.json', result)
    print(json.dumps(result), flush=True)


def formal():
    prereg = frozen()
    assert json.loads((OUT / 'smoke.json').read_text())['passed']
    assert not (OUT / 'formal').exists(), 'Refuse to overwrite formal data'
    folder = OUT / 'formal'
    folder.mkdir()
    meta = dict(start_utc=utc(), prereg_sha256=prereg, code_sha256=digest(Path(__file__)),
        numpy=np.__version__, python=os.sys.version, deltas=DELTAS.tolist(), seeds=SEEDS,
        eta=ETA, batch=B, beta1=.9, beta2=.95, adam_epsilon=EPS,
        environment={k: os.getenv(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE']},
        tasks=[], complete=False)
    start = time.monotonic()
    write_json(folder / 'metadata.json', meta)
    tasks = [(n, seed, 0, 0) for n in [32, 64, 128] for seed in SEEDS]
    tasks += [(n, seed, 1, ratio) for n in [32, 64, 128]
        for ratio in [16, 64, 256] for seed in SEEDS]
    try:
        for n, seed, mode, ratio in tasks:
            if time.monotonic() - start > 2850:
                raise TimeoutError('preregistered new-task deadline')
            risks, diagnostics, task = train(n, seed, mode, ratio)
            filename = f'm{mode}_n{n}_r{ratio}_s{seed}.npz'
            np.savez_compressed(folder / filename, risks=risks, diagnostics=diagnostics,
                deltas=DELTAS)
            task['file'] = filename
            meta['tasks'].append(task)
            meta['wall_seconds'] = time.monotonic() - start
            write_json(folder / 'metadata.json', meta)
            print(json.dumps(task), flush=True)
        meta['complete'] = True
    except Exception as exc:
        meta['error'] = repr(exc)
        raise
    finally:
        meta['end_utc'] = utc()
        meta['wall_seconds'] = time.monotonic() - start
        meta['peak_rss_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        write_json(folder / 'metadata.json', meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['smoke', 'formal'])
    args = parser.parse_args()
    (smoke if args.mode == 'smoke' else formal)()
