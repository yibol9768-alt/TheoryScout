"""Offline, sequential CPU experiment utilities. No network dependencies."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '3'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/week1-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')

import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'results'
FIG = ROOT / 'experiments' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def serial(x):
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, np.generic):
        return serial(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    return x


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w',encoding='utf-8') as handle:
        json.dump(serial(data),handle,ensure_ascii=False,allow_nan=False,separators=(',', ':'))
    tmp.replace(path)


def ci(x):
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    e = 1.96 * float(x.std(ddof=1)) / math.sqrt(x.size)
    return {'mean': m, 'ci95': [m-e, m+e], 'independent_repeats': x.size}


def binomial_ci(k, n):
    from scipy.special import betaincinv
    return [0.0 if k == 0 else float(betaincinv(k, n-k+1, .025)),
            1.0 if k == n else float(betaincinv(k+1, n-k, .975))]


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class Budget:
    def __init__(self):
        self.start = time.monotonic()
    def check(self, new_run=False):
        elapsed = time.monotonic()-self.start
        if rss_bytes() > 3_000_000_000:
            raise RuntimeError('Active RSS stop at 3.0 GB')
        if elapsed > (55 if new_run else 58)*60:
            raise TimeoutError('Preregistered execution budget reached')
    def elapsed(self):
        return time.monotonic()-self.start


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
