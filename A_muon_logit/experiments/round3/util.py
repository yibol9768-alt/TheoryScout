"""Round 3 offline utilities. Project imports are confined to this repository."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
import json
import hashlib
import resource
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXPERIMENTS = ROOT / 'experiments'
DATA = EXPERIMENTS / 'results'
sys.path.insert(0, str(EXPERIMENTS))
from model import make_bundle, forward, subset, SCALE, H, DH  # noqa: E402

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_prereg():
    expected = (ROOT / 'PREREG_ROUND3.sha256').read_text().split()[0]
    assert sha(ROOT / 'PREREG_ROUND3.md') == expected, 'Frozen prereg changed'
    return expected

def serial(x):
    if isinstance(x, np.ndarray):
        return serial(x.tolist())
    if isinstance(x, np.generic):
        return serial(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    return x

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(serial(obj), f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')
    temp.replace(path)

def rng(index, stage):
    return np.random.default_rng(np.random.SeedSequence([20261106, index, stage]))

def metadata(start):
    return dict(elapsed_seconds=time.monotonic()-start,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                executable=sys.executable, numpy_version=np.__version__,
                threads={k: os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                device='CPU; NumPy only; torch not imported',
                enforced_scope='systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0',
                prereg_sha256=verify_prereg())

def check_time(start, limit=9000):
    if time.monotonic()-start > limit:
        raise TimeoutError('Round3 stage time limit; keep partial artifacts')

def fit_beta(g, y, bound):
    """Constrained through-origin LS, same operation in every bootstrap."""
    cap = bound / np.maximum(np.max(np.abs(g), axis=-2), 1e-300)
    numerator = np.sum(g*y, axis=-2)
    denominator = np.sum(g*g, axis=-2)
    return np.clip(numerator / np.maximum(denominator, 1e-300), 0, cap)

def predict(g, beta, bound):
    return np.clip(g*beta, -bound, bound)

def max_acf(residual):
    """Input (...,450,C). Exact n-k numerator and n denominator, all 4 windows."""
    windows = residual[..., 50:450, :].reshape(*residual.shape[:-2], 4, 100, residual.shape[-1])
    z = windows - windows.mean(axis=-2, keepdims=True)
    ft = np.fft.rfft(z, n=256, axis=-2)
    corr = np.fft.irfft(ft * ft.conj(), n=256, axis=-2)[..., :21, :]
    denominator = corr[..., 0:1, :] / 100
    values = corr[..., 1:21, :] / np.arange(99, 79, -1)[:, None]
    values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
    return np.max(np.abs(values), axis=(-3,-2,-1))

def holm(pvalues):
    values = np.asarray(pvalues)
    order = np.argsort(values, kind='stable')
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values),0,-1)))
    return adjusted

def blocks(n, generator, repeats=500, length=10):
    starts = generator.integers(n, size=(repeats, (n+length-1)//length))
    return ((starts[...,None]+np.arange(length)) % n).reshape(repeats,-1)[:, :n]

def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if min(np.std(x), np.std(y)) < 1e-15:
        return None
    return float(np.corrcoef(x,y)[0,1])
