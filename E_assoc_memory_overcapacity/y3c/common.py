"""Y3C 公共约束；无随机抽样，所有产物排他创建。"""
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'y3c'
METHODS = ['GD', 'Adam2', 'Adam1', 'Adam.1', 'Sign']
TASKS = ['old', 'new']
PAIRS = [('GD', 'Sign'), ('Adam2', 'Adam.1'), ('Adam.1', 'Sign')]
DIMS = [128, 256, 512]
RS = [.5, 0., -.5]
BASE_ORDER = [(128, 0), (256, 0), (512, 0), (128, 1), (256, 1),
              (512, 1), (128, 2), (256, 2), (512, 2)]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def table(path, rows, fields=None):
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def environment():
    import numpy as np
    import scipy
    from scipy.stats import t  # 同时加载分析所用动态库
    for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        assert os.environ.get(k) == '2', k
    assert sys.version_info[:3] == (3, 12, 3)
    cg = next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    cp = Path('/sys/fs/cgroup') / cg.lstrip('/')
    assert (cp / 'memory.max').read_text().strip() == '3221225472'
    assert (cp / 'memory.swap.max').read_text().strip() == '0'
    libs = []
    for p in sorted({s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines()
                     if 'openblas' in s and '.so' in s}):
        lib = ctypes.CDLL(p)
        for symbol in ['scipy_openblas_get_num_threads64_', 'scipy_openblas_get_num_threads',
                       'openblas_get_num_threads64_', 'openblas_get_num_threads']:
            if hasattr(lib, symbol):
                fn = getattr(lib, symbol)
                fn.restype = ctypes.c_int
                threads = fn()
                assert 1 <= threads <= 2
                libs.append(dict(path=p, sha256=sha(p), threads=threads))
                break
        else:
            raise RuntimeError('无法查询 BLAS 线程数: ' + p)
    assert libs
    return dict(python=sys.version, executable=sys.executable, executable_sha256=sha(sys.executable),
                numpy=np.__version__, scipy=scipy.__version__, machine=platform.machine(),
                byteorder=sys.byteorder, blas=libs, float_dtype='float64',
                bit_generator='numpy.random.PCG64', memory_max=3221225472, memory_swap_max=0)

def verify():
    manifest = ROOT / 'Y3C_FROZEN.sha256'
    entries = manifest.read_text().splitlines()
    assert entries
    seen = set()
    for line in entries:
        h, rel = line.split('  ', 1)
        p = ROOT / rel
        assert p.resolve().is_relative_to(ROOT) and rel not in seen
        seen.add(rel)
        assert sha(p) == h, '冻结哈希不符: ' + rel
    required = {'Y3C_PREREG.md', 'Y3C_READY.md', 'y3c/config.json', 'y3c/environment.json',
                'y3c/predictions.csv', 'y3c/streams.json', 'y3c/run.py', 'y3c/analyze.py',
                'y3c/common.py','y3c/launch.py','y3c/predict.py','y3c/parameters.json',
                'y3c/integrity.py','y3c/publication.py'}
    assert required <= seen
    assert environment() == json.loads((OUT / 'environment.json').read_text()), '环境已改变'
    return sha(manifest)

def config():
    return json.loads((OUT / 'config.json').read_text())

def entropy(d, ir, seed, stream):
    return [20261106, 87410329, 83, d, ir, seed, stream]

def jobs(count):
    for seed in range(count):
        k = seed % 9
        for d, ir in BASE_ORDER[k:] + BASE_ORDER[:k]:
            yield d, ir, seed

def usage():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return dict(cpu_seconds=u.ru_utime + u.ru_stime, peak_rss_mib=u.ru_maxrss / 1024,
                monotonic=time.monotonic())
