import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/killer_v2'

def check_freeze():
    manifest = (ROOT/'PREREG_KILLER_V2.sha256').read_text()
    for line in manifest.splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    return manifest

def resources():
    status = Path('/proc/self/status').read_text().splitlines()
    threads = int(next(x.split(':')[1] for x in status if x.startswith('Threads:')))
    names = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
    limits = {key: os.environ[key] for key in names}
    assert all(0 < int(v) <= 2 for v in limits.values()) and threads <= 2
    cg = Path('/sys/fs/cgroup') / Path('/proc/self/cgroup').read_text().strip().split('::')[1].lstrip('/')
    meta = {'cgroup': str(cg), 'memory_max': (cg/'memory.max').read_text().strip(),
            'swap_max': (cg/'memory.swap.max').read_text().strip(), 'threads': threads,
            'thread_environment': limits, 'numpy': np.__version__, 'python': os.sys.version,
            'pid': os.getpid(), 'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    assert meta['memory_max'] == str(2*1024**3) and meta['swap_max'] == '0'
    return meta

def begin(mode):
    manifest = check_freeze()
    meta = resources()
    path = OUT/f'{mode}_meta.json'
    assert not path.exists(), f'Refuse overwrite {path}'
    meta.update({'mode': mode, 'complete': False, 'frozen_manifest': manifest})
    path.write_text(json.dumps(meta, indent=2))
    return time.monotonic(), meta, path

def finish(state, complete=True, **extra):
    start, meta, path = state
    meta.update({'complete': complete, 'elapsed_seconds': time.monotonic()-start,
                 'cpu_seconds': time.process_time(),
                 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **extra})
    # Recheck actual live thread count and cgroup limits at the end as well.
    meta['end_resources'] = resources()
    path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k:meta[k] for k in ('mode','complete','elapsed_seconds','max_rss_kib')}), flush=True)

def prior_elapsed(exclude):
    return sum(json.loads(p.read_text()).get('elapsed_seconds',0.)
               for p in OUT.glob('*_meta.json') if p.stem != exclude+'_meta')

def mean_se(v):
    x = np.asarray(v, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x)))
