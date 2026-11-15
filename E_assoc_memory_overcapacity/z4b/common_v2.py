"""Z4 paths, exclusive durable output, environment and freeze checks. No draws."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'z4b'
METHODS = ['GD', 'Adam2', 'Adam.1', 'Sign']
PAIRS = [('GD', 'Sign'), ('Adam2', 'Adam.1'), ('Adam.1', 'Sign')]
ARMS = ['empirical', 'analytic']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def read(path):
    def pairs(xs):
        d = {}
        for k, v in xs:
            if k in d: raise ValueError('duplicate key: ' + k)
            d[k] = v
        return d
    def bad(x): raise ValueError('nonfinite JSON: ' + x)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)

def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def commit(path, arrays):
    import numpy as np
    p = Path(path).with_suffix('.partial')
    with p.open('xb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.link(p, path); p.unlink()
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def usage():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return dict(cpu=u.ru_utime + u.ru_stime, mono=time.monotonic(), rss_mib=u.ru_maxrss / 1024)

def config(): return read(OUT / 'config.json')

def entropy(phase, d, method, seed, stream):
    # phase 0=calibration, 1=evaluation; method=-1 is shared evaluation data.
    return [20261106, 3187362049, 45, phase, d, method + 1, seed, stream]

def plan(cfg):
    jobs = []
    for d in cfg['dims']:
        for m in range(len(METHODS)):
            for s in range(cfg['n_cal']): jobs.append(['cal', d, m, s])
    jobs.append(['lock'])
    for s in range(cfg['n_eval']):
        dims = cfg['dims'][s % len(cfg['dims']):] + cfg['dims'][:s % len(cfg['dims'])]
        for d in dims: jobs.append(['eval', d, s])
    return jobs

def jobname(job): return '_'.join(map(str, job)) + ('.json' if job == ['lock'] else '.npz')

def environment():
    import ctypes
    import numpy as np
    import scipy
    from scipy.stats import t
    for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        if os.environ.get(k) != '2': raise ValueError('thread env: ' + k)
    cg = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    cp = Path('/sys/fs/cgroup') / cg.lstrip('/')
    if (cp / 'memory.max').read_text().strip() != '3221225472': raise ValueError('MemoryMax')
    if (cp / 'memory.swap.max').read_text().strip() != '0': raise ValueError('MemorySwapMax')
    pools=[]
    for path in sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if 'openblas' in line and '.so' in line}):
        lib=ctypes.CDLL(path)
        for symbol in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads',
                       'openblas_get_num_threads64_','openblas_get_num_threads']:
            if hasattr(lib,symbol):
                fn=getattr(lib,symbol);fn.restype=ctypes.c_int;threads=fn()
                if not 1<=threads<=2: raise ValueError('actual threads')
                pools.append(dict(path=path,sha256=sha(path),threads=threads));break
        else: raise ValueError('unknown BLAS thread symbol')
    if not pools: raise ValueError('no BLAS')
    return dict(python=sys.version, executable=sys.executable, executable_sha256=sha(sys.executable),
                numpy=np.__version__, scipy=scipy.__version__,
                pools=pools,
                memory_max=3221225472, memory_swap_max=0)

def verify():
    manifest = ROOT / 'Z4B_FROZEN.sha256'
    seen = set()
    for line in manifest.read_text().splitlines():
        h, rel = line.split('  ', 1)
        p = ROOT / rel
        if rel in seen or p.is_symlink() or not p.resolve().is_relative_to(ROOT): raise ValueError('manifest path')
        if not (rel.startswith('z4b/') or rel.startswith('Z4B_')): raise ValueError('manifest scope')
        if sha(p) != h: raise ValueError('freeze hash: ' + rel)
        seen.add(rel)
    required = {'Z4B_PREREG.md','Z4B_READY.md','Z4B_SEED_PROOF.md'} | {'z4b/' + n for n in
        ['common_v2.py','run_v2.py','analyze_v2.py','protocol_v2.py','launch_v2.py','entry.py',
         'model.py','config.json','environment.json','streams.json','seed_audit.json',
         'power.json','resources.json','port_synthetic_tests.json','resume_tests.json',
         'credential_tests.json','final_tests.json','revision_tests.json','volume_result.json',
         'input_hashes.json','baseline_check.json','freeze.py','seed_domain.py','seed_extension.json']}
    if not required <= seen: raise ValueError('incomplete manifest')
    if environment() != read(OUT / 'environment.json'): raise ValueError('environment mismatch')
    for rel,h in read(OUT/'input_hashes.json').items():
        if sha(ROOT/rel)!=h: raise ValueError('input hash: '+rel)
    if not read(OUT/'power.json')['design_qualified']: raise ValueError('power design failed')
    import numpy as np
    if sha(np.random.bit_generator.__file__) != read(OUT/'seed_extension.json')['binary_sha256']:
        raise ValueError('SeedSequence binary changed')
    return sha(manifest)

def scheduled(cfg, seed, target):
    return target == cfg['primary_target'] or seed < cfg['side_n_eval']

def expected_n(cfg, target):
    return cfg['n_eval'] if target == cfg['primary_target'] else min(cfg['side_n_eval'],cfg['n_eval'])
