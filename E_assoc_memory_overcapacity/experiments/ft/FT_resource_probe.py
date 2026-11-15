import os,ctypes,json,time,resource
from pathlib import Path
import numpy as np
import numpy._core._multiarray_umath as numpy_core
import scipy.linalg._fblas as scipy_blas
start=time.monotonic()
pools=[]
for name,module in [('numpy',numpy_core),('scipy',scipy_blas)]:
    lib=ctypes.CDLL(module.__file__)
    for symbol in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads',
                   'openblas_get_num_threads64_','openblas_get_num_threads']:
        try: fn=getattr(lib,symbol)
        except AttributeError:continue
        fn.restype=ctypes.c_int
        pools.append(dict(dependency=name,symbol=symbol,threads=fn()))
        break
    else:raise RuntimeError(name)
assert all(p['threads']<=2 for p in pools)
out=dict(pools=pools,seconds=time.monotonic()-start,
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
with (Path(__file__).resolve().parent/'FT_resource_probe.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
