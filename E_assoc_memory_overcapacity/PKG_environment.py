import json,os,sys,platform
from pathlib import Path
import numpy,scipy,mpmath
cg=next(s.split(':',2)[2].strip() for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
path=Path('/sys/fs/cgroup')/cg.lstrip('/')
report=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,mpmath=mpmath.__version__,machine=platform.machine(),
    memory={k:(path/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.peak','memory.events']},
    threads=len(list(Path('/proc/self/task').iterdir())),thread_environment={k:os.environ[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
assert report['memory']['memory.max']=='3221225472' and report['memory']['memory.swap.max']=='0' and report['threads']<=2
Path('PKG_environment.json').write_text(json.dumps(report,indent=2))
Path('supplement/versions.json').write_text(json.dumps({k:report[k] for k in ['python','numpy','scipy','mpmath','machine']},indent=2))
print(json.dumps(report))
