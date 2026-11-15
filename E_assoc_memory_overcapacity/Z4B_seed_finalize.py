"""新增种子域证明后的源绑定回归；仅合成，保留此前阳性证据。"""
import os
import resource
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'z4b'))
from common_v2 import OUT,read,dump,usage
assert read(OUT/'seed_extension.json')['passed']
(OUT/'final_tests.json').rename(OUT/'before_seed_extension_final_tests.json')
(OUT/'synthetic_final').rename(OUT/'synthetic_final_before_seed_extension')
code='''import numpy as np,scipy.stats,scipy.integrate,scipy.optimize,runpy,sys
def forbidden(*a,**k):raise RuntimeError('SYNTHETIC ONLY')
for name in ['Generator','PCG64','default_rng','RandomState','normal','standard_normal','choice']:
    setattr(np.random,name,forbidden)
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
p=subprocess.run([sys.executable,'-B','-c',code,str(OUT/'test_final.py')],cwd=OUT,capture_output=True,text=True)
u=resource.getrusage(resource.RUSAGE_CHILDREN)
dump(OUT/'seed_extension_test_receipt.json',dict(passed=p.returncode==0,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
     child_cpu=u.ru_utime+u.ru_stime,usage=usage(),training_or_query_draws=0))
assert p.returncode==0,p.stderr
if (OUT/'baseline_check.json').exists():(OUT/'baseline_check.json').rename(OUT/'baseline_check_before_seed_extension.json')
old=read(OUT/'seed_extension.json')['old_explicit_unique']
for name in ['Z4B_PREREG.md','Z4B_READY.md']:
    path=ROOT/name;path.write_text(path.read_text().replace('1125506',str(old)))
print(dict(passed=True,old_explicit_unique=old,child_cpu=u.ru_utime+u.ru_stime))
