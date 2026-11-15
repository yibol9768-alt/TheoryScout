"""顺序执行全部移植及新增测试；每个子进程禁止构造科学 RNG。"""
import json
import os
import resource
import subprocess
import sys
import time
from common_v2 import OUT,dump,read,usage,sha
start=usage();rows=[]
if len(sys.argv)==3 and sys.argv[1]=='--resume-from':
    previous=read(OUT/sys.argv[2])
    for row in previous['scripts']:
        if row['returncode']!=0:break
        assert sha(OUT/row['script'])==row['sha256']
        rows.append(row)
for script in ['test_synthetic.py','test_resume.py','test_credentials.py','test_final.py','test_revision.py','test_volume.py']:
    if any(row['script']==script for row in rows):continue
    before=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic()
    code='''import numpy as np,scipy.stats,scipy.integrate,scipy.optimize,runpy,sys
def forbidden(*a,**k):raise RuntimeError('SYNTHETIC ONLY: scientific RNG forbidden')
for name in ['Generator','PCG64','default_rng','RandomState','normal','standard_normal','choice']:
    setattr(np.random,name,forbidden)
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    p=subprocess.run([sys.executable,'-B','-c',code,str(OUT/script)],cwd=OUT,capture_output=True,text=True)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    row=dict(script=script,sha256=sha(OUT/script),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,
        child_cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,wall=time.monotonic()-wall)
    rows.append(row);print(dict(script=script,returncode=p.returncode,child_cpu=row['child_cpu']),flush=True)
    if p.returncode:
        print(p.stdout);print(p.stderr);break
dump(OUT/'all_tests.json',dict(passed=len(rows)==6 and all(r['returncode']==0 for r in rows),
     scripts=rows,start=start,end=usage(),training_or_query_draws=0,scientific_RNG_guard=True))
if len(rows)!=6 or any(r['returncode'] for r in rows):raise SystemExit(1)
