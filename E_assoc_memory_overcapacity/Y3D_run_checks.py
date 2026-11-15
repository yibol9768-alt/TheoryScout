"""在单一受限 scope 中串行重跑；不启动正式 launch。"""
import json,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
steps=[
 ['Y3D_prepare.py'],['Y3D_prepare_tests.py'],['Y3D_prepare_freeze.py'],
 ['y3d/test_synthetic.py','--name','synthetic_y3d'],
 ['y3d/test_synthetic.py','--name','synthetic_rerun'],
 ['y3d/test_protocol.py','--name','protocol_rerun'],
 ['y3d/test_final_edges.py'],['y3d/audit_independent.py'],
 ['y3d/audit_false_confirmation.py'],['Y3D_test_r3.py'],['y3d/archive_tests.py']]
results=[]
for i,args in enumerate(steps):
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
    log=ROOT/f'Y3D_check_{i:02d}.log'
    print(json.dumps(dict(stage=i,args=args,starting=True)),flush=True)
    with log.open('x') as f:
        child=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    results.append(dict(stage=i,args=args,returncode=child.returncode,log=log.name,wall_seconds=time.monotonic()-start,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    print(json.dumps(results[-1]),flush=True)
    if child.returncode:break
cg=next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
stat=(Path('/sys/fs/cgroup')/cg.lstrip('/')/'cpu.stat').read_text()
with (ROOT/'Y3D_EXECUTION.json').open('x') as f:
    json.dump(dict(passed=len(results)==len(steps) and all(s['returncode']==0 for s in results),stages=results,
        scope_cpu_stat_before_parent_exit=stat,no_formal_sampling=True),f,indent=2)
assert len(results)==len(steps) and all(s['returncode']==0 for s in results)
