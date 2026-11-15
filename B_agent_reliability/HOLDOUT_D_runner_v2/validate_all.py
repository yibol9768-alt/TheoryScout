"""单入口回归，约束全部写入、连接及实际资源。"""
import json
import os
from pathlib import Path
import sys
import time
import unittest
from protocol import ROOT, dump

OUT=ROOT/'HOLDOUT_D_runner_v2'
def audit(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            if not Path(os.fsdecode(path)).resolve().is_relative_to(OUT):raise RuntimeError('write_outside_v2:'+str(path))
    if event in ('os.mkdir','os.remove','os.rmdir','os.rename'):
        for p in (args[:2] if event=='os.rename' else args[:1]):
            if not Path(os.fsdecode(p)).resolve().is_relative_to(OUT):raise RuntimeError('mutation_outside_v2')
    if event in ('socket.connect','socket.getaddrinfo'):
        address=args[1] if event=='socket.connect' else args[:2]
        if not (isinstance(address,tuple) and address[0]=='127.0.0.1' and address[1]!=18319):
            raise RuntimeError('forbidden_network:'+repr(address))
sys.addaudithook(audit)
rel=Path('/proc/self/cgroup').read_text().strip().split('0::')[-1]
cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
resources=dict(cgroup=rel,memory_max=(cg/'memory.max').read_text().strip(),swap_max=(cg/'memory.swap.max').read_text().strip(),
               threads=len(list(Path('/proc/self/task').iterdir())),thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
assert resources['memory_max']=='2147483648' and resources['swap_max']=='0' and resources['threads']<=2
assert all(v in ('1','2') for v in resources['thread_env'].values())
import test_mock
import test_amend2
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_mock,test_amend2)])
start=time.time();result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),seconds=time.time()-start,
            resources=resources,artifacts=str(test_mock.BASE.relative_to(ROOT)),no_real_llm=True,no_real_quota=True,
            old_tests=22,old_smoke_positive_assertion_replaced='E2禁止测试伪造人工已审，改为拒绝未批准模板，其余旧断言保留',
            network='仅旧HTTP测试的fork本地随机端口mock；禁止18319和外部连接')
with (test_mock.BASE/'HOLDOUT_D_ALL_TEST_RESULTS.json').open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(dump(report));raise SystemExit(0 if result.wasSuccessful() else 1)
