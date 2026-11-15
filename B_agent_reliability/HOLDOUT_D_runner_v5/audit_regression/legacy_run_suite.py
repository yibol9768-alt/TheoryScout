import ast, time, types, unittest, threading
from common import *
guards(loopback=True)
RUNNER=ROOT/'HOLDOUT_D_runner_v2'
import common
common.RUNNER=RUNNER
modules()
source=(ROOT/'HOLDOUT_D_runner_v2')/'test_mock.py';tree=ast.parse(source.read_text(),str(source))
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in node.targets):
        node.value=ast.parse('Path('+repr(str(OUT/'v2_all_tests'))+')',mode='eval').body
ast.fix_missing_locations(tree)
m=types.ModuleType('test_mock');m.__file__=str(source);sys.modules['test_mock']=m
exec(compile(tree,str(source),'exec'),m.__dict__)
sys.path.insert(0,str(ROOT/'HOLDOUT_D_runner_v2'))
import test_amend2
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(x) for x in (m,test_amend2)])
r0=resources();thread_checks=[];original_start=threading.Thread.start
def checked_start(self,*args,**kw):
    result=original_start(self,*args,**kw)
    observed=len(list(Path('/proc/self/task').iterdir()));thread_checks.append(observed)
    assert observed<=2,('actual_thread_limit',observed)
    return result
threading.Thread.start=checked_start
start=time.time();result=unittest.TextTestRunner(verbosity=2).run(suite)
save('v2_suite_results.json',dict(tests=result.testsRun,failures=[(str(t),e) for t,e in result.failures],errors=[(str(t),e) for t,e in result.errors],thread_start_checks=thread_checks,
                            passed=result.wasSuccessful(),seconds=time.time()-start,resources=r0,end_resources=resources(),
                            mapping='Only test BASE, engine.allowed_dir, reconcile evidence path; no assertions modified.'))
raise SystemExit(0 if result.wasSuccessful() else 1)
