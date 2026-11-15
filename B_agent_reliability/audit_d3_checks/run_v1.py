import ast,time,types,unittest
from common import *
guards(loopback=True)
v1=ROOT/'HOLDOUT_D_runner';sys.path.insert(0,str(v1))
source=(v1/'engine.py').read_text().replace("ROOT/'HOLDOUT_D_runner'",'Path('+repr(str(OUT))+')')
m=types.ModuleType('engine');m.__file__=str(v1/'engine.py');sys.modules['engine']=m
exec(compile(source,m.__file__,'exec'),m.__dict__)
path=v1/'test_mock.py';tree=ast.parse(path.read_text(),str(path))
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in node.targets):
        node.value=ast.parse('Path('+repr(str(OUT/'v1_tests'))+')',mode='eval').body
ast.fix_missing_locations(tree)
test=types.ModuleType('test_mock');test.__file__=str(path);sys.modules['test_mock']=test
exec(compile(tree,str(path),'exec'),test.__dict__)
r0=resources();start=time.time();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test))
save('v1_results.json',dict(resources=r0,tests=result.testsRun,seconds=time.time()-start,passed=result.wasSuccessful(),
    failures=[(str(t),e) for t,e in result.failures],errors=[(str(t),e) for t,e in result.errors],
    mapping='Only BASE, allowed_dir and reconcile evidence directory. Original v1 assertions unchanged. Test-made v1 smoke approval is a fixture, not human release.'))
raise SystemExit(0 if result.wasSuccessful() else 1)
