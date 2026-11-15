import ast,types,unittest
from common import *
guards();modules()
# Use the same output-only mapping as the main suite.
p=RUNNER/'test_mock.py';tree=ast.parse(p.read_text(),str(p))
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in node.targets):node.value=ast.parse('Path('+repr(str(OUT/'dispatch_attacks'))+')',mode='eval').body
ast.fix_missing_locations(tree)
m=types.ModuleType('test_mock');m.__file__=str(p);sys.modules['test_mock']=m
exec(compile(tree,str(p),'exec'),m.__dict__)
from test_amend3 import Amend3Tests
names=[n for n in unittest.defaultTestLoader.getTestCaseNames(Amend3Tests) if n.startswith('test_R2_')]
result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Amend3Tests(n) for n in names))
save('repeated_dispatch.json',dict(resources=resources(),tests=result.testsRun,passed=result.wasSuccessful(),cases=names,projection='Only live identity and missing release checks projected; actual durable claims, locks and disk WAL checked at opener boundary. No socket.'))
assert result.wasSuccessful()
