"""执行原 test_mock.py；仅把 BASE 和 allowed_dir 的输出限制映射至审计目录。"""
import ast
from common import ROOT, OUT, install_guards, resource_info, runner_modules

install_guards(loopback_mock=True)
runner_modules()
print('RESOURCE', resource_info(), flush=True)
source = ROOT/'HOLDOUT_D_runner/test_mock.py'
tree = ast.parse(source.read_text(), str(source))
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BASE' for t in node.targets):
        node.value = ast.parse("Path(" + repr(str(OUT/'original_mock_r2')) + ")", mode='eval').body
ast.fix_missing_locations(tree)
exec(compile(tree, str(source), 'exec'), globals())
