"""独立审计运行约束；仅允许在本审计目录写入，模型端点始终禁用。"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'audit_d2_checks'

def dump_file(name, obj):
    p = OUT / name
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')

def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def install_guards(loopback_mock=False):
    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if isinstance(path, (str, bytes, os.PathLike)) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                if not Path(os.fsdecode(path)).resolve().is_relative_to(OUT):
                    raise RuntimeError('审计禁止在指定目录之外写入:' + os.fsdecode(path))
        if event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.rename'):
            paths = args[:2] if event == 'os.rename' else args[:1]
            for p in paths:
                if not Path(os.fsdecode(p)).resolve().is_relative_to(OUT):
                    raise RuntimeError('审计禁止目录外变更:' + str(p))
        if event == 'socket.connect':
            address = args[1]
            if not (loopback_mock and isinstance(address, tuple) and address[0] == '127.0.0.1' and address[1] != 18319):
                raise RuntimeError('审计阻止网络连接:' + repr(address))
        if event == 'socket.getaddrinfo':
            if not (loopback_mock and args[0] == '127.0.0.1' and args[1] != 18319):
                raise RuntimeError('审计阻止DNS/模型地址解析')
    sys.addaudithook(audit)

def resource_info():
    rel = Path('/proc/self/cgroup').read_text().strip().split('0::')[-1]
    cg = Path('/sys/fs/cgroup') / rel.lstrip('/')
    z = dict(cgroup=rel, memory_max=(cg/'memory.max').read_text().strip(),
             swap_max=(cg/'memory.swap.max').read_text().strip(),
             threads=len(list(Path('/proc/self/task').iterdir())))
    assert z['memory_max'] == '2147483648' and z['swap_max'] == '0' and z['threads'] <= 2
    return z

def runner_modules():
    import ast
    sys.path.insert(0, str(ROOT/'HOLDOUT_D_runner'))
    import engine
    def local_path(path):
        p = Path(path).resolve()
        if not p.is_relative_to(OUT) or p == OUT:
            raise ValueError('audit_output_only')
        return p
    engine.allowed_dir = local_path
    # 原方法只接受 runner 子目录中的证据；审计把同一目录边界映射为 OUT。
    tree = ast.parse((ROOT/'HOLDOUT_D_runner/engine.py').read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'reconcile_response')
    for n in ast.walk(method):
        if isinstance(n, ast.Constant) and n.value == 'HOLDOUT_D_runner':
            n.value = 'audit_d2_checks'
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(ROOT/'HOLDOUT_D_runner/engine.py'), 'exec'), engine.__dict__, namespace)
    engine.Engine.reconcile_response = namespace['reconcile_response']
    return engine

if __name__ == '__main__':
    install_guards()
    baseline = {}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and not p.is_relative_to(OUT) and p.name != 'AUDIT_HOLDOUT_D_AMEND1.md':
            baseline[str(p.relative_to(ROOT))] = sha_file(p)
    checks = {}
    for name in ('HOLDOUT_D_PREREG.sha256', 'HOLDOUT_D_AMEND1.sha256'):
        errors = []
        lines = (ROOT/name).read_text().splitlines()
        for line in lines:
            digest, path = line.split('  ', 1)
            p = (ROOT/path).resolve()
            if not p.is_relative_to(ROOT) or sha_file(p) != digest:
                errors.append(path)
        checks[name] = dict(entries=len(lines), errors=errors)
    dump_file('baseline.json', dict(files=baseline, manifests=checks, resources=resource_info()))
    print(json.dumps(dict(existing_files=len(baseline), manifests=checks, resources=resource_info()), ensure_ascii=False))
