import ast,hashlib,json,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(30,35))
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_y3b_checks'
ledger=json.loads((ROOT/'y3b/input_hashes.json').read_text())
extra=[]
for p in ROOT.rglob('*.py'):
    if OUT in p.parents or ROOT/'y3b' in p.parents or str(p.relative_to(ROOT)) in ledger:continue
    try:tree=ast.parse(p.read_text())
    except SyntaxError:continue
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['SeedSequence','default_rng','RandomState'] and node.args:
            try:v=ast.literal_eval(node.args[0])
            except (ValueError,TypeError):continue
            extra.append(dict(path=str(p.relative_to(ROOT)),line=node.lineno,literal=v))
with (OUT/'additional_seed_initializer_sources.json').open('x') as f:json.dump(extra,f,indent=2)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
artifacts=[ROOT/'AUDIT_Y3B.md']+[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='ARTIFACTS.sha256']
with (OUT/'ARTIFACTS.sha256').open('x') as f:
    for p in artifacts:f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
print(json.dumps(dict(artifacts=len(artifacts),additional_seed_sources=extra),ensure_ascii=False))
