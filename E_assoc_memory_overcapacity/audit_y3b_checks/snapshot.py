import hashlib, json, sys, resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
root = Path(__file__).resolve().parents[1]
out = root / 'audit_y3b_checks'
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def files():
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not p.is_symlink() and out not in p.parents
            and p.name != 'AUDIT_Y3B.md' and p.suffix != '.log'}
if sys.argv[1] == 'before':
    with (out/'baseline.json').open('x') as f: json.dump(files(), f, indent=2)
else:
    before = json.loads((out/'baseline.json').read_text()); after = files()
    result = dict(checked=len(before), changed=[p for p,h in before.items() if after.get(p)!=h],
                  new_existing_area_files=sorted(set(after)-set(before)),
                  formal_exists=(root/'y3b/formal').exists())
    with (out/'unchanged.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result))
