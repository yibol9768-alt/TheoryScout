"""本项目既有文件前后核验；不跟随指向项目外的链接。"""
import hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit_y3c_checks'
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def snapshot():
    result = {}
    for folder, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if Path(folder)/d != OUT)
        for name in sorted(files):
            p=Path(folder)/name
            if p == ROOT/'AUDIT_Y3C.md': continue
            rel=str(p.relative_to(ROOT))
            if p.is_symlink(): result[rel]={'symlink':os.readlink(p)}
            else: result[rel]={'size':p.stat().st_size, 'sha256':sha(p)}
    return result
mode=sys.argv[1]
now=snapshot()
if mode=='before':
    with (OUT/'baseline.json').open('x') as f: json.dump(now,f,indent=2)
    print(json.dumps({'baseline_files':len(now)}))
elif mode=='after':
    before=json.loads((OUT/'baseline.json').read_text())
    changes={k:{'before':v,'after':now.get(k)} for k,v in before.items() if now.get(k)!=v}
    new=sorted(set(now)-set(before))
    report={'original_files':len(before),'changed':changes,'unexpected_new':new,
            'formal_y3c_exists':(ROOT/'y3c/formal').exists(),
            'formal_y3b_exists':(ROOT/'y3b/formal').exists()}
    with (OUT/'unchanged.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))
