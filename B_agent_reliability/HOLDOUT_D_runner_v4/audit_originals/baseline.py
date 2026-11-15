from common import *
guards()
files={}
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and not p.is_relative_to(OUT) and p.name!='AUDIT_HOLDOUT_D_AMEND2.md':
        raw=p.read_bytes();files[str(p.relative_to(ROOT))]=dict(size=len(raw),sha256=digest(raw))
manifest={}
for name in ('HOLDOUT_D_PREREG.sha256','HOLDOUT_D_AMEND1.sha256','HOLDOUT_D_AMEND2.sha256'):
    lines=(ROOT/name).read_text().splitlines();bad=[]
    for line in lines:
        sha,rel=line.split('  ',1);p=ROOT/rel
        if not p.is_file() or digest(p.read_bytes())!=sha:bad.append(rel)
    manifest[name]=dict(entries=len(lines),bad=bad,sha256=digest((ROOT/name).read_bytes()))
save('baseline.json',dict(resources=resources(),files=files,manifests=manifest))
print(json.dumps(dict(files=len(files),manifests=manifest),ensure_ascii=False))
