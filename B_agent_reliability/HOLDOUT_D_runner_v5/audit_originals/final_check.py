from common import *
guards();base=json.loads((OUT/'baseline.json').read_text());changes=[];missing=[];new=[];unchanged=0
for name,old in base['files'].items():
    p=ROOT/name
    if not p.is_file():missing.append(name);continue
    raw=p.read_bytes()
    if digest(raw)==old['sha256']:unchanged+=1
    else:changes.append(dict(file=name,before=old['size'],after=len(raw),preserved_prefix=len(raw)>=old['size'] and digest(raw[:old['size']])==old['sha256']))
for p in ROOT.rglob('*'):
    if p.is_file() and not p.is_relative_to(OUT) and p.name!='AUDIT_HOLDOUT_D_AMEND2.md' and str(p.relative_to(ROOT)) not in base['files']:
        new.append(str(p.relative_to(ROOT)))
manifests={}
for name in ('HOLDOUT_D_PREREG.sha256','HOLDOUT_D_AMEND1.sha256','HOLDOUT_D_AMEND2.sha256'):
    bad=[];lines=(ROOT/name).read_text().splitlines()
    for line in lines:
        expected,rel=line.split('  ',1);p=ROOT/rel
        if not p.is_file() or digest(p.read_bytes())!=expected:bad.append(rel)
    manifests[name]=dict(entries=len(lines),bad=bad)
writers=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    for fd in ('1','2'):
        try:
            dest=os.readlink(p/'fd'/fd)
            if dest==str(ROOT/'codex_daudit3.log'):writers.append(dict(pid=int(p.name),fd=fd,comm=(p/'comm').read_text().strip(),target=dest))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
unexpected=[x for x in changes if x['file']!='codex_daudit3.log' or not x['preserved_prefix']]
save('final_integrity.json',dict(resources=resources(),baseline_files=len(base['files']),unchanged=unchanged,changes=changes,missing=missing,unexpected_new=new,
    unexpected_changes=unexpected,manifests=manifests,external_stdout_writers=writers,
    no_release=not (RUNNER/'HOLDOUT_D_LIVE_RELEASE.json').exists(),no_live_ledger=not (RUNNER/'HOLDOUT_D_live').exists()))
assert not missing and not new and not unexpected and all(not m['bad'] for m in manifests.values())
print(json.dumps(dict(unchanged=unchanged,changes=changes,manifests=manifests,external_stdout_writers=writers),ensure_ascii=False))
