import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from harness import OUT,ROOT,resources
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot():
    return {str(p.relative_to(ROOT)):dict(size=p.stat().st_size,sha256=sha(p)) for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink() and not p.is_relative_to(OUT) and p.name!='AUDIT_HOLDOUT_D_AMEND3.md'}
def manifests():
    results={}
    for name in ('HOLDOUT_D_PREREG.sha256','HOLDOUT_D_AMEND1.sha256','HOLDOUT_D_AMEND2.sha256','HOLDOUT_D_AMEND3.sha256'):
        lines=(ROOT/name).read_text().splitlines();bad=[]
        for line in lines:
            digest,path=line.split('  ',1)
            if sha(ROOT/path)!=digest:bad.append(path)
        results[name]=dict(entries=len(lines),mismatch=bad,sha256=sha(ROOT/name))
    return results
if __name__=='__main__':
    start=time.time();r=resources();baseline=snapshot();mf=manifests()
    (OUT/'baseline.json').write_text(json.dumps(dict(files=baseline,manifests=mf,resources=r),indent=2)+'\n')
    assert all(not z['mismatch'] for z in mf.values()),mf
    jobs=[['suite',v] for v in ('v1','v2','v3')]
    jobs += [['original',s] for s in ('attack_entries.py','endpoint_escape.py','url_resolution.py','repeated_dispatch.py')]
    jobs += [['ported',s] for s in ('attack_entries.py','endpoint_escape.py','url_resolution.py','repeated_dispatch.py','attack_smoke.py','check_scores.py')]
    jobs += [['boundary']]
    results=[]
    for args in jobs:
        label='_'.join(args).replace('.py','');path=OUT/(label+'.log');print('START',label,flush=True)
        command=[sys.executable,'-B',str(OUT/('boundary_tests.py' if args==['boundary'] else 'harness.py')),*([] if args==['boundary'] else args)]
        with path.open('x') as log:p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(job=args,exit=p.returncode,log=path.name));print('DONE',label,p.returncode,flush=True)
        (OUT/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    end=snapshot();changes=[]
    for name,z in baseline.items():
        if end.get(name)!=z:
            p=ROOT/name;prefix=hashlib.sha256(p.read_bytes()[:z['size']]).hexdigest() if p.exists() else None
            changes.append(dict(path=name,before=z,after=end.get(name),old_prefix_unchanged=prefix==z['sha256']))
    final=dict(results=results,passed=all(x['exit']==0 for x in results),seconds=time.time()-start,resources=r,manifests=manifests(),old_file_changes=changes,unexpected_new=sorted(set(end)-set(baseline)))
    (OUT/'final_results.json').write_text(json.dumps(final,indent=2,ensure_ascii=False)+'\n');print(json.dumps(final,ensure_ascii=False),flush=True)
