from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
root=Path.cwd();base=root/'supplement'
def manifest():
    paths=sorted(p for p in base.rglob('*') if p.is_file() and p.name!='manifest.json')
    data=dict(format=1,sha256={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},files=len(paths))
    (base/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
manifest();tmp=base/'relocation_validation';tmp.mkdir(exist_ok=False)
for src in list(base.iterdir()):
    if src==tmp:continue
    if src.is_dir():shutil.copytree(src,tmp/src.name)
    else:shutil.copy2(src,tmp/src.name)
checks=[('integrity','verify_package.py',[]),('independent_decisions','two_step_adam_certificate/check_adam.py',['--certificate-dir',str(tmp/'two_step_adam_certificate/reference'),'--output',str(tmp/'relocated_decisions.json')]),('manuscript_tables','experiments_replay/recompute_tables.py',['--output',str(tmp/'relocated_tables')])]
records=[]
for label,script,args in checks:
    start=time.monotonic();p=subprocess.run([sys.executable,'-B',str(root/'PKG_relocation_worker.py'),str(tmp),str(tmp/script),*args],capture_output=True,text=True)
    with (root/('PKG_relocated_'+label+'.log')).open('x') as f:f.write(p.stdout+p.stderr)
    assert p.returncode==0,(label,p.stdout,p.stderr)
    records.append(dict(check=label,status='PASS',seconds=time.monotonic()-start,original_project_accesses=0))
shutil.rmtree(tmp)
result=dict(status='PASS',checks=records,relocated_package_only=True,original_project_reads_blocked_by_audit_hook=True)
(root/'PKG_relocation.json').write_text(json.dumps(result,indent=2))
(base/'logs/relocation_validation.json').write_text(json.dumps(result,indent=2))
with (base/'VALIDATION.md').open('a') as f:f.write('\n脱离原目录复核：复制整个包至另一目录，并用 Python audit hook 禁止访问包外的原工程文件；完整性检查、独立 Adam 检查与全样本表格重算均通过。\n')
manifest()
p=subprocess.run([sys.executable,'-B',str(base/'verify_package.py')],capture_output=True,text=True);assert p.returncode==0,p.stderr
(root/'PKG_integrity_verification.json').write_text(json.dumps(dict(status='PASS',stdout=p.stdout.strip()),indent=2))
print(json.dumps(result));print(p.stdout)
