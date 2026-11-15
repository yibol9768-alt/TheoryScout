"""仅生成 Y3D 的归档/冻结辅助程序。"""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'y3d'
(OUT/'archive_tests.py').write_text('''"""归档本轮确定性重跑，历史副本保留。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,25))
import os,subprocess,json
from common import ROOT,OUT,sha,dump,usage
names=['synthetic_y3d','synthetic_rerun','protocol_rerun','final_rerun','audit_rerun','r3_rerun']
assert all((OUT/n).is_dir() for n in names)
for n in ['protocol_rerun','final_rerun','r3_rerun']:assert json.loads((OUT/n/'results.json').read_text())['passed']
path=OUT/'Y3D_test_fixtures.tar'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
subprocess.run(['tar','--sort=name','--mtime=@0','--owner=0','--group=0','--numeric-owner','-cf',str(path),
                *[str((OUT/n).relative_to(ROOT)) for n in names]],cwd=ROOT,check=True)
with path.open('rb') as f:os.fsync(f.fileno())
u=resource.getrusage(resource.RUSAGE_CHILDREN)
dump(OUT/'Y3D_archive_status.json',dict(passed=True,directories=names,archive_sha256=sha(path),usage=usage(),
     cpu_seconds_children=u.ru_utime+u.ru_stime,no_formal_sampling=True))
print(json.dumps(dict(passed=True,usage=usage(),cpu_seconds_children=u.ru_utime+u.ru_stime)))
''')
(OUT/'freeze.py').write_text('''"""Y3D 冻结；核验旧文件边界、科学不变性及本轮测试。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(45,50))
import ast,difflib,json,os
from pathlib import Path
from common import ROOT,OUT,sha,dump,usage,environment,config
assert not any((ROOT/n/'formal').exists() for n in ['y3b','y3c','y3d'])
manifest=ROOT/'Y3D_FROZEN.sha256';assert not manifest.exists()
assert environment()==json.loads((OUT/'environment.json').read_text())
baseline={}
for line in (ROOT/'Y3D_BASELINE.sha256').read_text().splitlines():
    h,rel=line.split('  ',1);baseline[rel]=h
changed=[rel for rel,h in baseline.items() if not (ROOT/rel).is_file() or sha(ROOT/rel)!=h]
assert not changed,changed
unexpected=[]
for base,dirs,names in os.walk(ROOT):
    dirs[:]=[n for n in dirs if Path(base)/n != OUT]
    for name in names:
        p=Path(base)/name;rel='./'+str(p.relative_to(ROOT))
        if rel not in baseline and p.suffix!='.log' and not name.startswith('Y3D_'):unexpected.append(rel)
assert not unexpected,unexpected
identical=[]
for name in ['config.json','parameters.json','predictions.csv','streams.json','seed_audit.json','rng_inventory.json',
             'power_summary.json','primary_power.json','planning_covariances.json','power_grid.csv','environment.json',
             'environment_packages.json','input_hashes.json','predict.py']:
    assert sha(OUT/name)==sha(ROOT/'y3c'/name),name
    identical.append(name)
assert sha(ROOT/'Y3D_PREREG.md')==sha(ROOT/'Y3C_PREREG.md')
identical.append('PREREG')
scientific={};diffs=[]
for name,allowed in [('analyze.py',{'main'}),('common.py',{'verify'}),('run.py',{'main'}),('predict.py',set())]:
    a=(ROOT/'y3c'/name).read_text();b=(OUT/name).read_text()
    def nodes(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    before,after=nodes(a),nodes(b)
    changed_fns={k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}
    assert changed_fns==allowed,(name,changed_fns)
    scientific[name]=dict(changed=sorted(changed_fns),unchanged=sorted(set(before)-changed_fns))
    diffs.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='y3c/'+name,tofile='y3d/'+name))
for name in ['publication.py','integrity.py','launch.py']:
    diffs.extend(difflib.unified_diff((ROOT/'y3c'/name).read_text().splitlines(True),(OUT/name).read_text().splitlines(True),fromfile='y3c/'+name,tofile='y3d/'+name))
with (ROOT/'Y3D_SOURCES.diff').open('x') as f:f.writelines(diffs)
results={}
for name in ['synthetic_y3d_tests.json','synthetic_rerun_tests.json','protocol_rerun/results.json','final_rerun/results.json','r3_rerun/results.json']:
    result=json.loads((OUT/name).read_text());assert result['passed'],name
    results[name]=dict(checks=len(result['checks']),usage=result['usage'],cpu_seconds_children=result['cpu_seconds_children'])
    if 'synthetic' in name:
        assert len(result['checks'])==59 and result['checks']==json.loads((ROOT/'y3c/synthetic_final_tests.json').read_text())['checks']
for folder in ['final_rerun','r3_rerun']:
    r=json.loads((OUT/folder/'results.json').read_text())
    for name,h in r['source_sha256'].items():
        if name!='freeze.py':assert sha(OUT/name)==h,(folder,name)
faults=json.loads((OUT/'audit_rerun/independent_faults/results.json').read_text())
for r in faults['results']:
    if r['case'] in ['report_mutation_before_accept','decision_mutation_before_accept']:assert not r['accepted'] and not r['verified']
false=json.loads((OUT/'audit_rerun/false_confirmation/results.json').read_text())
assert not false['accepted'] and not false['verification_succeeded'] and all(v is False for v in false['original_decisions'].values())
archive=json.loads((OUT/'Y3D_archive_status.json').read_text());assert archive['passed']
assert sha(OUT/'Y3D_test_fixtures.tar')==archive['archive_sha256']
audit=dict(schema=1,no_formal_sampling=True,no_llm_api=True,original_nonlog_files_checked=len(baseline),
           original_nonlog_files_changed=changed,unexpected_new_nonlog_files=unexpected,
           byte_identical=identical,scientific_functions=scientific,results=results,
           audit_faults=faults['results'],R3a_rejected=True,R3b_rejected=True,
           statistics_predictions_parameters_seeds_budget_unchanged=True,config=config(),
           validation_cpu_seconds=sum(r['usage']['cpu_seconds']+r['cpu_seconds_children'] for r in results.values())+
              faults['usage']['cpu_seconds']+faults['child_cpu_seconds']+false['usage']['cpu_seconds']+false['child_cpu_seconds']+
              archive['usage']['cpu_seconds']+archive['cpu_seconds_children'],
           accounting_boundary='本字段汇总列明测试/归档进程及其子进程；复制、快照、生成辅助脚本及本次冻结不包含，非准备总预算实测。历次修订验证不伪称累计小于原200秒。正式及准备预算配置均不变。',
           log_boundary='既有非log文件全量前后内容比对；未向既有log打开写入，会话自动追加日志不作零变化保证。',
           current_test_archive='y3d/Y3D_test_fixtures.tar',freeze_usage=usage())
audit['freeze_helper_correction']=dict(failure_log='Y3D_freeze.log',failed_process_returncode=1,failed_cpu_hard_upper_seconds=50,reason='冻结辅助程序缺少Path导入；首次完整旧文件哈希复核通过后，在枚举新增文件时失败。补齐辅助导入并记录该修正，所有正式模块仍须匹配测试时源码哈希。',source_hash_exception=['freeze.py'])
dump(ROOT/'Y3D_FREEZE_AUDIT.json',audit)
# 历史展开夹具由从Y3C复制的原test_fixtures.tar留存；只归档本轮新证据。
paths=sorted([p for p in OUT.iterdir() if p.is_file()]+[OUT/n for n in ['protocol_rerun/results.json','final_rerun/results.json','r3_rerun/results.json','audit_rerun/independent_faults/results.json','audit_rerun/false_confirmation/results.json']]+[p for p in ROOT.glob('Y3D_*') if p.is_file() and (p.suffix!='.log' or p.name=='Y3D_freeze.log')])
assert len(paths)==len(set(paths)) and all(not p.is_symlink() for p in paths)
with manifest.open('x') as f:
    for p in paths:f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\\n')
    f.flush();os.fsync(f.fileno())
os.chmod(manifest,0o444)
print(json.dumps(dict(frozen_files=len(paths),manifest_sha256=sha(manifest),usage=usage())))
''')
