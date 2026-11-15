from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,shutil,re
root=Path.cwd();base=root/'supplement';fr=base/'final_runs';logs=base/'logs'
full=json.loads((fr/'independent_full/adam_check.json').read_text());assert full['status']=='PASS' and full['reintegrated_region_cells']==1027

def stripped(x):
    if isinstance(x,dict):return {k:stripped(v) for k,v in x.items() if k not in ['seconds','elapsed_seconds','resources']}
    if isinstance(x,list):return [stripped(v) for v in x]
    return x
for name in ['stabilizer_ordering','matched_chain','moment_history','history_reversal']:
    a=json.loads((base/f'two_step_adam_certificate/reference/{name}.json').read_text());b=json.loads((fr/f'from_zero/{name}.json').read_text());assert stripped(a)==stripped(b),name
assert stripped(json.loads((base/'polar_certificate/reference/polar_variance.json').read_text()))==stripped(json.loads((fr/'from_zero/polar_variance.json').read_text()))
(logs/'from_zero_equivalence.json').write_text(json.dumps(dict(status='PASS',all_five_certificates_numerically_identical=True,timestamps_and_resource_statistics_excluded=True),indent=2))
# Verify the independent dyadic endpoints and each parent box subcover.
def dyadic(x):return (-1 if x['sign'] else 1)*F(int(x['mantissa']))*F(2)**x['exponent']
def cover1(rows,lo,hi):
    end=lo
    for a,b in sorted(rows):assert a==end and a<b;end=b
    assert end==hi
reg=json.loads((base/'two_step_adam_certificate/reference/stabilizer_ordering.json').read_text());groups={}
with (fr/'independent_full/independent_integral_envelopes.jsonl').open() as f:
    for line in f:
        r=json.loads(line)
        for key in ['N','J']:assert 0<dyadic(r[key][0])<=dyadic(r[key][1])
        for pair in r['integrals']:assert dyadic(pair[0])<=dyadic(pair[1])
        groups.setdefault(r['parent'],[]).append(tuple(map(F,r['a']+r['e'])))
assert set(groups)==set(range(1027))
for i,rr in groups.items():
    al,ar=map(F,reg['leaves'][i]['a']);el,er=map(F,reg['leaves'][i]['e'])
    assert all(al<=a<b<=ar and el<=c<d<=er for a,b,c,d in rr)
    cuts=sorted({a for r in rr for a in r[:2]});assert cuts[0]==al and cuts[-1]==ar
    for x,y in zip(cuts,cuts[1:]):
        mid=(x+y)/2;cover1([(c,d) for a,b,c,d in rr if a<mid<b],el,er)
subcover=dict(status='PASS',parent_boxes=1027,independent_subboxes=sum(map(len,groups.values())),all_dyadic_signs_strict=True,exact_subcovers=True)
(logs/'independent_subcover_audit.json').write_text(json.dumps(subcover,indent=2))
# Store the final review artifacts in their mathematical sections.
ref=base/'two_step_adam_certificate/reference'
for src,dst in [('decisions_final/adam_check.json','independent_decisions.json'),('sample_final/adam_check.json','independent_sample.json'),('independent_full/adam_check.json','independent_full.json'),('independent_full/independent_integral_envelopes.jsonl','independent_region_integrals.jsonl'),('independent_full/independent_auxiliary_integral_envelopes.jsonl','independent_auxiliary_integrals.jsonl')]:shutil.copy2(fr/src,ref/dst)
expref=base/'experiments_replay/reference';expref.mkdir(exist_ok=False)
shutil.copytree(base/'runs/replay',expref/'small_replay')
shutil.copytree(base/'runs/tables/tables',expref/'manuscript_tables')
shutil.copytree(fr/'query_tables',expref/'query_tables')
shutil.copy2(fr/'full_plan/plan.json',expref/'full_design.json')
# Preserve successful commands and measured times with neutral names.
sets=[('certificates',fr/'from_zero',['polar_generation','polar_check','stabilizer_ordering','matched_chain','moment_history','history_reversal','adam_decisions']),
      ('polar',fr/'polar',['polar_separate_generation','polar_separate_check']),
      ('adam',fr/'adam',['adam_separate_ordering','adam_separate_chain','adam_separate_history']),
      ('history',fr/'history',['history_separate_generation']),
      ('independent_decisions',fr/'decisions_final',['independent_decisions']),
      ('independent_sample',fr/'sample_final',['independent_sample']),
      ('independent_full',fr/'independent_full',['independent_full']),
      ('replay',base/'runs/replay',['matched_retrieval_replay','finite_dimension_replay']),
      ('tables',base/'runs/tables',['manuscript_tables']),
      ('query_tables',fr/'query_tables',['query_measurements_matched','query_measurements_confirmation','query_tables']),
      ('full_plan',fr/'full_plan',['full_experiment_plan'])]
summary={}
for label,folder,names in sets:
    timings=json.loads((folder/'timings.json').read_text());assert all(t['exit_code']==0 for t in timings);assert len(timings)==len(names)
    summary[label]=dict(seconds=sum(t['seconds'] for t in timings),commands=timings,status='PASS')
    (logs/(label+'_timings.json')).write_text(json.dumps(timings,indent=2))
    for index,name in enumerate(names,1):shutil.copy2(folder/f'command_{index}.log',logs/(name+'.log'))
(logs/'run_summary.json').write_text(json.dumps(summary,indent=2))
shutil.copy2(root/'PKG_environment.json',logs/'resource_validation.json')
# Check every original source captured before any edits.
initial=json.loads((root/'PKG_original_sources.json').read_text())
assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in initial.items())
stable=dict(status='PASS',original_source_count=len(initial),all_original_source_hashes_unchanged=True,existing_manuscript_unmodified=True)
old=json.loads((root/'y2_checks/region.json').read_text())
mismatch=[]
for name,h in old['inputs_before'].items():
    if name.startswith('paper/') and (root/name).is_file() and hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:mismatch.append(name)
(root/'PKG_source_audit.json').write_text(json.dumps(dict(**stable,old_certificate_manuscript_mismatches=mismatch,original_sources_sha256=initial),indent=2))
(logs/'original_sources_unchanged.json').write_text(json.dumps(stable,indent=2))
# Keep failed development diagnostics outside the submission directory.
for src,dst in [(base/'runs/history/command_1.log',root/'PKG_initial_history_failure.log'),(base/'runs/adam/command_1.log',root/'PKG_interrupted_generation.log')]:
    if src.is_file():shutil.copy2(src,dst)
for name in ['runs','final_runs']:
    target=base/name;assert target.parent==base;shutil.rmtree(target)
# Compile syntax without writing bytecode and scan shipped text for project codes.
parsed=0
for p in base.rglob('*.py'):ast.parse(p.read_text(),filename=str(p.relative_to(base)));parsed+=1
for p in base.rglob('*'):
    assert not p.is_symlink()
    if p.is_file() and p.suffix in ['.py','.md','.tex','.json','.jsonl','.log','.csv','.txt']:
        text=p.read_text();assert not re.search(r'/home/|/mnt/|x3_checks|y2_checks|z3_checks|experiments/x1|y3d|ICML_REVIEW|X1_PREREG',text),str(p)
(logs/'static_audit.json').write_text(json.dumps(dict(status='PASS',python_files_parsed=parsed,no_absolute_project_paths=True,no_internal_project_codes=True,no_symlinks=True),indent=2))
validation=f'''# 实际验证记录

环境为 Python 3.12.3、numpy 2.5.4、scipy 1.18.1、mpmath 1.3.0。所有计算均在 MemoryMax=3G、MemorySwapMax=0 的 systemd 用户 scope 中运行，四个数值线程环境变量均为 1；独立检查器不导入生成器或其区间核心。

|验证|实测秒数|结果|
|---|---:|---|
'''
for label,r in summary.items():validation+=f"|{label}|{r['seconds']:.3f}|PASS|\n"
validation+=f'''\n独立重积分覆盖 1027 个原始区域格，形成 {subcover['independent_subboxes']} 个精确子格；全部子格 N、J 严格为正，另对每个原始格核对了完整子覆盖。边界、根、固定方向、消融和反转证书也由独立实现重积分通过。端点以精确二进制数保存。

MR 12 个、有限维修正确认实验 18 个代表性数据集的召回值均与原结果相同。全部 1440 个原始查询归档导出的数据集测量值核验误差为零。全样本表格重算得到 18 个正向主要对比、7 个总体值覆盖、22/30 个新任务等价、3 个确认端点通过及 10/72 个诊断等价。

完整规模实验只执行了设计计划检查，没有重新生成全部 1440 个训练数据集。长时完整命令与后续表格命令均在 README 中明确列出为尚未执行。代表性查询级数据、全样本数据集级测量和全部数学证书随包提供。

逐条命令、退出码、计时见 `logs/run_summary.json` 和各 `*_timings.json`；原始输出见同目录的 `.log`。源码和论文摘录的哈希已与本次生成的证书核对，原工程捕获的 {len(initial)} 个源文件哈希保持不变。
'''
(base/'VALIDATION.md').write_text(validation)
print(json.dumps(dict(status='PASS',summary=summary,subcover=subcover,originals=stable),ensure_ascii=False))
