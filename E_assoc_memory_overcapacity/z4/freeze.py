"""最终排他冻结；只新建输出，检查旧字节、科学输入、测试与未抽样。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(45,50))
import ast
import os
from common_v2 import ROOT,OUT,read,dump,sha,environment,usage

manifest=ROOT/'Z4_FROZEN.sha256'
assert not manifest.exists() and not (OUT/'formal').exists()
baseline=read(OUT/'resume_baseline.json')
changed={p:dict(before=h,after=sha(ROOT/p)) for p,h in baseline.items() if sha(ROOT/p)!=h}
assert set(changed)<= {'PKG_continuation_resources.json'},changed
inputs=read(OUT/'input_hashes.json')
assert all(sha(ROOT/p)==h for p,h in inputs.items())
additional_inputs={'y3_checks/Y3_frozen_predictions.csv':sha(ROOT/'y3_checks/Y3_frozen_predictions.csv')}
dump(OUT/'additional_inputs.json',additional_inputs)
assert environment()==read(OUT/'environment.json')
assert read(OUT/'resume_final_audit.json')['passed']
tests=read(OUT/'resume_tests.json');creds=read(OUT/'credential_tests.json')
final=read(OUT/'final_tests.json')
assert tests['passed'] and len(tests['tests'])==34
assert creds['passed'] and len(creds['checks'])==8
assert final['passed'] and len(final['checks'])==5
assert all(sha(OUT/n)==h for n,h in final['source_sha256'].items())
assert tests['training_or_query_draws']==creds['training_or_query_draws']==0
# Check every source syntax without writing pyc or importing old runners.
for p in OUT.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
old=read(OUT/'planning_usage.json');new=read(OUT/'resume_usage.json')
ledger=dict(original_planning_recorded_cpu=old['end']['cpu'],
    original_synthetic_recorded_self_cpu=read(OUT/'synthetic_tests.json')['end']['cpu'],
    resume_build_recorded_self_cpu=read(OUT/'resume_build.json')['usage_cpu'],
    resume_tests_recorded_self_cpu=tests['end']['cpu'],
    resume_tests_including_children_conservative_charge=140,
    resume_offline_recorded_self_cpu=new['end']['cpu'],
    credentials_recorded_cpu=creds['end']['cpu']+creds['child_cpu'],
    final_audit_recorded_cpu=read(OUT/'resume_final_audit.json')['usage']['cpu'],
    this_freeze_hard_charge=50,postfreeze_verify_hard_charge=10,
    preparation_reserve=600,
    unknown_historical_preparation='not measured; no claim that missing history was zero or bounded by reserve')
audit=dict(passed=True,tests_passed=47,training_or_query_draws=0,formal_sampling=False,
    existing_baseline_files_checked=len(baseline),concurrent_noninput_changes=changed,
    scientific_inputs_checked=len(inputs),additional_test_input_hashes=additional_inputs,
    existing_z4_unchanged=True,environment_verified=True,
    authoritative_runtime=['entry.py','launch_v2.py','run_v2.py','analyze_v2.py','protocol_v2.py','common_v2.py','model.py'],
    original_failures_preserved=['synthetic_tests.json','resume_audit.json'],resource_ledger=ledger,
    usage=usage(),scope='Only new Z4_* and z4 paths written; root baseline excludes automatically changing logs.')
dump(OUT/'freeze_audit.json',audit)
paths=sorted([p for p in OUT.rglob('*') if p.is_file()]+[p for p in ROOT.glob('Z4_*') if p.is_file()])
assert len(paths)==len(set(paths)) and all(not p.is_symlink() for p in paths)
with manifest.open('x') as f:
    for p in paths:f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    f.flush();os.fsync(f.fileno())
os.chmod(manifest,0o444)
print(dict(passed=True,frozen_files=len(paths),manifest_sha256=sha(manifest),usage=usage()))
