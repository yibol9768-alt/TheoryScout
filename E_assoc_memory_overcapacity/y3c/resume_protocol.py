"""全套已运行到末尾；修正测试CSV恢复后核验保存证据并补完失败检查。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(7,8))
import ast,csv,json
from common import ROOT,OUT,sha,dump,usage,config
from publication import verify_accepted,finalize
from integrity import read_json
common_sources=['common.py','run.py','analyze.py','launch.py','integrity.py','publication.py','predict.py']
prior=read_json(OUT/'final_edges/results.json')
for name in common_sources:assert sha(OUT/name)==prior['source_sha256'][name]
TEST=OUT/'protocol_v2';mh=sha(ROOT/'Y3B_FROZEN.sha256');cfg=config();checks=[]
def check(n,c):
    assert c,n
    checks.append(n)
check('full_run_last_failure_preserved',(TEST/'FAILED.json').is_file())
# 保存的所有审计夹具与独立单故障结果都是最终执行模块实际运行的产物。
audits=sorted(TEST.glob('audit_*'));check('all_13_audit_fixtures',len(audits)==13)
for case in audits:
    data=case/'formal';p=read_json(data/'analysis/PENDING.json');s=read_json(data/'launch_status.json')
    check(case.name+'_rejected',not p['candidate_confirmation_valid'] and bool(p['integrity_errors']) and not s['publication_eligible'])
    with (data/'analysis/all_completed.csv').open() as f:n=sum(1 for _ in csv.DictReader(f))
    check(case.name+'_all_readable_disclosed',n==p['complete_cells']*10)
mutants=sorted(TEST.glob('*/integrity_test.json'));check('all_43_independent_mutants',len(mutants)==43)
for path in mutants:
    check(path.parent.name+'_integrity_errors',bool(read_json(path)['errors']))
    check(path.parent.name+'_no_accepted',not (path.parent/'formal/analysis/FINAL_ACCEPTED.json').exists())
for name in ['missing_events','legal_json_illegal_event','commit_hash_mismatch','event_status_mismatch']:
    data=TEST/name/'formal';p=read_json(data/'analysis/PENDING.json')
    check(name+'_analysis_invalid',not p['confirmation_valid'] and not p['candidate_confirmation_valid'])
    with (data/'analysis/all_completed.csv').open() as f:n=sum(1 for _ in csv.DictReader(f))
    check(name+'_all_288_disclosed',n==2880)
# 从另一个完整独立分析输出取回原始字节，并要求与原凭据哈希完全相同。
base=TEST/'control_normal/formal';target=base/'analysis/endpoints.csv'
expected=read_json(base/'analysis/FINAL_ACCEPTED.json')['evidence_sha256']['analysis/endpoints.csv']
reference=TEST/'actual_analysis_after_pending/formal/analysis/endpoints.csv'
check('restore_reference_matches_original_commit',sha(reference)==expected)
target.write_bytes(reference.read_bytes())
check('original_positive_credential_restored',verify_accepted(base,mh,cfg)['confirmation_valid'])
check('positive_all_three_pass',read_json(base/'analysis/FINAL_ACCEPTED.json')['primary_both_passed'] and read_json(base/'analysis/FINAL_ACCEPTED.json')['d512_passed'])
# 直接重新执行修正后的字节篡改/恢复检查。
original=target.read_bytes();target.write_bytes(original+b'tampered\n')
try:verify_accepted(base,mh,cfg)
except AssertionError:checks.append('post_commit_tamper_rejected')
else:raise AssertionError('tamper should fail')
finally:target.write_bytes(original)
check('byte_restore_passed',verify_accepted(base,mh,cfg)['confirmation_valid'])
try:finalize(base,mh,cfg,{})
except FileExistsError:checks.append('accepted_cannot_overwrite')
else:raise AssertionError('overwrite should fail')
for name in ['soft_stop_partial','soft_stop_empty']:
    data=TEST/name/'formal';a=verify_accepted(data,mh,cfg)
    check(name+'_valid_execution',a['confirmation_valid'])
    check(name+'_no_statistical_confirmation',not a['primary_both_passed'] and not a['d512_passed'])
for name in ['analysis_nonzero','analysis_hard_flag','analysis_negative_return','supervisor_over_budget',
             'total_cpu_over_budget','total_wall_over_budget','exit_evidence_disagrees',
             'changed_after_analysis','missing_report','missing_pending','analysis_cpu_over_budget']:
    data=TEST/name/'formal';s=read_json(data/'launch_status.json')
    check(name+'_supervisor_rejects',not s['publication_eligible'] and bool(s['publication_errors']) and not (data/'analysis/FINAL_ACCEPTED.json').exists())
injections=[]
for mode,code in [('after_pending',-9),('before_pending',-9),('nonzero',7)]:
    data=TEST/('actual_analysis_'+mode)/'formal';s=read_json(data/'launch_status.json')
    check(mode+'_actual_returncode',s['stages'][1]['returncode']==code)
    check(mode+'_no_credential',not s['publication_eligible'] and not (data/'analysis/FINAL_ACCEPTED.json').exists())
    if mode!='before_pending':
        p=read_json(data/'analysis/PENDING.json')
        check(mode+'_valid_pending_never_confirms',p['candidate_confirmation_valid'] and not p['confirmation_valid'])
    injections.append(dict(case=mode,returncode=code,accepted=False))
data=TEST/'actual_runner_kill_after_status/formal';s=read_json(data/'launch_status.json')
check('actual_runner_sigkill',s['stages'][0]['returncode']==-9 and not s['publication_eligible'])
old=read_json(ROOT/'y3b/synthetic_v4_tests.json');new=read_json(OUT/'synthetic_final_tests.json')
check('all_59_original_synthetic_checks',new['passed'] and old['checks']==new['checks'] and len(new['checks'])==59)
check('formal_never_started',not (OUT/'formal').exists() and not (ROOT/'y3b/formal').exists())
check('corrected_full_test_uses_byte_restore',"write_bytes(original_bytes)" in (OUT/'test_protocol.py').read_text())
dump(TEST/'results.json',dict(passed=True,checks=checks,injections=injections,
    completed_by='完整协议执行到末尾CSV恢复失败后，修正测试恢复并重核持久化证据；失败保留，未重新抽样。',
    original_full_run_cpu_seconds=51.152,original_full_run_failed_at='positive_credential_restored',
    authoritative_runtime_unchanged=True,source_sha256={p.name:sha(p) for p in OUT.glob('*.py')},
    usage=usage(),cpu_seconds_children=0.,no_formal_sampling=True,no_random_training_or_queries=True))
print(json.dumps(dict(passed=True,checks=len(checks),usage=usage())))
