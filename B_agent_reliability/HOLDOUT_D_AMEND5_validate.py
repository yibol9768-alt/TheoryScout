"""只读验收：六份冻结清单、旧件基线、v4原断言、新增路径与真实状态缺席。"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT/'HOLDOUT_D_runner_v5'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    record=sys.argv[1:]==['--record-before-freeze']
    assert not sys.argv[1:] or record
    counts={}
    for name in ['HOLDOUT_D_PREREG.sha256',*[f'HOLDOUT_D_AMEND{i}.sha256' for i in range(1,5 if record else 6)]]:
        lines=(ROOT/name).read_text().splitlines()
        for line in lines:
            digest,rel=line.split('  ',1);p=(ROOT/rel).resolve()
            assert p.is_relative_to(ROOT) and sha(p.read_bytes())==digest,(name,rel)
        counts[name]=len(lines)
    baseline=json.loads((ROOT/'HOLDOUT_D_AMEND5_BASELINE.json').read_text());external=[]
    for rel,z in baseline.items():
        p=ROOT/rel;raw=p.read_bytes()
        if sha(raw)==z['sha256']:continue
        assert p.parent==ROOT and p.name.startswith('codex_') and p.suffix=='.log' and sha(raw[:z['size']])==z['sha256'],('old_file_changed',rel)
        external.append(dict(path=rel,original_size=z['size'],current_size=len(raw),original_prefix_unchanged=True))
    new=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=str(p.relative_to(ROOT))
        if rel in baseline:continue
        assert p.is_relative_to(R) or (p.parent==ROOT and (p.name.startswith('HOLDOUT_D_AMEND5_') or p.name=='HOLDOUT_D_AMEND5.sha256')),('unexpected_new',rel)
        new.append(rel)
    old=ROOT/'HOLDOUT_D_runner_v4'
    assert (R/'amend5_checks/boundary_tests.py').read_text()==(old/'amend4_checks/boundary_tests.py').read_text().replace('HOLDOUT_D_runner_v4','HOLDOUT_D_runner_v5')
    assert (R/'amend5_checks/resume_tests.py').read_bytes()==(old/'amend4_checks/resume_tests.py').read_bytes()
    for name in ('test_amend2.py','test_amend3.py'):
        assert (R/name).read_bytes()==(old/name).read_bytes(),name
    assert (R/'test_mock.py').read_text()==(old/'test_mock.py').read_text().replace('HOLDOUT_D_runner_v4','HOLDOUT_D_runner_v5')
    for path in ('HOLDOUT_D_runner','HOLDOUT_D_runner_v2','HOLDOUT_D_runner_v3','HOLDOUT_D_runner_v4','HOLDOUT_D_runner_v5'):
        runner=ROOT/path
        if path!='HOLDOUT_D_runner':assert not (runner/'HOLDOUT_D_LIVE_RELEASE.json').exists()
        assert not (runner/'HOLDOUT_D_live').exists()
    all_results=json.loads((R/'amend5_checks/results/ALL_RESULTS.json').read_text())
    assert all_results['passed'] and len(all_results['results'])==14 and all(x['exit']==0 for x in all_results['results'])
    if not record:
        frozen={line.split('  ',1)[1] for line in (ROOT/'HOLDOUT_D_AMEND5.sha256').read_text().splitlines()}
        assert frozen==set(new)-{'HOLDOUT_D_AMEND5.sha256'},('unfrozen_new_files',sorted(set(new)-frozen-{'HOLDOUT_D_AMEND5.sha256'}))
    z=dict(passed=True,manifests=counts,baseline_files=len(baseline),new_files=len(new),external_append_only_logs=external,
           original_v4_assertions_unchanged=True,no_live_release_or_ledger=True,all_14_jobs_passed=True,stage='before_final_freeze' if record else 'final_readonly')
    if record:(ROOT/'HOLDOUT_D_AMEND5_INTEGRITY.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(z,ensure_ascii=False))
if __name__=='__main__':main()
