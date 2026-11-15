"""只读验收五份冻结清单、旧文件、v3原件及新增路径范围。"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNNER=ROOT/'HOLDOUT_D_runner_v4'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    record=sys.argv[1:]==['--record-before-freeze']
    counts={}
    names=['HOLDOUT_D_PREREG.sha256',*[f'HOLDOUT_D_AMEND{i}.sha256' for i in range(1,4 if record else 5)]]
    for name in names:
        lines=(ROOT/name).read_text().splitlines()
        for line in lines:
            digest,rel=line.split('  ',1);p=(ROOT/rel).resolve()
            assert p.is_relative_to(ROOT) and sha(p.read_bytes())==digest,(name,rel)
        counts[name]=len(lines)
    baseline=json.loads((ROOT/'HOLDOUT_D_AMEND4_BASELINE.json').read_text());external=[]
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
        assert p.is_relative_to(RUNNER) or (p.parent==ROOT and (p.name.startswith('HOLDOUT_D_AMEND4_') or p.name=='HOLDOUT_D_AMEND4.sha256')),('unexpected_new',rel)
        new.append(rel)
    for path in ('HOLDOUT_D_runner','HOLDOUT_D_runner_v2','HOLDOUT_D_runner_v3','HOLDOUT_D_runner_v4'):
        runner=ROOT/path
        if path!='HOLDOUT_D_runner':assert not (runner/'HOLDOUT_D_LIVE_RELEASE.json').exists()
        assert not (runner/'HOLDOUT_D_live').exists()
    # 独立新增12项只允许更换目标运行器，测试断言逐字保留。
    original=(ROOT/'audit_d4_checks/boundary_tests.py').read_text()
    assert (RUNNER/'amend4_checks/boundary_tests.py').read_text()==original.replace("engine_module(ROOT/'HOLDOUT_D_runner_v3')","engine_module(ROOT/'HOLDOUT_D_runner_v4')")
    if not record:
        frozen={line.split('  ',1)[1] for line in (ROOT/'HOLDOUT_D_AMEND4.sha256').read_text().splitlines()}
        assert frozen==set(new)-{'HOLDOUT_D_AMEND4.sha256'},('unfrozen_new_files',sorted(set(new)-frozen-{'HOLDOUT_D_AMEND4.sha256'}))
    z=dict(passed=True,manifests=counts,baseline_files=len(baseline),new_files=len(new),external_append_only_logs=external,
           original_12_assertions_unchanged=True,stage='before_final_freeze' if record else 'final_readonly')
    if record:(ROOT/'HOLDOUT_D_AMEND4_INTEGRITY.json').write_text(json.dumps(z,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(z,ensure_ascii=False))
if __name__=='__main__':main()
