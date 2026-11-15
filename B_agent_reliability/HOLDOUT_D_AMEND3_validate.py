"""只读核验四份冻结清单、全部旧文件与审计脚本原始副本。"""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    counts={}
    for name in ('HOLDOUT_D_PREREG.sha256','HOLDOUT_D_AMEND1.sha256','HOLDOUT_D_AMEND2.sha256','HOLDOUT_D_AMEND3.sha256'):
        lines=(ROOT/name).read_text().splitlines()
        for line in lines:
            digest,rel=line.split('  ',1);p=(ROOT/rel).resolve()
            assert p.is_relative_to(ROOT) and sha(p.read_bytes())==digest,(name,rel)
        counts[name]=len(lines)
    baseline=json.loads((ROOT/'HOLDOUT_D_AMEND3_BASELINE.json').read_text());external=[]
    for name,z in baseline.items():
        p=ROOT/name;raw=p.read_bytes()
        if sha(raw)==z['sha256']:continue
        assert p.suffix=='.log' and p.name.startswith('codex_') and sha(raw[:z['size']])==z['sha256'],name
        external.append(dict(file=name,old_size=z['size'],new_size=len(raw),original_prefix_unchanged=True))
    for p in (ROOT/'audit_d3_checks').glob('*.py'):
        assert p.read_bytes()==(ROOT/'HOLDOUT_D_runner_v3/audit_originals'/p.name).read_bytes(),p.name
    new_files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=str(p.relative_to(ROOT))
        if rel in baseline:continue
        assert (p.is_relative_to(ROOT/'HOLDOUT_D_runner_v3') or
                (p.parent==ROOT and p.name.startswith('HOLDOUT_D_AMEND3'))),('unexpected_new',rel)
        new_files.append(rel)
    for runner in ('HOLDOUT_D_runner','HOLDOUT_D_runner_v2','HOLDOUT_D_runner_v3'):
        if runner!='HOLDOUT_D_runner':assert not (ROOT/runner/'HOLDOUT_D_LIVE_RELEASE.json').exists()
        assert not (ROOT/runner/'HOLDOUT_D_live').exists()
    print(json.dumps(dict(passed=True,manifests=counts,baseline_files=len(baseline),new_files=len(new_files),external_append_only_logs=external),ensure_ascii=False))
if __name__=='__main__':main()
