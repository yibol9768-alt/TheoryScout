"""Administrative final integrity manifest, never issues model requests."""
import json
from pathlib import Path
import reliability_r2 as r

def main():
    r.verify('r2')
    r.verify('calib1')
    old=json.loads((r.ROOT/'FREEZE_DIFFICULTY.json').read_text())
    assert all(r.sha(r.ROOT/path)==h for path,h in old['sha256'].items())
    aux=json.loads((r.ROOT/'FREEZE_SUPPLEMENTARY_AUDIT_R2.json').read_text())
    assert all(r.sha(r.ROOT/path)==h for path,h in aux['sha256'].items())
    names=['difficulty_real','difficulty_r2_real']
    if (r.ROOT/'calib1_real/summary.json').exists():
        names.append('calib1_real')
    seeds=set()
    new_calls=0
    status=[]
    for name in names:
        out=r.ROOT/name
        s=json.loads((out/'summary.json').read_text())
        assert s.get('finished_utc')
        batch_seeds={x['seed'] for x in s['schedule']}
        assert len(batch_seeds)==len(s['schedule']) and not seeds.intersection(batch_seeds)
        seeds.update(batch_seeds)
        if name=='difficulty_real':
            continue
        gate=json.loads((out/'SMOKE_APPROVED.json').read_text())
        assert gate['approved']
        for path,h in gate['trace_sha256'].items():
            assert r.sha(out/path)==h
        assert len(gate['reviewed_ids'])==(3 if name=='difficulty_r2_real' else 2)
        b=[json.loads(x) for x in (out/'budget.jsonl').read_text().splitlines()]
        reserve=[x for x in b if x['event']=='reserve']
        assert len(reserve)==s['calls']
        new_calls+=s['calls']
        status.append(dict(name=name,calls=s['calls'],attempted=len(s['results']),
                           complete=sum(x['status']=='success' for x in s['results']),halt=s['halt']))
    assert new_calls<=6000
    result=dict(utc=r.d.now(),protocol_freezes_verified=True,old_protocol_unchanged=True,
                independent_seeds_verified=True,smoke_trace_hashes_verified=True,
                new_real_http_attempts=new_calls,phases=status)
    (r.ROOT/'FINAL_AUDIT_R2_CALIB1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    files=set()
    for pattern in ('*R2*','*CALIB1*','reliability_*.py','*reliability_r2.py'):
        files.update(p for p in r.ROOT.glob(pattern) if p.is_file())
    for name in ('difficulty_r2_real','difficulty_r2_mock','calib1_real','calib1_engine_mock'):
        files.update(p for p in (r.ROOT/name).glob('*') if p.is_file())
    manifest={str(p.relative_to(r.ROOT)):r.sha(p) for p in sorted(files)
              if p.name!='ARTIFACTS_R2_CALIB1.json'}
    (r.ROOT/'ARTIFACTS_R2_CALIB1.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
