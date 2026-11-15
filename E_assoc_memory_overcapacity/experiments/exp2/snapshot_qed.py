"""Only the user-specified QED outputs are read; nothing there is executed or written."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/exp2/qed_readonly'
base=Path('/home/liuyibo/auto-llm-driving/TheoryScout/qed_runs/out')
manifest=[]
for i in range(1,6):
    source=base/f'E_assoc_memory_overcapacity__v2_L{i}'
    problem=(source/'problem.tex').read_bytes()
    assert problem==(ROOT/f'lemmas_v2/L{i}.tex').read_bytes()
    copied=OUT/f'L{i}.problem.tex';assert copied.read_bytes()==problem
    manifest.append(dict(local=str(copied.relative_to(ROOT)),source=str(source/'problem.tex'),sha256=hashlib.sha256(problem).hexdigest(),matches_local_problem=True))
    leaf=source if i==4 else source/'decomposition/attempt_1/revision_1'/('proof_2' if i==2 else 'proof_1')
    for kind,filename in [('proof','proof.md')]+([] if i==4 else [('verification','detailed_verification.md')]):
        path=leaf/filename;data=path.read_bytes()
        local=OUT/(f'L{i}.proof2.md' if i==2 and kind=='proof' else f'L{i}.{kind}.md')
        assert local.read_bytes()==data
        manifest.append(dict(local=str(local.relative_to(ROOT)),source=str(path),sha256=hashlib.sha256(data).hexdigest(),
            overall_PASS=(b'Overall Verdict: PASS' in data) if kind=='verification' else None))
result=dict(read_time_utc=datetime.now(timezone.utc).isoformat(),scope='only explicitly requested E_assoc_memory_overcapacity__v2_* paths; read-only; no execution',files=manifest)
(OUT/'source_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(time=result['read_time_utc'],files=len(manifest),all_problems_match=True,verification_PASS=sum(r.get('overall_PASS') is True for r in manifest))))
