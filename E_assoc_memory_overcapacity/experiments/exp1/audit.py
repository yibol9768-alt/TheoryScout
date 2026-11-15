"""Post-run integrity audit, without re-running the experiment or changing gates."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/exp1'

def main():
    prereg=(ROOT/'PREREG_EXP1.md').read_text()
    assert hashlib.sha256((ROOT/'PREREG_EXP1.md').read_bytes()).hexdigest()==(ROOT/'PREREG_EXP1.sha256').read_text().split()[0]
    for path in (ROOT/'FORMULATION_V2.md',OUT/'moments.json'):
        assert hashlib.sha256(path.read_bytes()).hexdigest() in prereg
    phasecounts=Counter();methods={};pairs={};duplicate_checks=0;rows=0
    for line in (OUT/'formal_rows.jsonl').open():
        r=json.loads(line);rows+=1;phase=r['phase'];phasecounts[phase]+=1
        assert len(r['ranks'])==len(r['query_indices'])==len(r['true_scores'])==128
        assert len(set(r['query_indices']))==128
        assert all(1<=v<=r['n'] for v in r['ranks'])
        assert all(math.isfinite(x) for x in r['true_scores'])
        assert abs(r['mean_rank']-sum((v-1)/(r['n']-1) for v in r['ranks'])/128)<1e-14
        for q in (.01,.1,.5):
            assert r[f'top{q:g}']==sum(v<=math.ceil(q*r['n']) for v in r['ranks'])/128
        assert len(r['wrong_cdf_per_query'])==128
        for a in r['wrong_cdf_per_query']:
            assert len(a)==33 and all(0<=v<=1 for v in a)
            assert all(a[i]<=a[i+1] for i in range(32))
        key=(phase,r['d'],r['n'],r['seed']);methods.setdefault(key,set()).add(r['name'])
        if phase in ('A','B'):
            kk=(r['d'],r['n'],r['seed'],r['name'])
            compact=(r['ranks'],r['true_scores'],r['query_indices'])
            if kk in pairs:
                assert compact==pairs[kk];duplicate_checks+=1
            pairs[kk]=compact
    assert all(len(m)=={'A':9,'B':43,'C':3}[key[0]] for key,m in methods.items())
    assert dict(phasecounts)==dict(A=405,B=1290,C=180)
    assert len(methods)==135
    result=dict(prereg_unchanged=True,source_and_moments_unchanged=True,rows=rows,
        phase_counts=dict(phasecounts),complete_data_configs=len(methods),
        exact_A_B_duplicate_checks=duplicate_checks,all_rank_topq_and_cdf_invariants=True)
    (OUT/'integrity_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    files=[ROOT/'PREREG_EXP1.md',ROOT/'PREREG_EXP1.sha256',ROOT/'RESULTS_EXP1.md']+sorted(p for p in OUT.iterdir() if p.is_file() and p.name not in ('ARTIFACTS.sha256','audit.log'))
    (OUT/'ARTIFACTS.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files))

if __name__=='__main__':main()
