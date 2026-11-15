"""Final read-only checks of old inputs; exclusive new manifests and audit output."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    assert os.environ.get(k)=='2'
import csv,json,hashlib,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'y3_checks'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before=json.loads((OUT/'Y3_inputs.json').read_text())
checks={p:sha(ROOT/p)==h for p,h in before.items()}
assert all(checks.values())
counts={}
for name,wanted in [('Y3_coefficients.csv',30),('Y3_frozen_predictions.csv',90),
                    ('Y3_X1_contrasts.csv',18),('Y3_X1_new_comparison.csv',30),
                    ('Y3_X1_old_comparison.csv',30),('Y3_matching_tangent.csv',15)]:
    with (OUT/name).open() as f:rr=list(csv.DictReader(f))
    counts[name]=len(rr);assert len(rr)==wanted
with (OUT/'Y3_frozen_predictions.csv').open() as f:pred=list(csv.DictReader(f))
assert {r['d'] for r in pred}=={'128','256','512'}
assert len({(r['d'],r['ir'],r['method'],r['task']) for r in pred})==90
text=(ROOT/'Y3_FINITE_D_NONLINEAR.md').read_text()
draft=(ROOT/'Y3_PREREG_DRAFT.md').read_text()
assert '尚非定理' in text and '没有完成' in text and '未运行' in draft
assert '5400' in draft and '30917401' in draft
freeze=['Y3_FINITE_D_NONLINEAR.md','Y3_PREREG_DRAFT.md',
        'y3_checks/Y3_coefficients.py','y3_checks/Y3_decomposition.py',
        'y3_checks/Y3_verify.py','y3_checks/Y3_make_tables.py','y3_checks/Y3_audit.py',
        'y3_checks/Y3_coefficients.csv','y3_checks/Y3_frozen_predictions.csv',
        'experiments/x1/X1_population.json','X1_PREREG.md']
with (ROOT/'Y3_FROZEN.sha256').open('x') as f:
    for p in freeze:f.write(sha(ROOT/p)+'  '+p+'\n')
cg=Path('/proc/self/cgroup').read_text().strip().split('::')[-1]
cp=Path('/sys/fs/cgroup')/cg.lstrip('/')
audit=dict(old_inputs_unchanged=checks,output_row_counts=counts,
           new_mr_experiment_executed=False,recall_remainder_proved=False,
           frozen_manifest='Y3_FROZEN.sha256',
           memory_max=(cp/'memory.max').read_text().strip(),
           memory_swap_max=(cp/'memory.swap.max').read_text().strip(),
           cpu_seconds=time.process_time(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
assert audit['memory_max']=='3221225472' and audit['memory_swap_max']=='0'
with (OUT/'Y3_audit.json').open('x') as f:json.dump(audit,f,indent=2,ensure_ascii=False)
artifacts=sorted(list(ROOT.glob('Y3_*'))+list(OUT.glob('Y3_*')))
assert all(p.is_file() for p in artifacts)
with (ROOT/'Y3_ARTIFACTS.sha256').open('x') as f:
    for p in artifacts:f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
print(json.dumps(audit,ensure_ascii=False))
