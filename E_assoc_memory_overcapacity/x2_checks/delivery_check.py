import os
assert all(os.environ[s]=='1' for s in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])
import json,hashlib,re,time,ast
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent
start=time.monotonic()
for p in H.glob('*.py'):ast.parse(p.read_text())
s=(R/'X2_FINITE_D.md').read_text()
assert s.count('$$')%2==0
for part in s.split('$$')[1::2]:assert part.count('{')==part.count('}')
for line in s.splitlines():assert line.count('$')%2==0
tags=[int(x) for x in re.findall(r'\\tag\{(\d+)\}',s)]
assert sorted(tags)==list(range(1,37)),tags
hh=json.loads((H/'verified/input_hashes.json').read_text())
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==v for p,v in hh.items())
files=['verified/coefficients.json','verified/ft1_exact.json','verified/nonlinear_exact.json',
 'verified/w4_reanalysis.json','scatter_summary.json','rank_reduction.json','small_checks.json',
 'ft_rank_reduction.json','w4_retrieval.json']
equal={p:json.loads((H/p).read_text())==json.loads((H/'replay_check'/p).read_text()) for p in files}
assert all(equal.values()),equal
meta=[json.loads(p.read_text()) for p in H.rglob('*metadata.json') if 'replay_metadata' not in p.name]
assert all(m['controls']['memory.max']=='3221225472' and m['controls']['memory.swap.max']=='0' for m in meta)
out=dict(input_hashes_unchanged=True,replay_scientific_data_exactly_equal=equal,
 max_rss_mib=max(m['rss_mib'] for m in meta),max_cgroup_memory_peak=max(int(m['controls']['memory.peak']) for m in meta),
 first_analytic_threads=json.loads((H/'analytic_metadata.json').read_text())['process_threads'],
 subsequent_process_threads=sorted(set(m['process_threads'] for m in meta if m!=json.loads((H/'analytic_metadata.json').read_text()))),
 original_logged_seconds=sum(json.loads((H/p).read_text())['seconds'] for p in ['analytic_metadata.json','verified/analytic_metadata.json','scatter_metadata.json','rank_metadata.json','ft_rank_metadata.json','summary_metadata.json','audit.json']),
 replay_seconds=json.loads((H/'replay_check/replay_metadata.json').read_text())['total_seconds'],
 report_equations=len(tags),delivery_seconds=time.monotonic()-start)
with (H/'delivery.json').open('x') as f:json.dump(out,f,indent=2)
paths=[R/'X2_FINITE_D.md']+sorted(p for p in H.rglob('*') if p.is_file())
with (R/'X2_ARTIFACTS.sha256').open('x') as f:
    for p in paths:f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(R))+'\n')
print(json.dumps(out,indent=2))
