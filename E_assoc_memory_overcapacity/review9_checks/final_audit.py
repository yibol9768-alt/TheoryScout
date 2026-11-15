"""Data-only statistical audit; no execution/import of audited code."""
import os
assert all(os.environ.get(s)=='1' for s in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
import json
import math
import hashlib
import sys
from pathlib import Path
import numpy as np
from scipy.stats import t

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
load=lambda p:json.loads(p.read_text())
checks=load(HERE/'results.json')
coeff=checks['FT4']['independently_derived']
raw=[json.loads(line) for line in (ROOT/'experiments/w4/W4_raw.jsonl').read_text().splitlines()]
w4=[]
for d in (64,128,256,320):
    for mode in ('fixed','matched'):
        rows=[r for r in raw if r['kind']=='FT' and r['d']==d and r['mode']==mode]
        vv=np.array([r['old']['msr'] for r in rows])
        se=vv.std(ddof=1)/math.sqrt(len(vv))
        pred=coeff['mu']+coeff[mode]/d
        w4.append({'d':d,'mode':mode,'n':len(vv),'bias':float(vv.mean()-coeff['mu']),
                   'SE':float(se),'corrected_residual_SE':float((vv.mean()-pred)/se)})
reference=load(ROOT/'x2_checks/rank_reduction.json')
new=load(HERE/'student_rank_results.json')['rows']
comparison=[]
for old in reference:
    row=next(r for r in new if r['d']==old['d'])
    gap=old['mean_rank']-row['exact_rank_quadrature']
    comparison.append({'d':old['d'],'old_mean_rank':old['mean_rank'],
                       'new_mean_rank':row['exact_rank_quadrature'],
                       'gap_in_reported_SE':gap/old['rank_se'],
                       'nominal_95_t7_includes_new':bool(abs(gap)<t.ppf(.975,7)*old['rank_se']),
                       'old_effective_beta':old['rank_beta'],
                       'new_effective_beta':row['effective_beta']})
intervals=[]
for r in checks['FT4']['recomputed_rows']:
    center=r['d']*r['bias']
    hw=r['d']*r['bonferroni_8_halfwidth']
    intervals.append({'d':r['d'],'mode':r['mode'],'beta_interval':[center-hw,center+hw],
                      'halfwidth':hw})
original_hashes=load(HERE/'input_hashes.json')
assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in original_hashes.items())
cg=Path('/sys/fs/cgroup')/next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')).lstrip('/')
result={'original_W4_reanalysis':w4,'rank_numerical_comparison':comparison,
        'descriptive_95_Bonferroni_8_coefficient_intervals':intervals,
        'all_original_hashed_inputs_unchanged':True,
        'python':sys.executable,'threads':len(list(Path('/proc/self/task').iterdir())),
        'cgroup':{k:(cg/k).read_text().strip() for k in ('memory.max','memory.swap.max','memory.peak')}}
with (HERE/'final_audit_results.json').open('x') as f:
    json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
