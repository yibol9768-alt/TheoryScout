"""Transparent descriptive examples selected by largest recorded error/regret."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,resource
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
r=Path(__file__).resolve().parent; data=[json.loads(x) for x in (r/'formal/results.jsonl').read_text().splitlines()]
strata={}
for kind,predicate in [('E1_homogeneous_generator',lambda d:d['layer']=='E1' and d['channel'][1:]==[1.,1.]),('E1_heterogeneous_generator',lambda d:d['layer']=='E1' and d['channel'][1:]!=[1.,1.]),('E2',lambda d:d['layer']=='E2')]:
    selected=[d for d in data if predicate(d)]; rows=[x for d in selected for x in d['rows'] if x['policy'] not in ('padded_summary','cold_archive','route_raw')]
    if not rows:continue
    errors=np.array([x['abs_error'] for x in rows]); ext=np.array([abs(x['accuracy']-x['extended_predicted_accuracy']) for x in rows]);oracle=np.array([abs(x['accuracy']-x['oracle_accuracy']) for x in rows if 'oracle_accuracy' in x])
    strata[kind]=dict(configurations=len(selected),strategy_rows=len(rows),homogeneous_mae=float(errors.mean()),extended_mae=float(ext.mean()),homogeneous_exceedances=sum(x['abs_error']>max(.015,4*x['se']) for x in rows),extended_exceedances=sum(abs(x['accuracy']-x['extended_predicted_accuracy'])>max(.015,4*x['se']) for x in rows),oracle_exceedances=sum(abs(x['accuracy']-x['oracle_accuracy'])>max(.015,4*x['se']) for x in rows if 'oracle_accuracy' in x),oracle_mae=float(oracle.mean()) if len(oracle) else None)
candidates=[(x['regret'],d,x) for d in data if d['layer']=='E2' and d['dist']==0 for x in d['rows'] if x['policy']=='dp']
if candidates:
    _,d,x=max(candidates,key=lambda c:c[0]);example={k:d[k] for k in ('id','seed','G','channel','epsilon','nu_s','nu_r')}; example['selected_dp']=x; example['calibration']=d['calibration']['homogeneous'];example['extended_calibration']=d['calibration']['extended'];example['endpoints']=[x for x in d['rows'] if x['policy'].startswith('full_')]
    with np.load(r/'formal'/(d['id']+'.npz')) as a:
        pairs=a['counts']; eligible=4*pairs[:,0]+pairs[:,1]<=x['budget'];good=a['candidate_correct_counts'][:,eligible];pi,ci=np.unravel_index(np.argmax(good),good.shape); n,s=pairs[eligible][ci]
        example['best_count_scan']=dict(permutation=int(pi),n=int(n),s=int(s),accuracy=float(good[pi,ci]/8192))
else:example=None
(r/'diagnostic_examples.json').write_text(json.dumps({'selection_rule':'post hoc largest recorded uniform E2 DP regret, descriptive only; original thresholds unchanged','strata':strata,'largest_uniform_E2_regret':example},indent=2))
print(json.dumps({'strata':strata,'example_id':example['id'] if example else None}))
