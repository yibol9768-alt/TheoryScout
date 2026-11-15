"""Integrity checks of experiment outputs, not post hoc scientific thresholds."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,resource,hashlib
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
r=Path(__file__).resolve().parent; project=r.parent.parent
assert hashlib.sha256((project/'PREREG_EXP1.md').read_bytes()).hexdigest()==(project/'PREREG_EXP1.sha256').read_text().split()[0]
records=[json.loads(x) for x in (r/'formal/results.jsonl').read_text().splitlines()]
status=json.loads((r/'formal/status.json').read_text()); complete={x['id'] for x in status if x['state']=='COMPLETED'}
assert len(complete)==len(records)==len({x['id'] for x in records})
rows=0
for d in records:
    assert d['id'] in complete and d['n_test']==8192
    assert sum(x['n'] for x in d['calibration']['fit'])==2048 and sum(x['n'] for x in d['calibration']['validation'])==2048
    assert abs(sum(d['p'])-1)<1e-10 and abs(sum(d['p_hat'])-1)<1e-10
    with np.load(r/'formal'/(d['id']+'.npz')) as a:
        assert a['endpoint_correct'].shape==(5,8192)
        names=a['endpoint_names'].tolist();arr=a['endpoint_correct']
        assert np.array_equal(arr[names.index('cold_archive')],arr[names.index('full_summary')])
        for row in d['rows']:
            if row['policy'] in names: assert row['accuracy']==arr[names.index(row['policy'])].mean()
        if d['layer']=='E2' and d['dist']==0:
            assert len(a['counts'])==(d['G']+1)*(d['G']+2)//2
            assert a['candidate_correct_counts'].shape==(d['permutations_completed'],len(a['counts']))
    for row in d['rows']:
        rows+=1;assert 0<=row['accuracy']<=1 and 0<=row['predicted_accuracy']<=1
        if row['policy'] not in ('padded_summary','cold_archive','route_raw'):
            assert 4*row['n']+row['s']<=row['budget'] and row['n']+row['s']<=d['G']
meta=json.loads((r/'formal/metadata.json').read_text());assert meta['elapsed_seconds']<3540 and meta['peak_rss_mib']<2048
invalid=json.loads((r/'invalid_calibration_run/metadata.json').read_text())
assert meta['elapsed_seconds']+invalid['elapsed_seconds']<3600
expected={(d['id'],x['budget']) for d in records for x in d['rows'] if x['policy']=='dp'}
policies=[json.loads(line) for line in (r/'dp_policies.jsonl').read_text().splitlines()]
assert len(policies)==len(expected) and {(x['id'],x['budget']) for x in policies}==expected
out={'status':'PASS','completed_configurations':len(records),'verified_strategy_rows':rows,'prereg_unchanged':True,'all_completed_npz_verified':True,'formal_seconds':meta['elapsed_seconds'],'invalid_plus_valid_formal_seconds':meta['elapsed_seconds']+invalid['elapsed_seconds'],'peak_rss_mib':meta['peak_rss_mib']}
(r/'integrity_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
