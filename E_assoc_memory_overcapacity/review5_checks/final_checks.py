"""Exact analytic constants and final artifact/provenance checks."""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import math

num=json.loads(Path('review5_checks/numeric_results.json').read_text())
cert=json.loads(Path('w6_checks/certificate.json').read_text())
strict=json.loads(Path('review5_checks/interval_results.json').read_text())
refined=json.loads(Path('review5_checks/refined_audit.json').read_text())
assert cert['status']==strict['status']==refined['status']=='PASS'
assert hashlib.sha256(Path('w6_checks/adam_certificate.py').read_bytes()).hexdigest()==cert['certificate_source_sha256']
for p,h in num['baseline_hashes'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
constants=[]
for t in range(1,5):
    w=[]
    for b in [Q(9,10),Q(99,100)]:
        w.append([b**j/sum(b**i for i in range(t)) for j in range(t)])
    c2=sum(x*x/y for x,y in zip(*w))
    assert c2<4
    constants.append(str(c2))
assert 2*sum(Q(11,2)**j for j in range(4))==Q(1625,4)
assert Q(1625,4)<512
assert 6/Q(1,2)+4*24/(2*Q(1,8))==396<400
gaps=[]
for row,original in zip(num['results'],cert['results']):
    for cell,old in zip(row['msr'],original['comparisons']):
        r=cell['r']; k=1-math.sqrt(.75)
        b=(2-2*r)*k; v=(2-2*r)*k*k
        gd=(1-b/2)/math.sqrt(1-b+v)
        gap=gd-cell['MSR_at_root'][0]
        bounds=old['gd_minus_adam_old']['display_only']
        assert bounds[0]<gap<bounds[1]
        gaps.append(dict(e=row['e'],r=r,GD_minus_Adam_old_diagnostic=gap))
max_error=max(abs(x) for row in num['results'] for cell in row['msr'] for x in cell['actual_MSR_error'])
result=dict(status='PASS',source_certificate_hash_correct=True,reviewed_inputs_unchanged=True,
            direction_bound_squares=constants,diagnostic_old_MSR_gaps=gaps,
            max_diagnostic_MSR_quantization_error=max_error,
            independent_interval_source_hash_matches=(hashlib.sha256(Path('review5_checks/independent_intervals.py').read_bytes()).hexdigest()==strict['source_sha256']))
assert result['independent_interval_source_hash_matches']
with open('review5_checks/final_audit.json','x') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
