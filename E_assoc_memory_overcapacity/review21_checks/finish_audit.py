"""End-of-review input integrity and independent subdivision coverage."""
from pathlib import Path
from fractions import Fraction as F
import json, hashlib
from decimal_enclosures import resources

OUT=Path('review21_checks')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cover1(seq,key,start,end):
    ranges=sorted(tuple(map(F,c[key])) for c in seq)
    assert ranges and ranges[0][0]==start and ranges[-1][1]==end
    assert all(start<=l<h<=end for l,h in ranges)
    assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
def rectangle(seq,al,ah,bl,bh):
    assert seq
    knots=sorted({F(x) for c in seq for x in c['a']})
    assert knots[0]==al and knots[-1]==ah
    for c in seq:
        assert al<=F(c['a'][0])<F(c['a'][1])<=ah
    for l,h in zip(knots,knots[1:]):
        cover1([c for c in seq if F(c['a'][0])<(l+h)/2<F(c['a'][1])],'beta',bl,bh)
resources()
base=load(OUT/'input_hashes.json')
changes=[p for p,h in base.items() if sha(p)!=h]
assert not changes,changes
original=load('aa3_checks/AA3_centered_paths_certificate.json')
independent=load(OUT/'decimal_enclosures.json')
assert independent['status']=='PASS'
counts=[]
for old,new in zip(original['paths'],independent['paths']):
    assert old['e']==new['e'] and old['kind']==new['kind']
    assert {c['parent'] for c in new['cells']}==set(range(len(old['cells'])))
    for index,cell in enumerate(old['cells']):
        rectangle([c for c in new['cells'] if c['parent']==index],*map(F,cell['a']),*map(F,cell['beta']))
    rectangle(new['cells'],F(0),F(old['amax']),F(0),F(old['beta_max']))
    cover1(new['endpoints'],'beta',F(0),F(old['beta_max']))
    assert all(F(c['lambda'][0])>0 for c in new['cells'])
    positive=new['kind']==2 and new['e']=='1/2'
    assert all(F(c['Jnormalized'][0])>0 if positive else F(c['Jnormalized'][1])<0 for c in new['cells'])
    assert all(F(c['c1'][0])>F('11/20') for c in new['endpoints'])
    counts.append(dict(e=old['e'],kind=old['kind'],original_cells=len(old['cells']),independent_cells=len(new['cells']),
                       independent_endpoint_cells=len(new['endpoints']),
                       lambda_min=str(min(F(c['lambda'][0]) for c in new['cells'])),
                       c1_endpoint_min=float(min(F(c['c1'][0]) for c in new['endpoints']))))
for case in independent['near_threshold']:
    for arm in case['arms']:
        cover1(arm['monotonic'],'a',F(0),F(arm['root'][1]))
        assert all(F(c['lambda_bound'][0])>0 for c in arm['monotonic'])
    assert F(case['gap'][0])>0 if case['e']=='21/20' or case['delta']=='1/4' else F(case['gap'][1])<0
delivery=load('aa3_checks/AA3_delivery.json')
current_bindings={p:sha(p)==h for key in ['source_hashes','new_file_hashes'] for p,h in delivery[key].items()}
assert all(current_bindings.values())
result=dict(status='PASS',original_inputs_checked=len(base),input_changes=changes,current_delivery_bindings_match=True,
            independent_coverage=counts,resources=resources())
with (OUT/'input_integrity.json').open('x') as f:json.dump(result,f,indent=2)
manifest={str(p):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.suffix in ['.py','.json','.md']}
manifest['PROOF_REVIEW21.md']=sha('PROOF_REVIEW21.md')
with (OUT/'review21_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print(json.dumps(result,indent=2))
