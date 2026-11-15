"""Check the report's exact displayed bounds against certified integer endpoints."""
from certify import *
def read(p):
    with open(p) as f:return json.load(f)
def dec(v):return int(v['lo']),int(v['hi'])
def within(iv,lo,hi):
    assert Fraction(iv[0],S)>=Fraction(lo) and Fraction(iv[1],S)<=Fraction(hi),(show(iv),lo,hi)
reg=read('y2_checks/region.json');pt=read('y2_checks/points.json');hi=read('y2_checks/history.json');ve=read('y2_checks/verification.json')
assert all(v['status']=='PASS' for v in [reg,pt,hi,ve])
within(dec(reg['global_N']),'17/100','146/100');within(dec(reg['global_J']),'1/20000','535/100')
within(dec(pt['sign']['bounds']['c0']),'.84906207','.84906221')
for row,lo,up in zip(pt['points'],['.86752629','.89519116','.90477476'],['.86752642','.89519127','.90477484']):
    within(dec(row['c0']),lo,up)
    for ep in row['endpoints']:
        a=F(ep['a']);s=stats(a,[dec(v) for v in ep['integrals']]);assert s['B'][0]>0
within(dec(pt['GD_c0']),'.91651513899','.91651513900')
assert Fraction(dec(pt['adam_point1_minus_sign'])[0],S)>Fraction('.01846408')
for row,lo,up in zip(hi['models'],['.90603102','.90470411','.90611275'],['.90603111','.90470420','.90611283']):within(dec(row['c0']),lo,up)
for key,lower in [('adding_momentum_with_rms','.00133791'),('adding_rms_with_momentum','.00007056')]:assert Fraction(dec(hi['gaps'][key])[0],S)>Fraction(lower)
for row,lo in zip(pt['points'],['.00784959','.00234188','.00131216']):assert Fraction(dec(row['actual_minus_frozen'])[0],S)>Fraction(lo)
for obj in [reg,pt,hi,ve]:
    r=obj['resources'];assert r['memory.max']=='3221225472' and r['memory.swap.max']=='0' and r['threads']==1
    assert 'oom 0\n' in r['memory.events'] and 'oom_kill 0\n' in r['memory.events']
report=Path('Y2_MULTISTEP.md').read_text();assert '1027' in report and '3980/3971' in report
paths=[Path('Y2_MULTISTEP.md')]+sorted(Path('y2_checks').glob('*'))
assert all(p.is_file() for p in paths)
inputs={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in reg['inputs_before']};assert inputs==reg['inputs_before']
out=dict(status='PASS',displayed_bounds_outward=True,source_inputs_unchanged=True,resource_requirements_passed=True,files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},input_sha256=inputs,resources=resources())
with open('y2_checks/final_audit.json','x') as f:json.dump(out,f,indent=2)
print('PASS: report bounds, resource records, original inputs, artifact hashes',flush=True)
