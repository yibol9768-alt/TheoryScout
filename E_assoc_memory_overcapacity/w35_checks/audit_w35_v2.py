"""Complete W35 audit while explicitly retaining concurrent input drift."""
import hashlib,json,datetime
from check_w35 import ROOT,OUT,put,meta

read=lambda name:json.loads((OUT/name).read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
inputs={}
for line in (OUT/'input_hashes.sha256').read_text().splitlines():
    digest,name=line.split('  ',1);path=ROOT/name;current=sha(path)
    inputs[name]=dict(baseline_sha256=digest,current_sha256=current,unchanged=current==digest,
        mtime=datetime.datetime.fromtimestamp(path.stat().st_mtime,datetime.timezone.utc).isoformat())

matching=read('matching_rows.json');polar=read('polar_rows.json');pop=read('matching_population.json')
assert len(matching)==160 and len(polar)==40 and len(pop)==10
assert {(r['d'],r['seed']) for r in matching}=={(d,s) for d in (16,32,64,128) for s in range(4)}
assert {(r['d'],r['gamma'],r['seed']) for r in polar}=={(d,g,s) for d in (32,64,128,256) for g in (16,64) for s in range(5)}
assert all(r['selected'] is not None for r in matching)
assert all(abs(r['k']-r['k_entrywise'])<2e-12 and r['k']>=-1e-12 and r['k']<=r['k_finite_bound']+1e-12 for r in polar)

cert=read('stabilizer_certificate.json')
def bounds(x):
    assert x['denominator_power']==96
    lo,hi=int(x['numerator_lower']),int(x['numerator_upper'])
    assert lo<=hi
    return lo,hi
strict=[]
for row in cert['rows']:
    g=row['gamma'];a,b=bounds(row['efficiency_crossing_band'])
    assert row['proposed_e_band_certified']
    assert bounds(row['tested_c'][0])[1]<a and bounds(row['tested_c'][1])[0]>b
    for item in row['differences']:
        lo,hi=bounds(item['coordinate_minus_muon']);tau=item['tau']
        assert hi<0 if g==16 and tau==0 else lo>0
        strict.append(dict(gamma=g,tau=tau,sign='negative' if hi<0 else 'positive'))
for filename in ('matching_meta.json','stabilizer_meta.json','certificate_meta.json'):
    m=read(filename)
    assert m['controls']['memory.max']=='2147483648'
    assert m['controls']['memory.swap.max']=='0'
    assert all(p['threads']<=2 for p in m['pools'])
    ev=dict(line.split() for line in m['controls']['memory.events'].splitlines())
    assert ev['oom']=='0' and ev['oom_kill']=='0'

paths=sorted(list(ROOT.glob('W35_*'))+[p for p in OUT.iterdir() if p.is_file()])
manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
drift=[name for name,state in inputs.items() if not state['unchanged']]
put('audit_w35.json',dict(numerical_checks_passed=True,input_integrity_passed=not drift,
    detected_input_drift=drift,input_status=inputs,
    preliminary_audit='audit_w35.py stopped at changed paper/appendix/F.tex; no result file created',
    response_to_drift='Recorded without reverting or overwriting either original file.',
    matching_records=len(matching),polar_records=len(polar),certificate_strict_signs=strict,
    delivered_file_hashes=manifest,
    limitations=['Adam T=4 root uniqueness at h=.75 remains conditional',
                 'Fixed 24-step search has a quantized population-algorithm limit',
                 'Polar comparisons use the locked external mean integral identification',
                 'No retrieval theorem or fixed-physical-epsilon performance theorem'],
    resource=meta()))
print('Numerical audit passed: 160 matching records, 40 polar records, 9 strict signs.')
print('Input integrity:',len(inputs)-len(drift),'unchanged;',len(drift),'changed during task:',drift)
