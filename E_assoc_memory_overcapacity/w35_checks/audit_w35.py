"""Final W35 output audit, with an exclusive result file."""
import hashlib,json
from check_w35 import ROOT,OUT,put,meta

read=lambda name:json.loads((OUT/name).read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
originals={}
for line in (OUT/'input_hashes.sha256').read_text().splitlines():
    digest,name=line.split('  ',1)
    assert sha(ROOT/name)==digest,name
    originals[name]=digest

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
put('audit_w35.json',dict(passed=True,matching_records=len(matching),polar_records=len(polar),
    certificate_strict_signs=strict,unchanged_inputs=originals,delivered_file_hashes=manifest,
    limitations=['Adam T=4 root uniqueness at h=.75 remains conditional',
                 'Fixed 24-step search has a quantized population-algorithm limit',
                 'Polar comparisons use the locked external mean integral identification',
                 'No retrieval theorem or fixed-physical-epsilon performance theorem'],
    resource=meta()))
print('W35 audit passed: 160 matching records, 40 polar records, 9 strict certificate signs, 8 unchanged inputs.')
