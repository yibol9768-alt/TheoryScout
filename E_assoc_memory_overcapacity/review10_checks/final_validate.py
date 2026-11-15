"""Check all independent results and the submitted decimal/hex endpoint records."""
import hashlib
import json
from pathlib import Path
import mpmath as mp

base=Path(__file__).resolve().parent.parent
mp.mp.dps=90
own=json.loads((base/'review10_checks/independent_results.json').read_text())
original=json.loads((base/'x3_checks/polar_certificate.json').read_text())
summary={'all_15_independent_signs':True,'all_30_efficiency_boxes_nested':True,
         'all_independent_moment_boxes_nested':True,'hex_records_consistent':True,
         'input_files_still_unchanged':True,'independent_nu_decimal_boxes':[]}
for a,b in zip(own['rows'],original['rows']):
    assert a['gamma']==b['gamma']
    for field in ('mu','k','nu'):
        inner=a['independent_'+field+'_box']
        outer=b[field]
        assert mp.mpf(outer['lower'])<=mp.mpf(inner[0])<=mp.mpf(inner[1])<=mp.mpf(outer['upper'])
    for x in a['roots']:
        assert float(x['strict_lower_margin'])>0 and float(x['strict_upper_margin'])>0
        assert all(x['submitted_c_boxes_contain_own'])
    lo,hi=map(mp.mpf,a['independent_nu_box'])
    summary['independent_nu_decimal_boxes'].append([a['gamma'],
        mp.nstr(mp.floor(lo*10**12)/10**12,20),mp.nstr(mp.ceil(hi*10**12)/10**12,20)])

def walk(value):
    if isinstance(value,dict):
        if 'lower_hex' in value:
            assert float.fromhex(value['lower_hex'])==value['lower']
            assert float.fromhex(value['upper_hex'])==value['upper']
        for item in value.values():walk(item)
    elif isinstance(value,list):
        for item in value:walk(item)
walk(original)
for path,digest in own['input_hashes'].items():
    assert hashlib.sha256((base/path).read_bytes()).hexdigest()==digest,path
summary['smallest_independent_sign_margin']=min(float(x[k]) for a in own['rows'] for x in a['roots']
    for k in ('strict_lower_margin','strict_upper_margin'))
summary['source_files']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (base/'review10_checks').glob('*.py')}
(base/'review10_checks/final_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
