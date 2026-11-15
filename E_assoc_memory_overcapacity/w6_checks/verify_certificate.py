"""Independent structural/integer audit and NumPy input-node check of certificate.json."""
import hashlib
import argparse
import json
import os
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
os.chdir(root)
parser = argparse.ArgumentParser()
parser.add_argument('--certificate',default='w6_checks/certificate.json')
parser.add_argument('--output',default='w6_checks/verification.json')
args = parser.parse_args()
data = json.loads(Path(args.certificate).read_text())
assert data['status']=='PASS' and data['inputs_unchanged']
S = 1 << data['power']

def iv(v):
    assert v['power']==data['power']
    lo,hi = int(v['lo']),int(v['hi'])
    assert lo<=hi
    return lo,hi

def contains(v,x):
    lo,hi = iv(v)
    assert Fraction(lo,S)<=x<=Fraction(hi,S)

def float_fraction(x): return Fraction(*float(x).as_integer_ratio())

def inside_display(v,left,right):
    lo,hi = iv(v)
    assert Fraction(left)<=Fraction(lo,S)<=Fraction(hi,S)<=Fraction(right)

# Check the analytic uniform tail constant and derivative envelope with rationals.
for c in [Fraction(17),Fraction(2),Fraction(1,3),
          Fraction(1024)*Fraction(7,6),Fraction(512),Fraction(2048,12)]:
    assert c*Fraction(1,2**72)<Fraction(1,2**60)
assert 2*sum(Fraction(11,2)**j for j in range(4))==Fraction(1625,4)<512
assert Fraction(6)/Fraction(1,2)+Fraction(4*24)/(2*Fraction(1,8))==396<400

for p,h in data['inputs_before'].items():
    assert h==data['inputs_after'][p]
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, p
assert hashlib.sha256(Path('w6_checks/adam_certificate.py').read_bytes()).hexdigest()==data['certificate_source_sha256']
grid = np.linspace(0,.6/4,33)
summary = {'status':'PASS','numpy':np.__version__,'executable':sys.executable,'rows':[]}
for row in data['results']:
    band = iv(row['root'])
    assert len(row['slabs'])==32
    for j,slab in enumerate(row['slabs']):
        contains(slab['a'],Fraction(3*j,640))
        contains(slab['a'],Fraction(3*(j+1),640))
        ds = slab['Fprime_Bprime_Vprime']
        lo,hi = iv(ds[0]); assert -7*S<lo<=hi<-2*S
        for v in ds[1:]:
            lo,hi=iv(v); assert -4*S<lo<=hi<4*S
    assert iv(row['root_endpoints'][0]['residual'])[0]>0
    assert iv(row['root_endpoints'][1]['residual'])[1]<0
    if row['e']=='1/10':
        inside_display(row['global_derivatives'][0],'-6.400','-2.223')
        inside_display(row['root_endpoints'][0]['residual'],'1.916e-14','1.920e-14')
        inside_display(row['root_endpoints'][1]['residual'],'-2.730e-14','-2.727e-14')
    else:
        inside_display(row['global_derivatives'][0],'-3.605','-2.104')
        inside_display(row['root_endpoints'][0]['residual'],'4.495e-15','4.501e-15')
        inside_display(row['root_endpoints'][1]['residual'],'-2.289e-14','-2.287e-14')
    search = row['search24']
    for j,node in enumerate(search['scan']):
        contains(node['a'],float_fraction(grid[j]))
        assert float_fraction(grid[j])==float_fraction(j*((.6/4)/32))
    assert len(search['decisions'])==24
    lo,hi = [Fraction(iv(v)[0],S) for v in search['initial_bracket']]
    assert iv(search['initial_bracket'][0])[0]==iv(search['initial_bracket'][0])[1]
    for decision in search['decisions']:
        mid = float_fraction((float(lo)+float(hi))/2)
        contains(decision['a'],mid)
        a,b = iv(decision['a'])
        if decision['sign']==1:
            assert b<band[0]; lo=mid
        else:
            assert a>band[1]; hi=mid
        assert iv(decision['residual_abs_lower'])[0]>0
    contains(search['final_bracket'][0],lo)
    contains(search['final_bracket'][1],hi)
    assert hi-lo<Fraction(28,10**11)
    assert iv(search['all_cache_argmin_gap_lower'])[0]>0
    assert iv(search['minimum_decision_gap_lower'])[0]>0
    assert Fraction(iv(search['minimum_decision_gap_lower'])[0],S)>Fraction('1.63e-10' if row['e']=='1/10' else '1.53e-10')
    assert Fraction(iv(search['all_cache_argmin_gap_lower'])[0],S)>Fraction('5.37e-10' if row['e']=='1/10' else '3.45e-10')
    inside_display(search['loss_residual_upper'],'0','1.96e-9')
    inside_display(search['msr_error_upper_all_five_cases'],'0','1.12e-7')
    chosen = search['chosen_endpoint']
    rs = [iv(v) for v in search['final_residuals']]
    absolute = [rs[0],(-rs[1][1],-rs[1][0])]
    assert absolute[chosen][1]<absolute[1-chosen][0]
    for comparison in row['comparisons']:
        assert iv(comparison['gd_minus_adam_old'])[0]>S//10000
        bands = {'1/10':{'0':('.00773477828','.00773477830'),
                  '1/2':('.00349267662','.00349267663'),'-1/2':('.01311845760','.01311845761')},
                 '1':{'0':('.00176630390','.00176630392'),
                  '1/2':('.00079575044','.00079575046'),'-1/2':('.00300514031','.00300514032')}}
        inside_display(comparison['gd_minus_adam_old'],*bands[row['e']][comparison['r']])
    summary['rows'].append({
        'e':row['e'],'global_derivatives':[v['display_only'] for v in row['global_derivatives']],
        'root_display_only':row['root']['display_only'],
        'endpoint_residuals_display_only':[v['residual']['display_only'] for v in row['root_endpoints']],
        'initial_bracket_display_only':[v['display_only'] for v in search['initial_bracket']],
        'final_bracket_display_only':[v['display_only'] for v in search['final_bracket']],
        'selected_a_exact':str(Fraction(iv(search['chosen_a'])[0],S)),
        'selected_a_display_only':search['chosen_a']['display_only'],
        'minimum_decision_gap_lower_display_only':search['minimum_decision_gap_lower']['display_only'],
        'all_cache_gap_lower_display_only':search['all_cache_argmin_gap_lower']['display_only'],
        'final_width_exact':str(hi-lo),
        'loss_residual_bound_display_only':search['loss_residual_upper']['display_only'],
        'MSR_error_bound_display_only':search['msr_error_upper_all_five_cases']['display_only'],
        'comparisons_display_only':[{'r':c['r'],'adam_old_new':[v['display_only'] for v in c['adam_old_new']],
          'old_gap':c['gd_minus_adam_old']['display_only']} for c in row['comparisons']]})
summary['os_threads'] = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('Threads:'))
assert summary['os_threads']<=2
summary['reported_intervals_outward'] = True
summary['analytic_tail_and_envelope_checks'] = True
summary['report_sha256'] = hashlib.sha256(Path('W6_ADAM_ROOT.md').read_bytes()).hexdigest()
with open(args.output,'x') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
