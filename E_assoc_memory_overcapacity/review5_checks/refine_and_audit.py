"""Resolve whether the ORIGINAL displayed/root residual bands are also sound.

Imports only this review's implementation. No reuse of original proof code.
"""
import json
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal
import independent_intervals as own

source=json.loads(Path('w6_checks/certificate.json').read_text())
independent=json.loads(Path('review5_checks/interval_results.json').read_text())
numeric=json.loads(Path('review5_checks/numeric_results.json').read_text())
refined=[]
for src,new,num in zip(source['results'],independent['results'],numeric['results']):
    glob=[]
    for k in range(3):
        glob.append([str(min(Decimal(s['derivatives'][k][0]) for s in new['slabs'])),
                     str(max(Decimal(s['derivatives'][k][1]) for s in new['slabs']))])
    checks=[]
    for old,endpoint in zip(src['root_endpoints'],new['root_endpoint_checks']):
        vals,err=own.taylor_integrals(own.I(endpoint['a']),own.I(new['e']),cells=256)
        residual=vals[0]-own.I(Q(3,4))
        exact=old['residual']
        # Compare as rationals, without decimal conversion fuzz at the source bound.
        assert Q(int(exact['lo']),2**exact['power']) <= Q(residual.lo)
        assert Q(residual.hi) <= Q(int(exact['hi']),2**exact['power'])
        checks.append(dict(a=endpoint['a'],residual=residual.show(),
                           remainder=[x.show() for x in err],inside_original_raw_band=True))
    # Recompute all scalar inequalities behind the MSR bound exactly.
    assert 2*sum(Q(11,2)**j for j in range(4))==Q(1625,4)<512
    assert Q(6)/Q(1,2)+Q(4*24)/(2*Q(1,8))==396<400
    width=Q(new['width'])
    assert 400*width<Q('1.12e-7')
    print('e',new['e'],'global intervals',glob,'refined',checks,flush=True)
    refined.append(dict(e=new['e'],global_derivatives=glob,endpoint_checks=checks,
                        MSR_error_upper_fraction=str(400*width)))
for path,digest in numeric['baseline_hashes'].items():
    import hashlib
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
with open('review5_checks/refined_audit.json','x') as f:
    json.dump(dict(status='PASS',results=refined,reviewed_input_hashes_unchanged=True),f,indent=2)
