"""Final delivery audit only; no stochastic experiment."""
from AA3_numeric import resources,OUT
import json,hashlib
from pathlib import Path
from fractions import Fraction as F

if __name__=='__main__':
    resources()
    certs={name:json.loads((OUT/file).read_text()) for name,file in {
        'continuous':'AA3_continuous_certificate.json','points':'AA3_points_certificate.json',
        'paths':'AA3_centered_paths_certificate.json','independent':'AA3_independent.json',
        'audit':'AA3_final_audit.json','near_threshold':'AA3_near_threshold_certificate.json'}.items()}
    hashes={}
    for cert in certs.values():
        for p,h in cert.get('source_hashes',{}).items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
            hashes[p]=h
    assert all(certs[k]['status']=='PASS' for k in ['continuous','points','paths','independent','near_threshold'])
    assert len(certs['audit']['saved']['paths'])==4 and certs['audit']['saved']['points']==8
    summary=[]
    for row in certs['paths']['paths']:
        def lower(field):return min(F(int(c[field]['lo']),2**c[field]['power']) for c in row['cells'])
        summary.append(dict(e=row['e'],kind=row['kind'],amax=row['amax'],beta_max=row['beta_max'],delta_certified=row['delta_certified'],
                            cells=len(row['cells']),endpoint_cells=len(row['endpoint_cover']),lambda_lower_display=float(lower('lam')),
                            normalization=row['stored_J_normalization']))
    # Hash both authoritative records and clearly named intermediate audit history.
    files=[p for p in OUT.rglob('AA3_*') if p.is_file()]+[OUT.parent/'AA3_ADAM_PREDICTIONS.md',OUT.parent/'AA3_PREREG_DRAFT.md']
    own={str(p.relative_to(OUT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    r=dict(status='PASS',paths=summary,continuous_cells=[len(r['cells']) for r in certs['continuous']['continuous']],
           point_groups=len(certs['points']['points']),root_boxes=sum(len(r['rows']) for r in certs['points']['points']),
           additional_near_threshold_root_boxes=6,
           independent_root_boxes=6,independent_root_endpoints=12,source_hashes=hashes,new_file_hashes=own,
           resources=resources(),computation_seconds={k:certs[k]['seconds'] for k in ['continuous','points','paths','independent','near_threshold']},
           finite_dimensional_experiment_run=False,two_step_BR_proved=False,
           limits=['r=.25 and e in {.5,2}; specified history paths only','no explicit two-step finite-d remainder','power depends on declared variance/normal-approximation assumptions'])
    with (OUT/'AA3_delivery.json').open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps({k:r[k] for k in ['status','paths','continuous_cells','root_boxes','computation_seconds','resources']},indent=2),flush=True)
