"""Stdlib exact audit of shared-threshold flip components and joint polygon."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse

import certify_energy_flip_components as cert

def references():
    cases=0;envelopes=0
    thresholds=list(map(F,[-3,-1,0,1,3]))
    for d,r,mu in product(map(F,[-2,'-1/2',0,'1/2',2]),map(F,[0,'1/2',1,2]),map(F,['1/2',1,2])):
        actual=[d>=r*t for t in thresholds];frozen=[d>=mu*t for t in thresholds]
        pos=F(sum(a and not b for a,b in zip(actual,frozen)),len(thresholds))
        neg=F(sum(b and not a for a,b in zip(actual,frozen)),len(thresholds))
        net=F(sum(actual)-sum(frozen),len(thresholds))
        assert pos==max(net,0) and neg==max(-net,0) and pos+neg==abs(net)
        assert pos+neg-abs(pos-neg)==2*min(pos,neg);cases+=1
        grid=sorted(set([F(-1),F(-1,2),F(0),F(1,2),F(1),net]))
        for lo in grid:
            for hi in grid:
                if lo<=net<=hi:
                    e=cert.envelopes(lo,hi)
                    for name,x in [('positive',pos),('negative',neg),('total',pos+neg)]:assert e[name][0]<=x<=e[name][1]
                    envelopes+=1
    # Same marginal recall with independently resampled thresholds can have
    # positive flip probability despite zero signed conditional difference.
    shared=F(sum((0>=t)!=(0>=t) for t in [-1,1]),2)
    independent=F(sum((0>=a)!=(0>=b) for a,b in product([-1,1],repeat=2)),4)
    assert shared==0 and independent==F(1,2)
    polys=0
    for p,n in [(F(0),F(0)),(F(1,5),F(1,10)),(F(1,5),F(2,5)),(F(1,2),F(1,2))]:
        for slack in [F(0),F(1,10),F(1)]:
            bounds={k:[str(max(F(0),x-slack)),str(min(F(1),x+slack))] for k,x in [('positive',p),('negative',n),('total',p+n)]}
            net=[str(max(F(-1),p-n-slack)),str(min(F(1),p-n+slack))]
            poly=cert.polygon(bounds,net)
            for name,x in [('positive',p),('negative',n),('total',p+n),('net',p-n)]:
                a,b=map(F,poly['projected_intervals'][name]);assert a<=x<=b
            a,b=map(F,poly['cancelled_total_flip_mass_interval']);assert a<=2*min(p,n)<=b
            for a,b,c in cert.halfplanes(bounds,net):assert a*p+b*n<=c
            polys+=1
    return dict(nested_shared_threshold_cases_with_atoms_negative_and_zero_energy=cases,component_cell_envelope_cases=envelopes,
                joint_polygon_and_cancellation_reference_cases=polys,independent_threshold_counterexample_detected=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];saved=cert.base.read(a.certificate/'RESULT.json');design_saved=cert.base.read(a.certificate/'DESIGN.json')
    assert saved['source_hashes']==design_saved['source_hashes']==cert.input_hashes(root)
    design,rows,kernels=cert.energy.build(root)
    for k,v in design.items():assert design_saved[k]==v,k
    receipt=cert.base.read(root/'revisions/R45_HISTORY.json')
    manifest=cert.base.read(root/receipt['path']/'manifest.json')
    old_design_path='revisions/r45_two_sided_energy/DESIGN.json'
    assert cert.base.sha(root/old_design_path)==manifest['files'][old_design_path]
    old_design=cert.base.read(root/old_design_path)
    for k,v in design.items():assert old_design[k]==v,k
    assert saved['constraints']==len(rows)==82 and saved['cells']==design['good_box_cells']+1
    records=[]
    for record in saved['rows']:
        length=record['L'];cells=[cert.envelopes(v[0],v[1]) for v in kernels[length]]
        for name,component in record['components'].items():
            raw=[]
            for side,index,sign in [('lower',0,1),('upper',1,-1)]:
                proof=component['proofs'][side];cost=[sign*v[name][index] for v in cells]
                bound,parts=cert.base.simplex_dual(cost,rows,proof['multipliers'])
                assert parts==proof['parts'] and sign*bound==F(proof['certified_bound']);raw.append(sign*bound)
            assert component['raw_dual_interval']==list(map(str,raw))
            interval=[max(F(0),raw[0]),min(F(1),raw[1])]
            assert component['interval']==list(map(str,interval)) and interval[0]<=interval[1]
        prior=cert.base.read(root/'revisions/r45_two_sided_energy/TWO_SIDED.json')
        net=next(r for r in prior['rows'] if r['L']==length)['actual_C0_energy_contrast_interval']
        assert record['prior_net_interval']==net
        polygon=cert.polygon({k:v['interval'] for k,v in record['components'].items()},net)
        assert polygon==record['joint_outer_polygon']
        records.append(dict(L=length,component_intervals={k:v['interval'] for k,v in record['components'].items()},
                            projected_intervals=polygon['projected_intervals'],cancelled_total_flip_mass_interval=polygon['cancelled_total_flip_mass_interval']))
    cert.base.save(a.out,dict(reference=references(),source_hashes=len(saved['source_hashes']),cells=saved['cells'],constraints=len(rows),
        R45_historical_design_sha256=cert.base.sha(root/old_design_path),
        exact_simplex_duals_reconstructed=12,all_three_component_kernels_reconstructed=True,joint_polygons_and_projections_reconstructed=True,
        R45_density_CDF_spectral_moment_C0_inputs_inherited=True,new_probability_integrals_recomputed=0,optimizer_required=False,
        records=records,independent_proof_review=False))
    print('PASS 12 exact component duals, shared-threshold flip identities and joint polygons')

if __name__=='__main__':main()
