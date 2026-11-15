"""Certify an exact zero lower-objective witness of the frozen energy LP.

The witness consists of cell masses, not atoms of a true C0 joint law.
Known variance/covariance bounds prove that it cannot lift to that law.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse

import certify_energy_flip_components as flips

base=flips.base

def sources(root):
    result=base.read(root/'revisions/r46_energy_flips/RESULT.json')
    hashes=flips.input_hashes(root)
    assert result['source_hashes']==hashes
    receipt=base.read(root/'revisions/R46_HISTORY.json');manifest=base.read(root/receipt['path']/'manifest.json')
    names=list(hashes)+['revisions/r45_two_sided_energy/DESIGN.json','revisions/r46_energy_flips/RESULT.json']
    for name in names:assert base.sha(root/name)==manifest['files'][name],name
    return {n:base.sha(root/n) for n in names}

def audit(root,mass):
    design,rows,kernels=flips.energy.build(root)
    old=base.read(root/'revisions/r45_two_sided_energy/DESIGN.json')
    for k,v in design.items():assert old[k]==v,k
    count=design['good_box_cells']+1;weights={}
    for index,value in mass:
        assert isinstance(index,int) and 0<=index<count and index not in weights
        weights[index]=F(value);assert weights[index]>0
    assert sum(weights.values())==1 and design['good_box_cells'] not in weights
    ds=list(map(F,design['D_grid']));vs=list(map(F,design['V_grid']));nv=len(vs)-1
    strips={(vs[i%nv],vs[i%nv+1]) for i in weights};assert len(strips)==1
    strip=next(iter(strips));ml,mh=map(F,design['mean_V_interval'])
    assert strip[0]<=ml<=mh<=strip[1]
    slacks=[]
    for row in rows:
        lhs=sum(row['coefficients'][i]*p for i,p in weights.items());slack=row['rhs']-lhs
        assert slack>=0,row['label']
        slacks.append(dict(label=row['label'],lhs=str(lhs),rhs=str(row['rhs']),slack=str(slack)))
    proofs=[];comparisons=0
    for length,values in kernels.items():
        cell_envelopes=[flips.envelopes(v[0],v[1]) for v in values]
        for component in ['positive','negative','total']:
            cost=[v[component][0] for v in cell_envelopes]
            assert all(c>=0 for c in cost);comparisons+=len(cost)
            assert all(cost[i]==0 for i in weights)
            primal=sum(cost[i]*p for i,p in weights.items());assert primal==0
            dual,parts=base.simplex_dual(cost,rows,[]);assert dual==primal==0
            proofs.append(dict(L=length,component=component,primal_objective='0',nonnegative_zero_multiplier_dual='0',
                               dual_parts=parts,relaxed_lower_objective_optimum_exactly_zero=True))
    mom=base.read(root/'revisions/r28_conditional_moments_v2/RESULT.json')['C0_moments']
    variance_lower=base.lo(base.restore(mom['variance_V']));variance_D_upper=base.hi(base.restore(mom['variance_D']))
    covariance_lower=base.lo(base.restore(mom['covariance_DV']));lift_bound=(strip[1]-strip[0])**2/4
    assert variance_lower>lift_bound
    assert covariance_lower>0 and covariance_lower**2>variance_D_upper*lift_bound
    return dict(cells=count,constraints=len(rows),positive_cell_masses=len(weights),mass_sum='1',outside_cell_mass='0',
        energy_strip=list(map(str,strip)),full_mean_V_interval=design['mean_V_interval'],slacks=slacks,
        minimum_raw_row_slack=str(min(F(r['slack']) for r in slacks)),raw_row_slacks_not_probability_accuracy=True,
        component_lower_cost_nonnegativity_comparisons=comparisons,six_lower_objective_optimality_certificates=proofs,
        any_lift_variance_V_upper=str(lift_bound),true_C0_variance_V_lower=str(variance_lower),
        true_C0_variance_D_upper=str(variance_D_upper),true_C0_covariance_DV_lower=str(covariance_lower),
        variance_and_Cauchy_contradictions_exclude_true_C0_lift=True,
        witness_is_cell_mass_vector_not_an_actual_C0_distribution=True,actual_zero_energy_flips_claimed=False,
        actual_energy_net_or_recall_enclosures_changed=False,independent_proof_review=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];hashes=sources(root)
    diagnostic=base.read(root/'revisions/r51_zero_feasibility_diagnostic.json')
    chosen=next(r for r in diagnostic['trials'] if r['label']=='frozen_no_tail')
    mass=chosen['atoms'];result=audit(root,mass)
    assert result['minimum_raw_row_slack']==chosen['minimum_exact_slack']
    result['mass_by_cell']=mass
    result['source_hashes']=dict(hashes,**{n:base.sha(root/n) for n in [
        'tools/certify_energy_zero_witness.py','revisions/r51_zero_feasibility_diagnostic.json']})
    base.save(a.out,result)
    print('EXACT WITNESS',result['positive_cell_masses'],'cells; six relaxed minima zero; actual C0 lift excluded',flush=True)

if __name__=='__main__':main()
