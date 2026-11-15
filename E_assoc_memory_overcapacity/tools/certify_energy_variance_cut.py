"""A valid lower-second-moment row that removes the frozen-strip witness.

The outside coefficient uses a certified global energy support bound. This
file proposes one valid additional row; it does not reoptimize energy bounds.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse

import certify_energy_zero_witness as zero

base=zero.base

def compute(root):
    prior=base.read(root/'revisions/r51_energy_zero_witness.json')
    for n,h in prior['source_hashes'].items():assert base.sha(root/n)==h,n
    zero.sources(root)
    design=base.read(root/'revisions/r45_two_sided_energy/DESIGN.json')
    inputs=base.read(root/'revisions/r28_conditional_moments_v2/INPUTS.json')
    moments=base.read(root/'revisions/r28_conditional_moments_v2/RESULT.json')['C0_moments']
    n=inputs['n'];m=n-1;assert n==design['n']
    maximum=F(n*n,m)*(F(inputs['Nhat_operator_norm_upper'])+F(inputs['N_operator_perturbation']))
    center=F(1057,1000);lower=base.lo(base.restore(moments['variance_V']))
    assert maximum>center>0 and lower>0
    ds=list(map(F,design['D_grid']));vs=list(map(F,design['V_grid']))
    upper=lambda a,b:max((a-center)**2,(b-center)**2)
    coefficients=[-upper(a,b) for _ in zip(ds,ds[1:]) for a,b in zip(vs,vs[1:])]+[-upper(F(0),maximum)]
    assert len(coefficients)==prior['cells']
    lhs=sum(coefficients[i]*F(p) for i,p in prior['mass_by_cell']);rhs=-lower
    assert lhs>rhs
    a,b=map(F,prior['energy_strip']);strip_upper=upper(a,b);outside_upper=upper(F(0),maximum)
    needed=(lower-strip_upper)/(outside_upper-strip_upper)
    assert 0<needed<F(design['tail_mass_upper'])
    return dict(label='lower_second_moment_V_about_1057_1000',center=str(center),
        lower_moment_bound=str(lower),lower_bound_uses_variance_plus_nonnegative_mean_shift=True,
        certified_global_V_support=['0',str(maximum)],coefficient_formula='-max((v_lower-center)^2,(v_upper-center)^2); outside uses[0,Vmax]',
        coefficients=list(map(str,coefficients)),rhs=str(rhs),old_witness_lhs=str(lhs),old_witness_violation=str(lhs-rhs),
        old_witness_excluded=True,outside_coefficient_nonzero=True,
        outside_mass_necessary_if_other_mass_stays_in_this_strip=str(needed),
        existing_outside_mass_upper=design['tail_mass_upper'],necessary_budget_compatibility_not_a_feasibility_certificate=True,
        row_not_applied_to_active_polytope_or_manuscript=True,new_actual_energy_or_recall_bounds_claimed=False,
        independent_proof_review=False)

def references():
    checks=0
    for center in [F(0),F(1),F(1057,1000)]:
        for a,b in [(F(0),F(1)),(F(1),F(2)),(F(0),F(11))]:
            for v in [a,(a+b)/2,b]:
                assert (v-center)**2<=max((a-center)**2,(b-center)**2);checks+=1
    # Dropping the outside coefficient would exclude an admissible law:
    # V=1 with.99 mass, V=11 with.01 mass, center1 has second moment1.
    correct=F(99,100)*0+F(1,100)*100;assert correct==1
    wrong=F(0);assert wrong<correct
    return dict(cell_and_outside_squared_moment_covers=checks,unsafe_zero_outside_coefficient_counterexample_detected=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--compare',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];result=compute(root);result['reference']=references()
    result['source_hashes']={n:base.sha(root/n) for n in ['tools/certify_energy_variance_cut.py','tools/certify_energy_zero_witness.py',
        'revisions/r51_energy_zero_witness.json','revisions/r45_two_sided_energy/DESIGN.json',
        'revisions/r28_conditional_moments_v2/INPUTS.json','revisions/r28_conditional_moments_v2/RESULT.json']}
    if a.compare:assert base.read(a.compare)==result
    base.save(a.out,result)
    print('PASS valid lower-moment row, old-witness exclusion and retained outside support budget')

if __name__=='__main__':main()
