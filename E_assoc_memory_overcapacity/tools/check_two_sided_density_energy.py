"""Stdlib replay of sphere density minorants and signed energy envelopes."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import math
import sys

import certify_two_sided_density_energy as new
import check_energy_outer_polytope as audit
import check_shared_cdf_energy as old_audit

def multinomial_references():
    checks=0
    values=list(map(F,[-2,0,1,3]));weights=[F(1,10),F(2,10),F(3,10),F(4,10)]
    for size in range(1,5):
        for a,b in [(F(1,2),F(1)),(F(1),F(2)),(F(1,2),F(3))]:
            lo=sum(w for v,w in zip(values,weights) if v<=a)
            hi=sum(w for v,w in zip(values,weights) if v>b)
            for k in range(size+1):
                total=F(0)
                for choice in product(range(4),repeat=size):
                    vector=[values[j] for j in choice]
                    if sum(v>a for v in vector)==sum(v>b for v in vector)==k:
                        total+=math.prod(weights[j] for j in choice)
                assert total==math.comb(size,k)*hi**k*lo**(size-k);checks+=1
    return dict(exact_multinomial_empty_band_cases=checks)

def interval_references():
    bins=[dict(left=str(j),right=str(j+1),density_upper=str(w)) for j,w in enumerate([F(1,4),F(1,2),F(1,4)])]
    upper=dict(m=4,rows=[dict(L=1,bins=bins,tail_density_upper='0')],grid_end='3')
    lower=dict(m=4,rows=[dict(L=1,bins=[dict(left=b['left'],right=b['right'],density_lower=b['density_upper']) for b in bins],tail_density_lower='0')])
    model=new.TwoSided(upper,lower);checks=0;positive=0;negative=0
    def cdf(t):return sum(F(b['density_upper'])*max(F(0),min(F(1),t-F(b['left']))) for b in bins)
    for dl,dh in [(F(0),F(1,2)),(F(1,2),F(1)),(F(1),F(2)),(F(2),F(4)),(F(4),F(10))]:
        for rl,rh in [(F(1,2),F(3,4)),(F(1),F(3,2)),(F(3,2),F(2))]:
            for ml,mh in [(F(3,4),F(5,4)),(F(1),F(3,2))]:
                a,b=model.signed_minorant(1,(dl,dh,rl*rl,rh*rh),(ml*ml,mh*mh))
                positive+=int(a>0);negative+=int(b<0)
                for d,r,mean in product([dl,(dl+dh)/2,dh],[rl,(rl+rh)/2,rh],[ml,(ml+mh)/2,mh]):
                    assert a<=cdf(d/r)-cdf(d/mean)<=b;checks+=1
    assert positive>0 and negative>0
    return dict(signed_local_density_and_full_mean_interval_checks=checks,strict_positive_cells=positive,strict_negative_cells=negative)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    upper=new.base.read(root/'revisions/r44_sphere_density.json')
    upper_rebuilt=new.old.density.compute();upper_rebuilt['references']=new.old.density.references()
    for k,v in upper_rebuilt.items():assert upper[k]==v,k
    lower=new.base.read(root/'revisions/r45_sphere_density_lower_v2.json')
    rebuilt=new.minorant.compute(upper);rebuilt['references']=new.minorant.references()
    for k,v in rebuilt.items():assert lower[k]==v,k
    for table in [upper,lower]:
        for name,digest in table['source_hashes'].items():assert new.base.sha(root/name)==digest
    saved=new.base.read(a.certificate/'TWO_SIDED.json')
    for name,digest in saved['source_hashes'].items():assert new.base.sha(root/name)==digest
    audit.build=new.build;sys.argv=[sys.argv[0],str(a.certificate),'--out',str(a.out/'base_audit.json')];audit.main()
    assert new.certificate(root,a.certificate)==saved
    new.base.save(a.out/'CHECKS.json',dict(upper_and_lower_density_arithmetic_recomputed=True,
        upper_density_references=upper_rebuilt['references'],lower_density_references=rebuilt['references'],
        multinomial_references=multinomial_references(),signed_interval_references=interval_references(),
        old_continuity_references=new.old.references(),old_interval_references=old_audit.interval_references(),
        all_82_constraint_rows_and_actual_recall_cell_bounds_unchanged=True,cost_comparisons=saved['cost_comparisons'],
        strictly_tightened_cost_coefficients=saved['strictly_tightened_cost_coefficients'],
        four_new_exact_duals_and_four_old_dual_lifts_reconstructed=True,final_intersection_reconstructed=True,
        base_audit=new.base.read(a.out/'base_audit.json'),optimizer_required=False,
        prior_CDF_spectral_moment_C0_inputs_inherited=True,independent_proof_review=False))
    print('PASS upper/lower density, signed kernels, unchanged rows and exact duals')

if __name__=='__main__':main()
