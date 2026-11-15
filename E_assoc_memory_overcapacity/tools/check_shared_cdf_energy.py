"""Stdlib replay: density arithmetic, same-CDF envelopes and exact duals."""
from pathlib import Path
import argparse
import sys

import certify_shared_cdf_energy as shared
import check_energy_outer_polytope as audit

def interval_references():
    F=shared.F
    bins=[dict(left=str(j),right=str(j+1),density_upper=str(w)) for j,w in enumerate([F(1,4),F(1,2),F(1,4)])]
    table=dict(rows=[dict(L=1,bins=bins,tail_density_upper='0')],grid_end='3')
    model=shared.SharedCDF(table);checks=0
    def cdf(t):return sum(F(b['density_upper'])*max(F(0),min(F(1),t-F(b['left']))) for b in bins)
    for dl,dh in [(F(0),F(1,2)),(F(1,2),F(1)),(F(1),F(2)),(F(2),F(4)),(F(4),F(10))]:
        for rl,rh in [(F(1,2),F(1)),(F(1),F(3,2)),(F(5,4),F(2))]:
            for ml,mh in [(F(3,4),F(5,4)),(F(1),F(3,2))]:
                cap=model.cap(1,(dl,dh,rl*rl,rh*rh),(ml*ml,mh*mh))
                for d in [dl,(dl+dh)/2,dh]:
                    for r in [rl,(rl+rh)/2,rh]:
                        for mean_r in [ml,(ml+mh)/2,mh]:
                            assert abs(cdf(d/r)-cdf(d/mean_r))<=cap
                            checks+=1
    return dict(local_piecewise_density_D_V_and_full_mean_interval_checks=checks)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    table=shared.base.read(root/'revisions/r44_sphere_density.json')
    rebuilt=shared.density.compute();rebuilt['references']=shared.density.references()
    for key,value in rebuilt.items():assert table[key]==value,key
    for name,digest in table['source_hashes'].items():assert shared.base.sha(root/name)==digest
    saved=shared.base.read(a.certificate/'SHARED_CDF.json')
    for name,digest in saved['source_hashes'].items():assert shared.base.sha(root/name)==digest
    audit.build=shared.build
    sys.argv=[sys.argv[0],str(a.certificate),'--out',str(a.out/'base_audit.json')];audit.main()
    assert shared.certificate(root,a.certificate)==saved
    shared.base.save(a.out/'CHECKS.json',dict(density_arithmetic_recomputed=True,density_references=rebuilt['references'],
        shared_CDF_references=shared.references(),interval_references=interval_references(),all_82_constraint_rows_unchanged=True,
        cost_comparisons=saved['cost_comparisons'],strictly_tightened_cost_coefficients=saved['strictly_tightened_cost_coefficients'],
        four_new_exact_duals_and_four_old_dual_lifts_reconstructed=True,final_intersection_reconstructed=True,
        base_audit=shared.base.read(a.out/'base_audit.json'),optimizer_required=False,
        prior_CDF_spectral_and_C0_inputs_inherited=True,independent_proof_review=False))
    print('PASS density, shared CDF, unchanged rows and exact duals')

if __name__=='__main__':main()
