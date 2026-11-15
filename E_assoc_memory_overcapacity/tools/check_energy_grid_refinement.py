"""Replay fine-grid rational bounds and the exact coarse-to-fine lift."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import sys

import check_energy_outer_polytope as original_audit
import certify_energy_outer_polytope as base
from refine_energy_outer_grid import BASE_BUILD,refinement_certificate


def lift_references():
    mapping=[0,0,1,1,2];checks=0
    for weights,increment,multipliers in product([[1,2,3,4,1],[0,1,0,2,1]],[F(0),F(1,7),F(1)],
        [[],[(0,'1/7'),(1,'13/2')],[(0,'1000'),(1,'1/1000'),(2,'2')]]):
        p=[F(x,sum(weights)) for x in weights]
        coarse_a=[[F(1),F(-1),F(0)],[F(-2),F(3),F(0)],[F(0),F(0),F(1)]]
        fine_a=[[row[j]+(increment if i<2 and k in [1,2] else 0) for k,j in enumerate(mapping)] for i,row in enumerate(coarse_a)]
        rhs=[sum(a*x for a,x in zip(row,p))+F(1,10) for row in fine_a]
        coarse_rows=[dict(coefficients=a,rhs=b) for a,b in zip(coarse_a,rhs)]
        fine_rows=[dict(coefficients=a,rhs=b) for a,b in zip(fine_a,rhs)]
        coarse_cost=[F(-1),F(1,5),F(1)]
        fine_cost=[coarse_cost[j]+(increment if i in [1,2] else 0) for i,j in enumerate(mapping)]
        coarse_probability=[sum(x for x,j in zip(p,mapping) if j==i) for i in range(3)]
        assert all(sum(a*x for a,x in zip(row,coarse_probability))<=b for row,b in zip(coarse_a,rhs))
        old,_=base.simplex_dual(coarse_cost,coarse_rows,multipliers)
        new,_=base.simplex_dual(fine_cost,fine_rows,multipliers)
        assert old<=new<=sum(a*x for a,x in zip(fine_cost,p));checks+=1
    return dict(arbitrary_nonnegative_multiplier_refinement_lift_cases=checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];certificate=args.certificate.resolve()
    saved=base.read(certificate/'REFINEMENT.json');factor=saved['subdivisions']
    assert base.sha(root/'tools/certify_energy_outer_polytope.py')==saved['current_compatible_builder_sha256']
    receipt=base.read(root/'revisions/R42_HISTORY.json');manifest=base.read(root/receipt['path']/'manifest.json')
    assert base.sha(root/'revisions/r43_previous_certify_energy_outer_polytope.py')==saved['prior_builder_historical_sha256']==manifest['files']['tools/certify_energy_outer_polytope.py']
    original_audit.build=lambda path:BASE_BUILD(path,factor)
    sys.argv=[sys.argv[0],str(certificate),'--out',str(args.out.resolve()/'fine_grid_audit.json')]
    original_audit.main()
    actual=refinement_certificate(root,certificate,factor);assert actual==saved
    base.save(args.out/'CHECKS.json',dict(base_fine_grid_audit=base.read(args.out/'fine_grid_audit.json'),
        reference=lift_references(),exact_constraint_coefficient_dominance_comparisons=saved['exact_coefficient_dominance_comparisons'],
        exact_cost_coefficient_dominance_comparisons=4*(saved['new_good_cells']+1),
        four_old_dual_lifts_reconstructed=True,final_old_lift_and_new_bounds_intersection_reconstructed=True,
        source_data_unchanged=True,no_new_matrices_samples_or_integrals=True,independent_proof_review=False))
    print('PASS nested-grid containment, old-dual lift and final intersection')


if __name__=='__main__':main()
