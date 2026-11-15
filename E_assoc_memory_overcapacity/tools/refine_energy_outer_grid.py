"""Nested cell refinement and exact lift of the frozen R42 dual certificates."""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_right
import argparse
import sys

import certify_energy_outer_polytope as base

BASE_BUILD=base.build


def refinement_certificate(root,directory,subdivisions):
    coarse_design,coarse_rows,coarse_kernels=BASE_BUILD(root)
    fine_design,fine_rows,fine_kernels=BASE_BUILD(root,subdivisions)
    previous_path=root/'revisions/r42_energy_outer_v2/RESULT.json';previous=base.read(previous_path)
    expected_design=base.read(root/'revisions/r42_energy_outer_v2/DESIGN.json')
    for key,value in coarse_design.items():assert expected_design[key]==value
    assert [r['label'] for r in coarse_rows]==[r['label'] for r in fine_rows]
    assert [r['rhs'] for r in coarse_rows]==[r['rhs'] for r in fine_rows]
    ds=list(map(F,coarse_design['D_grid']));vs=list(map(F,coarse_design['V_grid']));fd=list(map(F,fine_design['D_grid']));fv=list(map(F,fine_design['V_grid']))
    assert set(ds)<=set(fd) and set(vs)<=set(fv)
    mapping=[]
    for a,b in zip(fd,fd[1:]):
        i=bisect_right(ds,a)-1;assert ds[i]<=a<b<=ds[i+1]
        for c,d in zip(fv,fv[1:]):
            j=bisect_right(vs,c)-1;assert vs[j]<=c<d<=vs[j+1]
            mapping.append(i*(len(vs)-1)+j)
    assert len(mapping)==fine_design['good_box_cells']
    mapping.append(coarse_design['good_box_cells'])
    comparisons=0
    for coarse,fine in zip(coarse_rows,fine_rows):
        for j,parent in enumerate(mapping):assert fine['coefficients'][j]>=coarse['coefficients'][parent];comparisons+=1
    new=base.read(directory/'RESULT.json');rows=[]
    assert new['source_hashes']==previous['source_hashes']
    for name,digest in previous['source_hashes'].items():assert base.sha(root/name)==digest
    for original in previous['rows']:
        L=original['L'];fresh=next(row for row in new['rows'] if row['L']==L);lifts={}
        for side,index,sign in [('lower',0,1),('upper',1,-1)]:
            old_cost=[sign*v[index] for v in coarse_kernels[L]];new_cost=[sign*v[index] for v in fine_kernels[L]]
            assert all(v>=old_cost[p] for v,p in zip(new_cost,mapping))
            proof=original['proofs'][side];old_bound,old_parts=base.simplex_dual(old_cost,coarse_rows,proof['multipliers'])
            assert old_parts==proof['parts'] and sign*old_bound==F(proof['certified_bound'])
            lifted,parts=base.simplex_dual(new_cost,fine_rows,proof['multipliers'])
            assert lifted>=old_bound
            lifts[side]=dict(bound=str(sign*lifted),parts=parts,old_bound=str(sign*old_bound),certified_gain_in_bound=str(lifted-old_bound))
        old=list(map(F,original['actual_C0_energy_contrast_interval']));candidate=list(map(F,fresh['actual_C0_energy_contrast_interval']))
        final=[max(old[0],candidate[0],F(lifts['lower']['bound'])),min(old[1],candidate[1],F(lifts['upper']['bound']))]
        assert old[0]<=final[0]<=final[1]<=old[1]
        rows.append(dict(L=L,old_actual_interval=list(map(str,old)),new_solver_interval=list(map(str,candidate)),old_dual_lifts=lifts,
            actual_C0_energy_contrast_interval=list(map(str,final)),approximate=list(map(float,final))))
    return dict(subdivisions=subdivisions,old_good_cells=coarse_design['good_box_cells'],new_good_cells=fine_design['good_box_cells'],
        constraints=len(fine_rows),exact_coefficient_dominance_comparisons=comparisons,tail_cell_unchanged=True,
        prior_result='revisions/r42_energy_outer_v2/RESULT.json',prior_result_sha256=base.sha(previous_path),
        prior_builder_historical_sha256=base.sha(root/'revisions/r43_previous_certify_energy_outer_polytope.py'),
        current_compatible_builder_sha256=base.sha(root/'tools/certify_energy_outer_polytope.py'),
        fine_solver_result_sha256=base.sha(directory/'RESULT.json'),rows=rows,
        no_new_samples_matrices_or_integrals=True,old_bounds_preserved=True,independent_proof_review=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--subdivisions',type=int,default=2,choices=[2,4]);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.argv=[sys.argv[0],'--out',str(args.out)]
    base.build=lambda directory:BASE_BUILD(directory,args.subdivisions)
    base.main()
    result=refinement_certificate(root,args.out,args.subdivisions)
    base.save(args.out/'REFINEMENT.json',result)
    print('GRID REFINEMENT',[(r['L'],r['approximate']) for r in result['rows']],flush=True)


if __name__=='__main__':main()
