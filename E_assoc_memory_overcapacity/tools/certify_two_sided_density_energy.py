"""Add signed energy-kernel minorants from actual threshold density lower bounds."""
from bisect import bisect_left,bisect_right
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import argparse
import sys

import certify_shared_cdf_energy as old
import certify_sphere_density_lower as minorant

base=old.base

class TwoSided(old.SharedCDF):
    def __init__(self,upper,lower):
        super().__init__(upper);self.lower={r['L']:r for r in lower['rows']}
        assert upper['m']==lower['m']
        for row in lower['rows']:
            assert [F(b['left'])**2 for b in row['bins']]==self.left
            assert [F(b['right'])**2 for b in row['bins']]==self.right

    @lru_cache(None)
    def minimum(self,length,start,end,tail):
        row=self.lower[length]
        values=[F(b['density_lower']) for b in row['bins'][start:end]]
        if tail:values.append(F(row['tail_density_lower']))
        assert values
        return min(values)

    @lru_cache(None)
    def inverse_interval(self,vl,vh,ml,mh):
        I=old.density.I;S=old.density.S
        delta=1/I.raw(I(vl).lo,I(vh).hi).sqrt()-1/I.raw(I(ml).lo,I(mh).hi).sqrt()
        return F(delta.lo,S),F(delta.hi,S)

    def signed_minorant(self,length,box,meanv):
        dl,dh,vl,vh=box;ml,mh=meanv
        assert 0<=dl<=dh and 0<vl<=vh and 0<ml<=mh
        xmin2=dl*dl/max(vh,mh);xmax2=dh*dh/min(vl,ml)
        start=bisect_left(self.right,xmin2);end=bisect_right(self.left,xmax2)
        bound=self.minimum(length,start,end,xmax2>=F(self.table['grid_end'])**2)
        al,ah=self.inverse_interval(vl,vh,ml,mh)
        if al>=0:return min(F(1),bound*dl*al),F(1)
        if ah<=0:return F(-1),max(F(-1),bound*dl*ah)
        return F(-1),F(1)

    def kernel(self,row,box,meanv,negative):
        previous=super().kernel(row,box,meanv,negative)
        if box[1]<=0:return previous
        lower,upper=self.signed_minorant(row['L'],box,meanv)
        lower=max(lower,previous[0]);upper=min(upper,previous[1])
        assert lower<=upper
        return lower,upper,*previous[2:]

def load(root):
    upper=base.read(root/'revisions/r44_sphere_density.json')
    lower=base.read(root/'revisions/r45_sphere_density_lower_v2.json')
    for table in [upper,lower]:
        for name,digest in table['source_hashes'].items():assert base.sha(root/name)==digest
    assert upper['m']==base.read(root/'revisions/r28_conditional_moments_v2/INPUTS.json')['n']-1
    return TwoSided(upper,lower)

def build(root):
    model=load(root);previous=base.kernel_bounds
    try:
        base.kernel_bounds=model.kernel
        return old.BASE_BUILD(root,2)
    finally:base.kernel_bounds=previous

def certificate(root,directory):
    old_design,old_rows,old_kernels=old.build(root)
    design,rows,kernels=build(root)
    assert old_design==design and old_rows==rows
    previous=base.read(root/'revisions/r44_shared_cdf_energy_v2/RESULT.json')
    retained=base.read(root/'revisions/r44_shared_cdf_energy_v2/SHARED_CDF.json')
    fresh=base.read(directory/'RESULT.json');answer=[];tightened=0
    assert fresh['source_hashes']==previous['source_hashes']
    for name,digest in previous['source_hashes'].items():assert base.sha(root/name)==digest
    for original in previous['rows']:
        length=original['L'];current=next(r for r in fresh['rows'] if r['L']==length);lifts={};counts={}
        assert all(a[2:]==b[2:] for a,b in zip(old_kernels[length],kernels[length]))
        for side,index,sign in [('lower',0,1),('upper',1,-1)]:
            old_cost=[sign*v[index] for v in old_kernels[length]];cost=[sign*v[index] for v in kernels[length]]
            assert all(b>=a for a,b in zip(old_cost,cost))
            counts[side]=sum(b>a for a,b in zip(old_cost,cost));tightened+=counts[side]
            proof=original['proofs'][side]
            before,parts=base.simplex_dual(old_cost,old_rows,proof['multipliers'])
            assert parts==proof['parts'] and sign*before==F(proof['certified_bound'])
            lifted,parts=base.simplex_dual(cost,rows,proof['multipliers']);assert lifted>=before
            lifts[side]=dict(bound=str(sign*lifted),parts=parts,improvement=str(lifted-before))
        old_interval=list(map(F,next(r for r in retained['rows'] if r['L']==length)['actual_C0_energy_contrast_interval']))
        candidate=list(map(F,current['actual_C0_energy_contrast_interval']))
        final=[max(old_interval[0],candidate[0],F(lifts['lower']['bound'])),min(old_interval[1],candidate[1],F(lifts['upper']['bound']))]
        assert old_interval[0]<=final[0]<=final[1]<=old_interval[1]
        answer.append(dict(L=length,previous_interval=list(map(str,old_interval)),fresh_interval=list(map(str,candidate)),
            old_dual_lifts=lifts,strictly_tightened_by_side=counts,actual_C0_energy_contrast_interval=list(map(str,final)),approximate=list(map(float,final))))
    names=['tools/certify_two_sided_density_energy.py','tools/certify_sphere_density_lower.py',
           'tools/certify_shared_cdf_energy.py','tools/certify_energy_outer_polytope.py',
           'revisions/r45_sphere_density_lower_v2.json','revisions/r44_sphere_density.json',
           'revisions/r44_shared_cdf_energy_v2/RESULT.json','revisions/r44_shared_cdf_energy_v2/SHARED_CDF.json']
    return dict(source_hashes={n:base.sha(root/n) for n in names},solver_result_sha256=base.sha(directory/'RESULT.json'),
        constraints=len(rows),cells=design['good_box_cells']+1,all_constraint_coefficients_and_rhs_identical=True,
        cost_comparisons=4*(design['good_box_cells']+1),strictly_tightened_cost_coefficients=tightened,
        actual_recall_cell_bounds_unchanged=True,old_bounds_retained=True,rows=answer,
        independent_proof_review=False,no_certified_LP_optimality_claim=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    base.build=build;sys.argv=[sys.argv[0],'--out',str(a.out)];base.main()
    result=certificate(root,a.out);base.save(a.out/'TWO_SIDED.json',result)
    print('TWO SIDED DENSITY',[(r['L'],r['approximate']) for r in result['rows']],flush=True)

if __name__=='__main__':main()
