"""Use one actual threshold CDF on both sides of the energy contrast."""
from bisect import bisect_left,bisect_right
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import argparse
import sys

import certify_energy_outer_polytope as base
import certify_sphere_order_density as density

BASE_BUILD=base.build
BASE_KERNEL=base.kernel_bounds

class SharedCDF:
    def __init__(self,table):
        self.table=table
        self.by_length={r['L']:r for r in table['rows']}
        self.left=[F(b['left'])**2 for b in table['rows'][0]['bins']]
        self.right=[F(b['right'])**2 for b in table['rows'][0]['bins']]
        assert all([F(b['left'])**2 for b in r['bins']]==self.left and
                   [F(b['right'])**2 for b in r['bins']]==self.right for r in table['rows'])

    @lru_cache(None)
    def maximum(self,length,start,end,tail):
        row=self.by_length[length]
        values=[F(b['density_upper']) for b in row['bins'][start:end]]
        if tail:values.append(F(row['tail_density_upper']))
        assert values
        return max(values)

    @lru_cache(None)
    def inverse_difference(self,vl,vh,ml,mh):
        first=1/density.I.raw(density.I(vl).lo,density.I(vh).hi).sqrt()
        second=1/density.I.raw(density.I(ml).lo,density.I(mh).hi).sqrt()
        delta=first-second
        return F(max(abs(delta.lo),abs(delta.hi)),density.S)

    def cap(self,length,box,meanv):
        dl,dh,vl,vh=box;ml,mh=meanv
        assert 0<=dl<=dh and 0<vl<=vh and 0<ml<=mh
        xmin2=dl*dl/max(vh,mh);xmax2=dh*dh/min(vl,ml)
        start=bisect_left(self.right,xmin2);end=bisect_right(self.left,xmax2)
        tail=xmax2>=F(self.table['grid_end'])**2
        bound=self.maximum(length,start,end,tail)
        return min(F(1),bound*dh*self.inverse_difference(vl,vh,ml,mh))

    def kernel(self,row,box,meanv,negative):
        old=BASE_KERNEL(row,box,meanv,negative)
        if box[1]<=0:return old
        cap=self.cap(row['L'],box,meanv)
        lower=max(old[0],-cap);upper=min(old[1],cap)
        assert lower<=upper
        return lower,upper,*old[2:]

def load(root):
    table=base.read(root/'revisions/r44_sphere_density.json')
    for name,digest in table['source_hashes'].items():assert base.sha(root/name)==digest
    assert table['m']==base.read(root/'revisions/r28_conditional_moments_v2/INPUTS.json')['n']-1
    return SharedCDF(table)

def build(root):
    shared=load(root);previous=base.kernel_bounds
    try:
        base.kernel_bounds=shared.kernel
        return BASE_BUILD(root,2)
    finally:base.kernel_bounds=previous

def certificate(root,directory):
    old_design,old_rows,old_kernels=BASE_BUILD(root,2)
    new_design,new_rows,new_kernels=build(root)
    assert old_design==new_design and old_rows==new_rows
    previous=base.read(root/'revisions/r43_energy_grid2/RESULT.json')
    retained=base.read(root/'revisions/r43_energy_grid2/REFINEMENT.json')
    fresh=base.read(directory/'RESULT.json');answer=[];improved=0
    assert fresh['source_hashes']==previous['source_hashes']
    for name,digest in previous['source_hashes'].items():assert base.sha(root/name)==digest
    for original in previous['rows']:
        length=original['L'];current=next(r for r in fresh['rows'] if r['L']==length);lifts={}
        for side,index,sign in [('lower',0,1),('upper',1,-1)]:
            old=[sign*v[index] for v in old_kernels[length]];new=[sign*v[index] for v in new_kernels[length]]
            assert all(b>=a for a,b in zip(old,new));improved+=sum(b>a for a,b in zip(old,new))
            proof=original['proofs'][side]
            old_bound,old_parts=base.simplex_dual(old,old_rows,proof['multipliers'])
            assert old_parts==proof['parts'] and sign*old_bound==F(proof['certified_bound'])
            lifted,parts=base.simplex_dual(new,new_rows,proof['multipliers']);assert lifted>=old_bound
            lifts[side]=dict(bound=str(sign*lifted),parts=parts,improvement=str(lifted-old_bound))
        old=list(map(F,next(r for r in retained['rows'] if r['L']==length)['actual_C0_energy_contrast_interval']))
        candidate=list(map(F,current['actual_C0_energy_contrast_interval']))
        final=[max(old[0],candidate[0],F(lifts['lower']['bound'])),min(old[1],candidate[1],F(lifts['upper']['bound']))]
        assert old[0]<=final[0]<=final[1]<=old[1]
        answer.append(dict(L=length,previous_interval=list(map(str,old)),fresh_interval=list(map(str,candidate)),
                           old_dual_lifts=lifts,actual_C0_energy_contrast_interval=list(map(str,final)),approximate=list(map(float,final))))
    names=['tools/certify_shared_cdf_energy.py','tools/certify_sphere_order_density.py','tools/certify_energy_outer_polytope.py',
           'revisions/r44_sphere_density.json','revisions/r43_energy_grid2/RESULT.json','revisions/r43_energy_grid2/REFINEMENT.json']
    return dict(source_hashes={n:base.sha(root/n) for n in names},solver_result_sha256=base.sha(directory/'RESULT.json'),
                constraints=len(new_rows),cells=new_design['good_box_cells']+1,all_constraint_coefficients_and_rhs_identical=True,
                cost_comparisons=4*(new_design['good_box_cells']+1),strictly_tightened_cost_coefficients=improved,
                actual_recall_cell_bounds_unchanged=True,old_bounds_retained=True,rows=answer,
                independent_proof_review=False,no_certified_LP_optimality_claim=True)

def references():
    # Continuous uniform and triangular CDFs with exact density maxima.
    checks=0
    def uniform(x):return max(F(0),min(F(1),x))
    def triangular(x):
        x=max(F(0),min(F(1),x))
        return 2*x*x if x<=F(1,2) else 1-2*(1-x)**2
    for fn,M in [(uniform,F(1)),(triangular,F(2))]:
        for d in [F(0),F(1,4),F(3,4),F(2)]:
            for r in [F(1,2),F(1),F(3,2)]:
                for mean_r in [F(3,4),F(1),F(5,4)]:
                    box=(d,d,r*r,r*r);meanv=(mean_r*mean_r,mean_r*mean_r)
                    fake=dict(rows=[dict(L=1,bins=[dict(left='0',right='6',density_upper=str(M))],tail_density_upper=str(M))],grid_end='6')
                    cap=SharedCDF(fake).cap(1,box,meanv)
                    contrast=fn(d/r)-fn(d/mean_r)
                    assert abs(contrast)<=cap
                    if r==mean_r:assert cap<F(1,1<<140)
                    checks+=1
    return dict(continuous_shared_CDF_cases=checks)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    base.build=build;sys.argv=[sys.argv[0],'--out',str(a.out)];base.main()
    result=certificate(root,a.out);base.save(a.out/'SHARED_CDF.json',result)
    print('SHARED CDF',[(r['L'],r['approximate']) for r in result['rows']],flush=True)

if __name__=='__main__':main()
