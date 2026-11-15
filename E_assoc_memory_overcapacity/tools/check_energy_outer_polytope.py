"""Standard-library audit of actual-C0 energy-contrast outer bounds."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import json

from certify_energy_outer_polytope import read,sha,save,build,kernel_bounds,simplex_dual,frozen_recall_bounds,linear_range,square_min


def success(d,v,t):
    if v==0 or t==0:return d>=0
    if t>0:return d>=0 and d*d>=t*t*v
    return d>=0 or d*d<=t*t*v


def references():
    ts=[F(-1),F(0),F(1,2),F(1),F(2),F(3)];grid=ts[2:]
    ds=[F(-2),F(-1),F(0),F(1,2),F(1),F(2),F(4)];vs=[F(1,4),F(1),F(9,4),F(4)]
    kernel_cases=event_cases=moment_cases=dual_cases=frozen_cases=0
    for weights,error in product([[1]*6,[0,1,2,0,3,1]],[F(0),F(1,20)]):
        law=[F(x,sum(weights)) for x in weights];cdf=lambda t:sum(p for x,p in zip(ts,law) if x<=t)
        row=dict(t=list(map(str,grid)),sweeps=[dict(lower_CDF=[str(max(F(0),cdf(t)-error)) for t in grid],upper_CDF=[str(min(F(1),cdf(t)+error)) for t in grid])])
        for dl,dh,vl,vh,meanv in product(ds[:-1],ds[1:],vs[:-1],vs[1:],[F(1),F(9,4)]):
            if ds.index(dh)!=ds.index(dl)+1 or vs.index(vh)!=vs.index(vl)+1:continue
            box=(dl,dh,vl,vh);low,high,ql,qh=kernel_bounds(row,box,(meanv,meanv),cdf(0))
            for d,v in product([dl,(dl+dh)/2,dh],[vl,(vl+vh)/2,vh]):
                qa=sum(p*success(d,v,t) for t,p in zip(ts,law));qr=sum(p*success(d,meanv,t) for t,p in zip(ts,law))
                assert low<=qa-qr<=high and ql<=qa<=qh;kernel_cases+=1
                for a,b in [(F(1),F(0)),(F(0),F(1)),(F(1),F(-2))]:
                    assert (a*d+b*v-F(1,3))**2>=square_min(a,b,-F(1,3),box);moment_cases+=1
                for t in [F(1,2),F(2)]:
                    lower=dl>=0 and dl*dl>=t*t*vh;upper=dh>=0 and dh*dh>=t*t*vl
                    assert int(lower)<=int(success(d,v,t))<=int(upper);event_cases+=1
        for dw in [[1,1,1],[1,2,3]]:
            support=[F(-1),F(1),F(3)];prob=[F(x,sum(dw)) for x in dw]
            nodes=[]
            for d in grid:
                value=sum(p for x,p in zip(support,prob) if x>=d)
                nodes.extend([dict(threshold=str(d),side='lower',C0_survival_bound=str(max(F(0),value-error))),dict(threshold=str(d),side='upper',C0_survival_bound=str(min(F(1),value+error)))])
            for meanv in [F(1),F(9,4)]:
                low,high=frozen_recall_bounds(nodes,row,(meanv,meanv))
                exact=sum(p*q*success(d,meanv,t) for d,p in zip(support,prob) for t,q in zip(ts,law))
                assert low<=exact<=high;frozen_cases+=1
    for weights,slack,cost,multipliers in product([[1,1,1],[1,2,3],[0,1,2]],[F(0),F(1,10)],[[F(-1),F(1,3),F(7,10)],[F(2),F(-3),F(1)]],[[],[(0,'1/7'),(1,'13/2')],[(0,'1000'),(1,'1/1000')]]):
        p=[F(x,sum(weights)) for x in weights];a=[[F(1),F(0),F(-1)],[F(0),F(1),F(1)]]
        rows=[dict(coefficients=row,rhs=sum(x*y for x,y in zip(row,p))+slack) for row in a]
        bound,_=simplex_dual(cost,rows,multipliers)
        assert bound<=sum(x*y for x,y in zip(cost,p));dual_cases+=1
    tail_cases=0
    for weights,mask in product([[1,1,1],[2,1,1]],product([0,1],repeat=3)):
        p=[F(x,sum(weights)) for x in weights];z=[F(-2),F(0),F(3)];delta=sum(w*m for w,m in zip(p,mask));value=sum(w*m*x for w,m,x in zip(p,mask,z))
        assert value*value<=delta*sum(w*x*x for w,x in zip(p,z));tail_cases+=1
    joint_covers=0
    for dl,dh in zip(ds,ds[1:]):
        for vl,vh in zip(vs,vs[1:]):
            for d,v,c,t in product([dl,(dl+dh)/2,dh],[vl,(vl+vh)/2,vh],[F(1,4),F(3,10)],[F(-1,10),F(0),F(1,10)]):
                n=100;minimum=c*dl-(dl*dl+(n-1)*vh)/n;maximum=c*dh-(dh*dh+(n-1)*vl)/n
                value=c*d-(d*d+(n-1)*v)/n
                assert minimum<=value<=maximum and int(minimum>=t)<=int(value>=t)<=int(maximum>=t)
                joint_covers+=1
    return dict(cell_kernel_CDF_references_with_atoms_and_negative_thresholds=kernel_cases,
        normalized_event_cover_checks=event_cases,nonnegative_moment_minimum_checks=moment_cases,
        arbitrary_multiplier_simplex_dual_checks=dual_cases,Cauchy_tail_reference_checks=tail_cases,
        frozen_energy_staircase_checks=frozen_cases,joint_event_cell_cover_checks=joint_covers)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];result=read(args.certificate/'RESULT.json');saved_design=read(args.certificate/'DESIGN.json')
    assert result['source_hashes']==saved_design['source_hashes']
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    design,rows,kernels=build(root)
    for key,value in design.items():assert saved_design[key]==value
    assert len(rows)==result['constraints'] and design['good_box_cells']==result['design_cells']
    audit=[]
    for record in result['rows']:
        L=record['L'];values=kernels[L]
        for side,index,sign in [('lower',0,1),('upper',1,-1)]:
            proof=record['proofs'][side];cost=[sign*v[index] for v in values]
            bound,parts=simplex_dual(cost,rows,proof['multipliers'])
            assert parts==proof['parts'] and F(proof['certified_bound'])==sign*bound
        raw=[F(record['proofs']['lower']['certified_bound']),F(record['proofs']['upper']['certified_bound'])]
        assert record['cell_dual_interval']==list(map(str,raw))
        cdf=next(v for v in read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows'] if v['L']==L)
        qrl,qru=frozen_recall_bounds(read(root/'revisions/r32_frozen_energy/RESULT.json')['nodes'],cdf,list(map(F,design['mean_V_interval'])))
        assert record['frozen_recall_interval']==list(map(str,[qrl,qru]))
        previous=next(v for v in read(root/'revisions/r37_joint_radial_v2/RESULT.json')['rows'] if v['L']==L)
        unpaired=[F(previous['conditional_recall_lower'])-qru,F(previous['conditional_recall_upper'])-qrl]
        assert record['unpaired_interval']==list(map(str,unpaired))
        final=[max(raw[0],unpaired[0]),min(raw[1],unpaired[1])]
        assert record['actual_C0_energy_contrast_interval']==list(map(str,final)) and final[0]<=final[1]
        audit.append(dict(L=L,actual_energy_contrast_interval=list(map(str,final)),dual_counts={k:len(v['multipliers']) for k,v in record['proofs'].items()}))
    save(args.out,dict(reference=references(),source_hashes=len(result['source_hashes']),good_box_cells=design['good_box_cells'],tail_cell_count=1,constraint_rows=len(rows),
        exact_simplex_dual_certificates=4,tail_mass_and_negative_threshold_retained=True,full_moment_intervals_not_selected_midpoints=True,
        actual_C0_outer_enclosure_not_finite_atomic_witness=True,solver_required=False,audits=audit,independent_proof_review=False))
    print('PASS actual energy outer polytope, tail and four exact duals')


if __name__=='__main__':main()
