"""Standard-library replay of joint-mixture rational certificates, without LP."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import json

from certify_joint_radial_transfer import read,sha,save,mixture_constraints,objective,rational_dual
from refine_sphere_cdf import RadialCDF
import certify_sphere_threshold_cdf as sphere


def references():
    ts=[F(1,2),F(1),F(2),F(3)]
    support=[F(-1),F(0),F(1,2),F(3,2),F(3),F(4)]
    radii=[F(1,2),F(1),F(3,2)]
    mixture_cases=dual_cases=0
    for tw,rw in product([[1]*6,[0,1,2,0,3,1],[2,0,0,1,1,2]],[[1,1,1],[0,1,2],[2,1,0]]):
        tprob=[F(x,sum(tw)) for x in tw];rprob=[F(x,sum(rw)) for x in rw]
        truth=[sum(p for t,p in zip(support,tprob) if t<=node) for node in ts]
        records=[]
        for r in radii:
            value=sum(p for s,p in zip(radii,rprob) if s<=r)
            records.append(dict(radius=str(r),tail='lower',probability=dict(lower_integer=str(int(value*sphere.S)),upper_integer=str(-((-value.numerator*sphere.S)//value.denominator)))))
        radial=RadialCDF(records)
        for x,error in product([F(1,4),F(1,2),F(1),F(2),F(3),F(6)],[F(0),F(1,20)]):
            h=sum(p*q for t,p in zip(support,tprob) for r,q in zip(radii,rprob) if r*t<=x)
            node=dict(x=str(x),lower=str(max(F(0),h-error)),upper=str(min(F(1),h+error)))
            constraints=mixture_constraints(ts,radial,node)
            for a,b in constraints:
                assert sum(v*f for v,f in zip(a,truth))<=b;mixture_cases+=1
            low=[max(F(0),v-F(1,10)) for v in truth];high=[min(F(1),v+F(1,10)) for v in truth]
            for c in [[F(1),F(2),F(0),F(1)],[-F(1),F(0),F(1),-F(2)]]:
                for y in [[(0,'0')],[(0,'1/7'),(1,'3/2')],[(0,'1001/10'),(1,'2/3')]]:
                    bound,_=rational_dual(c,low,high,constraints,y)
                    assert bound<=sum(v*f for v,f in zip(c,truth));dual_cases+=1
    decomposition_cases=0
    for weights in [[1]*6,[0,1,2,0,3,1],[2,0,0,1,1,2]]:
        law=[F(x,sum(weights)) for x in weights]
        cdf=[sum(p for t,p in zip(support,law) if t<=node) for node in ts]
        for dweights in [[1,1,1],[1,2,1],[0,1,3]]:
            pairs=[(F(-1),F(1,2)),(F(1),F(1)),(F(3),F(3,2))]
            p=[F(x,sum(dweights)) for x in dweights]
            kernel=lambda t:sum(w for w,(d,r) in zip(p,pairs) if d>=t*r)
            heights=[kernel(t) for t in ts];jumps=[heights[j]-(heights[j+1] if j+1<len(ts) else 0) for j in range(len(ts))]
            lower=dict(t=list(map(str,ts)),jumps=list(map(str,jumps)))
            upper=dict(all_point_uppers={str(t):str(kernel(t)) for t in ts})
            def profile(t,kind):
                const,c=objective(ts,kind,lower,upper)
                return const+sum(w for x,w in zip(ts,c) if t<=x)
            assert all(profile(t,'lower')<=kernel(t)<=profile(t,'upper') for t in support)
            expectations={kind:sum(w*profile(t,kind) for t,w in zip(support,law)) for kind in ['lower','upper','gap']}
            for kind,value in expectations.items():
                const,c=objective(ts,kind,lower,upper);assert const+sum(v*f for v,f in zip(c,cdf))==value
            assert expectations['upper']-expectations['lower']==expectations['gap']
            for error in [F(0),F(1,20),F(1,10)]:
                a=expectations['lower']-error;b=expectations['upper']+error
                assert b-a==(b-expectations['upper'])+expectations['gap']+(expectations['lower']-a)
                assert b-a-expectations['gap']==2*error;decomposition_cases+=1
    return dict(mixture_inequalities_with_negative_T_and_atoms=mixture_cases,
                arbitrary_nonnegative_duals_and_box_correction=dual_cases,
                exact_staircase_expectations_and_gap_decompositions=decomposition_cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('certificate',type=Path)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];result=read(args.certificate/'RESULT.json');design=read(args.certificate/'DESIGN.json')
    assert result['source_hashes']==design['source_hashes']
    for name,digest in result['source_hashes'].items():assert sha(root/name)==digest
    assert sha(args.certificate/'GAUSSIAN.json')==result['gaussian_file_sha256']
    saved=read(args.certificate/'GAUSSIAN.json');cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json')
    lower=read(root/'revisions/r26_grid_nodes/RESULT.json');upper=read(root/'revisions/r36_right_tail/RESULT.json')
    radial=RadialCDF(read(root/'revisions/r21_sphere_cdf/RADII.json')['records'])
    m=read(root/'revisions/r21_sphere_cdf/RESULT.json')['m'];assert m==result['m']==saved['m']
    pi=sphere.integer.pi_interval();gaussians=0;audits=[];profile_checks=0
    assert [r['L'] for r in result['rows']]==[1,5]
    for row in result['rows']:
        length=row['L'];source=next(v for v in cdf['rows'] if v['L']==length)
        ts=list(map(F,source['t']));last=source['sweeps'][-1]
        const,gap_weights=objective(ts,'gap',lower,upper)
        probes=[ts[0]-1,*ts,*[(a+b)/2 for a,b in zip(ts,ts[1:])],ts[-1]+1]
        for t in probes:
            assert const+sum(v for x,v in zip(ts,gap_weights) if t<=x)>=0
            profile_checks+=1
        low=list(map(F,last['lower_CDF']));high=list(map(F,last['upper_CDF']));constraints=[]
        nodes=next(v for v in saved['rows'] if v['L']==length)['nodes']
        wanted=sorted({t*(1+F(step,1000)) for t in ts for step in design['steps_per_thousand']})
        assert list(map(lambda n:F(n['x']),nodes))==wanted
        for node in nodes:
            h=sphere.gaussian_order_cdf(m,length,F(node['x']),pi)
            assert (F(node['lower']),F(node['upper']))==(F(h.lo,sphere.S),F(h.hi,sphere.S));gaussians+=1
            constraints.extend(mixture_constraints(ts,radial,node))
        for j in range(len(ts)-1):
            a=[F(0)]*len(ts);a[j]=F(1);a[j+1]=F(-1);constraints.append((a,F(0)))
        assert len(constraints)==row['constraints']
        specifications={'lower':('lower',1),'upper':('upper',-1),'gap_lower':('gap',1),'gap_upper':('gap',-1)}
        for kind,proof in row['duals'].items():
            base,sign=specifications[kind];constant,c=objective(ts,base,lower,upper)
            bound,parts=rational_dual([sign*v for v in c],low,high,constraints,proof['multipliers'])
            assert parts==proof['parts'] and F(proof['constant'])==constant
            assert constant+sign*bound==F(proof['certified_staircase_expectation'])
        assert set(row['duals'])==set(specifications)
        previous=next(v for v in upper['rows'] if v['L']==length)
        assert row['previous_interval']==[previous['conditional_recall_lower'],previous['conditional_recall_upper']]
        lo=max(F(previous['conditional_recall_lower']),F(row['duals']['lower']['certified_staircase_expectation']))
        hi=min(F(previous['conditional_recall_upper']),F(row['duals']['upper']['certified_staircase_expectation']))
        assert 0<=lo<=hi<=1
        assert [F(row['conditional_recall_lower']),F(row['conditional_recall_upper'])]==[lo,hi]
        gaplo=max(F(0),F(row['duals']['gap_lower']['certified_staircase_expectation']))
        gaphi=min(hi-lo,F(row['duals']['gap_upper']['certified_staircase_expectation']))
        assert 0<=gaplo<=gaphi<=hi-lo
        assert row['envelope_gap_expectation']==list(map(str,[gaplo,gaphi]))
        assert row['combined_threshold_propagation_slack']==list(map(str,[hi-lo-gaphi,hi-lo-gaplo]))
        audits.append(dict(L=length,exact_dual_bounds=4,constraints=len(constraints),
             dual_nonzero_multipliers={k:len(v['multipliers']) for k,v in row['duals'].items()}))
    assert gaussians==result['new_gaussian_order_CDF_evaluations']
    output=dict(reference=references(),source_hashes=len(result['source_hashes']),
        gaussian_order_CDFs_recomputed_with_integer_arithmetic=gaussians,audits=audits,
        nonnegative_actual_envelope_gap_profile_checks=profile_checks,
        LP_or_scipy_required_for_replay=False,full_box_residual_correction_audited=True,
        inherited_radial_and_spectral_certificates_not_recomputed=True,independent_proof_review=False)
    save(args.out,output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()
