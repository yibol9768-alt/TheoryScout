"""Exact references and arithmetic audit for conditional Z4B remainder bounds."""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import argparse
import json

from bound_z4b_matching_remainder import read,sha,save,planes,envelope,frozen_staircase_bounds
from audit_z4b_matching import exact,METHODS,PAIRS


def references(points,data):
    taylor=0;interior=0;axes=0;staircases=0
    for theta in [F(-1,50),F(-1,100),F(0),F(1,100),F(1,50)]:
        for eps in [F(0),F(1,2),F(2)]:
            for curve in [F(0),F(1),F(10)]:
                for dk in [-eps,F(0),eps]:
                    for second in [-curve,F(0),curve]:
                        p=F(1,2);k=F(-1);actual=p-(k+dk)*theta+second*theta**2/2;proxy=p-k*theta
                        allowance=eps*abs(theta)+curve*theta**2/2
                        assert proxy-allowance<=actual<=proxy+allowance;taylor+=1
    probes=[points[j] for j in [0,5,13,22,43]]
    for a,b in PAIRS:
        ia=METHODS.index(a);ib=METHODS.index(b);ka=exact(data['slopes'][a]['k']);kb=exact(data['slopes'][b]['k'])
        old=next(r for r in data['old'] if r['pair']==a+'-'+b);L=exact(old['lower']);U=exact(old['upper'])
        ps=planes(points,ka,kb,ia,ib,L,U)
        ecrit=min(F(r['base_lower'])/F(r['slope_weight']) for r in ps if F(r['slope_weight']))
        ccrit=min(F(r['base_lower'])/F(r['curvature_weight']) for r in ps if F(r['curvature_weight']))
        for e,c in [(ecrit,F(0)),(F(0),ccrit)]:
            assert F(envelope(ps,e,c)['lower'])==0
            assert F(envelope(ps,e*F(9,10),c*F(9,10))['lower'])>0
            assert F(envelope(ps,e*F(11,10),c*F(11,10))['lower'])<0;axes+=3
        for e,c in [(F(0),F(0)),(F(1),F(100)),(F(5),F(1000))]:
            bound=envelope(ps,e,c)
            for v,w in combinations(probes,2):
                for fraction in [F(1,4),F(1,2),F(3,4)]:
                    x=[fraction*p+(1-fraction)*q for p,q in zip(v,w)]
                    correction=-ka*x[ia]+kb*x[ib];allow=e*(abs(x[ia])+abs(x[ib]))+c*(x[ia]**2+x[ib]**2)/2
                    assert F(bound['lower'])<=L+correction-allow
                    assert U+correction+allow<=F(bound['upper']);interior+=1
    # Same locks, same true slope, monotone old curves and unique roots;
    # different curvature can still reverse the exactly matched ordering.
    theta=F(1,100);k=F(-1);curvature_coefficient=F(1,4)/theta**2
    def old(p,c,x):return p+k*(x-theta)+c*(x-theta)**2
    for p in [F(1,2),F(3,5)]:
        for c in [F(0),curvature_coefficient]:
            assert old(p,c,theta)==p
            for j in range(11):
                x=theta*F(j,10);assert 0<=old(p,c,x)<=1 and k+2*c*(x-theta)<0
    first=old(F(1,2),F(0),F(0))-old(F(3,5),curvature_coefficient,F(0))
    second=old(F(1,2),curvature_coefficient,F(0))-old(F(3,5),F(0),F(0))
    assert first==F(-7,20) and second==F(3,20)
    # Existing marginal screen: exact finite laws with atoms and negatives.
    D=[F(-2),F(0),F(1),F(2),F(4)];T=[F(-1),F(0),F(1,2),F(3,2),F(3)]
    grid=[F(1,4),F(1,2),F(1),F(3,2),F(2),F(3),F(4)]
    for dp in [[F(1,5)]*5,[F(1,10),F(1,10),F(1,5),F(1,5),F(2,5)]]:
        for tp in [[F(1,5)]*5,[F(0),F(0),F(1,4),F(1,4),F(1,2)]]:
            cdf=lambda x:sum(p for p,t in zip(tp,T) if t<=x)
            row=dict(t=list(map(str,grid)),sweeps=[dict(lower_CDF=list(map(lambda x:str(cdf(x)),grid)),upper_CDF=list(map(lambda x:str(cdf(x)),grid)))])
            nodes=[dict(threshold=str(x),side=side,C0_survival_bound=str(sum(p for p,d in zip(dp,D) if d>=x))) for x in [F(1,2),F(3,2),F(5,2)] for side in ['lower','upper']]
            lo,hi=frozen_staircase_bounds(nodes,row,F(1),F(3,2))
            for scale in [F(1),F(5,4),F(3,2)]:
                actual=sum(p*q for p,d in zip(dp,D) for q,t in zip(tp,T) if d>=scale*t)
                assert lo<=actual<=hi;staircases+=1
    return dict(exact_quadratic_Taylor_checks=taylor,interior_convex_combination_checks=interior,
                exact_axis_boundary_checks=axes,finite_staircase_screen_checks=staircases,
                monotone_curve_counterexample_matched_contrasts=[str(first),str(second)],independent_review=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('certificate',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];inputs=read(args.certificate/'INPUTS.json');result=read(args.certificate/'RESULT.json');frontier=read(args.certificate/'FRONTIER.json')
    for name,digest in inputs['source_hashes'].items():assert sha(root/name)==digest
    data=read(root/'revisions/r14_z4b_replay/INPUTS.json');points=[[F(x) for x in p] for p in inputs['vertices']]
    for j,method in enumerate(METHODS):
        domain=inputs['curve_domains'][j];a=min(F(0),min(v[j] for v in points));b=max(F(0),max(v[j] for v in points))
        assert domain['method']==method and domain['new_recall_minus_target_domain']==[str(a),str(b)]
        assert domain['new_recall_domain']==[str(F(17,40)+a),str(F(17,40)+b)]
        assert domain['locked_model_slope']==str(exact(data['slopes'][method]['k']))
    for row,(a,b) in zip(result['rows'],PAIRS):
        old=next(r for r in data['old'] if r['pair']==a+'-'+b)
        ps=planes(points,exact(data['slopes'][a]['k']),exact(data['slopes'][b]['k']),METHODS.index(a),METHODS.index(b),exact(old['lower']),exact(old['upper']))
        assert ps==frontier['curves'][row['pair']]
        for example in row['examples']:assert envelope(ps,F(example['common_slope_error']),F(example['common_curvature']))==example['range']
        assert min(F(r['base_lower'])/F(r['slope_weight']) for r in ps if F(r['slope_weight']))==F(row['common_slope_error_axis_limit'])
        assert min(F(r['base_lower'])/F(r['curvature_weight']) for r in ps if F(r['curvature_weight']))==F(row['common_curvature_axis_limit'])
    assert result['registered_decisions_unchanged']==data['decisions']
    assert result['inverse_branches_actual_slopes_and_curvatures_verified'] is False
    out=dict(reference=references(points,data),arithmetic_audit=dict(source_hashes=len(inputs['source_hashes']),pairs=3,vertices=44,frontier_inequalities=132,illustrative_parameter_ranges=9,axis_limits=6,registered_decisions_unchanged=True,actual_curve_assumptions_not_verified=True))
    save(args.out,out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
