"""Conditional nonlinear Z4B matching sensitivity, with exact rational geometry.

The true inverse branches, slope error and curvature bounds are ASSUMPTIONS,
not inferred from the archived point data. Coverage remains nominal t-model
coverage from R14. No actual exactly-matched ranking is newly confirmed.
"""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left,bisect_right
from datetime import datetime,timezone
import argparse
import hashlib
import json
import time

from audit_z4b_matching import constraints,vertices,exact,METHODS,PAIRS


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def planes(points,ka,kb,ia,ib,old_lower,old_upper):
    out=[]
    for j,v in enumerate(points):
        correction=-ka*v[ia]+kb*v[ib];slope_weight=abs(v[ia])+abs(v[ib]);curvature_weight=(v[ia]**2+v[ib]**2)/2
        out.append(dict(vertex=j,base_lower=str(old_lower+correction),base_upper=str(old_upper+correction),
                        slope_weight=str(slope_weight),curvature_weight=str(curvature_weight)))
    return out


def envelope(rows,slope_error,curvature):
    assert slope_error>=0 and curvature>=0
    lo=[F(r['base_lower'])-slope_error*F(r['slope_weight'])-curvature*F(r['curvature_weight']) for r in rows]
    hi=[F(r['base_upper'])+slope_error*F(r['slope_weight'])+curvature*F(r['curvature_weight']) for r in rows]
    a=min(lo);b=max(hi)
    return dict(lower=str(a),upper=str(b),approximate_pp=[100*float(a),100*float(b)],
                lower_vertex=lo.index(a),upper_vertex=hi.index(b),strictly_positive=a>0)


def frozen_staircase_bounds(nodes,row,rlo,rhi):
        assert 0<rlo<=rhi
        t=list(map(F,row['t']));fl=list(map(F,row['sweeps'][-1]['lower_CDF']));fu=list(map(F,row['sweeps'][-1]['upper_CDF']))
        lower=sorted((F(n['threshold'])/rhi,F(n['C0_survival_bound'])) for n in nodes if n['side']=='lower')
        upper=sorted((F(n['threshold'])/rlo,F(n['C0_survival_bound'])) for n in nodes if n['side']=='upper')
        levels=[max(v for _,v in lower[j:]) for j in range(len(lower))]+[F(0)];lb=F(0)
        for j,(x,_) in enumerate(lower):
            k=bisect_right(t,x)-1;Fbound=fl[k] if k>=0 else F(0);lb+=(levels[j]-levels[j+1])*Fbound
        prev=F(1);ub=F(0)
        for x,value in upper:
            current=min(prev,value);k=bisect_left(t,x);Fbound=fu[k] if k<len(t) else F(1);ub+=(prev-current)*Fbound;prev=current
        ub+=prev;assert 0<=lb<=ub<=1
        return lb,ub


def tail_screen(root):
    score=read(root/'revisions/r32_frozen_energy/RESULT.json');cdf=read(root/'revisions/r23_cdf_dense8/RESULT.json')
    actual=read(root/'revisions/r27_recall_upper/RESULT.json');r=score['true_RMS'];den=1<<r['denominator_power2']
    rlo=F(r['lower_integer'])/den;rhi=F(r['upper_integer'])/den;out=[]
    assert actual['source_hashes']['revisions/r23_cdf_dense8/RESULT.json']==sha(root/'revisions/r23_cdf_dense8/RESULT.json')
    for row in cdf['rows']:
        lb,ub=frozen_staircase_bounds(score['nodes'],row,rlo,rhi)
        old=next(x for x in actual['rows'] if x['L']==row['L']);a=F(old['conditional_recall_lower'])-ub;b=F(old['conditional_recall_upper'])-lb
        out.append(dict(L=row['L'],Qr_lower=str(lb),Qr_upper=str(ub),Qr_approximate=[float(lb),float(ub)],
                        actual_minus_RMS_frozen_lower=str(a),actual_minus_RMS_frozen_upper=str(b),remainder_approximate=[float(a),float(b)],sign_undetermined=a<=0<=b))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic();protected=[]
    groups=[('R14',['revisions/r14_z4b_replay/INPUTS.json','revisions/r14_z4b_replay/VERTICES.json','revisions/r14_z4b_replay/RESULTS.json','tools/audit_z4b_matching.py']),
            ('R32',['revisions/r32_frozen_energy/RESULT.json']),('R23',['revisions/r23_cdf_dense8/RESULT.json']),('R27',['revisions/r27_recall_upper/RESULT.json'])]
    for revision,names in groups:
        receipt=read(root/f'revisions/{revision}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];protected.append(name)
    data=read(root/'revisions/r14_z4b_replay/INPUTS.json');assert data['methods']==METHODS
    points=vertices(constraints(data['matching']));saved=read(root/'revisions/r14_z4b_replay/VERTICES.json')
    assert [[str(x) for x in v] for v in points]==saved['matching'] and len(points)==44
    prior=read(root/'revisions/r14_z4b_replay/RESULTS.json');target=F(17,40)
    domains=[]
    for j,method in enumerate(METHODS):
        a=min(F(0),min(v[j] for v in points));b=max(F(0),max(v[j] for v in points))
        domains.append(dict(method=method,new_recall_minus_target_domain=[str(a),str(b)],new_recall_domain=[str(target+a),str(target+b)],
                            locked_model_slope=str(exact(data['slopes'][method]['k'])),true_inverse_branch_and_derivative_bounds_unverified=True))
    all_planes={};rows=[];settings=[(F(0),F(0)),(F(1),F(100)),(F(5),F(1000))]
    for a,b in PAIRS:
        ia=METHODS.index(a);ib=METHODS.index(b);ka=exact(data['slopes'][a]['k']);kb=exact(data['slopes'][b]['k'])
        old=next(r for r in data['old'] if r['pair']==a+'-'+b);name=a+'-'+b
        ps=planes(points,ka,kb,ia,ib,exact(old['lower']),exact(old['upper']));all_planes[name]=ps
        baseline=envelope(ps,F(0),F(0));oldrow=next(r for r in prior['results'] if r['pair']==name)['linearized_old_proxy_range']
        assert baseline['lower']==oldrow['lower'] and baseline['upper']==oldrow['upper']
        slope_limits=[(F(r['base_lower'])/F(r['slope_weight']),r['vertex']) for r in ps if F(r['slope_weight'])>0]
        curve_limits=[(F(r['base_lower'])/F(r['curvature_weight']),r['vertex']) for r in ps if F(r['curvature_weight'])>0]
        slope_limit,sv=min(slope_limits);curve_limit,cv=min(curve_limits);assert slope_limit>0 and curve_limit>0
        examples=[dict(common_slope_error=str(e),common_curvature=str(c),range=envelope(ps,e,c),assumptions_NOT_empirically_verified=True) for e,c in settings]
        rows.append(dict(pair=name,common_slope_error_axis_limit=str(slope_limit),slope_limit_vertex=sv,
                         common_curvature_axis_limit=str(curve_limit),curvature_limit_vertex=cv,
                         axis_limits_approximate=[float(slope_limit),float(curve_limit)],axis_limits_not_a_joint_rectangle=True,examples=examples))
    common_slope=min(F(r['common_slope_error_axis_limit']) for r in rows);common_curvature=min(F(r['common_curvature_axis_limit']) for r in rows)
    save(args.out/'FRONTIER.json',dict(inequality='base_lower - epsilon*slope_weight - Lambda*curvature_weight > 0 for every pair/vertex',
        curves=all_planes,vertex_count=len(points),epsilon_and_Lambda_nonnegative=True,curve_bounds_are_assumptions=True))
    save(args.out/'TAIL_SCREEN.json',dict(rows=tail_screen(root),scope='Existing marginal staircase screen only; broad signed remainder, not proof of sign or new integrals'))
    save(args.out/'INPUTS.json',dict(source_hashes={name:sha(root/name) for name in protected},curve_domains=domains,
        vertices=[[str(x) for x in v] for v in points],slope_error_units='old/new recall derivative ratio; fractions',curvature_units='old recall fraction / (new recall fraction)^2'))
    result=dict(rows=rows,all_pair_common_axis_limits=[str(common_slope),str(common_curvature)],
        joint_example=dict(epsilon='5',Lambda='1000',all_three_conditional_lower_bounds_positive=all(r['examples'][2]['range']['strictly_positive'] for r in rows)),
        geometry='Exact rational convex/concave envelope over the same 44-vertex nominal joint region',
        nominal_coverage_budgets=prior['coverage_error_budgets'],coverage_still_Student_t_approximation=True,
        inverse_branches_actual_slopes_and_curvatures_verified=False,actual_exact_matching_ranking_confirmed=False,
        new_samples=0,new_tests=0,registered_decisions_unchanged=data['decisions'],independent_review=False,
        scope='Conditional sensitivity frontier for nonlinear matching under explicitly unverified true-curve assumptions; not confirmation of exact matching',wall_seconds=time.monotonic()-start)
    save(args.out/'RESULT.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
