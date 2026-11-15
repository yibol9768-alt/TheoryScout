"""Moment majorants for common-threshold actual/RMS-energy success flips.

Affine squares with a negative constant/score-square term allow local
majorants. Fourth moments explicitly pay for their possible negative tails.
SciPy proposes weights; an exact cell residual repairs feasibility.
"""
from pathlib import Path
from fractions import Fraction as F
from bisect import bisect_left,bisect_right
from datetime import datetime,timezone
import argparse
import hashlib
import json
import math
import time

import certify_weighted_chisquare as integer


S=integer.SCALE
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def restore(r):return integer.I.raw(int(r['lower_integer']),int(r['upper_integer']))


def setup(root):
    inputs=read(root/'revisions/r28_conditional_moments_v2/INPUTS.json')
    moments=read(root/'revisions/r28_conditional_moments_v2/RESULT.json')['C0_moments']
    bad=read(root/'revisions/r27_recall_upper/BAD_EVENTS.json')
    n=inputs['n'];m=n-1;eta=F(inputs['eta']);kappa=F(inputs['common_K_norm']);nu=F(inputs['N_operator_perturbation'])
    error=F(n*n,m)*(nu+2*kappa*eta)
    vlo=F(19,25);vhi=F(139,100);dmax=F(10)
    assert vlo<F(bad['l0'])**2-error and vhi>F(bad['u0'])**2+error
    assert F(bad['center'])+F(bad['radius'])+n*eta<dmax
    assert F(bad['center'])-F(bad['radius'])-n*eta>F(-3)
    meanv=restore(moments['mean_V']);mu=[F(meanv.lo,S),F(meanv.hi,S)]
    ds=[F(-3),F(-2),F(-1),F(0),F(2)]+[F(j,10) for j in range(21,71)]+[dmax]
    vs=[F(j,100) for j in range(76,140)]
    assert ds==sorted(set(ds)) and vs[0]==vlo and vs[-1]==vhi
    directions=[(0,10),(1,0),(1,5),(1,-5),(1,10),(1,-10),(1,20),(1,-20)]
    offsets=[F(-3),F(-2),F(-1),F(-1,2),F(0),F(1,2),F(1),F(2),F(3)]
    d0=F(3834524,1000000);v0=F(1056992,1000000)
    features=[dict(a='0',b='0',c='1')]
    for a,b in directions:
        for offset in offsets:features.append(dict(a=str(a),b=str(b),c=str(-a*d0-b*v0-offset)))
    features.extend([dict(a='0',b='0',c='1',negative=True),dict(a='1',b='0',c=str(-d0),negative=True)])
    costs=[];penalties=[];delta=F(bad['union_upper'])
    for feature in features:
        a,b,c=[F(feature[k]) for k in ['a','b','c']]
        value=a*a*restore(moments['variance_D'])+2*a*b*restore(moments['covariance_DV'])+b*b*restore(moments['variance_V'])
        value=value+(a*restore(moments['mean_D'])+b*meanv+c).square()
        assert value.hi>=0
        if feature.get('negative',False):
            assert b==0
            shift=a*restore(moments['mean_D'])+c
            fourth=a**4*restore(moments['fourth_central_D'])+4*a**3*shift*restore(moments['third_central_D'])+6*a*a*shift.square()*restore(moments['variance_D'])+shift.square().square()
            penalty=(fourth*delta).sqrt()
            costs.append(-F(value.lo,S));penalties.append(F(penalty.hi,S))
        else:costs.append(F(value.hi,S));penalties.append(F(0))
    negative=F(1,1<<100)
    for length in [1,5]:assert (1<<100)*sum(math.comb(m,j) for j in range(length))<=(1<<m)
    return dict(n=n,energy_operator_error=str(error),good_event_failure_upper=bad['union_upper'],
        negative_threshold_upper=str(negative),mean_V=list(map(str,mu)),D_grid=list(map(str,ds)),V_grid=list(map(str,vs)),
        features=features,feature_expectation_uppers=list(map(str,costs)),negative_feature_bad_tail_penalties=list(map(str,penalties)))


def square_min(feature,dlo,dhi,vlo,vhi):
    a,b,c=[F(feature[k]) for k in ['a','b','c']]
    low=min(a*dlo,a*dhi)+min(b*vlo,b*vhi)+c
    high=max(a*dlo,a*dhi)+max(b*vlo,b*vhi)+c
    if feature.get('negative',False):return -max(low*low,high*high)
    return F(0) if low<=0<=high else min(low*low,high*high)


def cdf_bound(row,d,energy,side):
    """CDF(D/sqrt(energy)) from rational squared comparisons, D>=0."""
    assert d>=0 and energy>0
    if '_prepared_r39' not in row:
        row['_prepared_r39']=([F(t)**2 for t in row['t']],list(map(F,row['sweeps'][-1]['lower_CDF'])),list(map(F,row['sweeps'][-1]['upper_CDF'])))
    squared,lower,upper=row['_prepared_r39'];ratio=d*d/energy
    if side=='lower':
        index=bisect_right(squared,ratio)-1
        return F(0) if index<0 else lower[index]
    index=bisect_left(squared,ratio)
    return F(1) if index==len(squared) else upper[index]


def flip_upper(row,dlo,dhi,vlo,vhi,meanv,side):
    ml,mh=meanv
    if dhi<=0:return F(0) # global F_T(0) allowance covers this entire side
    assert dlo>=0
    if side=='gain':
        if vlo>=mh:return F(0)
        return max(F(0),cdf_bound(row,dhi,vlo,'upper')-cdf_bound(row,dlo,mh,'lower'))
    assert side=='loss'
    if vhi<=ml:return F(0)
    return max(F(0),cdf_bound(row,dhi,ml,'upper')-cdf_bound(row,dlo,vhi,'lower'))


def cells(design):
    ds=list(map(F,design['D_grid']));vs=list(map(F,design['V_grid']))
    return [(a,b,c,d) for a,b in zip(ds,ds[1:]) for c,d in zip(vs,vs[1:])]


def repair(row,design,side,weights,minimums=None):
    boxes=cells(design);mu=list(map(F,design['mean_V']));features=design['features']
    weights=list(map(F,weights));assert len(weights)==len(features) and min(weights)>=0
    residual=F(0);witness=None
    for index,box in enumerate(boxes):
        minimum=minimums[index] if minimums is not None else [square_min(f,*box) for f in features]
        gap=flip_upper(row,*box,mu,side)-sum(w*x for w,x in zip(weights,minimum))
        if gap>residual:residual=gap;witness=index
    cost=sum(w*(F(x)+F(p)) for w,x,p in zip(weights,design['feature_expectation_uppers'],design['negative_feature_bad_tail_penalties']))+residual
    final=min(F(1),cost+F(design['good_event_failure_upper'])+F(design['negative_threshold_upper']))
    assert final>=0
    return dict(weights=list(map(str,weights)),constant_feasibility_repair=str(residual),repair_witness_cell=witness,
        majorant_expectation_and_bad_tail_upper=str(cost),flip_probability_upper=str(final),
        approximate_upper=float(final),bad_event_and_negative_threshold_both_retained=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];start=time.monotonic()
    groups={'R28':['revisions/r28_conditional_moments_v2/INPUTS.json','revisions/r28_conditional_moments_v2/RESULT.json','tools/certify_weighted_chisquare.py'],
            'R27':['revisions/r27_recall_upper/BAD_EVENTS.json'],'R23':['revisions/r23_cdf_dense8/RESULT.json']}
    hashes={}
    for rev,names in groups.items():
        receipt=read(root/f'revisions/{rev}_HISTORY.json');manifest=read(root/receipt['path']/'manifest.json')
        for name in names:assert sha(root/name)==manifest['files'][name];hashes[name]=sha(root/name)
    design=setup(root);design.update(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,
        multiplier_proposal_only_float=True,probability_target='Ggain=P(actual succeeds,RMS fails),Gloss=P(RMS succeeds,actual fails) using common auxiliary spherical T')
    save(args.out/'DESIGN.json',design)
    boxes=cells(design);minimums=[[square_min(feature,*box) for feature in design['features']] for box in boxes]
    import numpy as np
    from scipy.optimize import linprog
    matrix=-np.array([[float(x) for x in values] for values in minimums]);cost=np.array([float(F(x)+F(p)) for x,p in zip(design['feature_expectation_uppers'],design['negative_feature_bad_tail_penalties'])])
    rows=[]
    for row in read(root/'revisions/r23_cdf_dense8/RESULT.json')['rows']:
        proofs={}
        for side in ['gain','loss']:
            rhs=-np.array([float(flip_upper(row,*box,list(map(F,design['mean_V'])),side)) for box in boxes])
            fit=linprog(cost,A_ub=matrix,b_ub=rhs,bounds=(0,None),method='highs');assert fit.success,fit.message
            weights=[F.from_float(float(max(0,w))) for w in fit.x]
            proofs[side]=repair(row,design,side,weights,minimums)
            print(row['L'],side,proofs[side]['approximate_upper'],flush=True)
        rows.append(dict(L=row['L'],proofs=proofs,signed_actual_minus_RMS_interval=[str(-F(proofs['loss']['flip_probability_upper'])),proofs['gain']['flip_probability_upper']]))
    save(args.out/'RESULT.json',dict(rows=rows,source_hashes=hashes,grid_cells=len(boxes),quadratic_features=len(design['features']),negative_features_with_fourth_moment_tail_budget=2,
        no_D_V_independence_assumed=True,actual_joint_moments_used=True,no_gaussian_correct_score_approximation=True,
        new_samples=0,new_matrices=0,new_probability_integrals=0,new_CDF_integrals=0,independent_proof_review=False,
        actual_energy_remainder_sign_resolved=False,wall_seconds=time.monotonic()-start,
        scope='Conditional C0 one-query common-threshold energy flip budgets; inherited R13/R27/R28/R23 numerical assumptions'))


if __name__=='__main__':main()
