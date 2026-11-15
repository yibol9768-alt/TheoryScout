"""Exact references for scatter-only GD schedules with no step lower bound.

Rational finite algebra, not random experiments or an independent review of
the Gaussian asymptotic transfer theorem. All output files are exclusive.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import hashlib
import json

from check_uniform_gd_geometry import mul,transpose,scale,add,sqnorm,inner,inverse2

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def steps(family,cap,count,trace=F(2)):
    result=[]
    for j in range(1,count+1):
        if family=='constant':a=cap/2
        elif family=='harmonic':a=cap/F(j+1)
        elif family=='tiny_harmonic':a=cap/F((1<<40)*(j+1))
        elif family=='alternating':a=cap/(1 if j%2 else 7)
        elif family=='scatter_harmonic':a=cap/(j+1+trace)
        elif family=='increasing':a=cap*F(j+1,j+2)
        else:raise ValueError(family)
        assert 0<a<=cap;result.append(a)
    return result

FAMILIES=['constant','harmonic','tiny_harmonic','alternating','scatter_harmonic','increasing']
TIMES=[1,2,3,4,8,16,32,64]

def scalar_checks():
    checks=0;derivatives=0;cap=F(4,5);records=[]
    for family in FAMILIES:
        schedule=steps(family,cap,64)
        for s in [F(j,16) for j in range(12,21)]:
            q=F(1);dq=F(0);ddq=F(0);p=F(0);ref=F(1);cumulative=F(0)
            for t,eta in enumerate(schedule,1):
                factor=1-eta*s;assert 0<=factor<=1
                ddq,dq,q=ddq*factor-2*eta*dq,dq*factor-eta*q,q*factor
                ref*=1-eta;cumulative+=eta
                p=p*factor+eta;k=1-ref
                assert p==(1-q)/s and k>0
                assert -dq>=0 and -ddq<=0 and 0<=-s*dq<=1-q;derivatives+=1
                assert min(F(1),1/s)<=p/k<=max(F(1),1/s)
                delta=abs(s-1);assert abs(p/k-1)<=delta/(1-delta)
                assert q<=1/(1+s*cumulative);checks+=1
                if t in TIMES:records.append(dict(family=family,s=str(s),T=t,filter_ratio=str(p/k)))
    return dict(rational_scalar_filter_and_cumulative_product_cases=checks,
                exact_increasing_concave_numerator_cases=derivatives,finite_filter_records=len(records))

def matrix_checks():
    eye=[[F(1),F(0)],[F(0),F(1)]];z=[[F(1),F(1,5)],[F(-1,10),F(2)]];qz=sqnorm(z)
    rotations=[eye,[[F(3,5),F(-4,5)],[F(4,5),F(3,5)]]]
    endpoints=0;identities=0;ls_cases=0
    for delta,rot in product([F(1,8),F(1,32),F(1,128)],rotations):
        eigen=[1-delta,1+delta];s=mul(mul(rot,[[eigen[0],F(0)],[F(0),eigen[1]]]),transpose(rot))
        cap=1/(1+delta);epsilon=delta/(1-delta)
        for family in FAMILIES:
            u=scale(z,F(0));polynomial=scale(eye,F(0));power=eye;scalar=[F(0),F(0)];ref=F(1)
            for t,eta in enumerate(steps(family,cap,64,s[0][0]+s[1][1]),1):
                residual=add(eye,scale(s,-eta))
                u=add(mul(u,residual),scale(z,eta))
                polynomial=add(mul(polynomial,residual),scale(eye,eta));power=mul(power,residual)
                scalar=[p*(1-eta*v)+eta for p,v in zip(scalar,eigen)];ref*=1-eta;k=1-ref
                assert u==mul(z,polynomial)
                assert polynomial==mul(mul(rot,[[scalar[0],F(0)],[F(0),scalar[1]]]),transpose(rot))
                assert add(mul(u,s),scale(z,-1))==scale(mul(z,power),-1);identities+=1
                if t in TIMES:
                    assert max(abs(p/k-1) for p in scalar)<=epsilon
                    qu=sqnorm(u);dot=inner(u,z)
                    assert k*k*(1-epsilon)**2*qz<=qu<=k*k*(1+epsilon)**2*qz
                    assert dot>0 and (1-2*epsilon**2)**2*qu*qz<=dot*dot<=qu*qz
                    endpoints+=1
            pinv=inverse2(s);ls=mul(z,pinv);assert add(mul(ls,s),scale(z,-1))==scale(z,F(0))
            assert max(abs(1/v-1) for v in eigen)<=epsilon;ls_cases+=1
    return dict(rational_matrix_recurrence_residual_and_spectral_representation_cases=identities,
                exact_normalization_and_squared_center_bound_cases=endpoints,least_squares_endpoint_cases=ls_cases)

def stopping_checks():
    z=[[F(1),F(1,5)],[F(-1,10),F(2)]];qz=sqnorm(z);eye=[[F(1),F(0)],[F(0),F(1)]]
    rot=[[F(3,5),F(-4,5)],[F(4,5),F(3,5)]];cases=[];checks=0
    for delta in [F(1,8),F(1,32)]:
        low,high=1-delta,1+delta;s=mul(mul(rot,[[low,F(0)],[F(0),high]]),transpose(rot));cap=1/high
        for tol in [F(3,4),F(1,2),F(1,4)]:
            u=scale(z,F(0));power=eye;total=F(0);history=[]
            for t,eta in enumerate(steps('harmonic',cap,256),1):
                residual=add(eye,scale(s,-eta));u=add(mul(u,residual),scale(z,eta));power=mul(power,residual);total+=eta
                gradient=add(mul(u,s),scale(z,-1));norm=sqnorm(gradient)
                assert gradient==scale(mul(z,power),-1)
                assert norm<=qz/(1+low*total)**2;checks+=1;history.append(norm)
                if norm<=tol*tol*qz:break
            else:raise AssertionError('Finite rational harmonic test did not stop')
            assert all(v>tol*tol*qz for v in history[:-1])
            cases.append(dict(delta=str(delta),schedule='harmonic',tolerance=str(tol),selected_T=t))
    return dict(exact_harmonic_residual_stop_cases=cases,exact_residual_and_cumulative_bound_checks=checks)

def limits_and_selection_references():
    # Summable positive steps have residual product bounded below by1/2 at S=I.
    q=F(1);total=F(0)
    for j in range(1,65):
        eta=F(1,(j+1)*(j+2));total+=eta;q*=1-eta
        assert total==F(1,2)-F(1,j+2) and q>=1-total>F(1,2)
    # The tight ratio bound needs nonnegative factors, even though legacy
    # constant eta=3/2 is stable near the identity under its older bound.
    eta=F(3,2);s=F(11,10);p=2*eta-eta*eta*s;k=2*eta-eta*eta
    assert abs(1-eta*s)<1 and p/k<1/s
    # Finite scatter-only schedule/horizon policy: select6 endpoints on3
    # scatter states, independently of3 residual-Haar states.
    pc=[F(1,6),F(1,3),F(1,2)];ph=[F(1,3)]*3;target=F(1,2)
    loss=lambda c,a,h:(F((c+2*a+h)%5,4)-target)**2
    risks=[[sum(p*loss(c,a,h) for h,p in enumerate(ph)) for a in range(6)] for c in range(3)]
    envelope=sum(p*max(r) for p,r in zip(pc,risks));cases=0
    for policy in product(range(6),repeat=3):
        actual=sum(p*w*loss(c,policy[c],h) for c,p in enumerate(pc) for h,w in enumerate(ph))
        assert actual==sum(p*risks[c][policy[c]] for c,p in enumerate(pc))<=envelope;cases+=1
    return dict(summable_step_product_lower_bound_cases=64,overshoot_ratio_counterexample_detected=True,
                scatter_schedule_and_horizon_policy_cases=cases,rank_or_Haar_peeking_policy_permitted=False)

def compute():
    return dict(scalar=scalar_checks(),matrix=matrix_checks(),stopping=stopping_checks(),
                limitations=limits_and_selection_references(),random_datasets_generated=0,
                finite_references_not_independent_proof_review=True,
                scope='Uniform normalized nonovershooting GD filters; finite rational checks; no nonlinear growing-time or realized time-supremum claim')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--compare',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];result=compute()
    old=read(root/'revisions/R34_HISTORY.json');manifest=read(root/old['path']/'manifest.json')
    names=['tools/check_uniform_gd_geometry.py','tools/certify_weighted_chisquare.py','paper/appendix/AD_uniform_GD.tex',
           'revisions/R34_SCATTER_SELECTION_AUDIT.md']
    for n in names:assert sha(root/n)==manifest['files'][n],n
    result['source_hashes']={n:sha(root/n) for n in names+['tools/check_gd_varying_schedule.py']}
    if a.compare:assert read(a.compare)==result
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('PASS nonovershooting variable-step filter, normalized geometry, harmonic stopping and schedule-policy references')

if __name__=='__main__':main()
