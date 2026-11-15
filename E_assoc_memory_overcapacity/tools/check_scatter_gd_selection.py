"""Exact algebra/selection references for scatter-measurable GD horizons.

No random datasets or claim of independent proof review. Counterexamples
concern generic bounded losses, not failure examples of the actual GD model.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import argparse
import json

from check_uniform_gd_geometry import mul,transpose,scale,add,sqnorm,inner,inverse2,op2
from certify_weighted_chisquare import I,SCALE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    eye=[[F(int(i==j)) for j in range(2)] for i in range(2)];cases=[];identities=0;ls_cases=0
    for e in [F(1,128),F(1,256),F(1,512)]:
        factor=[[F(1),F(0),F(0),F(0)],[e,F(1),F(0),F(0)],[2*e,-e,F(1),F(0)],[e,2*e,-e,F(1)]]
        fx,fy=factor[:2],factor[2:];s=mul(fx,transpose(fx));z=scale(mul(fy,transpose(fx)),F(8));qz=sqnorm(z)
        deltaS=max(sum(abs(s[i][j]-int(i==j)) for j in range(2)) for i in range(2))
        ls=mul(z,inverse2(s));assert sqnorm(add(mul(ls,s),scale(z,-1)))==0;ls_cases+=1
        for eta in [F(1,2),F(1),F(3,2)]:
            residual=add(eye,scale(s,-eta));r=op2(residual);assert r<1
            rho=max(abs(1-eta*F(7,8)),abs(1-eta*F(9,8)));constant=eta**2/((1-rho)**2*(1-abs(1-eta)))
            epsilon=constant*deltaS;assert epsilon<F(1,2)
            for tol in [F(1,2),F(1,16),F(1,256),F(1,4096)]:
                u=scale(z,F(0));power=eye;polynomial=scale(eye,F(0));squares=[]
                for t in range(1,257):
                    u=add(u,scale(add(z,scale(mul(u,s),-1)),eta))
                    polynomial=add(polynomial,scale(power,eta));power=mul(power,residual)
                    gradient=add(mul(u,s),scale(z,-1));squared=sqnorm(gradient)
                    assert u==mul(z,polynomial) and gradient==scale(mul(z,power),-1)
                    assert squared<=qz*r**(2*t);identities+=1;squares.append(squared)
                    if squared<=tol*tol*qz:break
                else:raise AssertionError('Positive tolerance did not stop on stable rational test')
                assert all(v>tol*tol*qz for v in squares[:-1])
                qu=sqnorm(u);dot=inner(u,z);assert dot>0 and dot*dot<=qu*qz
                center_loss=1-I(dot)/I(qu*qz).sqrt();assert F(center_loss.hi,SCALE)<=2*epsilon**2
                cases.append(dict(factor_epsilon=str(e),eta=str(eta),tolerance=str(tol),selected_T=t,
                                  selected_residual_relative_squared=str(squares[-1]/qz),contraction_upper=str(r),center_relative_loss=center_loss.record()))
    # Conditional selection identity, finite countable-time analogue.
    pc=[F(1,6),F(1,3),F(1,2)];pq=[F(1,3)]*3;target=F(1,2)
    recall=lambda c,t,q:F((c+2*t+q)%5,4)
    risks=[[sum(p*(recall(c,t,q)-target)**2 for q,p in enumerate(pq)) for t in range(3)] for c in range(3)]
    envelope=sum(p*max(row) for p,row in zip(pc,risks));selected=[]
    for selector in product(range(3),repeat=3):
        direct=sum(p*w*(recall(c,selector[c],q)-target)**2 for c,p in enumerate(pc) for q,w in enumerate(pq))
        conditional=sum(p*risks[c][selector[c]] for c,p in enumerate(pc))
        assert direct==conditional<=envelope;selected.append(direct)
    assert max(selected)==envelope
    counterexamples=[]
    for count in [2,4,8]:
        # Scatter-only selection can defeat a bound with sup OUTSIDE E.
        fixed=max(sum(F(int(c==t),count) for c in range(count)) for t in range(count))
        scatter_adaptive=sum(F(int(c==c),count) for c in range(count));assert fixed==F(1,count) and scatter_adaptive==1
        # Rank/Q peeking can defeat even E sup_T E[loss_T|C] with C constant.
        conditional_envelope=F(1,count);rank_peeking=sum(F(int(q==q),count) for q in range(count));assert rank_peeking==1
        counterexamples.append(dict(states=count,sup_fixed_time_risk=str(fixed),scatter_selected_risk=str(scatter_adaptive),
            conditional_risk_envelope_for_constant_scatter=str(conditional_envelope),Q_peeking_risk=str(rank_peeking)))
    # Exact fourth-moment Wick pairing counts behind the old Z-norm bound.
    wick=[]
    for n in [1,2,3,5,8]:
        aa=ab=bb=0
        for k,l,r,s in product(range(n),repeat=4):
            a=int(k==l and r==s);b=int(k==r and l==s)+int(k==s and l==r)
            aa+=a*a;ab+=a*b;bb+=b*b
        assert (aa,ab,bb)==(n*n,2*n,2*n*n+2*n)
        for d in [1,2,3,5]:
            second=F(d**4*aa+2*d**3*ab+d*d*bb,n*n);variance=second-d**4
            assert variance==2*d*d*(1+F(2*d+1,n))
            wick.append(dict(d=d,n=n,variance_Z_norm_squared=str(variance)))
    out=dict(exact_residual_stopping_cases=len(cases),exact_recurrence_residual_and_contraction_checks=identities,
        least_squares_residual_cases=ls_cases,conditional_selection_identity_checks=len(selected),
        conditional_envelope_optimal_in_finite_reference=True,generic_selection_counterexamples=counterexamples,
        exact_Wick_pairing_norm_variance_cases=len(wick),random_datasets_generated=0,
        rank_selected_GD_failure_claimed=False,independent_proof_review=False,
        scope='Exact algebra and finite conditional-selection references, not a proof of the Gaussian asymptotic corollary',cases=cases,wick_cases=wick)
    with args.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['cases','wick_cases']},indent=2))


if __name__=='__main__':main()
