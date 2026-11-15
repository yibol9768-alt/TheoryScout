"""Exact-factor checks of the GD filter and transfer trace identities.

Deterministic rational matrices, no random training datasets. Square-root
comparisons use the standard-library integer interval module from R9.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse
import json

from certify_weighted_chisquare import I,SCALE


def tr(a):return sum(a[i][i] for i in range(len(a)))
def transpose(a):return list(map(list,zip(*a)))
def mul(a,b):return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
def add(a,b):return [[x+y for x,y in zip(ar,br)] for ar,br in zip(a,b)]
def scale(a,s):return [[s*x for x in row] for row in a]
def sqnorm(a):return sum(x*x for row in a for x in row)
def inner(a,b):return sum(x*y for ar,br in zip(a,b) for x,y in zip(ar,br))
def op2(a):
    assert a[0][1]==a[1][0]
    radius=I((a[0][0]-a[1][1])**2+4*a[0][1]**2).sqrt()
    plus=(I(tr(a))+radius)/2;minus=(I(tr(a))-radius)/2
    return F(max(abs(plus.lo),abs(plus.hi),abs(minus.lo),abs(minus.hi)),SCALE)
def inverse2(a):
    determinant=a[0][0]*a[1][1]-a[0][1]*a[1][0];assert determinant>0
    return scale([[a[1][1],-a[0][1]],[-a[1][0],a[0][0]]],1/determinant)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    records=[];eye=[[F(int(i==j)) for j in range(2)] for i in range(2)]
    wanted=[1,2,3,4,8,32,128];n=64
    for e in [F(1,64),F(1,128),F(1,256)]:
        factor=[[F(1),F(0),F(0),F(0)],[e,F(1),F(0),F(0)],[2*e,-e,F(1),F(0)],[e,2*e,-e,F(1)]]
        c=mul(factor,transpose(factor));fx,fy=factor[:2],factor[2:]
        s=mul(fx,transpose(fx));z=scale(mul(fy,transpose(fx)),F(8))
        delta=max(sum(abs(c[i][j]-int(i==j)) for j in range(4)) for i in range(4));assert delta<F(1,8)
        deltaS=max(sum(abs(s[i][j]-int(i==j)) for j in range(2)) for i in range(2))
        for eta in [F(1,2),F(1),F(3,2)]:
            rho=max(abs(1-eta*F(7,8)),abs(1-eta*F(9,8)));lip=eta**2/(1-rho)**2;kmin=1-abs(1-eta);constant=lip/kmin
            epsilon=constant*deltaS;assert epsilon<1
            residual=add(eye,scale(s,-eta));power=eye;p=scale(eye,F(0));u=scale(z,F(0));endpoints=[]
            for t in range(1,max(wanted)+1):
                p=add(p,scale(power,eta));power=mul(power,residual)
                u=add(u,scale(add(z,scale(mul(u,s),-1)),eta))
                if t in wanted:
                    assert u==mul(z,p)
                    endpoints.append((t,u,p,1-(1-eta)**t))
            pinv=inverse2(s);endpoints.append(('infinity',mul(z,pinv),pinv,F(1)))
            for t,u,p,k in endpoints:
                assert k>=kmin and p[0][0]>0 and p[0][0]*p[1][1]-p[0][1]*p[1][0]>0
                assert op2(add(p,scale(eye,-k)))<=lip*deltaS
                qu,qz=sqnorm(u),sqnorm(z);dot=inner(u,z)
                assert dot>0 and dot*dot<=qu*qz
                center_loss=1-I(dot)/I(qu*qz).sqrt()
                assert F(center_loss.hi,SCALE)<=2*epsilon**2
                ratio=I(op2(mul(u,transpose(u))))/I(qu)
                zratio=I(op2(mul(z,transpose(z))))/I(qz)
                ratio_bound=I(((1+epsilon)/(1-epsilon))**2)*zratio
                assert F(ratio.hi,SCALE)<=F(ratio_bound.lo,SCALE)
                raw=mul(mul(transpose(fy),u),fx)
                assert tr(raw)==dot/8
                normB=sqnorm(raw)/qu;traceB2=tr(mul(raw,raw))/qu;mu2=tr(raw)**2/qu
                twiceK=normB+traceB2
                assert (1-delta)**2<=normB<=(1+delta)**2
                assert 1-2*delta<=twiceK<=1+2*delta+2*delta**2
                nn=mul(transpose(raw),raw);traceN2=tr(mul(nn,nn))/(qu*qu)
                bound=I((1+delta)**4)*ratio
                assert traceN2<=F(bound.lo,SCALE)
                spectral_variance=twiceK-2*mu2/n;true_variance=F(n,n+2)*spectral_variance
                meanV=(n*normB-mu2-true_variance)/(n-1);assert true_variance>0 and meanV>0
                records.append(dict(factor_epsilon=str(e),eta=str(eta),T=t,scatter_delta_upper=str(delta),
                     center_relative_loss=center_loss.record(),spectral_sigma2=str(spectral_variance),
                     true_sigma2=str(true_variance),expected_V=str(meanV)))
    result=dict(exact_matrix_cases=len(records),rational_recurrence_filter_checks=63,
                positive_filters=True,center_bound_and_transfer_trace_checks=True,
                random_datasets_generated=0,scope='Algebra/interval checks only; not proof review or Monte Carlo evidence',cases=records)
    with args.out.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
