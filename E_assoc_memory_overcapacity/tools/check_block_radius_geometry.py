"""Exact references for block-radius conditioning and energy comparisons.

No Gaussian approximation, random samples or actual C0 joint integration.
"""
from fractions import Fraction as F
from pathlib import Path
from itertools import product
import argparse
import json
import math


def beta_polynomial(alpha,beta):
    n=alpha+beta-1;coeff=[F(0)]*(n+1)
    for j in range(alpha,n+1):
        for k in range(n-j+1):coeff[j+k]+=math.comb(n,j)*math.comb(n-j,k)*(-1)**k
    return coeff


def beta_cdf(alpha,beta,x):
    if x<=0:return F(0)
    if x>=1:return F(1)
    return sum(c*x**j for j,c in enumerate(beta_polynomial(alpha,beta)))


def direct_success(d,v,t):
    assert v>=0
    if v==0 or t==0:return d>=0
    if t>0:return d>=0 and d*d>=t*t*v
    return d>=0 or d*d<=t*t*v


def radial_threshold(a,b,n,t,r,actual):
    if b==0:return ('all' if actual or t<=0 else 'none'),None
    if t==0:return ('all' if a>=0 else 'none'),None
    if t>0 and a<=0:return 'none',None
    if t<0 and a>=0:return 'all',None
    z=t*t*b/((n-1+t*t)*a*a) if actual else r*t/(n*a)
    return ('upper' if t>0 else 'lower'),z


def event(kind,z,s):
    return {'all':True,'none':False,'upper':s>=z if z is not None else False,'lower':s<=z if z is not None else False}[kind]


def kernel(kind,z,alpha,beta):
    if kind=='all':return F(1)
    if kind=='none':return F(0)
    value=beta_cdf(alpha,beta,z)
    return 1-value if kind=='upper' else value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    matrices=[[[F(1),F(0)],[F(0),F(1)]],[[F(1),F(0)],[F(0),F(-1)]],
        [[F(0),F(1)],[F(-1),F(0)]],[[F(1),F(2)],[F(-1),F(1,2)]],
        [[F(1),F(0)],[F(0),F(0)]]]
    directions=[(F(1),F(0)),(F(0),F(1)),(F(3,5),F(4,5)),(F(3,5),-F(4,5)),
        (F(5,13),F(12,13)),(-F(5,13),F(12,13))]
    radial=[F(1,100),F(1,4),F(9,25),F(16,25),F(3,4),F(99,100)]
    thresholds=list(map(F,[-3,-1,0,1,3]))+[F(-1,2),F(1,2)]
    radii=[F(1,2),F(1),F(3,2)]
    event_checks=pencil_checks=saturation_checks=zero_kernel_checks=0;beta_checks=0
    for alpha,beta in product(range(1,5),repeat=2):
        p=beta_polynomial(alpha,beta);derivative=[(j+1)*p[j+1] for j in range(len(p)-1)]
        expected=[F(0)]*len(derivative);scale=F(math.factorial(alpha+beta-1),math.factorial(alpha-1)*math.factorial(beta-1))
        for k in range(beta):expected[alpha-1+k]=scale*math.comb(beta-1,k)*(-1)**k
        assert derivative==expected and sum(p)==1 and p[0]==0;beta_checks+=1
        values=[beta_cdf(alpha,beta,F(k,10)) for k in range(11)]
        assert values==sorted(values) and values[0]==0 and values[-1]==1
    for matrix,u,n in product(matrices,directions,[4,6,8]):
        assert sum(x*x for x in u)==1
        bu=[sum(x*y for x,y in zip(row,u)) for row in matrix]
        a=sum(x*y for x,y in zip(u,bu));b=sum(x*x for x in bu);assert a*a<=b
        alpha=1;beta=(n-2)//2
        for t,r in product(thresholds,radii):
            ka,za=radial_threshold(a,b,n,t,r,True);kr,zr=radial_threshold(a,b,n,t,r,False)
            pa=kernel(ka,za,alpha,beta);pr=kernel(kr,zr,alpha,beta)
            assert 0<=pa<=1 and 0<=pr<=1
            if b==0:
                assert a==0 and pa==1 and pr==int(t<=0);zero_kernel_checks+=1
            if t>0 and a>0:
                c=r*(n-1+t*t)/(n*t);pencil=c*a-b
                assert (zr-za)*pencil>=0 and (zr==za)==(pencil==0)
                assert (pa-pr)*pencil>=0
                if za>=1 and zr>=1:assert pa==pr==0;saturation_checks+=1
                pencil_checks+=1
            for s in radial:
                d=n*s*a;v=F(n*n,n-1)*(s*b-s*s*a*a)
                assert v>=0
                assert direct_success(d,v,t)==event(ka,za,s)
                assert (d>=r*t)==event(kr,zr,s);event_checks+=2
    # Containing block dimension is distinct from rank: q=4, rank(B)=2.
    # The event algebra is identical but the beta law changes to Beta(2,2).
    for t in thresholds:
        a=F(1);b=F(1);n=8;r=F(1)
        kind,z=radial_threshold(a,b,n,t,r,True)
        assert 0<=kernel(kind,z,2,2)<=1
    factorizations=0
    for q,n,k in product([1,2,4],[6,8,10],range(1,5)):
        radius_moment=math.prod(F(q+2*j,n+2*j) for j in range(k))
        direction_moment=math.prod(F(1+2*j,q+2*j) for j in range(k))
        ambient_moment=math.prod(F(1+2*j,n+2*j) for j in range(k))
        assert radius_moment*direction_moment==ambient_moment;factorizations+=1
    output=dict(beta_CDF_density_polynomial_checks=beta_checks,
        exact_direct_vs_radius_event_checks=event_checks,exact_pencil_sign_checks=pencil_checks,
        both_thresholds_saturated_checks=saturation_checks,zero_direction_kernel_checks=zero_kernel_checks,
        exact_radius_direction_monomial_factorizations=factorizations,
        negative_thresholds_zero_t_skew_and_rank_deficient_matrices_covered=True,
        containing_dimension_not_matrix_rank=True,actual_C0_joint_integration_performed=False,
        actual_C0_recall_or_energy_sign_changed=False,independent_proof_review=False)
    with args.out.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
