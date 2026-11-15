"""Exact references for adaptive secant endpoints and upper CDF transfer."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from bisect import bisect_right
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    secants=bad=0
    for m,a,R,l0,u0,beta in product([1,15,16594],map(F,[-1,0,3]),map(F,[1,6]),[F(1,2),F(7,8)],[F(47,40),F(3,2)],[F(1,3),F(2)]):
        for S in [u0+l0/2,u0+l0,u0+2*l0]:
            t=beta*S;l=min(l0,S-u0);u=S-l;assert 0<l<=l0<u0<=u
            for root in [l0,(l0+u0)/2,u0]:
                for d in [a-R,a,a+R]:
                    v=root*root;w=v+d*d/m
                    upper=(1+2*beta*a/m)*d-beta*w+beta*(R*R-a*a)/m-beta*l*u
                    assert upper>=d-t*root
                    scaled=(4*beta*m+8*beta*beta*a)*d-4*beta*beta*m*w+4*beta*beta*(R*R-a*a)-4*beta*beta*m*l*u
                    assert scaled==4*beta*m*upper;secants+=1
        for d,root in product([a-2*R,a-R,a,a+R,a+2*R],[l0/2,l0,(l0+u0)/2,u0,2*u0]):
            v=root*root;w=v+d*d/m;good=abs(d-a)<=R and l0*l0<=v<=u0*u0
            cover=d<=a-R or d>=a+R or w<=l0*l0+(abs(a)+R)**2/m or w>=u0*u0
            assert good or cover;bad+=1
    ts=[F(1,2),F(1),F(2)];support=[F(-1),F(1,2),F(3,4),F(1),F(3,2),F(2),F(3)]
    laws=[tuple(F(v,4) for v in weights) for weights in product(range(5),repeat=len(support)) if sum(weights)==4]
    transfers=0
    for law in laws:
        cdf=[sum(w for x,w in zip(support,law) if x<=t) for t in ts]
        for raw in product([F(0),F(1,2),F(1)],repeat=3):
            b=[];last=F(1)
            for value in raw:last=min(last,value);b.append(last)
            expected=sum(w*(b[j] if (j:=bisect_right(ts,x)-1)>=0 else 1) for x,w in zip(support,law))
            formula=b[-1]+sum(((b[j-1] if j else 1)-b[j])*cdf[j] for j in range(3))
            assert expected<=formula<=1;transfers+=1
    result=dict(exact_secant_and_scaling_cases=secants,exact_bad_cover_cases=bad,upper_CDF_transfer_cases_with_atoms=transfers,
                tests_are_not_independent_mathematical_review=True)
    with args.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
