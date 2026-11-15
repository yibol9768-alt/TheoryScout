"""Exact rational checks of arbitrary-anchor lower/upper event inclusions."""
from fractions import Fraction as F
from pathlib import Path
import argparse
import hashlib
import json
import random


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    seed=202611071957;rng=random.Random(seed);cases=[];upper_checked=0
    for index in range(512):
        n=4;m=n-1
        b=[[F(rng.randrange(-8,9),8) for _ in range(n)] for _ in range(n)]
        u=[F(rng.randrange(-8,9),4) for _ in range(n-1)]
        s=sum(z*z for z in u);w=[2*z/(1+s) for z in u]+[(1-s)/(1+s)]
        assert sum(z*z for z in w)==1
        a=F(rng.randrange(-40,41),4);alpha=F(rng.randrange(1,17),8);t=F(rng.randrange(1,33),8)
        anchor=t/(2*alpha)
        bw=[sum(b[i][j]*w[j] for j in range(n)) for i in range(n)]
        D=n*sum(w[i]*bw[i] for i in range(n))
        W=F(n*n,m)*sum(z*z for z in bw);V=W-D*D/m
        assert V>=0
        L=(1+2*alpha*a/m)*D-alpha*W-alpha*a*a/m-t*anchor/2
        gap=D-L
        assert gap==alpha*(V+(D-a)**2/m+anchor**2)
        assert gap>=0 and gap*gap>=t*t*V
        # Cross-check division-free matrix/constant representation.
        A=4*alpha*m+8*alpha*alpha*a;Nfactor=4*alpha*alpha*n
        matrix=[[A*(b[i][j]+b[j][i])/2-Nfactor*sum(b[k][i]*b[k][j] for k in range(n))
                 for j in range(n)] for i in range(n)]
        constant=-4*alpha*alpha*a*a-m*t*t
        quadratic=n*sum(w[i]*matrix[i][j]*w[j] for i in range(n) for j in range(n))+constant
        assert quadratic==4*alpha*m*L
        # Arbitrary rational square-root endpoints and arbitrary center/radius.
        low,high,R=F(1,100),F(100),F(100)
        if low*low<=V<=high*high and abs(D-a)<=R:
            beta=t/(low+high)
            U=(1+2*beta*a/m)*D-beta*W+beta*(R*R-a*a)/m-beta*low*high
            assert U>=D or (D-U)**2<=t*t*V
            upper_checked+=1
        cases.append(dict(index=index,D=str(D),V=str(V),center=str(a),sqrt_energy_anchor=str(anchor),
                          lower_score=str(L),scaled_score=str(quadratic)))
    result=dict(seed=seed,exact_rational_cases=len(cases),lower_inclusions_checked=len(cases),
                division_free_matrix_checks=len(cases),upper_inclusions_checked=upper_checked,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                statistical_inference=False,cases=cases)
    with args.out.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
