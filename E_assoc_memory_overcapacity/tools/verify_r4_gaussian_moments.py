"""Deterministic Gaussian quadrature for the finite moments used in R4."""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


def grid(count,order):
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    nodes=nodes*math.sqrt(2);weights=weights/math.sqrt(math.pi)
    for indices in itertools.product(range(order),repeat=count):
        index=np.array(indices)
        yield nodes[index],float(np.prod(weights[index]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    records=[]
    for d,n in ((1,2),(2,2),(2,3)):
        a=np.arange(1,d*n+1,dtype=float).reshape(d,n)/(d*n)+np.eye(d,n)
        s=a@a.T/n
        p=.7*(np.eye(d)+(np.eye(d)-.7*s)+(np.eye(d)-.7*s)@(np.eye(d)-.7*s))
        m=s@p@p;h=s@s@p@p
        mean_den=mean_num=0.
        for values,weight in grid(d*n,3):
            b=values.reshape(d,n);u=b@a.T@p/math.sqrt(n)
            mean_den+=weight*float(np.sum(u*u))
            mean_num+=weight*float(np.trace(u@s@u.T@(b@b.T/n)))
        den_formula=d*float(np.trace(m));num_formula=d*(n+d+1)/n*float(np.trace(h))
        tau3=tau_delta2=0.
        for values,weight in grid(d*n,4):
            q=values.reshape(d,n);ss=q@q.T/n
            tau3+=weight*float(np.trace(ss@ss@ss))/d
            tau_delta2+=weight*float(np.sum((ss-np.eye(d))**2))/d
        third_formula=1+3*(d+1)/n+(d*d+3*d+4)/(n*n)
        errors=[abs(mean_den-den_formula),abs(mean_num-num_formula),
                abs(tau3-third_formula),abs(tau_delta2-(d+1)/n)]
        if max(errors)>1e-9:
            raise RuntimeError('Finite Gaussian moment identity failed')
        records.append(dict(d=d,n=n,errors=errors,denominator=mean_den,numerator=mean_num,
                            third_moment=tau3,centered_second=tau_delta2))
    result=dict(scope='Deterministic quadrature of Gaussian polynomials; no training dataset sampling',
                cases=records,max_error=max(e for r in records for e in r['errors']))
    with args.out.open('x') as handle:
        json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
