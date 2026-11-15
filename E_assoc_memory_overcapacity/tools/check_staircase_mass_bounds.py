"""Exact finite-law references for nonnegative staircase-difference bounds."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    laws=[tuple(F(x,4) for x in v) for v in product(range(5),repeat=3) if sum(v)==4]
    steps=[tuple(F(x,2) for x in v) for v in product(range(3),repeat=3) if v[0]>=v[1]>=v[2]]
    checks=0
    for law in laws:
        cdf=[sum(law[:j+1]) for j in range(3)]
        lower=[max(F(0),x-F(1,8)) for x in cdf];upper=[min(F(1),x+F(1,8)) for x in cdf]
        for old in steps:
            for new in steps:
                difference=[a-b for a,b in zip(new,old)]
                if min(difference)<0:continue
                exact=sum(d*w for d,w in zip(difference,law))
                lo=sum(d*max(F(0),lower[j]-(upper[j-1] if j else 0)) for j,d in enumerate(difference))
                hi=sum(d*min(F(1),upper[j]-(lower[j-1] if j else 0)) for j,d in enumerate(difference))
                assert 0<=lo<=exact<=hi;checks+=1
    with args.out.open('x') as f:json.dump(dict(exact_finite_law_checks=checks,atoms_included=True,all_bounds_contain_exact_expectation=True),f,indent=2);f.write('\n')
    print('PASS',checks,'exact finite-law staircase differences')


if __name__=='__main__':main()
