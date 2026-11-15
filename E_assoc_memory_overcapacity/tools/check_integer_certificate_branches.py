"""Exact tests of the dyadic primitives and odd-degree CF branches."""
from fractions import Fraction as F
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    path=Path(__file__).with_name('certify_weighted_chisquare.py')
    spec=importlib.util.spec_from_file_location('integer_certificate',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    I=module.I
    intervals=[(F(-3,2),F(-1,3)),(F(-1,10),F(1,10)),(F(0),F(0)),(F(1,7),F(2,3)),(F(100),F(101))]
    checks=0
    for low,high in intervals:
        first=I.raw(I(low).lo,I(high).hi)
        for left,right in intervals:
            second=I.raw(I(left).lo,I(right).hi)
            for a in (low,(low+high)/2,high):
                for b in (left,(left+right)/2,right):
                    assert (first+second).contains(a+b)
                    assert (first-second).contains(a-b)
                    assert (first*second).contains(a*b);checks+=3
                    if not left<=0<=right:
                        assert (first/second).contains(a/b);checks+=1
    pi=module.pi_interval();nodes=module.gauss8()
    cases=[]
    # Integer/half-integer beta survival at z=3/4 has a rational value.
    p,q=4,4
    beta=F(1)
    for j in range(p):beta*=F(2*q+1+2*j,2)
    beta=F(6)/beta
    reference=sum(F((-1)**j*__import__('math').comb(p-1,j))*F(1,2**(2*q+2*j+1))/F(2*q+2*j+1,2) for j in range(p))/beta
    for values,counts,target in (([F(1),F(-3)],[8,9],reference),
                                  ([F(1),F(-1,3)],[9,8],1-reference),
                                  ([F(1),F(-1)],[7,7],F(1,2))):
        record,enclosure=module.certify(values,counts,pi,nodes)
        assert enclosure.contains(target)
        record.update(exact_reference=str(target),contains_exact_reference=True)
        cases.append(record)
    # Beta(7/2,9/2) survival at 1/2 is 1/2 - 16/(35*pi).
    record,enclosure=module.certify([F(1),F(-1)],[7,9],pi,nodes)
    target=I(F(1,2))-16/(35*pi)
    assert enclosure.lo<=target.lo and enclosure.hi>=target.hi
    record.update(reference_interval=target.record(),contains_reference_interval=True)
    cases.append(record)
    args.out.mkdir(parents=True,exist_ok=False)
    result=dict(primitive_corner_checks=checks,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cases=cases,all_checks_passed=True,
                scope='Exact rational interval checks; no floating numerical reference used')
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS',checks,'exact primitive checks and four odd-degree certificates')


if __name__=='__main__':main()
