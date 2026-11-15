"""Additional independent exact endpoint, sign-series, and manuscript checks."""
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path
from mpmath import iv
from audit import Exact, decode, dyadic_range, stats, contained
from environment import resources

iv.dps=55


def rat(x):
    x=F(x)
    return iv.mpf(x.numerator)/iv.mpf(x.denominator)


def span(lo,hi):
    return iv.mpf([rat(lo).a,rat(hi).b])


def fraction_tuple(t):
    sign,mantissa,exponent,_=t
    return (-1 if sign else 1)*F(mantissa)*F(2)**exponent


def endpoints(x):
    return tuple(fraction_tuple(t) for t in x._mpi_)


def record(x):
    return dict(exact_dyadic=[list(t) for t in x._mpi_],display_only=list(map(float,endpoints(x))))


def sign_values(lo,hi):
    a=span(lo,hi)
    density=iv.sqrt(2/iv.pi)*iv.exp(-a*a/2)
    # Taylor polynomial through k=24 with a symmetric, absolute next-term
    # bound: 0<=a<=.21 guarantees alternating decreasing scalar terms.
    series=iv.mpf(0)
    for k in range(25):
        series+=(-1)**k*a**(2*k+1)/rat(2**k*math.factorial(k)*(2*k+1))
    upper=F(hi)
    error=rat(upper**51/F(2**25*math.factorial(25)*51))
    mass=iv.sqrt(2/iv.pi)*(series+iv.mpf([-error.b,error.b]))
    B=2*a*density
    V=4*a*a*(1-mass)
    q=1-2*B+2*V
    return {'B':B,'V':V,'q':q,'H':B*B-rat(F(4,25))*q,'c0':(1-B)/iv.sqrt(q),'c1':B/iv.sqrt(q)}


def main():
    pts=json.loads(Path('y2_checks/points.json').read_text())
    hist=json.loads(Path('y2_checks/history.json').read_text())
    validated=[]
    for row in pts['points']+hist['models']:
        st=stats(dyadic_range(row['a']),list(map(decode,row['integrals'])))
        B,q=st['B'],st['q']; cc=decode(row['c0'])
        assert q.lo>0 and B.hi<1 and cc.lo>0
        # Compare radicals by exact rational squaring, with all signs checked.
        assert cc.lo**2*q.hi <= (1-B.hi)**2
        assert cc.hi**2*q.lo >= (1-B.lo)**2
        validated.append(dict(beta=row.get('beta',['9/10','99/100']),e=row['e'],c0=cc.floats()))
    sign=[]
    for i,endpoint in enumerate(pts['sign']['endpoints']):
        ss=sign_values(endpoint['a'],endpoint['a'])
        assert endpoints(ss['B'])[0]>0
        assert endpoints(ss['H'])[1]<0 if i==0 else endpoints(ss['H'])[0]>0
        for name,value in ss.items():
            lo,hi=endpoints(value); old=decode(endpoint['bounds'][name])
            assert max(lo,old.lo)<=min(hi,old.hi)
        sign.append(dict(a=endpoint['a'],bounds={k:record(v) for k,v in ss.items()}))
    sign_bracket=sign_values(*pts['sign']['a'])
    sc=sign_bracket['c0']
    assert endpoints(sc)[0] > F('.84906207') and endpoints(sc)[1] < F('.84906221')
    assert decode(pts['points'][0]['c0']).lo-endpoints(sc)[1]>F('.01846408')
    c2=(F(9,19)**2/F(99,199))+(F(10,19)**2/F(100,199))
    assert c2==F(3980,3971)<4
    assert all(F(c,2**72)<F(1,2**60) for c in (3,9,30,180,1080))
    signN=F(7,5)*F(29,50)-16*F(21,100)**2
    assert signN==F(133,1250)>0
    actual=decode(pts['points'][2]['c0'])
    stateless,momonly,rmsonly=[decode(r['c0']) for r in hist['models']]
    assert rmsonly.lo>stateless.hi>actual.hi and stateless.lo>actual.hi and actual.lo>momonly.hi
    manifest=json.loads(Path('review12_checks/input_sha256.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    out=dict(status='PASS',six_c0_enclosures_verified_by_exact_squaring=validated,sign_endpoints=sign,sign_c0=record(sc),cauchy_squared=str(c2),sign_N_lower=str(signN),history_order=True,all_reviewed_originals_unchanged=True,resources=resources())
    with open('review12_checks/extra_checks.json','x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
