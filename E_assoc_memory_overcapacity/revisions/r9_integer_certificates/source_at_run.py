"""Integer interval certificates for weighted chi-square sign probabilities.

Standard library only. Exact rational input coefficients, dyadic outward
rounding, isolated Gauss-Legendre roots/weights, analytic quadrature and
infinite-tail bounds. This does NOT certify floating matrix construction.
"""
from datetime import datetime,timezone
from fractions import Fraction as F
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

BITS=160
SCALE=1<<BITS


class I:
    __slots__=('lo','hi')
    def __init__(self,value=0):
        if isinstance(value,I):self.lo,self.hi=value.lo,value.hi;return
        if isinstance(value,float):raise TypeError('Use explicit exact rational coefficients')
        value=F(value)
        self.lo=value.numerator*SCALE//value.denominator
        self.hi=-((-value.numerator*SCALE)//value.denominator)
    @classmethod
    def raw(cls,lo,hi):
        assert lo<=hi
        out=object.__new__(cls);out.lo,out.hi=int(lo),int(hi);return out
    def __add__(self,other):
        other=I(other);return I.raw(self.lo+other.lo,self.hi+other.hi)
    __radd__=__add__
    def __neg__(self):return I.raw(-self.hi,-self.lo)
    def __sub__(self,other):return self+-I(other)
    def __rsub__(self,other):return I(other)+-self
    def __mul__(self,other):
        other=I(other)
        products=[a*b for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        return I.raw(min(products)//SCALE,-((-max(products))//SCALE))
    __rmul__=__mul__
    def __truediv__(self,other):
        other=I(other)
        if other.lo<=0<=other.hi:raise ZeroDivisionError('Interval denominator contains zero')
        corners=[(a*SCALE,b) for a in (self.lo,self.hi) for b in (other.lo,other.hi)]
        return I.raw(min(a//b for a,b in corners),max(-((-a)//b) for a,b in corners))
    def __rtruediv__(self,other):return I(other)/self
    def square(self):
        low=0 if self.lo<=0<=self.hi else min(self.lo*self.lo,self.hi*self.hi)
        high=max(self.lo*self.lo,self.hi*self.hi)
        return I.raw(low//SCALE,-((-high)//SCALE))
    def __pow__(self,exponent):
        if exponent<0:return I(1)/(self**(-exponent))
        result=I(1);base=self
        while exponent:
            if exponent&1:result=result*base
            exponent//=2
            if exponent:base=base.square()
        return result
    def sqrt(self):
        if self.lo<0:raise ValueError('Negative square-root interval')
        low=math.isqrt(self.lo*SCALE);high=math.isqrt(self.hi*SCALE)
        if high*high<self.hi*SCALE:high+=1
        return I.raw(low,high)
    def contains(self,value):
        value=F(value)
        return self.lo*value.denominator<=value.numerator*SCALE<=self.hi*value.denominator
    def upper_le(self,value):
        value=F(value);return self.hi*value.denominator<=value.numerator*SCALE
    def record(self):
        return dict(lower_integer=str(self.lo),upper_integer=str(self.hi),denominator_power2=BITS,
                    approximate=[self.lo/SCALE,self.hi/SCALE],width_approximate=(self.hi-self.lo)/SCALE)


def atan_bounds(q):
    total=F(0);index=0
    while True:
        total+=(-1)**index*q**(2*index+1)/(2*index+1)
        index+=1
        next_term=(-1)**index*q**(2*index+1)/(2*index+1)
        if abs(next_term)<F(1,1<<(BITS+16)):
            lower,upper=sorted((total,total+next_term))
            return I.raw(I(lower).lo,I(upper).hi)


def pi_interval():return 16*atan_bounds(F(1,5))-4*atan_bounds(F(1,239))


def legendre_sign(numerator):
    x=numerator;s=SCALE
    return 6435*x**8-12012*x**6*s**2+6930*x**4*s**4-1260*x*x*s**6+35*s**8


def gauss8():
    positive=[]
    for left,right in ((F(18,100),F(19,100)),(F(52,100),F(53,100)),
                       (F(79,100),F(80,100)),(F(96,100),F(97,100))):
        lo,hi=I(left).lo,I(right).hi
        assert legendre_sign(lo)*legendre_sign(hi)<0
        sign=legendre_sign(lo)>0
        while hi-lo>1:
            mid=(lo+hi)//2;value=legendre_sign(mid)
            if value==0:lo=hi=mid;break
            if (value>0)==sign:lo=mid
            else:hi=mid
        positive.append(I.raw(lo,hi))
    nodes=[-x for x in positive[::-1]]+positive
    assert all(nodes[j].hi<nodes[j+1].lo for j in range(7))
    weighted=[]
    for x in nodes:
        x2=x.square()
        derivative=x*(((51480*x2-72072)*x2+27720)*x2-2520)/128
        weight=2/((1-x2)*derivative.square())
        assert weight.lo>0
        weighted.append((x,weight))
    for degree in range(16):
        moment=sum(weight*x**degree for x,weight in weighted)
        assert moment.contains(F(2,degree+1) if degree%2==0 else 0)
    return weighted


def cmul(first,second):
    a,b=first;c,d=second
    return a*c-b*d,a*d+b*c


def cpow(value,exponent):
    result=(I(1),I(0))
    while exponent:
        if exponent&1:result=cmul(result,value)
        exponent//=2
        if exponent:value=cmul(value,value)
    return result


def integrand(u,coefficients,counts):
    result=(I(1),I(0))
    for a,count in zip(coefficients,counts):
        ua=u*a;denominator=1+4*ua.square()
        inverse=(1/denominator,2*ua/denominator)
        factor=cpow(inverse,count//2)
        if count%2:
            radius=denominator.sqrt();q=((radius+1)/2).sqrt()
            factor=cmul(factor,(q/radius,ua/(q*radius)))
        result=cmul(result,factor)
    return result[1]/u


def mgf(tau,coefficients,counts):
    result=I(1)
    for a,count in zip(coefficients,counts):
        base=1-2*I(tau)*a
        assert base.lo>0
        factor=base**(-(count//2))
        if count%2:factor=factor/base.sqrt()
        result=result*factor
    return result


def certify(values,counts,pi,nodes,target=F(1,10**8)):
    original=[F(x) for x in values]
    assert len(original)==len(counts) and all(c>0 for c in counts)
    variance=2*sum(c*a*a for a,c in zip(original,counts))
    scale=F(1)
    while scale*scale<variance:scale*=2
    while scale*scale>=4*variance:scale/=2
    exact=[a/scale for a in original];coefficients=[I(a) for a in exact]
    tau=1/(4*max(abs(a) for a in exact))
    moment=I(math.factorial(17))*I(tau)**(-17)*(mgf(tau,coefficients,counts)+mgf(-tau,coefficients,counts))
    derivative=moment/17
    top=[]
    for a,count in zip(exact,counts):
        if a:top.extend([abs(a)]*min(count,32))
    top=sorted(top,reverse=True)[:32];k=len(top)-len(top)%2
    assert k>=2
    a_star=top[k-1];cutoff=1
    def tail_bound(u):return (2/(pi*k))*(I(2*a_star*u)**(-(k//2)))
    while not tail_bound(cutoff).upper_le(target/8):cutoff*=2
    constant=F(math.factorial(8)**4,17*math.factorial(16)**3)
    panels=cutoff
    def finite_bound(n):return I(constant)*derivative*I(cutoff)**17/I(n)**16/pi
    while not finite_bound(panels).upper_le(target/8):panels*=2
    total=I(0);h=F(cutoff,panels)
    for panel in range(panels):
        subtotal=I(0);left=F(cutoff*panel,panels)
        for node,weight in nodes:
            u=I(left)+I(h)*(1+node)/2
            subtotal=subtotal+weight*integrand(u,coefficients,counts)
        total=total+I(h)*subtotal/2
    arithmetic=I(F(1,2))+total/pi
    finite=finite_bound(panels);tail=tail_bound(cutoff)
    error=finite.hi+tail.hi
    enclosure=arithmetic+I.raw(-error,error)
    assert enclosure.hi-enclosure.lo<=I(target).lo
    return dict(coefficients=[str(a) for a in original],counts=counts,exact_scale=str(scale),
                tau=str(tau),cutoff=cutoff,panels=panels,quadrature_nodes=8*panels,
                probability=enclosure.record(),arithmetic_only=arithmetic.record(),
                finite_quadrature_error=finite.record(),infinite_tail_error=tail.record(),
                derivative_bound=derivative.record(),target_width=str(target)),enclosure


def exact_even_case(positive,negative,ratio):
    p,q=positive//2,negative//2;z=F(ratio)/(1+F(ratio));n=p+q-1
    return sum(F(math.comb(n,j))*z**j*(1-z)**(n-j) for j in range(p))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    pi=pi_interval();nodes=gauss8()
    setup=dict(created_utc=datetime.now(timezone.utc).isoformat(),bits=BITS,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Exact rational input coefficients only; matrix construction and eigenvalues excluded',
        pi=pi.record(),nodes=[dict(root=x.record(),weight=w.record()) for x,w in nodes],
        checked_monomial_degrees=list(range(16)))
    (args.out/'setup.json').write_text(json.dumps(setup,indent=2)+'\n')
    records=[]
    for positive,negative,ratio in ((8,8,F(1)),(8,12,F(7,10)),(10,6,F(7,4)),(20,20,F(2))):
        record,enclosure=certify([F(1),-ratio],[positive,negative],pi,nodes)
        reference=exact_even_case(positive,negative,ratio)
        assert enclosure.contains(reference)
        record.update(exact_reference=str(reference),contains_exact_reference=True)
        records.append(record)
        (args.out/f'case_{positive}_{negative}.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'{positive}/{negative}, ratio {ratio}: {enclosure.lo/SCALE:.12f} .. {enclosure.hi/SCALE:.12f}; {record["quadrature_nodes"]} nodes',flush=True)
    result=dict(scope=setup['scope'],cases=records,wall_seconds=time.monotonic()-start,
                all_exact_references_contained=True,
                max_enclosure_width=max(r['probability']['width_approximate'] for r in records),
                caveat='Implementation and mathematical derivation still require independent review; no certification of the R8 optimizer matrices')
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
