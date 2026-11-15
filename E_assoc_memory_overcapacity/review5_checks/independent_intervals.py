"""Fresh Decimal directed-rounding certification; never imports/executes w6 code.

Differences: decimal endpoint arithmetic; correctly rounded Decimal exp/sqrt;
Adam expressed with explicit normalized history weights; generic dual numbers;
sqrt Taylor coefficients obtained via binomial expansion. Root/source certificate
JSON is used only after new enclosures are computed, for comparisons.
"""
from decimal import Decimal, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import os
import resource
import time

START=time.monotonic()
DOWN=Context(prec=55,rounding=ROUND_FLOOR)
UP=Context(prec=55,rounding=ROUND_CEILING)
NEAR=Context(prec=55,rounding=ROUND_HALF_EVEN)

class I:
    __slots__=('lo','hi')
    def __init__(self,x=0,hi=None):
        if isinstance(x,I): self.lo,self.hi=x.lo,x.hi; return
        if hi is not None: self.lo,self.hi=Decimal(x),Decimal(hi); return
        q=Q(x)
        self.lo=DOWN.divide(Decimal(q.numerator),Decimal(q.denominator))
        self.hi=UP.divide(Decimal(q.numerator),Decimal(q.denominator))
    def __add__(self,b):
        b=I(b); return I(DOWN.add(self.lo,b.lo),UP.add(self.hi,b.hi))
    __radd__=__add__
    def __neg__(self): return I(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self,b): return self+-I(b)
    def __rsub__(self,b): return I(b)+-self
    def __mul__(self,b):
        b=I(b)
        return I(min(DOWN.multiply(x,y) for x in [self.lo,self.hi] for y in [b.lo,b.hi]),
                 max(UP.multiply(x,y) for x in [self.lo,self.hi] for y in [b.lo,b.hi]))
    __rmul__=__mul__
    def __truediv__(self,b):
        b=I(b); assert b.lo>0
        return self*I(DOWN.divide(Decimal(1),b.hi),UP.divide(Decimal(1),b.lo))
    def square(self):
        lo=Decimal(0) if self.lo<=0<=self.hi else min(DOWN.multiply(x,x) for x in [self.lo,self.hi])
        return I(lo,max(UP.multiply(x,x) for x in [self.lo,self.hi]))
    def sqrt(self):
        assert self.lo>=0,(self.lo,self.hi)
        # Decimal sqrt/exp are correctly rounded to nearest; one neighbor outward
        # encloses the exact function value regardless of the half-even direction.
        return I(max(Decimal(0),NEAR.next_minus(NEAR.sqrt(self.lo))),NEAR.next_plus(NEAR.sqrt(self.hi)))
    def exp(self):
        return I(NEAR.next_minus(NEAR.exp(self.lo)),NEAR.next_plus(NEAR.exp(self.hi)))
    def absmax(self): return max(self.lo.copy_abs(),self.hi.copy_abs())
    def show(self): return [str(self.lo),str(self.hi)]

ZERO=I(0); ONE=I(1)
def hull(a,b): return I(min(a.lo,b.lo),max(a.hi,b.hi))
def atan_bounds(inv):
    # Entire computation exact rational; first omitted term bounds the remainder.
    s=Q(0)
    for k in range(100): s+=(-1)**k*Q(1,(2*k+1)*inv**(2*k+1))
    next_term=Q(1,201*inv**201)
    return hull(I(s),I(s+next_term))
PI=16*atan_bounds(5)-4*atan_bounds(239)
PREF=(I(2)/PI).sqrt()
TAIL=I(Q(1,2**60))

def rho(z): return PREF*(-z.square()/2).exp()

# Dual arithmetic differentiates in a. All history values are intervals.
class Dual:
    def __init__(self,v=0,d=0): self.v,self.d=I(v),I(d)
    def __add__(self,b):
        if not isinstance(b,Dual): b=Dual(b)
        return Dual(self.v+b.v,self.d+b.d)
    __radd__=__add__
    def __neg__(self): return Dual(-self.v,-self.d)
    def __sub__(self,b): return self+-b if isinstance(b,Dual) else self+Dual(-I(b))
    def __mul__(self,b):
        if not isinstance(b,Dual): b=Dual(b)
        return Dual(self.v*b.v,self.d*b.v+self.v*b.d)
    __rmul__=__mul__
    def __truediv__(self,b):
        if not isinstance(b,Dual): b=Dual(b)
        quotient=self.v/b.v
        return Dual(quotient,(self.d-quotient*b.d)/b.v)
    def square(self): return Dual(self.v.square(),2*self.v*self.d)
    def sqrt(self):
        v=self.v.sqrt(); return Dual(v,self.d/(2*v))

class Poly:
    def __init__(self,c): self.c=[I(x) for x in c]
    def __add__(self,b):
        if not isinstance(b,Poly): b=constant(b,len(self.c)-1)
        return Poly([x+y for x,y in zip(self.c,b.c)])
    __radd__=__add__
    def __neg__(self): return Poly([-x for x in self.c])
    def __sub__(self,b): return self+(-b if isinstance(b,Poly) else -I(b))
    def __mul__(self,b):
        if not isinstance(b,Poly): return Poly([x*I(b) for x in self.c])
        n=len(self.c)
        return Poly([sum((self.c[j]*b.c[k-j] for j in range(k+1)),ZERO) for k in range(n)])
    __rmul__=__mul__
    def inv(self):
        out=[ONE/self.c[0]]
        for k in range(1,len(self.c)):
            out.append(-sum((self.c[j]*out[k-j] for j in range(1,k+1)),ZERO)/self.c[0])
        return Poly(out)
    def __truediv__(self,b):
        if not isinstance(b,Poly): return Poly([x/I(b) for x in self.c])
        return self*b.inv()
    def square(self):
        result=self*self; result.c[0]=self.c[0].square(); return result
    def sqrt(self):
        # sqrt(c0)*sum binomial(1/2,k)*(sum_{j>=1} cj/c0 x^j)^k.
        n=len(self.c)-1
        u=Poly([ZERO]+[x/self.c[0] for x in self.c[1:]])
        power=constant(1,n); result=constant(1,n); coeff=Q(1)
        for k in range(1,n+1):
            coeff*=Q(3-2*k,2*k)
            power=power*u
            result=result+power*I(coeff)
        return result*self.c[0].sqrt()

def constant(x,n): return Poly([I(x)]+[ZERO]*n)

WEIGHTS={}
for t in range(1,5):
    for beta in [Q(9,10),Q(99,100)]:
        WEIGHTS[t,beta]=[I(beta**(t-1-k)/sum(beta**j for j in range(t))) for k in range(t)]

def history_map(a,e,z,mode):
    # Direct normalized histories, not the recursive moment code under review.
    if mode=='dual':
        p=Dual(); aa=Dual(a,1); zz=Dual(z); ee=Dual(e)
    else:
        n=len(z.c)-1
        p=constant(0,n); aa=I(a); zz=z; ee=constant(e,n)
    history=[]
    for t in range(1,5):
        history.append(p-1)
        mean=sum((h*w for h,w in zip(history,WEIGHTS[t,Q(9,10)])),p*0)
        rms=sum((h.square()*w for h,w in zip(history,WEIGHTS[t,Q(99,100)])),p*0).sqrt()
        p=p-mean/(rms*zz+ee)*aa
    return p

def derivative_integrals(a,e,nz=256):
    sums=[ZERO,ZERO,ZERO]
    for j in range(nz):
        l,r=I(Q(12*j,nz)),I(Q(12*(j+1),nz))
        z=I(l.lo,r.hi)
        p=history_map(a,e,z,'dual')
        fns=[2*z.square()*(p.v-1)*p.d,z.square()*p.d,2*z.square()*p.v*p.d]
        # density is decreasing on the nonnegative half-axis
        density=I(rho(r).lo,rho(l).hi)
        for k,x in enumerate(fns): sums[k]+=x*density*I(Q(12,nz))
    return [x+I(-TAIL.hi,TAIL.hi) for x in sums]

def poly_integrands(a,e,z0,n):
    z=Poly([z0,ONE]+[ZERO]*(n-1))
    p=history_map(a,e,z,'poly')
    # exp(-z0 x-x^2/2), via the differential equation for exp polynomial.
    exponent=-(z*z)/2
    ex=[exponent.c[0].exp()]
    for k in range(1,n+1):
        ex.append(sum((I(j)*exponent.c[j]*ex[k-j] for j in range(1,k+1)),ZERO)/I(k))
    weighted=z.square()*Poly(ex)*PREF
    return [(p-1).square()*weighted,p*weighted,p.square()*weighted]

def taylor_integrals(a,e,cells=192,order=10):
    totals=[ZERO]*3; errors=[ZERO]*3
    for j in range(cells):
        left=Q(12*j*j,cells*cells); right=Q(12*(j+1)**2,cells*cells)
        center=I((left+right)/2); radius=Q(right-left,2)
        domain=hull(I(left),I(right))
        mid=poly_integrands(a,e,center,order-1)
        rem=poly_integrands(a,e,domain,order)
        for k in range(3):
            main=sum((mid[k].c[q]*I(2*radius**(q+1)/Q(q+1)) for q in range(0,order,2)),ZERO)
            error=I(rem[k].c[order].absmax())*I(2*radius**(order+1)/Q(order+1))
            totals[k]+=main+I(-error.hi,error.hi)
            errors[k]+=error
    return [x+(I(-TAIL.hi,TAIL.hi) if k==1 else I(0,TAIL.hi)) for k,x in enumerate(totals)],errors

def source_interval(x):
    return hull(I(Q(int(x['lo']),2**x['power'])),I(Q(int(x['hi']),2**x['power'])))

def main():
    # Exact rational tail checks, including all three derivative integrals.
    tail_coefficients=[Q(17),Q(2),Q(1,3),Q(1024)*Q(7,6),Q(512),Q(2048,12)]
    assert all(c*Q(1,2**72)<Q(1,2**60) for c in tail_coefficients)
    weights=[sum(Q(a.lo)*Q(a.lo)/Q(b.hi) for a,b in zip(WEIGHTS[t,Q(9,10)],WEIGHTS[t,Q(99,100)])) for t in range(1,5)]
    # Randomized rational endpoints test the Decimal implementation independently.
    import random
    rng=random.Random(8295)
    for _ in range(300):
        xx=sorted(Q(rng.randrange(-100,101),rng.randrange(1,20)) for _ in range(2))
        yy=sorted(Q(rng.randrange(1,101),rng.randrange(1,20)) for _ in range(2))
        x=hull(I(xx[0]),I(xx[1])); y=hull(I(yy[0]),I(yy[1]))
        for actual in [u*v for u in xx for v in yy]:
            v=x*y; assert Q(v.lo)<=actual<=Q(v.hi)
        for actual in [u/v for u in xx for v in yy]:
            v=x/y; assert Q(v.lo)<=actual<=Q(v.hi)
        v=y.sqrt(); assert Q(v.lo)**2<=yy[0] and Q(v.hi)**2>=yy[1]
    out=[]
    source=json.loads(Path('w6_checks/certificate.json').read_text())
    for ei,estr in enumerate(['.1','1']):
        e=I(estr); slabs=[]
        for j in range(32):
            a=hull(I(Q(3*j,640)),I(Q(3*(j+1),640)))
            ds=derivative_integrals(a,e)
            assert ds[0].lo>Decimal(-7) and ds[0].hi<Decimal(-2),(estr,j,[x.show() for x in ds])
            assert all(x.absmax()<4 for x in ds[1:])
            slabs.append(dict(a=a.show(),derivatives=[x.show() for x in ds]))
        print('INDEPENDENT MONOTONICITY PASS',estr,flush=True)
        endpoints=['.04819026078294','.04819026078295'] if estr=='.1' else ['.08287875576148','.08287875576149']
        vals=[]
        for k,x in enumerate(endpoints):
            v,err=taylor_integrals(I(x),e)
            residual=v[0]-I(Q(3,4))
            assert residual.lo>0 if k==0 else residual.hi<0
            vals.append(dict(a=x,F_B_V=[q.show() for q in v],residual=residual.show(),Taylor_error=[q.show() for q in err]))
            print('INDEPENDENT ENDPOINT',estr,x,residual.show(),flush=True)
        # All 24 branch decisions checked using new root endpoints and exact rationals.
        src=source['results'][ei]
        L,U=map(Q,endpoints)
        cache=[]
        initial=None
        import numpy as np
        for j,x in enumerate(np.linspace(0,.6/4,33)):
            q=Q.from_float(float(x)); cache.append(q)
            if q>U:
                initial=(cache[-2],q); break
            assert q<L
        lo,hi=initial
        for k in range(24):
            # Reproduce midpoint rounding only; branch decisions are rational comparisons.
            mid=Q.from_float((float(lo)+float(hi))/2)
            assert mid<L or mid>U
            cache.append(mid)
            recorded=source_interval(src['search24']['decisions'][k]['a'])
            assert Q(recorded.lo)<=mid<=Q(recorded.hi)
            if mid<L: lo=mid
            else: hi=mid
        width=hi-lo; assert width<Q('2.8e-10')
        fv=[]
        for q in [lo,hi]:
            v,err=taylor_integrals(I(q),e)
            fv.append(v[0]-I(Q(3,4)))
        assert fv[0].lo>0 and fv[1].hi<0
        ab=[fv[0],-fv[1]]
        chosen=0 if ab[0].hi<ab[1].lo else 1
        assert ab[chosen].hi<ab[1-chosen].lo
        gap=ab[1-chosen]-ab[chosen]
        other_distance=min((lo-q if q<lo else q-hi) for q in cache if q!=lo and q!=hi)
        all_gap=min(Q(gap.lo),2*other_distance)
        sign_gap=2*min((L-q if q<L else q-U) for q in cache)
        assert all_gap>Q('5.37e-10' if estr=='.1' else '3.45e-10')
        assert sign_gap>Q('1.63e-10' if estr=='.1' else '1.53e-10')
        assert chosen==src['search24']['chosen_endpoint']
        chosen_q=[lo,hi][chosen]
        recorded=source_interval(src['search24']['chosen_a'])
        assert Q(recorded.lo)<=chosen_q<=Q(recorded.hi)
        print('INDEPENDENT SEARCH PASS',estr,str(chosen_q),flush=True)
        out.append(dict(e=estr,slabs=slabs,root_endpoint_checks=vals,final_bracket=[str(lo),str(hi)],
                        final_residuals=[x.show() for x in fv],chosen=str(chosen_q),sign_gap_lower=str(sign_gap),
                        cache_gap_lower=str(all_gap),width=str(width),MSR_error_upper=str(400*width)))
    cg=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('::',1)[1].strip().lstrip('/')
    resources={k:(cg/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.peak','memory.events']}
    resources.update(seconds=time.monotonic()-START,executable=os.sys.executable,
                     threads=next(int(l.split()[1]) for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('Threads:')),
                     maxrss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    assert resources['memory.max']=='2147483648' and resources['memory.swap.max']=='0' and resources['threads']<=2
    with open('review5_checks/interval_results.json','x') as f:
        json.dump(dict(status='PASS',arithmetic='55 significant decimal digits, directed rounding',
                       pi=PI.show(),results=out,resources=resources,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
    print('DONE',resources,flush=True)

if __name__=='__main__': main()
