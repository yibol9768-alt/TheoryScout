"""mpmath.iv re-integration; sparse multivariate power series, not audited jets.

Two moment updates are propagated after dividing residuals by positive z.
The formal variables are dz, da, de. Only first parameter derivatives are
retained; dz terms are retained through the requested Taylor order.
Inverse, square root, exponential use finite nilpotent binomial series.
"""
import json
import math
import time
from fractions import Fraction as F
from pathlib import Path
from mpmath import iv
from environment import resources

iv.dps = 45
ORDER = 6
CELLS = 96


def number(x):
    x = F(x)
    return iv.mpf(x.numerator)/iv.mpf(x.denominator)


def interval(a,b):
    return iv.mpf([number(a).a,number(b).b])


class Poly:
    def __init__(self, terms=None):
        self.t = terms or {}
    @classmethod
    def constant(cls,x):
        return cls({(0,0): x})
    def __add__(self, other):
        if not isinstance(other, Poly): other=Poly.constant(number(other))
        out=self.t.copy()
        for k,v in other.t.items(): out[k]=out.get(k,iv.mpf(0))+v
        return Poly(out)
    __radd__=__add__
    def __neg__(self):
        return Poly({k:-v for k,v in self.t.items()})
    def __sub__(self,other): return self+-as_poly(other)
    def __rsub__(self,other): return as_poly(other)+-self
    def __mul__(self,other):
        other=as_poly(other)
        out={}
        for (j,p),u in self.t.items():
            for (k,q),v in other.t.items():
                if j+k > ORDER or (p and q): continue
                key=(j+k,p or q)
                out[key]=out.get(key,iv.mpf(0))+u*v
        return Poly(out)
    __rmul__=__mul__
    def analytic(self,kind):
        base=self.t.get((0,0),iv.mpf(0))
        rest=Poly({k:v for k,v in self.t.items() if k!=(0,0)})
        power=Poly.constant(iv.mpf(1))
        if kind=='exp':
            out=Poly.constant(iv.exp(base)); coeff=iv.exp(base)
            for k in range(1,ORDER+2):
                power=power*rest
                coeff=coeff/k
                out=out+power*Poly.constant(coeff)
            return out
        assert base.a > 0
        rest=rest*Poly.constant(1/base)
        out=Poly.constant(iv.sqrt(base) if kind=='sqrt' else 1/base)
        coeff=iv.sqrt(base) if kind=='sqrt' else 1/base
        exponent=F(1,2) if kind=='sqrt' else F(-1)
        for k in range(1,ORDER+2):
            power=power*rest
            coeff=coeff*number((exponent-(k-1))/k)
            out=out+power*Poly.constant(coeff)
        return out
    def __truediv__(self,other):
        return self*as_poly(other).analytic('inverse')
    def __rtruediv__(self,other):
        return as_poly(other)*self.analytic('inverse')


def as_poly(x):
    return x if isinstance(x,Poly) else Poly.constant(number(x))


def integrands(arange,erange,zrange):
    a=Poly({(0,0):arange,(0,1):iv.mpf(1)})
    e=Poly({(0,0):erange,(0,2):iv.mpf(1)})
    z=Poly({(0,0):zrange,(1,0):iv.mpf(1)})
    b1,b2=F(9,10),F(99,100)
    coefficient=Poly(); m=Poly(); v=Poly()
    for t in (1,2):
        g=a*coefficient-1
        m=b1*m+(1-b1)*g
        v=b2*v+(1-b2)*(g*g)
        mean=m/(1-b1**t)
        rms=(v/(1-b2**t)).analytic('sqrt')
        coefficient=coefficient-mean/(z*rms+e)
    w=z*coefficient
    density=((-F(1,2))*(z*z)).analytic('exp')*Poly.constant(iv.sqrt(2/iv.pi))
    aw=z*w*density
    qw=w*w*density
    # Extract da,de from the full integrand, including density (whose
    # parameter coefficients vanish). No separate derivative formula is used.
    return [aw,qw]


def dyadic(v):
    sign,mantissa,exponent,_=v
    return (-1 if sign else 1)*F(mantissa)*(F(2)**exponent)


def exact_bounds(x):
    return tuple(dyadic(v) for v in x._mpi_)


def encode(x):
    lo,hi=exact_bounds(x)
    return {'exact_dyadic_endpoints':[list(t) for t in x._mpi_], 'display_only':[float(lo),float(hi)]}


def integrate(a_bounds,e_bounds):
    ar=interval(*a_bounds); er=interval(*e_bounds)
    totals=[iv.mpf(0) for _ in range(6)]
    remainders=[iv.mpf(0) for _ in range(6)]
    for i in range(CELLS):
        left=F(12*i*i,CELLS*CELLS); right=F(12*(i+1)**2,CELLS*CELLS)
        center=(left+right)/2; radius=(right-left)/2
        mid=integrands(ar,er,number(center))
        full=integrands(ar,er,interval(left,right))
        for col,(j,p) in enumerate(((0,0),(1,0),(0,1),(1,1),(0,2),(1,2))):
            value=iv.mpf(0)
            for k in range(0,ORDER,2):
                value+=mid[j].t.get((k,p),iv.mpf(0))*number(2*radius**(k+1)/F(k+1))
            bound=abs(full[j].t.get((ORDER,p),iv.mpf(0))).b
            error=bound*number(2*radius**(ORDER+1)/F(ORDER+1))
            totals[col]+=value+iv.mpf([-error.b,error.b])
            remainders[col]+=error
    tail=number(F(1,2**60))
    totals=[x+iv.mpf([-tail.b,tail.b]) for x in totals]
    A,Q,Aa,Qa,Ae,Qe=totals
    B=ar*A; V=ar**2*Q
    N=(A+ar*Aa)*(1-B+2*V)-B*(2*ar*Q+ar**2*Qa)
    J=2*Ae*Q-A*Qe+ar*(Ae*Qa-Aa*Qe)
    q=1-2*B+2*V
    return totals,remainders,{'N':N,'J':J,'B':B,'H5':B**2-q/4}


def main():
    started=time.monotonic()
    reg=json.loads(Path('y2_checks/region.json').read_text())
    rows=reg['leaves']
    selected={min(range(len(rows)),key=lambda i:int(rows[i]['bounds'][key]['lo'])) for key in ('N','J')}
    selected.add(max(range(len(rows)),key=lambda i:int(rows[i]['bounds']['J']['hi'])))
    for a,e in ((F(0),F(1,10)),(F(0),F(1)),(F(1,2),F(1,10)),(F(1,2),F(1)),(F(1,4),F(1,2)),(F(2,5),F(1,5))):
        selected.add(next(i for i,r in enumerate(rows) if F(r['a'][0])<=a<=F(r['a'][1]) and F(r['e'][0])<=e<=F(r['e'][1])))
    results=[]
    def check_box(index, a_bounds, e_bounds, depth=0):
        row=rows[index]
        vals,errs,st=integrate(a_bounds,e_bounds)
        overlap=[]
        for x,y in zip(vals,row['integrals']):
            lo,hi=exact_bounds(x); oldlo=F(int(y['lo']),2**160); oldhi=F(int(y['hi']),2**160)
            overlap.append(max(lo,oldlo)<=min(hi,oldhi))
        assert all(overlap)
        if exact_bounds(st['N'])[0] <= 0 or exact_bounds(st['J'])[0] <= 0:
            print('inconclusive natural enclosure; subdividing',index,a_bounds,e_bounds,'N',st['N'],'J',st['J'],flush=True)
            assert depth < 3
            al,ar=map(F,a_bounds); el,er=map(F,e_bounds)
            am=(al+ar)/2; em=(el+er)/2
            for ap in ((al,am),(am,ar)):
                for ep in ((el,em),(em,er)):
                    check_box(index,[str(x) for x in ap],[str(x) for x in ep],depth+1)
            return
        result=dict(index=index,a=a_bounds,e=e_bounds,depth=depth,integrals=[encode(x) for x in vals],remainders=[encode(x) for x in errs],bounds={k:encode(v) for k,v in st.items()},all_integral_intersections_nonempty=all(overlap))
        results.append(result)
        print('rectangle',index,a_bounds,e_bounds,'N',st['N'],'J',st['J'],'seconds',time.monotonic()-started,flush=True)
    for index in sorted(selected):
        row=rows[index]
        check_box(index,row['a'],row['e'])
        sub=[r for r in results if r['index']==index]
        area=sum((F(r['a'][1])-F(r['a'][0]))*(F(r['e'][1])-F(r['e'][0])) for r in sub)
        assert area==(F(row['a'][1])-F(row['a'][0]))*(F(row['e'][1])-F(row['e'][0]))
    edges=[]
    for index in (0,63):
        row=reg['edge'][index]
        vals,errs,st=integrate(['1/2','1/2'],row['e'])
        assert exact_bounds(st['B'])[0]>0 and exact_bounds(st['H5'])[0]>0
        edges.append(dict(index=index,e=row['e'],B=encode(st['B']),H5=encode(st['H5'])))
        print('edge',index,'H5',st['H5'],flush=True)
    output=dict(status='PASS',library='mpmath.iv',version=__import__('mpmath').__version__,iv_dps=iv.dps,cells=CELLS,order=ORDER,tail='2^-60',selected_original_indices=sorted(selected),results=results,edges=edges,seconds=time.monotonic()-started,resources=resources())
    with open('review12_checks/interval_recheck.json','x') as f: json.dump(output,f,indent=2)


if __name__=='__main__': main()
