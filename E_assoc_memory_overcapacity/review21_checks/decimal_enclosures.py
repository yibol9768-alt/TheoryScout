"""Independent Decimal outward intervals and polynomial quadrature for review 21.

Does not import audited code or any previous review implementation. Decimal
exp/sqrt are correctly rounded (ROUND_HALF_EVEN); adjacent representable numbers
are used as conservative bounds. Basic arithmetic uses separate directed contexts.
Only parameters/root boxes, never saved integrals, are read from subject records.
"""
from decimal import Decimal as D, Context, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from fractions import Fraction as F
from pathlib import Path
import json, time, sys

PREC=48
DOWN=Context(prec=PREC,rounding=ROUND_FLOOR)
UP=Context(prec=PREC,rounding=ROUND_CEILING)
NEAR=Context(prec=PREC,rounding=ROUND_HALF_EVEN)
class I:
    __slots__=('l','h')
    def __init__(self,l,h=None):
        if isinstance(l,I):self.l,self.h=l.l,l.h;return
        if h is None:
            x=F(l); self.l=DOWN.divide(D(x.numerator),D(x.denominator));self.h=UP.divide(D(x.numerator),D(x.denominator))
        else:self.l,self.h=I(l).l,I(h).h
    @classmethod
    def pair(cls,l,h):
        obj=object.__new__(cls);obj.l,obj.h=l,h;assert l<=h;return obj
    def __add__(self,b):
        if isinstance(b,Series):return NotImplemented
        b=I(b);return I.pair(DOWN.add(self.l,b.l),UP.add(self.h,b.h))
    __radd__=__add__
    def __neg__(self):return I.pair(self.h.copy_negate(),self.l.copy_negate())
    def __sub__(self,b):
        if isinstance(b,Series):return NotImplemented
        return self+-I(b)
    def __rsub__(self,b):return I(b)+-self
    def __mul__(self,b):
        if isinstance(b,Series):return NotImplemented
        b=I(b)
        low=min(DOWN.multiply(x,y) for x in (self.l,self.h) for y in (b.l,b.h))
        high=max(UP.multiply(x,y) for x in (self.l,self.h) for y in (b.l,b.h))
        return I.pair(low,high)
    __rmul__=__mul__
    def __truediv__(self,b):
        if isinstance(b,Series):return NotImplemented
        b=I(b);assert b.l>0
        return I.pair(min(DOWN.divide(x,y) for x in (self.l,self.h) for y in (b.l,b.h)),max(UP.divide(x,y) for x in (self.l,self.h) for y in (b.l,b.h)))
    def __rtruediv__(self,b):return I(b)/self
    def sqrt(self):
        assert self.l>=0
        l=NEAR.sqrt(self.l);h=NEAR.sqrt(self.h)
        return I.pair(NEAR.next_minus(l) if l else l,NEAR.next_plus(h))
    def exp(self):return I.pair(NEAR.next_minus(NEAR.exp(self.l)),NEAR.next_plus(NEAR.exp(self.h)))
    def mag(self):return max(self.l.copy_abs(),self.h.copy_abs())
    def encode(self):return [str(self.l),str(self.h)]

class Series:
    __slots__=('c',)
    def __init__(self,c):self.c=c
    def constant(self,x):return Series([I(x)]+[I(0)]*(len(self.c)-1))
    def cast(self,x):return x if isinstance(x,Series) else self.constant(x)
    def __add__(self,b):
        b=self.cast(b);return Series([x+y for x,y in zip(self.c,b.c)])
    __radd__=__add__
    def __neg__(self):return Series([-x for x in self.c])
    def __sub__(self,b):return self+-self.cast(b)
    def __rsub__(self,b):return self.cast(b)+-self
    def __mul__(self,b):
        if not isinstance(b,Series):return Series([v*b for v in self.c])
        return Series([sum((self.c[j]*b.c[n-j] for j in range(n+1)),I(0)) for n in range(len(self.c))])
    __rmul__=__mul__
    def power(self,r):
        # From f*g' = r*f'*g, valid for positive f[0].
        r=F(r);assert self.c[0].l>0
        if r==F(1,2):g=[self.c[0].sqrt()]
        elif r==-1:g=[1/self.c[0]]
        else:raise ValueError(r)
        for n in range(1,len(self.c)):
            g.append(sum((self.c[k]*g[n-k]*((r+1)*k-n) for k in range(1,n+1)),I(0))/(self.c[0]*n))
        return Series(g)
    def __truediv__(self,b):
        return self*self.cast(b).power(-1) if isinstance(b,Series) else Series([v/I(b) for v in self.c])
    def __rtruediv__(self,b):return self.power(-1)*b
    def exp(self):
        # g' = f'*g, generic exponential, independent of Gaussian recurrence.
        g=[self.c[0].exp()]
        for n in range(1,len(self.c)):
            g.append(sum((self.c[k]*g[n-k]*k for k in range(1,n+1)),I(0))/n)
        return Series(g)

def atan_bracket(den):
    n=90
    val=sum((F((-1)**k,(2*k+1)*den**(2*k+1)) for k in range(n)),F(0))
    rem=F(1,(2*n+1)*den**(2*n+1))
    return I(val,val+rem)  # n even: partial sum below.
PI=16*atan_bracket(5)-4*atan_bracket(239)
NORMAL=(2/PI).sqrt()
CACHE={}
def variable_and_density(l,h,n):
    key=(l,h,n)
    if key not in CACHE:
        z=Series([I(l,h),I(1)]+[I(0)]*(n-1))
        density=(-(z*z)/2).exp()*NORMAL
        CACHE[key]=z,density
    return CACHE[key]

def functions(l,h,n,a,e,b1,b2,kind,mode):
    z,rho=variable_and_density(l,h,n)
    inv=1/(z+e); t=inv*a;residual=1-t
    # Algebraically cancel the identical beta terms in numerator/denominator
    # BEFORE interval evaluation. They follow from the raw moment recursion.
    numerator=1-t/(1+b1)
    L=(1+t*(t-2)/(1+b2)).power(F(1,2))
    denominator=z*L+e
    w=z*(inv+numerator/denominator)
    if mode=='value':return [z*w*rho,w*w*rho]
    residual_a=-inv
    numerator_a=residual_a/(1+b1)
    L_a=residual*residual_a/(L*(1+b2))
    w_a=z*(numerator_a/denominator-numerator*z*L_a/(denominator*denominator))
    values=[z*w,w*w,z*w_a,2*w*w_a]
    if mode=='path':
        if kind==1:vbar=z*inv/denominator
        else:vbar=z*z*numerator*(-2*inv+a*inv*inv)/(2*L*denominator*denominator)
        # Exact row/column operations of a 2x2 moment determinant.
        k=1 if e.l==D('.5') else 2
        ell=F(1,4) if k==1 and kind==1 else F(-1,5) if k==1 else F(1,8) if kind==1 else F(-1,20)
        da=w+a*w_a; f=z-k*w; g=vbar-ell*da
        values.extend([f*g,f*da,w*g])
    return [v*rho for v in values]

def quadrature(a,e,b1,b2,kind=1,mode='path',cells=64,order=6):
    count=7 if mode=='path' else 2 if mode=='value' else 4
    totals=[I(0) for _ in range(count)]
    for i in range(cells):
        l=F(12*i*i,cells*cells);h=F(12*(i+1)**2,cells*cells);mid=(l+h)/2;rad=I((h-l)/2)
        center=functions(mid,mid,order-1,a,e,b1,b2,kind,mode)
        remainder=functions(l,h,order,a,e,b1,b2,kind,mode)
        powers=[I(1)]
        for j in range(order+1):powers.append(powers[-1]*rad)
        for j in range(count):
            poly=sum((center[j].c[k]*2*powers[k+1]/(k+1) for k in range(0,order,2)),I(0))
            err=I(remainder[j].c[order].mag())*2*powers[order+1]/(order+1)
            totals[j]=totals[j]+poly+I.pair(err.h.copy_negate(),err.h)
    # Analytic envelope valid in the audited domain, separately checked in report.
    tail=I(F(1,2**60))
    return [v+I.pair(tail.h.copy_negate(),tail.h) for v in totals]

def state(a,e,b1,b2,kind=1,mode='path',cells=64,order=6):
    vals=quadrature(a,e,b1,b2,kind,mode,cells,order)
    A,Q=vals[:2];B=a*A;V=a*a*Q;q=1+F(3,2)*(V-B)
    old=1-F(3,4)*B;new=F(1,4)+F(3,4)*B
    result=dict(c0=old/q.sqrt(),c1=new/q.sqrt(),q=q)
    if mode!='value':
        Aa,Qa=vals[2:4];Ba=A+a*Aa;Va=2*a*Q+a*a*Qa
        result['lambda']=Ba*(q+new)-new*Va
    if mode=='path':
        fg,fda,wg=vals[4:];Y=Q+a*Qa/2
        result['Jnormalized']=2*(fg*Y-fda*wg)
    return result
def load(path):return json.loads(Path(path).read_text())
def encode(x):return {k:v.encode() for k,v in x.items()}
def resources():
    root=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('0::')[1].strip().lstrip('/')
    r={k:(root/k).read_text().strip() for k in ['memory.max','memory.swap.max','memory.peak','memory.events']}
    r['threads']=len(list(Path('/proc/self/task').iterdir()))
    assert r['memory.max']=='3221225472' and r['memory.swap.max']=='0' and r['threads']<=2
    return r
def main():
    start=time.monotonic();resources();result={}
    cert=load('aa3_checks/AA3_centered_paths_certificate.json')
    rows=[]
    for path in cert['paths']:
        e=I(path['e']);kind=path['kind'];cells=[];ends=[]
        for index,cell in enumerate(path['cells']):
            stack=[(*map(F,cell['a']),*map(F,cell['beta']),0)]
            while stack:
                al,ah,bl,bh,depth=stack.pop()
                a=I(al,ah);beta=I(bl,bh)
                b1,b2=(beta,I('99/100')) if kind==1 else (I('9/10'),beta)
                st=state(a,e,b1,b2,kind)
                good=st['lambda'].l>0 and (st['Jnormalized'].l>0 if kind==2 and path['e']=='1/2' else st['Jnormalized'].h<0)
                if good:
                    cells.append(dict(parent=index,a=[str(al),str(ah)],beta=[str(bl),str(bh)],**encode(st)))
                else:
                    assert depth<12,encode(st)
                    if (ah-al)/F(path['amax'])>=(bh-bl)/F(path['beta_max']):
                        mid=(al+ah)/2;stack.extend([(al,mid,bl,bh,depth+1),(mid,ah,bl,bh,depth+1)])
                    else:
                        mid=(bl+bh)/2;stack.extend([(al,ah,bl,mid,depth+1),(al,ah,mid,bh,depth+1)])
            if index%8==0:print('path',path['e'],kind,index+1,'/',len(path['cells']),'seconds',time.monotonic()-start,flush=True)
        for cell in path['endpoint_cover']:
            stack=[(*map(F,cell['beta']),0)]
            while stack:
                bl,bh,depth=stack.pop();beta=I(bl,bh)
                b1,b2=(beta,I('99/100')) if kind==1 else (I('9/10'),beta)
                st=state(I(path['amax']),e,b1,b2,kind,mode='value',cells=96,order=8)
                if st['c1'].l>D('.55'):
                    ends.append(dict(beta=[str(bl),str(bh)],**encode(st)))
                else:
                    assert depth<12,encode(st)
                    mid=(bl+bh)/2;stack.extend([(bl,mid,depth+1),(mid,bh,depth+1)])
        rows.append(dict(e=path['e'],kind=kind,cells=cells,endpoints=ends))
        with Path('review21_checks/path_'+path['e'].replace('/','_')+'_'+str(kind)+'.json').open('x') as f:
            json.dump(rows[-1],f,indent=2)
        print('PATH PASS',path['e'],kind,flush=True)
    result['paths']=rows
    # Independent integration of the six especially delicate near-threshold roots.
    near=load('aa3_checks/AA3_near_threshold_certificate.json');cases=[]
    for group in near['cases']:
        arms=[]
        for row in group['rows']:
            e=I(group['e']);b1=I('9/10');b2=I(row['beta2']);target=I(F('1/4')+F(group['delta']))
            low,high=row['root']
            endpoints=[state(I(v),e,b1,b2,mode='value',cells=160,order=10) for v in [low,high]]
            assert endpoints[0]['c1'].h<target.l and endpoints[1]['c1'].l>target.h
            st=state(I(low,high),e,b1,b2,mode='value',cells=160,order=10)
            # Independently certify every root-preceding cell (not sampled points).
            monotonic=[]
            for cell in row['monotonic']:
                mono=state(I(*cell['a']),e,b1,b2,mode='mono',cells=64,order=6)
                assert mono['lambda'].l>0
                monotonic.append(dict(a=cell['a'],lambda_bound=mono['lambda'].encode()))
            arms.append(dict(beta2=row['beta2'],root=row['root'],state=encode(st),endpoints=[encode(v) for v in endpoints],monotonic=monotonic))
        gap=I.pair(D(arms[1]['state']['c0'][0]),D(arms[1]['state']['c0'][1]))-I.pair(D(arms[0]['state']['c0'][0]),D(arms[0]['state']['c0'][1]))
        assert gap.l>0 if group['positive'] else gap.h<0
        cases.append(dict(e=group['e'],delta=group['delta'],arms=arms,gap=gap.encode()))
        print('NEAR PASS',group['e'],group['delta'],gap.encode(),flush=True)
    result['near_threshold']=cases
    result.update(status='PASS',resources=resources(),seconds=time.monotonic()-start,precision=PREC)
    with Path('review21_checks/decimal_enclosures.json').open('x') as f:json.dump(result,f,indent=2)
    print('ALL PASS',result['seconds'],result['resources'],flush=True)
if __name__=='__main__':main()
