"""Second implementation using mpmath's directed interval arithmetic.
Independent formulas, no submitted modules executed/imported. Exact Fraction
postprocessing checks the saved certificate, then all 80 monotonicity cells
and every root box are re-integrated with this different arithmetic backend.
"""
import json, hashlib, time
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp
from mpmath import iv
from numerical import ROOT, OUT, resources, save

iv.dps=32
mp.mp.dps=70

def I(q):
    q=F(q); return iv.mpf(q.numerator)/iv.mpf(q.denominator)

def box(a,b):
    x,y=I(a),I(b)
    return iv.mpf([x.a,y.b])

def rational_endpoint(t):
    sign,mantissa,exp,_=t
    return F((-1 if sign else 1)*mantissa)*F(2)**exp

def enc(x):
    lo,hi=map(rational_endpoint,x._mpi_)
    return dict(lo=str(lo),hi=str(hi),decimal=[float(lo),float(hi)])

def positive(x): return x.a>0

class Series:
    # Ordinary power series in z-z0, not the submission's fixed-point jets.
    def __init__(self,c): self.c=c; self.n=len(c)-1
    @classmethod
    def const(cls,x,n): return cls([x]+[I(0)]*n)
    def __add__(self,b):
        if not isinstance(b,Series): b=Series.const(b,self.n)
        return Series([x+y for x,y in zip(self.c,b.c)])
    __radd__=__add__
    def __neg__(self): return Series([-x for x in self.c])
    def __sub__(self,b): return self+-b
    def __mul__(self,b):
        if not isinstance(b,Series): return Series([x*b for x in self.c])
        return Series([sum((self.c[j]*b.c[k-j] for j in range(k+1)),I(0)) for k in range(self.n+1)])
    __rmul__=__mul__
    def reciprocal(self):
        c=[1/self.c[0]]
        for k in range(1,self.n+1): c.append(-sum((self.c[j]*c[k-j] for j in range(1,k+1)),I(0))/self.c[0])
        return Series(c)
    def __truediv__(self,b):
        return self*(b.reciprocal() if isinstance(b,Series) else 1/b)
    def sqrt(self):
        c=[iv.sqrt(self.c[0])]
        for k in range(1,self.n+1):
            c.append((self.c[k]-sum((c[j]*c[k-j] for j in range(1,k)),I(0)))/(2*c[0]))
        return Series(c)
    def exp(self):
        c=[iv.exp(self.c[0])]
        for k in range(1,self.n+1):
            c.append(sum((j*self.c[j]*c[k-j] for j in range(1,k+1)),I(0))/k)
        return Series(c)

def ingredients(z0,n):
    z=Series([z0,I(1)]+[I(0)]*(n-1))
    density=(-(z*z)/I(2)).exp()*iv.sqrt(2/iv.pi)
    return z,density

def moments(z0,n,e):
    z,rho=ingredients(z0,n); h=(z+e).reciprocal(); ans=[]; v=z*z*rho
    for j in range(4): v=v*h; ans.append(v)
    return ans

def endpoint(z0,n,a,e,b1,b2):
    z,rho=ingredients(z0,n); recip=(z+e).reciprocal()
    t=recip*a; f=Series.const(I(1),n)-t
    L=((f*f+b2)/(1+b2)).sqrt()
    # Differentiate the second bias-corrected mean and RMS explicitly in a.
    La=-(f*recip/L)/(1+b2)
    numer=Series.const(I(1),n)-t/(1+b1)
    den=z*L+e
    P=recip+numer/den
    Pa=-recip/(den*(1+b1))-numer*z*La/(den*den)
    weight=z*z*rho
    return [weight*P,weight*P*P,weight*Pa,weight*2*P*Pa]

def integrate(fun,cells,order):
    total=[I(0)]*4
    for j in range(cells):
        l=F(12*j*j,cells*cells); r=F(12*(j+1)**2,cells*cells)
        h=I((r-l)/2)
        center=fun(I((r+l)/2),order-1); full=fun(box(l,r),order)
        for k in range(4):
            part=sum((center[k].c[m]*(2*h**(m+1)/(m+1)) for m in range(0,order,2)),I(0))
            bound=max(abs(full[k].c[order].a),abs(full[k].c[order].b))
            error=bound*2*h**(order+1)/(order+1)
            total[k]+=part+iv.mpf([-error.b,error.b])
    return [x+box(-F(1,2**60),F(1,2**60)) for x in total]

def derived(vals,a,r):
    A,Q,Aa,Qa=vals; s=1-r; B=a*A; V=a*a*Q
    Ba=A+a*Aa; Va=2*a*Q+a*a*Qa
    q=1+2*s*(V-B); new=r+s*B
    lam=Ba*(q+new)-new*Va
    return dict(c0=(1-s*B)/iv.sqrt(q),lam=lam,H=new*new-I(F(1,4))*q,new=new,q=q)

def exact_saved(cert):
    def read(x): return (F(int(x['lo']),2**x['power']),F(int(x['hi']),2**x['power']))
    def plus(x,y): return (x[0]+y[0],x[1]+y[1])
    def neg(x): return (-x[1],-x[0])
    def times(x,y):
        a=[s*t for s in x for t in y]; return min(a),max(a)
    def square(x): return (F(0) if x[0]<=0<=x[1] else min(t*t for t in x),max(t*t for t in x))
    def minus(x,y): return plus(x,neg(y))
    source=[]
    for name,h in cert['source_hashes'].items():
        source.append(dict(path=name,match=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h))
    # Staleness is a reportable finding, not a reason to skip mathematical checks.
    for row in cert['threshold']:
        m1,m2,m3,m4=map(read,row['moments']); e=(F(row['e']),)*2
        D=minus(times(m1,m3),square(m2)); H=minus(D,times(e,minus(times(m1,m4),times(m2,m3))))
        stored=read(row['T']); assert stored[0]<=H[0]<=H[1]<=stored[1]
        assert H[0]>0 if row['e']=='21/20' else H[1]<0
    coverage=[]
    for ex in cert['examples']:
        for row in ex['rows']:
            l,h=map(F,row['root']); cells=row['monotonic']
            assert F(cells[0]['a'][0])==0 and F(cells[-1]['a'][1])==h
            for i,c in enumerate(cells):
                assert F(c['a'][0])<F(c['a'][1]) and read(c['N'])[0]>0
                if i: assert F(cells[i-1]['a'][1])==F(c['a'][0])
            assert read(row['endpoints_H'][0])[1]<0<read(row['endpoints_H'][1])[0]
            coverage.append(dict(e=ex['e'],beta2=row['beta2'],cells=len(cells),saved_lambda_min=float(min(read(x['N'])[0] for x in cells))))
        ca,cb=[read(x['c0']) for x in ex['rows']]; gap=minus(cb,ca); stored=read(ex['history_gap'])
        assert stored[0]<=gap[0]<=gap[1]<=stored[1]
        assert gap[0]>0 if ex['e']=='1/2' else gap[1]<0
    return dict(source_hashes=source,coverage=coverage)

if __name__=='__main__':
    started=time.monotonic(); resources()
    cert=json.loads((ROOT/'z3_checks/certificate.json').read_text())
    result=dict(saved_checks=exact_saved(cert),threshold=[],examples=[])
    for ee in ['21/20','11/10']:
        e=I(ee); m=integrate(lambda z,n:moments(z,n,e),128,8)
        m1,m2,m3,m4=m; D=m1*m3-m2*m2; H=D-e*(m1*m4-m2*m3)
        assert positive(H) if ee=='21/20' else positive(-H)
        x=dict(e=ee,H=enc(H),moments=[enc(t) for t in m]); result['threshold'].append(x)
        print('threshold',ee,x['H']['decimal'],flush=True)
    for ex in cert['examples']:
        e=I(ex['e']); r=I(ex['r']); rows=[]
        for row in ex['rows']:
            b1=I(row['beta1']); b2=I(row['beta2']); l,h=map(F,row['root'])
            integrand=lambda z,n:endpoint(z,n,aa,e,b1,b2)
            signs=[]
            for a in [l,h]:
                aa=I(a); vals=integrate(integrand,128,8); st=derived(vals,aa,r)
                assert positive(st['new']); signs.append(st['H'])
            assert positive(-signs[0]) and positive(signs[1])
            aa=box(l,h); vals=integrate(integrand,128,8); st=derived(vals,aa,r)
            c0=st['c0']; monot=[]
            for i,cell in enumerate(row['monotonic']):
                aa=box(*cell['a']); vals=integrate(integrand,64,4); st=derived(vals,aa,r)
                assert positive(st['lam']), (ex['e'],row['beta2'],i,enc(st['lam']))
                monot.append(enc(st['lam']))
                if (i+1)%5==0: print('monotonic',ex['e'],row['beta2'],i+1,flush=True)
            x=dict(beta2=row['beta2'],root=row['root'],c0=enc(c0),endpoint_H=[enc(t) for t in signs],monotonic_lambdas=monot)
            rows.append(x); print('root',ex['e'],row['beta2'],x['c0']['decimal'],flush=True)
        gap=box(rows[1]['c0']['lo'],rows[1]['c0']['hi'])-box(rows[0]['c0']['lo'],rows[0]['c0']['hi'])
        assert positive(gap) if ex['e']=='1/2' else positive(-gap)
        result['examples'].append(dict(e=ex['e'],rows=rows,gap=enc(gap)))
    result.update(resources=resources(),seconds=time.monotonic()-started,status='PASS')
    save('interval_audit.json',result); print('PASS seconds',result['seconds'],flush=True)
