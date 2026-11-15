"""Independent verification: mpmath.iv arithmetic and rational geometry.

This module imports no generator or integer interval module. Optional integration
uses a new dual-number power-series implementation with mpmath.iv primitives.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
from mpmath import iv
iv.dps=60
BASE=Path(__file__).resolve().parents[1]
AUX_AUDIT=None

def q(x):
    f=F(str(x));return iv.mpf(f.numerator)/f.denominator

def box(a,b=None):
    a=q(a);b=a if b is None else q(b)
    return iv.mpf([a.a,b.b])

def stored(x):
    assert x['power']==160 and int(x['lo'])<=int(x['hi'])
    return iv.mpf([iv.mpf(int(x['lo'])).a,iv.mpf(int(x['hi'])).b])/2**160

def pos(x):assert x.a>0, str(x)
def neg(x):assert x.b<0, str(x)
def overlap(x,y):assert x.a<=y.b and y.a<=x.b,(str(x),str(y))
def encode(x):
    return [dict(sign=v[0],mantissa=str(v[1]),exponent=v[2]) for v in x._mpi_]

def read(p):
    obj=json.loads(p.read_text());assert obj['status']=='PASS'
    for name,h in obj['source_hashes'].items():
        path=BASE/name
        assert path.resolve().is_relative_to(BASE)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==h, name
    return obj

def signals(a,v,r='0'):
    A,Q,Aa,Qa=v[:4];s=1-q(r)
    B=a*A;V=a*a*Q;Ba=A+a*Aa;Va=2*a*Q+a*a*Qa
    variance=1+2*s*(V-B);new=q(r)+s*B
    N=Ba*(1+q(r)+s*(2*V-B))-new*Va
    result=dict(B=B,q=variance,N=N,new=new,c0=(1-s*B)/iv.sqrt(variance))
    if len(v)==6:
        Ae,Qe=v[4:];result['J']=2*Ae*Q-A*Qe+a*(Ae*Qa-Aa*Qe)
    return result

def cover1(rows,lo,hi):
    end=F(lo)
    for a,b in sorted((F(a),F(b)) for a,b in rows):
        assert a==end and b>a,(a,b,end)
        end=b
    assert end==F(hi)

def exact_cover(rows):
    rects=[]
    for row in rows:
        al,ar=map(F,row['a']);el,er=map(F,row['e'])
        assert 0<=al<ar<=F('1/2') and F('1/10')<=el<er<=1
        rects.append((al,ar,el,er))
    cuts=sorted({p for r in rects for p in r[:2]})
    assert cuts[0]==0 and cuts[-1]==F('1/2')
    for a,b in zip(cuts,cuts[1:]):
        mid=(a+b)/2
        cover1([(el,er) for al,ar,el,er in rects if al<mid<ar],'1/10','1')
    return len(cuts)-1

class Dual:
    """A coefficient and its first parameter derivatives, all intervals."""
    def __init__(self,x=0,da=0,de=0):
        self.x=iv.mpf(x);self.da=iv.mpf(da);self.de=iv.mpf(de)
    def __add__(self,y):return Dual(self.x+y.x,self.da+y.da,self.de+y.de)
    def __neg__(self):return Dual(-self.x,-self.da,-self.de)
    def __sub__(self,y):return self+-y
    def __mul__(self,y):return Dual(self.x*y.x,self.da*y.x+self.x*y.da,self.de*y.x+self.x*y.de)
    def square(self):return Dual(self.x**2,2*self.x*self.da,2*self.x*self.de)
    def inverse(self):
        v=1/self.x;return Dual(v,-self.da*v*v,-self.de*v*v)
    def __truediv__(self,y):return self*y.inverse()
    def root(self):
        v=iv.sqrt(self.x);return Dual(v,self.da/(2*v),self.de/(2*v))

class Series:
    """Truncated powers in z; dual coefficients carry parameter derivatives."""
    def __init__(self,c):self.c=c
    @classmethod
    def constant(cls,x,n):return cls([x]+[Dual() for _ in range(n)])
    def __add__(self,y):return Series([a+b for a,b in zip(self.c,y.c)])
    def __neg__(self):return Series([-a for a in self.c])
    def __sub__(self,y):return self+-y
    def __mul__(self,y):
        return Series([sum((self.c[j]*y.c[k-j] for j in range(k+1)),Dual()) for k in range(len(self.c))])
    def __truediv__(self,y):
        v=[]
        for k,x in enumerate(self.c):
            v.append((x-sum((y.c[j]*v[k-j] for j in range(1,k+1)),Dual()))/y.c[0])
        return Series(v)
    def square(self):
        value=self*self;value.c[0]=self.c[0].square();return value
    def root(self):
        v=[self.c[0].root()]
        for k in range(1,len(self.c)):
            rem=sum((v[j]*v[k-j] for j in range(1,k)),Dual())
            v.append((self.c[k]-rem)/(Dual(2)*v[0]))
        return Series(v)

def independent_jets(a,e,z0,n,betas=('9/10','99/100'),moment=False,frozen=False):
    def c(x=0,da=0,de=0):return Series.constant(Dual(x,da,de),n)
    z=Series([Dual(z0),Dual(1)]+[Dual() for _ in range(n-1)])
    density=[iv.sqrt(2/iv.pi)*iv.exp(-z0**2/2)]
    for k in range(1,n+1):density.append((-z0*density[-1]-(density[k-2] if k>1 else 0))/k)
    weight=z*z*Series([Dual(v) for v in density])
    if moment:
        out=[];v=weight
        for _ in range(4):v=v/(z+c(e));out.append([x.x for x in v.c])
        return out
    if frozen:
        p=c(2)/(z+c(e));return [[x.x for x in v.c] for v in (p*weight,p*p*weight)]
    # Derive the endpoint from the two residuals and bias-corrected histories.
    b1,b2=map(q,betas);t=c(a,1)/(z+c(e,0,1));residual=c(1)-t
    rms=((c(b2)+residual.square())/c(1+b2)).root()
    p=c(1)/(z+c(e,0,1))+(c(1)-t/c(1+b1))/(z*rms+c(e,0,1))
    linear=p*weight;squared=p.square()*weight
    return [[getattr(v,attr) for v in poly.c] for attr in ('x','da','de') for poly in (linear,squared)]

def reintegrate(a,e,cells,order,betas=('9/10','99/100'),moment=False,frozen=False):
    count=4 if moment else 2 if frozen else 6
    answer=[iv.mpf(0) for _ in range(count)]
    for i in range(cells):
        left=F(12*i*i,cells*cells);right=F(12*(i+1)**2,cells*cells)
        center=q((left+right)/2);radius=q((right-left)/2)
        mid=independent_jets(a,e,center,order-1,betas,moment,frozen)
        whole=independent_jets(a,e,box(left,right),order,betas,moment,frozen)
        for j in range(count):
            value=sum((2*mid[j][k]*radius**(k+1)/(k+1) for k in range(0,order,2)),iv.mpf(0))
            remainder=abs(whole[j][order])*2*radius**(order+1)/(order+1)
            answer[j]+=value+iv.mpf([-remainder.b,remainder.b])
    tail=q(F(1,2**60))
    result=[x+iv.mpf([-tail.b,tail.b]) for x in answer]
    if AUX_AUDIT is not None:
        AUX_AUDIT.write(json.dumps(dict(a=encode(a),e=encode(e),cells=cells,order=order,betas=list(betas),moment=moment,frozen=frozen,integrals=[encode(v) for v in result]))+'\n')
    return result

def certify_rectangle(row, audit, index, depth=0):
    al,ar=map(F,row['a']);el,er=map(F,row['e'])
    a=box(al,ar);e=box(el,er);v=reintegrate(a,e,64,4);st=signals(a,v)
    if st['N'].a>0 and st['J'].a>0:
        audit.write(json.dumps(dict(parent=index,a=list(map(str,(al,ar))),e=list(map(str,(el,er))),
            integrals=[encode(x) for x in v],N=encode(st['N']),J=encode(st['J'])))+'\n')
        return 1
    assert depth<8, (index,depth,str(st['N']),str(st['J']))
    # Exact bisection certifies a complete subcover when dependency widens a box.
    if (ar-al)*2 > (er-el)/(el+F('1/10')):
        am=(al+ar)/2;children=[dict(a=[str(al),str(am)],e=row['e']),dict(a=[str(am),str(ar)],e=row['e'])]
    else:
        em=(el+er)/2;children=[dict(a=row['a'],e=[str(el),str(em)]),dict(a=row['a'],e=[str(em),str(er)])]
    return sum(certify_rectangle(child,audit,index,depth+1) for child in children)

def root_check(row,betas,recompute=False):
    a=box(*row['a']);e=q(row['e'])
    for k,ep in enumerate(row['endpoints']):
        vals=[stored(v) for v in ep['integrals']]
        st=signals(q(ep['a']),vals);pos(st['B']);h=st['B']**2-q('4/25')*st['q']
        (neg if k==0 else pos)(h);overlap(h,stored(ep['H']))
        if recompute:
            vv=reintegrate(q(ep['a']),e,192,10,betas)
            s=signals(q(ep['a']),vv);(neg if k==0 else pos)(s['B']**2-q('4/25')*s['q'])
    st=signals(a,[stored(v) for v in row['integrals']]);overlap(st['c0'],stored(row['c0']))
    if recompute:
        vv=reintegrate(a,e,192,10,betas);fresh=signals(a,vv);overlap(fresh['c0'],stored(row['c0']));st=fresh
    return st['c0']

def sign_signal(a):
    # Direct analytic density and alternating mass series, independent of generator.
    mass=iv.mpf(0)
    for k in range(30):mass+=(-1)**k*a**(2*k+1)/(2**k*__import__('math').factorial(k)*(2*k+1))
    error=abs(a**61/(2**30*__import__('math').factorial(30)*61))
    mass=(mass+iv.mpf([-error.b,error.b]))*iv.sqrt(2/iv.pi)
    B=2*a*iv.sqrt(2/iv.pi)*iv.exp(-a*a/2);V=4*a*a*(1-mass);var=1-2*B+2*V
    return (1-B)/iv.sqrt(var),B*B-q('4/25')*var

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--certificate-dir',type=Path,default=BASE/'two_step_adam_certificate/reference')
    parser.add_argument('--reintegrate',choices=['none','sample','all'],default='none')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();folder=args.certificate_dir
    reg=read(folder/'stabilizer_ordering.json');pts=read(folder/'matched_chain.json');hist=read(folder/'moment_history.json');rev=read(folder/'history_reversal.json')
    assert len(reg['leaves'])==1027 and reg['beta1']=='9/10' and reg['beta2']=='99/100'
    assert pts['u']=='2/5' and pts['r']=='0' and pts['beta1']=='9/10' and pts['beta2']=='99/100'
    assert [r['beta'] for r in hist['models']]==[['0','0'],['9/10','0'],['0','99/100']]
    assert [r['e'] for r in rev['threshold']]==['21/20','11/10']
    assert [r['e'] for r in rev['examples']]==['1/2','2']
    strips=exact_cover(reg['leaves']);cover1([row['e'] for row in reg['edge']],'1/10','1')
    count=0;durations=[];subcells=0
    audit=args.output.with_name('independent_integral_envelopes.jsonl').open('x')
    for index,row in enumerate(reg['leaves']):
        a=box(*row['a']);e=box(*row['e']);st=signals(a,[stored(v) for v in row['integrals']])
        for key in ['N','J']:pos(st[key]);overlap(st[key],stored(row['bounds'][key]))
        if args.reintegrate=='all' or (args.reintegrate=='sample' and index in [0,len(reg['leaves'])-1]):
            then=time.monotonic();subcells+=certify_rectangle(row,audit,index)
            durations.append(time.monotonic()-then);count+=1
        if index%100==0:print('cover/signs',index,'independent integrals',count,'seconds',time.monotonic()-start,flush=True)
    global AUX_AUDIT
    region_count=count
    all_integrals=args.reintegrate=='all'
    if all_integrals:AUX_AUDIT=args.output.with_name('independent_auxiliary_integral_envelopes.jsonl').open('x')
    for row in reg['edge']:
        v=[stored(x) for x in row['integrals']]
        if all_integrals:v=reintegrate(q('1/2'),box(*row['e']),64,4);count+=1
        st=signals(q('1/2'),v);pos(st['B']);pos(st['B']**2-st['q']/4)
    c0=[root_check(row,('9/10','99/100'),all_integrals) for row in pts['points']]
    sign_lo,sign_hi=pts['sign']['a'];sign_c,_=sign_signal(box(sign_lo,sign_hi))
    neg(sign_signal(q(sign_lo))[1]);pos(sign_signal(q(sign_hi))[1])
    # The analytic first-root bound for Sign is strictly positive.
    assert F(7,5)*F(29,50)-16*F(21,100)**2==F(133,1250)>0
    pos(c0[0]-sign_c);pos(c0[1]-c0[0]);pos(c0[2]-c0[1]);pos(iv.sqrt(q('21/25'))-c0[2])
    ablations=[]
    for row in hist['models']:
        cover1([s['a'] for s in row['derivative_slabs']],'0','2/5')
        for slab in row['derivative_slabs']:
            v=[stored(x) for x in slab['integrals']]
            if all_integrals:v=reintegrate(box(*slab['a']),q(1),64,4,row['beta']);count+=1
            pos(signals(box(*slab['a']),v)['N'])
        ablations.append(root_check(row,row['beta'],all_integrals))
    stateless,momentum,rms=ablations;pos(rms-stateless);pos(stateless-c0[2]);pos(c0[2]-momentum)
    for row,cc in zip(pts['points'],c0):
        fr=row['frozen_direction'];pos(cc-stored(fr['c0']))
        if all_integrals:
            A,Q=reintegrate(q(0),q(row['e']),192,10,frozen=True)
            u=q('2/5');a=u/(u*A+iv.sqrt((1+u*u)*A*A-2*u*u*Q))
            pos(cc-u*(1/(a*A)-1))
    for j,row in enumerate(rev['threshold']):
        v=[stored(x) for x in row['moments']]
        if all_integrals:v=reintegrate(q(0),q(row['e']),128,10,moment=True);count+=1
        m1,m2,m3,m4=v;D=m1*m3-m2*m2;H=D-q(row['e'])*(m1*m4-m2*m3)
        pos(D);(pos if j==0 else neg)(H)
    for example in rev['examples']:
        r=example['r'];e=q(example['e']);u=q(example['u']);cc=[]
        for row in example['rows']:
            cover1([s['a'] for s in row['monotonic']],'0',row['root'][1])
            for slab in row['monotonic']:
                v=[stored(x) for x in slab['integrals']]
                if all_integrals:v=reintegrate(box(*slab['a']),e,64,4,(row['beta1'],row['beta2']))[:4];count+=1
                pos(signals(box(*slab['a']),v,r)['N'])
            for k,ep in enumerate(row['endpoints_H']):
                v=[stored(x) for x in ep['integrals']]
                if all_integrals:v=reintegrate(q(row['root'][k]),e,128,10,(row['beta1'],row['beta2']))[:4];count+=1
                s=signals(q(row['root'][k]),v,r);pos(s['new']);(neg if k==0 else pos)(s['new']**2-u*u*s['q'])
            v=[stored(x) for x in row['root_integrals']]
            if all_integrals:v=reintegrate(box(*row['root']),e,128,10,(row['beta1'],row['beta2']))[:4];count+=1
            cc.append(signals(box(*row['root']),v,r)['c0'])
        (pos if example['e']=='1/2' else neg)(cc[1]-cc[0])
    audit.close()
    if AUX_AUDIT is not None:AUX_AUDIT.close()
    result=dict(status='PASS',arithmetic='mpmath.iv 60 decimal digits; Fraction exact coverage',
        rectangles=1027,edge_cells=64,parameter_strips=strips,all_coverage_and_signs_recomputed=True,
        independent_reintegration=args.reintegrate,reintegrated_region_cells=region_count,
        independent_region_subcells=subcells,sample_cell_seconds=durations[:2],estimated_region_seconds=(sum(durations[:2])/len(durations[:2])*1027 if durations else None),
        seconds=time.monotonic()-start)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
