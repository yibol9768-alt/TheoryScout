"""Validated T=2 FT retrieval certificate. Integer outward intervals only.
Core arithmetic is copied, unchanged, from w6_checks/adam_certificate.py.
"""
from interval_core import *

class AD:
    """First a,e derivatives of z-Taylor jets."""
    def __init__(self,v,da=None,de=None):
        self.v=v; n=len(v.c)-1
        self.a=Jet.const(Z,n) if da is None else da
        self.e=Jet.const(Z,n) if de is None else de
    def __add__(self,o):return AD(self.v+o.v,self.a+o.a,self.e+o.e)
    def __neg__(self):return AD(-self.v,-self.a,-self.e)
    def __sub__(self,o):return self+-o
    def __mul__(self,o):return AD(self.v*o.v,self.a*o.v+self.v*o.a,self.e*o.v+self.v*o.e)
    def __truediv__(self,o):
        v=self.v/o.v
        return AD(v,(self.a-v*o.a)/o.v,(self.e-v*o.e)/o.v)
    def square(self):
        return AD(self.v.square(),(self.v*self.a).scale(rat(2)),(self.v*self.e).scale(rat(2)))
    def root(self):
        v=self.v.root();den=v.scale(rat(2))
        return AD(v,self.a/den,self.e/den)
    def scale(self,x):return AD(self.v.scale(x),self.a.scale(x),self.e.scale(x))

def jets(a,e,z0,n,betas=(rat(9,10),rat(99,100))):
    def C(x):return AD(Jet.const(x,n))
    one=C(O);z=AD(Jet([z0,O]+[Z]*(n-1)))
    aa=C(a);aa.a=Jet.const(O,n)
    ee=C(e);ee.e=Jet.const(O,n)
    b1,b2=betas
    t=aa/(z+ee)
    f=one-t
    root=( (f.square()+C(b2)).scale(div(O,add(O,b2))) ).root()
    P=one/(z+ee)+(one-t.scale(div(O,add(O,b1))))/(z*root+ee)
    dc=[density(z0)]
    for k in range(1,n+1):
        dc.append(neg(div(add(mul(z0,dc[k-1]),dc[k-2] if k>=2 else Z),rat(k))))
    weight=z.v.square()*Jet(dc)
    pp=P.square()
    return [v*weight for v in (P.v,pp.v,P.a,pp.a,P.e,pp.e)]

# Each tail |integral_{12}^infty| < 2^-60 for A,Q,A_a,Q_a,A_e,Q_e.
# The global proof for 0<=a<=1/2, 1/10<=e<=1 is in Y2_MULTISTEP.md.

def integ(a,e,cells=64,order=4,betas=(rat(9,10),rat(99,100))):
    sums=[Z]*6;errs=[Z]*6
    for i in range(cells):
        center=rat(6*(i*i+(i+1)**2),cells*cells)
        cell=(rat(12*i*i,cells*cells)[0],rat(12*(i+1)**2,cells*cells)[1])
        rad=rat(6*(2*i+1),cells*cells)
        powers=[O]
        for _ in range(order+1):powers.append(mul(powers[-1],rad))
        mid=jets(a,e,center,order-1,betas)
        full=jets(a,e,cell,order,betas)
        for j in range(6):
            val=sum_iv([mul(mid[j].c[k],div(mul(rat(2),powers[k+1]),rat(k+1))) for k in range(0,order,2)])
            err=mul((abmax(full[j].c[order]),)*2,div(mul(rat(2),powers[order+1]),rat(order+1)))
            sums[j]=add(sums[j],add(val,(-err[1],err[1])))
            errs[j]=add(errs[j],err)
    return [add(s,neg(TAIL) if False else (-TAIL[1],TAIL[1])) for s in sums],errs

def stats(a,vals):
    A,Q,Aa,Qa,Ae,Qe=vals
    B=mul(a,A);V=mul(sq(a),Q)
    Ba=add(A,mul(a,Aa));Va=add(mul(mul(rat(2),a),Q),mul(sq(a),Qa))
    Be=mul(a,Ae);Ve=mul(sq(a),Qe)
    q=add(sub(O,mul(rat(2),B)),mul(rat(2),V))
    # N=q^(3/2)*dc1/da. J=a^-2 (B_e V_a - B_a V_e).
    N=sub(mul(Ba,add(sub(O,B),mul(rat(2),V))),mul(B,Va))
    J=add(sub(mul(rat(2),mul(Ae,Q)),mul(A,Qe)),mul(a,sub(mul(Ae,Qa),mul(Aa,Qe))))
    H=sub(sq(B),mul(rat(4,25),q))
    return dict(A=A,Q=Q,Aa=Aa,Qa=Qa,Ae=Ae,Qe=Qe,B=B,V=V,Ba=Ba,Va=Va,Be=Be,Ve=Ve,q=q,N=N,J=J,H=H)

def show(x):return dict(lo=str(x[0]),hi=str(x[1]),power=P,display_only=[x[0]/S,x[1]/S])
def F(x):return rat(Fraction(x).numerator,Fraction(x).denominator)
def resources():
    p=Path('/sys/fs/cgroup')/Path(next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')).lstrip('/'))
    d={s:(p/s).read_text().strip() for s in ('memory.max','memory.swap.max','memory.peak','memory.events')}
    d['threads']=len(list(Path('/proc/self/task').iterdir()));d['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert d['memory.max']=='3221225472' and d['memory.swap.max']=='0' and d['threads']<=2
    return d

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',default='probe');parser.add_argument('--output',default='y2_checks/certificate.json');args=parser.parse_args()
    start=time.monotonic()
    if args.mode=='probe':
        for ee,aa in [('0.1','.2234'),('.5','.2953'),('1','.3806')]:
            vals,err=integ(F(aa),F(ee))
            s=stats(F(aa),vals)
            print(ee,aa,{k:show(s[k])['display_only'] for k in ('A','Q','N','J','H')},'errs',[show(x)['display_only'] for x in err],flush=True)
        a=hull(F('.29'),F('.30'));e=hull(F('.49'),F('.51'))
        vals,err=integ(a,e);print('rectangle',{k:show(v)['display_only'] for k,v in stats(a,vals).items()},flush=True)
        print('seconds',time.monotonic()-start,resources(),flush=True)
