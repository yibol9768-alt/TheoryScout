"""Rational, outward-rounded certificate for the strict F4/F5 inequalities.

Every interval endpoint is an integer / 2**96. No float arithmetic is
used in the certificate, except conversion for an explicitly labeled display.
"""
import math
from check_w35 import put,meta

P=96; SCALE=1<<P
def rat(n,d=1): return (n*SCALE//d,-((-n*SCALE)//d))
ZERO=rat(0); ONE=rat(1)
def add(a,b):return (a[0]+b[0],a[1]+b[1])
def neg(a):return (-a[1],-a[0])
def sub(a,b):return add(a,neg(b))
def mul(a,b):
    vv=[x*y for x in a for y in b]
    return (min(vv)//SCALE,-((-max(vv))//SCALE))
def div(a,b):
    assert b[0]>0
    lo=min(x*SCALE//y for x in a for y in b)
    hi=max(-((-x*SCALE)//y) for x in a for y in b)
    return (lo,hi)
def sqrt(a):
    assert a[0]>=0
    lo=math.isqrt(a[0]*SCALE); hi=math.isqrt(a[1]*SCALE)
    if hi*hi<a[1]*SCALE:hi+=1
    return lo,hi
def sq(a):
    assert a[0]>=0
    return mul(a,a)
def expneg(x):
    # 0<=x<=32; Taylor at x/64 in [0,1/2], then six squarings.
    assert x[0]>=0 and x[1]<=32*SCALE
    y=div(x,rat(64)); term=ONE; acc=ONE; upper=None
    for j in range(1,28):
        term=div(mul(term,y),rat(j))
        acc=add(acc,term) if j%2==0 else sub(acc,term)
        if j==26:upper=acc[1]
    ans=(acc[0],upper)
    for _ in range(6):ans=sq(ans)
    return ans

def atan(x):
    term=x; acc=ZERO; x2=sq(x); upper=None
    # P_80 is an upper bound, P_81 a lower bound for atan(x).
    for j in range(82):
        tt=div(term,rat(2*j+1)); acc=add(acc,tt) if j%2==0 else sub(acc,tt)
        if j==80:upper=acc[1]
        term=mul(term,x2)
    return (acc[0],upper)

PI=sub(mul(rat(16),atan(rat(1,5))),mul(rat(4),atan(rat(1,239))))
HALF_DEN=sqrt(div(rat(2),PI))

def efficiency(e,N=32768):
    # Both rational factors t^2/(t+e)^k increase in t>=0.
    step=rat(8,N); ia=ZERO;iq=ZERO
    exprev=ONE
    for i in range(N):
        left=rat(8*i,N);right=rat(8*(i+1),N)
        exnext=expneg(div(sq(right),rat(2)))
        la=div(sq(left),add(left,e));ra=div(sq(right),add(right,e))
        lq=div(sq(left),sq(add(left,e)));rq=div(sq(right),sq(add(right,e)))
        ia=add(ia,mul(step,(mul(la,exnext)[0],mul(ra,exprev)[1])))
        iq=add(iq,mul(step,(mul(lq,exnext)[0],mul(rq,exprev)[1])))
        exprev=exnext
    # exp(-32)<2^-32. The two unnormalized tails <=exp(-32), exp(-32)/8.
    ia=(ia[0],ia[1]+rat(1,1<<32)[1]);iq=(iq[0],iq[1]+rat(1,8*(1<<32))[1])
    aa=mul(HALF_DEN,ia);qq=mul(HALF_DEN,iq)
    return div(aa,sqrt(qq))

def margin(g,N=16384):
    total=ZERO
    for i in range(N):
        u=(rat(i,N)[0],rat(i+1,N)[1]);u2=sq(u)
        root=sqrt(sub(ONE,div(u2,rat(g))))
        val=add(mul(root,add(ONE,u2)),div(mul(u2,sub(ONE,u2)),mul(rat(g),root)))
        total=add(total,div(val,rat(N)))
    return mul(div(rat(2),PI),total)

def show(v):
    return dict(numerator_lower=str(v[0]),numerator_upper=str(v[1]),denominator_power=P,
                approximate=[v[0]/SCALE,v[1]/SCALE])

def invroot_band(g,mu):
    # mu uncertainty passed monotonically rather than lost in a symmetric +/- bound.
    mu2=sq(mu);gg=rat(g*g)
    vlo=sub(ONE,div(mu2,gg))
    kcoef=div(rat(8),mul(gg,sq(sub(ONE,rat(1,g)))))
    vhi=add(vlo,mul(kcoef,sub(ONE,mu2)))
    # Use rectangular bounds; harmless slight enlargement accounts for mu dependency.
    cmin=div(mu,sqrt(sub((vhi[0],vhi[1]),div(mul(rat(2),mu2),rat(g)))))
    cmax=div(mu,sqrt(sub((vlo[0],vlo[1]),div(mul(rat(2),mu2),rat(g)))))
    return vlo,vhi,(cmin[0],cmax[1])

c=efficiency(rat(1,10));rows=[]
for g in (16,64,256):
    mu=margin(g);lo,hi,cross=invroot_band(g,mu)
    differences=[]
    for tau in (0,1,2):
        xx=rat(tau*tau)
        f=div(c,sqrt(add(add(ONE,div(mul(rat(2),sq(c)),rat(g))),xx)))
        mlo=div(mu,sqrt(add(hi,xx)));mhi=div(mu,sqrt(add(lo,xx)))
        diff=(sub(f,mhi)[0],sub(f,mlo)[1])
        if g==16 and tau==0: assert diff[1]<0
        if g==16 and tau in (1,2): assert diff[0]>0
        if g in (64,256):assert diff[0]>0
        differences.append(dict(tau=tau,coordinate_minus_muon=show(diff)))
    # A broad rational grid certificate of the unperturbed F4 band.
    # No numerical root finding enters the proof of these band endpoints.
    if g==16:es=(rat(120,1000),rat(145,1000))
    elif g==64:es=(rat(64,1000),rat(67,1000))
    else:es=(rat(53,1000),rat(55,1000))
    cs=[efficiency(e) for e in es]
    f4ok=(cs[0][1]<cross[0] and cs[1][0]>cross[1])
    rows.append(dict(gamma=g,mu=show(mu),variance_lower=show(lo),variance_upper=show(hi),
        efficiency_crossing_band=show(cross),differences=differences,
        tested_e=[show(e) for e in es],tested_c=[show(v) for v in cs],
        proposed_e_band_certified=f4ok))
    print('certificate',g,'F4 band',f4ok,flush=True)
put('stabilizer_certificate.json',dict(arithmetic='integer dyadic outward intervals',pi=show(PI),
    c_e01=show(c),rows=rows,efficiency_cells=32768,margin_cells=16384,
    f5_sign_assertions_passed=True))
put('certificate_meta.json',meta())
