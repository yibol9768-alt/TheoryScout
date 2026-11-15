"""Exact finite-d Gaussian Wick sums; no simulation or imported project code."""
from collections import defaultdict
from itertools import product

def pairings(xs):
    if not xs:
        yield []; return
    for j in range(1,len(xs)):
        for rest in pairings(xs[1:j]+xs[j+1:]):
            yield [(xs[0],xs[j])]+rest

def score_terms(k,wrong=False,tag='A'):
    # (species, task, coordinate, sample); query sample labels @0 and @1 fixed.
    a,b,c,l,m=[tag+s for s in 'abclm']
    j='@1' if wrong else '@0'
    head=[('y',k,a,j),('x',k,b,'@0')]
    return [
      (1,0,0,head+[('y',0,a,l),('x',0,b,l)]),
      (-1,1,-1,head+[('y',0,a,l),('x',0,c,l),('x',1,c,m),('x',1,b,m)]),
      (1,1,0,head+[('y',1,a,l),('x',1,b,l)])]

def moment(k,order,wrong=False):
    out=defaultdict(int)
    for terms in product(*[score_terms(k,wrong,str(i)) for i in range(order)]):
        sign=1; pa=pn=0; vs=[]
        for sg,aa,nn,vs0 in terms:
            sign*=sg; pa+=aa; pn+=nn; vs+=vs0
        coords={v[2] for v in vs}; samples={v[3] for v in vs}
        for px in pairings([v for v in vs if v[0]=='x']):
          for py in pairings([v for v in vs if v[0]=='y']):
            par={s:s for s in coords|samples}
            def find(s):
                while par[s]!=s: s=par[s]
                return s
            def union(s,t):
                s,t=find(s),find(t)
                if s!=t: par[t]=s
            rx=ry=0
            for v,w in px+py:
                union(v[2],w[2]); union(v[3],w[3])
                if v[1]!=w[1]:
                    if v[0]=='x': rx+=1
                    else: ry+=1
            fixed={find(s) for s in samples if s.startswith('@')}
            if len(fixed)!=len([s for s in samples if s.startswith('@')]): continue
            dc=len({find(s) for s in coords})-order
            nc=len({find(s) for s in samples}-fixed)+pn
            # n^(-order/2) factored out separately.
            out[dc,nc,pa,rx,ry]+=sign
    return {key:v for key,v in out.items() if v}

def evaluate(poly,d,n,alpha,rx,ry,order):
    return sum(c*d**dd*n**nn*alpha**aa*rx**xx*ry**yy
               for (dd,nn,aa,xx,yy),c in poly.items())/n**(order/2)

def coefficient(poly,power,kappa,alpha,rx,ry,order):
    return sum(c*kappa**(nn-order/2)*alpha**aa*rx**xx*ry**yy
      for (dd,nn,aa,xx,yy),c in poly.items() if dd+2*nn-order==power)

def display(poly,power,order):
    return [dict(d=dd,n=nn,alpha=aa,rx=xx,ry=yy,coefficient=c)
      for (dd,nn,aa,xx,yy),c in sorted(poly.items()) if dd+2*nn-order==power]
