import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert int(os.environ[key])<=2
import json, math, time, signal, resource, hashlib, sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.linalg import eigh_tridiagonal
from wick import moment, evaluate, coefficient, display

ROOT=Path(__file__).resolve().parent.parent
HERE=Path(os.environ.get('X2_OUTPUT',str(Path(__file__).resolve().parent)))
START=time.monotonic()
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('4500 second limit')))
signal.alarm(4500)

def dump(name,obj):
    with (HERE/name).open('x') as f: json.dump(obj,f,indent=2,allow_nan=False)

def rank_coeff(mu,m1,l1,h,tau,kappa=1.,rho=.1):
    nu1=m1-mu*l1/2-mu/4
    def bv(z):
        r=mu+z
        b=nu1+3*mu/4+z/2-mu*mu*z/2-r*(z*z-1+h/kappa+2*mu*z)/2
        v=tau+2*(mu+z/2)**2+r*r/2
        vp=3*mu+2*z
        return b,v,vp
    phi=lambda z:math.exp(-z*z/2)/math.sqrt(2*math.pi)
    def fn(z):
        r=mu+z; b,v,_=bv(z)
        return phi(z)*phi(r)*(-b+r*v/2)
    rank=quad(fn,-12,12,epsabs=2e-13)[0]
    t=ndtri(1-rho)-mu; b,v,vp=bv(t)
    rec=phi(t)*(b-(vp-t*v)/2)
    return dict(mean_rank_0=float(ndtr(-mu/math.sqrt(2))),mean_rank_1=rank,
                recall_0=float(ndtr(-t)),recall_1=float(rec),nu1=nu1,tau=tau)

def gd(T,eta,kappa=1.):
    u=(1-eta)**T; c=1-u
    # f(s)=(1-(1-eta*s)^T)/s, normalized by f(1).
    fp=T*eta*(1-eta)**(T-1)-c
    fpp=(-T*(T-1)*eta**2*(1-eta)**(T-2) if T>1 else 0)-2*T*eta*(1-eta)**(T-1)+2*c
    A=fp/c; B=fpp/(2*c); K=1+4*A+A*A+2*B
    mu=1/math.sqrt(kappa); m1=mu*(A+B)/kappa; l1=K/kappa
    return dict(T=T,eta=eta,A=A,B=B,mu=mu,m1=m1,l1=l1,
      correct_var1=(K+3)/kappa,wrong_var1=(K+1)/kappa,
      **rank_coeff(mu,m1,l1,1.,mu*mu/2,kappa))

def ft1(k,alpha,rx,ry,kappa=1.):
    pol={s:moment(k,o,w) for s,o,w in [('mean',1,False),('second',2,False),('wrong',2,True)]}
    r=rx*ry; u=1-alpha; q=u*u+alpha*alpha+2*r*u*alpha
    a=u+r*alpha if k==0 else r*u+alpha
    mu=a/math.sqrt(kappa*q)
    m1=coefficient(pol['mean'],-1,kappa,alpha,rx,ry,1)/math.sqrt(q)
    w1=coefficient(pol['wrong'],-1,kappa,alpha,rx,ry,2)/q
    sec1=coefficient(pol['second'],-1,kappa,alpha,rx,ry,2)/q
    h=((u*u+ry*ry*alpha*alpha+2*r*u*alpha) if k==0 else
       (ry*ry*u*u+alpha*alpha+2*r*u*alpha))/q
    e2=ry*ry*(alpha if k==0 else u)**2*(1-rx*rx)/(kappa*q)
    l1=w1-h/kappa
    v1=sec1-2*mu*m1
    assert abs(v1-(l1+3*mu*mu+e2))<1e-10
    return dict(k=k,alpha=alpha,rx=rx,ry=ry,q=q,mu=mu,m1=m1,l1=l1,h=h,
      correct_var1=v1,wrong_var1=w1,polynomials={s:[dict(d=x[0],n=x[1],alpha=x[2],rx=x[3],ry=x[4],c=c) for x,c in p.items()] for s,p in pol.items()},
      **rank_coeff(mu,m1,l1,h,mu*mu/2+e2,kappa))

def ft_independent(T,eta,k=0):
    u=(1-eta)**T; v=1-u; q=u*u+v*v
    ap=-T*eta*(1-eta)**(T-1)
    app=T*(T-1)*eta*eta*(1-eta)**(T-2)/2 if T>1 else 0.
    bp=-ap-v; bpp=-app+ap+v
    Lo=ap*ap+2*u*app+u*u+2*v*bp+bp*bp+2*v*bpp
    Ln=ap*ap+2*u*app+2*u*ap+v*v+4*v*bp+bp*bp+2*v*bpp
    mu=(u if k==0 else v)/math.sqrt(q)
    m1=(app if k==0 else -app)/math.sqrt(q)
    l1=(Lo if k==0 else Ln)/q; h=(u*u if k==0 else v*v)/q
    w1=l1+h
    L1=ap*ap+2*u*app+u*ap
    ueta=-T*(1-eta)**(T-1)
    shift=-L1/(2*u*ueta)
    dmu=(v if k==0 else -u)*ueta/q**1.5
    return dict(T=T,eta=eta,k=k,mu=mu,m1=m1,l1=l1,h=h,q=q,
      correct_var1=l1+3*mu*mu,wrong_var1=w1,msr1=m1-mu*w1/2,
      loss1=L1,eta_match1=shift,matched_msr1=m1-mu*w1/2+dmu*shift,
      **rank_coeff(mu,m1,l1,h,mu*mu/2))

def radial_moments(d,e,N):
    n=d*d; shape=n/2
    j=np.arange(N,dtype=float)
    nodes,vec=eigh_tridiagonal(2*j+shape,np.sqrt(j[1:]*(j[1:]+shape-1)))
    weights=vec[0]**2; vs=2*nodes/n
    cs=math.sqrt(2/math.pi)
    def gauss(v):
        if e==0: return cs,1.,1.
        def integ(power,den):
            return quad(lambda z:cs*math.exp(-z*z/2)*z**power/(math.sqrt(v)*z+e)**den,
                        0,12,epsabs=3e-13,epsrel=3e-13)[0]
        return math.sqrt(v)*integ(2,1),v*integ(2,2),v*integ(4,2)
    a,q0,q2=np.array([gauss(v) for v in vs]).T
    a0,q,_=gauss(1.)
    L=float(weights@(np.sqrt(vs)*a))/math.sqrt(n)
    B=float(weights@(vs*a*a))/(n+2)
    C=float(weights@(vs*(3*q2+(n-1)*q0)))/(n+2)
    Cw=float(weights@(vs*(q2+(n+1)*q0)))/(n+2)
    Ccross=float(weights@(vs*(q2-q0)))/(n+2)
    return dict(d=d,e=e,mean=d*L/math.sqrt(q),variance=(C+6*(d-1)*B-(2*d-1)*L*L)/q,
      sample_variance=(C-Ccross+4*(d-1)*B)/q,wrong_variance=(Cw+2*(d-1)*B)/q,
      a=a0,q=q,c=a0/math.sqrt(q),mean1=0.,var1=4*a0*a0/q,wrong1=2*a0*a0/q)

def main():
    inputs=['ICML_REVIEW_SIM3.md','W8_FINITE_D.md','PROOF_REVIEW7.md','W4_RESULTS.md',
       'FT_FORMULATION.md','FT_MOMENT_THEOREM.md','W4_GD_VARIANCE_NOTE.md',
       'experiments/w4/W4_raw.jsonl','experiments/w4/repro/W4_core.py']
    hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in inputs}
    dump('input_hashes.json',hashes)
    gds=[gd(1,1.),gd(4,.3),gd(4,.6)]
    fts=[ft1(k,.14,rx,ry) for rx,ry in [(0.,0.),(1.,.5),(.6,.7)] for k in [0,1]]
    dump('coefficients.json',dict(GD=gds,FT1=fts,FT4=[ft_independent(4,1-.75**(1/8),k) for k in [0,1]]))
    exact=[]
    for d in [64,128,256,512]:
      for row in fts:
        k,alpha,rx,ry,q=[row[s] for s in ['k','alpha','rx','ry','q']]
        pol={s:moment(k,o,w) for s,o,w in [('mean',1,False),('second',2,False),('wrong',2,True)]}
        m=evaluate(pol['mean'],d,d*d,alpha,rx,ry,1)/math.sqrt(q)
        vv=evaluate(pol['second'],d,d*d,alpha,rx,ry,2)/q-m*m
        w=evaluate(pol['wrong'],d,d*d,alpha,rx,ry,2)/q
        exact.append(dict(d=d,k=k,rx=rx,ry=ry,alpha=alpha,mean=m,variance=vv,wrong=w,
          mean_beta=d*(m-row['mu']),var_beta=d*(vv-1),wrong_beta=d*(w-1)))
    dump('ft1_exact.json',exact)
    nonlinear=[]
    for d in [64,128,256,512]:
      for e in [0.,.1]:
        r=radial_moments(d,e,48); s=radial_moments(d,e,80)
        s['quadrature_difference']=max(abs(r[k]-s[k]) for k in ['mean','variance','wrong_variance','sample_variance'])
        nonlinear.append(s)
    dump('nonlinear_exact.json',nonlinear)
    # Recompute old seed-level evidence without writing into experiments/.
    raw=[json.loads(s) for s in (ROOT/'experiments/w4/W4_raw.jsonl').read_text().splitlines()]
    summ=[]
    for d in [64,128,256,320]:
      for mode in ['fixed','matched']:
        rr=[r for r in raw if r['kind']=='FT' and r['d']==d and r['mode']==mode]
        if rr:
          v=np.array([r['old']['msr']-r['prediction'] for r in rr])
          summ.append(dict(d=d,mode=mode,n=len(v),bias=float(v.mean()),se=float(v.std(ddof=1)/math.sqrt(len(v)))))
    dump('w4_reanalysis.json',summ)
    assert all(hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h for s,h in hashes.items())
    cg=Path('/sys/fs/cgroup')/next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')).lstrip('/')
    dump('analytic_metadata.json',dict(seconds=time.monotonic()-START,python=sys.executable,
      rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      controls={s:(cg/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak']},
      process_threads=len(list(Path('/proc/self/task').iterdir()))))
    print(json.dumps(dict(GD=gds,FT4=ft_independent(4,1-.75**(1/8)),nonlinear=nonlinear),indent=2))

if __name__=='__main__': main()
