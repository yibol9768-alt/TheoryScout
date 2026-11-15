"""Deterministic population predictions only. No random experiments."""
import os, json, time, resource
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre, ndtr, ndtri
from scipy.optimize import brentq

OUT=Path(__file__).resolve().parent

def resources():
    c=Path('/sys/fs/cgroup')/next(s.split(':',2)[2].strip().lstrip('/') for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
    ans={s:(c/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak']}
    ans['threads']=len(list(Path('/proc/self/task').iterdir()))
    ans['maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert ans['memory.max']=='3221225472' and ans['memory.swap.max']=='0' and ans['threads']<=2
    assert all(int(os.environ[s])<=2 for s in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'])
    return ans

x,w=roots_legendre(600); z=6*(x+1); weight=6*w*np.sqrt(2/np.pi)*np.exp(-z*z/2)

def endpoint(a,e,b1,b2):
    t=a/(z+e); p1=1/(1+b1);p2=1/(1+b2)
    L=np.sqrt(1-2*p2*t+p2*t*t); den=z*L+e
    La=p2*(t-1)/(L*(z+e))
    P=1/(z+e)+(1-p1*t)/den
    Pa=-p1/((z+e)*den)-(1-p1*t)*z*La/den**2
    d=a*z*P; da=z*(P+a*Pa)
    v1=z*p1*p1/((z+e)*den)
    v2=-z*z*(1-p1*t)*p2*p2*(2/(z+e)-a/(z+e)**2)/(2*L*den**2)
    return d,da,a*a*v1,a*a*v2

def signals(a,e,b1,b2,r):
    d,da,db1,db2=endpoint(a,e,b1,b2);dt=np.array([da,db1,db2])
    B=weight@(z*d);V=weight@(d*d);Bt=dt@(weight*z);Vt=dt@(weight*2*d)
    s=1-r;q=1+2*s*(V-B);qt=2*s*(Vt-Bt)
    nums=np.array([1-s*B,r+s*B]);nt=np.array([-s*Bt,s*Bt])
    c=nums/np.sqrt(q);deriv=nt/np.sqrt(q)-nums[:,None]*qt/(2*q**1.5)
    return c,deriv,q

def match(r,delta,e,b1,b2):
    hi=e*.49
    while signals(hi,e,b1,b2,r)[0][1]<r+delta:
        hi*=1.1
        assert hi<e*.9
    a=brentq(lambda a:signals(a,e,b1,b2,r)[0][1]-r-delta,0,hi,xtol=2e-15)
    c,dc,q=signals(a,e,b1,b2,r)
    return dict(r=r,delta=delta,u=r+delta,e=e,beta1=b1,beta2=b2,a=a,c0=c[0],c1=c[1],q=q,
                slope=dc[0,0]/dc[1,0],c1a=dc[1,0],matched_derivatives=(dc[0,1:]-dc[0,0]*dc[1,1:]/dc[1,0]).tolist())

def influence(row):
    # Integrate out independent H~N(0,1) exactly: coefficients of H^0,H^1,H^2.
    r=row['r'];sigma=np.sqrt(2*(1-r));C=np.sqrt((1+r)/2);q=row['q'];c=row['c0']
    d=endpoint(row['a'],row['e'],row['beta1'],row['beta2'])[0]
    v=sigma*(d-z/2);g0=-sigma*z/2
    return np.array([v*g0/np.sqrt(q)-c*v*v/(2*q)-c/2,
                     C*(v+g0)/np.sqrt(q)-c*C*v/q,
                     np.full_like(z,C*C/np.sqrt(q)-c*C*C/(2*q))])

def influence_sd(A,B):
    v=influence(B)-influence(A)
    # E H^2=1, E H^4=3; odd terms vanish.
    return np.sqrt(weight@(v[0]**2+v[1]**2+3*v[2]**2+2*v[0]*v[2]))

def main():
    started=time.monotonic();resources();rows=[];contrasts=[]
    for e in [.5,1.05,1.1,2.]:
        for delta in [.01,.05,.1,.2,.25,.3]:
            group=[match(.25,delta,e,*beta) for beta in [(0,.99),(.9,0),(.9,.99),(.4,.7)]]
            rows.extend(group)
            for kind,i,j in [('beta1',0,2),('beta2',1,2)]:
                A,B=group[i],group[j];gap=B['c0']-A['c0'];sd=influence_sd(A,B)
                rho=.1;kappa=.5
                recall=lambda c:ndtr(c/np.sqrt(kappa)-ndtri(1-rho))
                contrasts.append(dict(e=e,delta=delta,kind=kind,gap=gap,recall_gap=recall(B['c0'])-recall(A['c0']),
                                      iid_influence_sd=sd,power_B128_d2048_inflation4=ndtr(abs(gap)*2048*np.sqrt(128)/(4*sd)-ndtri(1-.05/4))))
    result=dict(rows=rows,contrasts=contrasts,resources=resources(),seconds=time.monotonic()-started)
    with (OUT/'AA3_numeric.json').open('x') as f:json.dump(result,f,indent=2)
    for row in contrasts:
        if row['e'] in [.5,2] and row['delta'] in [.1,.25,.3]:print(row,flush=True)
    print('resources',result['resources'],flush=True)

if __name__=='__main__':main()
