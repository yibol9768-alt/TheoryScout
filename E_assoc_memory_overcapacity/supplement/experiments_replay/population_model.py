"""Population first-root matching and the candidate finite-size correction."""
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr,ndtri
METHODS=['GD','Adam2','Adam1','Adam.1','Sign']
CORRELATIONS=[.5,0.,-.5]
PAIRS=[('GD','Sign'),('Adam2','Adam.1'),('Adam.1','Sign')]

def expect(f):return quad(lambda z:math.sqrt(2/math.pi)*math.exp(-z*z/2)*f(z),0,np.inf,epsabs=2e-12,epsrel=2e-12)[0]
def population():
    rows=[]
    for ir,r in enumerate(CORRELATIONS):
        sigma=math.sqrt(2*(1-r))
        for method in METHODS:
            if method=='GD':A=Q=1.
            elif method=='Sign':A=math.sqrt(2/math.pi);Q=1.
            else:
                e={'Adam2':2.,'Adam1':1.,'Adam.1':.1}[method]
                A=expect(lambda z:z*z/(z+e));Q=expect(lambda z:(z/(z+e))**2)
            c=A/math.sqrt(Q);k=sigma*c/2;B=(1+r)*k/math.sqrt(1-k*k);M=r*r+B*B
            c0=(r*.68+B*math.sqrt(M-.68**2))/M
            theta=math.acos(c0);magnitude=math.sin(theta)/(math.sqrt(1-k*k)*c0+k*math.sin(theta))
            alpha=magnitude/(sigma if method=='GD' else math.sqrt(Q))
            variance=1-2*k*magnitude+magnitude*magnitude
            assert abs((r+k*magnitude)/math.sqrt(variance)-.68)<1e-12
            rows.append(dict(ir=ir,r=r,method=method,A=A,Q=Q,amplitude=alpha,q=variance,
                             old_recall=float(ndtr(2*c0+ndtri(.1))),new_recall=float(ndtr(1.36+ndtri(.1)))))
    return rows

def response(method,sigma):
    if method=='GD':return 1.,0.,sigma*sigma,1.
    if method=='Sign':
        b=math.sqrt(2/math.pi)/sigma;return b,-b/(2*sigma*sigma),1.,0.
    e={'Adam.1':.1,'Adam1':1.,'Adam2':2.}[method]
    A=expect(lambda z:z*z/(z+e));A3=expect(lambda z:z**4/(z+e))
    Q=expect(lambda z:(z/(z+e))**2);J=expect(lambda z:z*z*(z/(z+e))**2)
    return A/sigma,(A3-3*A)/(2*sigma**3),Q,(J-Q)/(2*sigma*sigma)

def predictions(params):
    result=[];kappa=.25;Z=float(ndtri(.9))
    for p in params:
        r=p['r'];alpha=p['amplitude'];sigma=math.sqrt(2*(1-r));b=3-2*r
        beta,bp,Q,Qp=response(p['method'],sigma)
        variance=1-alpha*sigma*sigma*beta+alpha*alpha*Q
        ell=(1+2*alpha*(beta*(r-3)+(r-1)*b*bp)+alpha*alpha*(2*sigma*sigma*beta*beta+b*Qp))/(kappa*variance)
        for task,r0,r1 in [('old',1.,r),('new',r,1.)]:
            ak=r0+alpha*(r1-r0)*beta;mu=ak/math.sqrt(kappa*variance)
            m=alpha*((r1-r0)*b*bp-r0*beta)/(kappa**1.5*math.sqrt(variance));w=ell+mu*mu;t=Z-mu
            correction=math.exp(-t*t/2)/math.sqrt(2*math.pi)*(m-mu*w/2-mu/2-mu*mu*t/2)
            for d in [128,256,512]:
                limit=float(ndtr(mu-Z))
                result.append(dict(d=d,ir=p['ir'],method=p['method'],task=task,mu=mu,m=m,w=w,q=variance,
                                   B_recall=correction,recall_limit=limit,recall_first_order=limit+correction/d))
    return result
