"""Numerical integration diagnostics and independent weighted slope fits."""
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad

from independent_checks import GaussianIntegration, predicted, resources

HERE=Path(__file__).resolve().parent
assert Path.cwd()==HERE.parent
assert not (HERE/'analysis.json').exists()
rng=np.random.default_rng(150015)
gi=GaussianIntegration(100)
fine=GaussianIntegration(180)
checks=[]
# Compare response quadrature and directional derivatives without using any
# asymptotic coefficient. Perturb full Gaussian covariances, holding S fixed.
for d in (8,16,48):
    n=d*d//4
    x=rng.normal(size=(d,n))
    S=x@x.T/n
    for r in (-.5,.0,.7):
        J=r*S-S@S
        K=S@S@S-2*r*S@S+S
        dz=S
        dj=J
        dk=(r*np.eye(d)-S)@S@(r*np.eye(d)-S)
        for method,e in (('Sign',1.),('Adam',.1),('Adam',1.),('Adam',2.)):
            eps=e*math.sqrt(2*(1-r))
            v=gi.row_energy(S,S,J,K,method,.6,eps,[(dz,dj,dk)])
            w=fine.row_energy(S,S,J,K,method,.6,eps,[(dz,dj,dk)])
            step=2e-4
            fp=gi.row_energy(S,S+step*dz,J+step*dj,K+step*dk,method,.6,eps)[0]
            fm=gi.row_energy(S,S-step*dz,J-step*dj,K-step*dk,method,.6,eps)[0]
            err=abs((fp-fm)/(2*step)-v[2][0])
            checks.append(dict(d=d,r=r,method=method,e=e,
                               quadrature_difference=abs(v[0]-w[0]),
                               derivative_quadrature_difference=abs(v[2][0]-w[2][0]),
                               derivative_finite_difference_error=err,tail_bound=v[3]))

# A separate double-density integral of the positive-quadrant bivariate law.
# This check neither expands Hermite polynomials nor samples a Gaussian pair.
bivariate=[]
for rho in (.2,.6,.85):
    v,w,eps=1.7,2.4,.1414213562373095
    den=1-rho*rho
    def outer(x):
        hx=math.sqrt(v)*x/(math.sqrt(v)*x+eps)
        def inner(y):
            hy=math.sqrt(w)*y/(math.sqrt(w)*y+eps)
            plus=math.exp(-(x*x+y*y-2*rho*x*y)/(2*den))
            minus=math.exp(-(x*x+y*y+2*rho*x*y)/(2*den))
            return hx*hy*(plus-minus)/(math.pi*math.sqrt(den))
        return quad(inner,0,12,epsabs=1e-11,epsrel=1e-11)[0]
    direct=quad(outer,0,12,epsabs=2e-10,epsrel=2e-10)[0]
    bd=gi.response_data(np.array([v,w]),'Adam',eps)
    approx=sum(bd[4][0,j]*bd[4][1,j]*rho**(2*j+1) for j in range(32))
    bivariate.append(dict(rho=rho,density_integral=direct,hermite_integral=approx,
                          difference=abs(direct-approx),parseval_tail_bound=math.sqrt(bd[1][0]*bd[1][1])*rho**65))

a=json.loads((HERE/'main.json').read_text())
raw=[]
for row in a['raw_conditional_checks']:
    q=predicted(row['method'],-.35,.63,row['e'],41/25)['q']
    diag_exact=row['all_energy_exact']-4/5*row['contraction_exact']/q
    zcross=(row['contraction_mc']-row['contraction_exact'])/row['contraction_se']
    zall=(row['all_energy_mc']-row['all_energy_exact'])/row['all_energy_se']
    raw.append(dict(method=row['method'],e=row['e'],task=row['task'],
                    contraction_z=zcross,all_energy_z=zall,diagonal_exact=diag_exact,
                    diagonal_mc=row['diagonal_mc'],row_energy_mc=row['row_energy_mc']))

groups={}
for row in a['monte_carlo']:
    key=(row['r'],row['method'],row['e'],row['task'],row['kind'])
    groups.setdefault(key,[]).append(row)
fits=[]
for key,rows in groups.items():
    rows=sorted(rows,key=lambda r:r['d'])
    r,method,e,task,kind=key
    used=[x for x in rows if x['d']>=48]
    ds=np.array([x['d'] for x in used],float)
    ys=np.array([x['mc']-x['limit'] for x in used])
    ses=np.array([x['se'] for x in used])
    if np.max(np.abs(ys))<1e-12:
        fits.append(dict(r=r,method=method,e=e,task=task,kind=kind,exact_zero=True,
                         target=rows[0]['coefficient']))
        continue
    # Fit in rescaled variables d*(E[T]-limit). This exposes the 1/d
    # coefficient as the intercept and reduces the matrix condition number.
    cy=ys*ds; cs=ses*ds
    X=np.array([np.ones_like(ds),48/ds,(48/ds)**2]).T
    A=X/cs[:,None]
    fit=np.linalg.lstsq(A,cy/cs,rcond=None)[0]
    cov=np.linalg.inv(A.T@A)
    residual=cy-X@fit
    # Also allow a d^-3/2 term rather than assuming it vanishes.
    H=np.array([np.ones_like(ds),np.sqrt(48/ds),48/ds]).T
    B=H/cs[:,None]
    fit15=np.linalg.lstsq(B,cy/cs,rcond=None)[0]
    cov15=np.linalg.inv(B.T@B)
    target=rows[0]['coefficient']
    rr=np.array([x['mc']-x['limit']-target/x['d'] for x in used])
    # Descriptive only: no claim that a finite fitted power proves big-O.
    exponent=float(np.polyfit(np.log(ds),np.log(np.maximum(np.abs(rr),1e-30)),1)[0])
    fits.append(dict(r=r,method=method,e=e,task=task,kind=kind,target=target,
                     fitted_1d=float(fit[0]),fitted_1d_se=float(math.sqrt(cov[0,0])),
                     fitted_1d_allow_d15=float(fit15[0]),fitted_1d_allow_d15_se=float(math.sqrt(cov15[0,0])),
                     chi2=float(np.sum((residual/cs)**2)),fit_df=len(ds)-3,
                     descriptive_residual_exponent=exponent,
                     scaled_residuals=[dict(d=x['d'],value=x['scaled_residual'],se=x['scaled_residual_se']) for x in rows]))

gx,gw=np.polynomial.hermite.hermgauss(9)
AA=1+math.sqrt(2)*gx[:,None]
BB=1+math.sqrt(2)*(.5*gx[:,None]+math.sqrt(.75)*gx[None,:])
FF,GG=AA*AA-2,BB*BB-2
WW=np.outer(gw,gw)/math.pi
counter_cov=float(np.sum(WW*FF*GG))
counter_bound=float(.25*math.sqrt(np.sum(WW*FF*FF)*np.sum(WW*GG*GG)))
assert counter_cov>counter_bound+.9

output=dict(resources=resources(),integration_checks=checks,bivariate_checks=bivariate,
            raw_checks=raw,fits=fits,
            noncentered_counterexample=dict(mean_A=1.,mean_B=1.,variance_A=1.,variance_B=1.,rho=.5,
                                            function='F(x)=G(x)=x*x-2',covariance=counter_cov,
                                            stated_bound_if_mean_not_zero=counter_bound,
                                            explanation='4*m*m*rho+2*rho*rho versus rho*rho*(4*m*m+2)'),
            max_quadrature_difference=max(x['quadrature_difference'] for x in checks),
            max_derivative_quadrature_difference=max(x['derivative_quadrature_difference'] for x in checks),
            max_finite_difference_error=max(x['derivative_finite_difference_error'] for x in checks))
with (HERE/'analysis.json').open('x') as f:
    json.dump(output,f,indent=2,ensure_ascii=False)
    f.write('\n')
print(json.dumps({k:v for k,v in output.items() if k.startswith('max_')},indent=2))
for row in fits:
    print(json.dumps(row,ensure_ascii=False))
