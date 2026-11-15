"""Independent symbolic expansion and adversarial scope probes."""
import json
import numpy as np
import sympy as sy
from scipy.optimize import brentq
from numerical import match,raw,signals,save,resources

resources()
r,a=sy.symbols('r a',real=True)
m1,m2,m3,m4,p1,p2,C=sy.symbols('m1 m2 m3 m4 p1 p2 C',real=True)
s=1-r
b1=2*m1
b2=p2*(m2-m3*sy.Symbol('e'))-p1*m2
e=sy.Symbol('e')
v2=4*m2
v3=4*(p2*(m3-e*m4)-p1*m3)
q1=-2*s*b1; q2=2*s*(v2-b2); q3=2*s*(v3-C)
inv=[sy.Integer(1),-q1/2,-q2/2+3*q1**2/8,-q3/2+3*q1*q2/4-5*q1**3/16]
c02=inv[2]-s*b1*inv[1]-s*b2
c03=inv[3]-s*b1*inv[2]-s*b2*inv[1]-s*C
c11=s*b1+r*inv[1]
c12=s*b2+s*b1*inv[1]+r*inv[2]
matched3=c03/c11**3-2*c02*c12/c11**4
D=m1*m3-m2**2; H=D-e*(m1*m4-m2*m3)
K=(1-r)/(2*(1-r*r)**3*m1**4)
checks={
    'initial_slope':sy.simplify(c11-2*(1-r*r)*m1)==0,
    'unknown_third_endpoint_term_cancels':sy.simplify(sy.diff(matched3,C))==0,
    'matched_p1_coefficient':sy.simplify(sy.diff(matched3,p1)-K*D)==0,
    'matched_p2_coefficient':sy.simplify(sy.diff(matched3,p2)+K*H)==0,
    'quadratic_term_history_free':sy.simplify(sy.diff(c02/c11**2,p1))==0 and sy.simplify(sy.diff(c02/c11**2,p2))==0,
}
B,V=sy.symbols('B V',real=True)
q=1+2*s*(V-B); c0=(1-s*B)/sy.sqrt(q); c1=(r+s*B)/sy.sqrt(q)
checks['jacobian']=sy.simplify(sy.Matrix([c0,c1]).jacobian([B,V]).det()-s*s*(1+r)/q**2)==0
assert all(checks.values())

# Compare endpoint formula to raw recursion, including sign-changing residuals.
rng=np.random.default_rng(17003); errors=[]
z=np.r_[np.geomspace(1e-8,10,80),[.125,.5,2]]
for j in range(100):
    aa,ee=rng.uniform(0,2),rng.uniform(.1,3)
    bb1,bb2=rng.uniform(0,.999,2)
    if j==0: aa,ee,bb1,bb2=.5,.5,0,0
    got,_=raw(z,aa,ee,bb1,bb2)
    t=aa/(z+ee)
    predicted=aa*z*(1/(z+ee)+(1-t/(1+bb1))/(z*np.sqrt((bb2+(1-t)**2)/(1+bb2))+ee))
    errors.append(float(np.max(abs(got-predicted))))
assert max(errors)<1e-12

finite_thresholds=[]
for rr in [-.5,.25,.5]:
    for delta in [.01,.1,.25]:
        if rr+delta>=1: continue
        root=brentq(lambda ee:match(rr,rr+delta,ee,.4,.7)['matched_derivatives'][1],.8,1.8,xtol=1e-9)
        finite_thresholds.append(dict(r=rr,delta=delta,beta=[.4,.7],derivative_zero_e=root))

scope_example=[match(.25,.5,1.1,.9,b2) for b2 in [0,.99]]
out=dict(symbolic=checks,max_endpoint_absolute_error=max(errors),endpoint_cases=100,
         finite_target_thresholds=finite_thresholds,finite_target_scope_example=scope_example,resources=resources())
save('algebra_checks.json',out)
print(json.dumps(out,indent=2))
