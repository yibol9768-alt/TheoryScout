"""Symbolic identities, exact table widening checks, Decimal primitive audit."""
from pathlib import Path
from fractions import Fraction as F
import json, time
import sympy as s
from decimal_enclosures import I, PI, resources

start=time.monotonic();resources();checks={}
r,B,V,Ba,Va,Bb,Vb=s.symbols('r B V Ba Va Bb Vb',real=True)
q=1+2*(1-r)*(V-B);c0=(1-(1-r)*B)/s.sqrt(q);c1=(r+(1-r)*B)/s.sqrt(q)
partial=lambda f,x,y:s.diff(f,B)*x+s.diff(f,V)*y
lam=Ba*(q+r+(1-r)*B)-(r+(1-r)*B)*Va
checks['c1_slope']=s.simplify(partial(c1,Ba,Va)-(1-r)*lam/q**s.Rational(3,2))==0
matched=partial(c0,Bb,Vb)-partial(c0,Ba,Va)*partial(c1,Bb,Vb)/partial(c1,Ba,Va)
checks['matched_jacobian']=s.simplify(matched-(1-r)*(1+r)*(Bb*Va-Ba*Vb)/(s.sqrt(q)*lam))==0
A,Q,Aa,Qa,C,T,a=s.symbols('A Q Aa Qa C T a',real=True)
checks['normalization_at_zero']=s.expand((a*a*C)*(2*a*Q+a*a*Qa)-(A+a*Aa)*(2*a*a*a*T)-a**3*(C*(2*Q+a*Qa)-2*(A+a*Aa)*T))==0
k,l,X,Y=s.symbols('k l X Y')
checks['centering']=s.expand((C-k*T-l*X+k*l*Y)*Y-(X-k*Y)*(T-l*Y)-(C*Y-X*T))==0
M1,M2,M3,M4,e=s.symbols('M1 M2 M3 M4 e')
J=C*(2*Q+a*Qa)-2*(A+a*Aa)*T
checks['small_a_beta1']=s.simplify(J.subs({a:0,A:2*M1,Q:4*M2,C:M2,T:2*M3})+8*(M1*M3-M2*M2))==0
checks['small_a_beta2']=s.simplify(J.subs({a:0,A:2*M1,Q:4*M2,C:-M2+e*M3,T:-2*M3+2*e*M4})-8*(M1*M3-M2*M2-e*(M1*M4-M2*M3)))==0
assert all(checks.values())
def saved(x):return F(x['lo'])/2**x['power'],F(x['hi'])/2**x['power']
main_bounds=[
    [('-4.032e-8','-4.029e-8'),('1.067e-8','1.070e-8')],
    [('-4.330671e-5','-4.330668e-5'),('1.188897e-5','1.188900e-5')],
    [('-8.120238e-4','-8.120237e-4'),('2.352130e-4','2.352131e-4')],
    [('-1.524786e-3','-1.524785e-3'),('4.4982010e-4','4.4982013e-4')],
    [('-7.961e-9','-7.953e-9'),('-1.943e-9','-1.934e-9')],
    [('-8.410353e-6','-8.410343e-6'),('-1.924238e-6','-1.924229e-6')],
    [('-1.525528e-4','-1.525527e-4'),('-3.136997e-5','-3.136995e-5')],
    [('-2.826381e-4','-2.826380e-4'),('-5.597129e-5','-5.597127e-5')]]
table_checks=[]
cert=json.loads(Path('aa3_checks/AA3_points_certificate.json').read_text())
for row,bounds in zip(cert['points'],main_bounds):
    for key,(l,h) in zip(['gap1','gap2'],bounds):
        il,ih=saved(row[key]);ok=F(l)<=il<=ih<=F(h)
        table_checks.append(dict(e=row['e'],delta=row['delta'],key=key,outward=ok));assert ok
near=json.loads(Path('aa3_checks/AA3_near_threshold_certificate.json').read_text())
for row,(l,h) in zip(near['cases'],[('7.235e-11','7.335e-11'),('-2.266e-10','-2.255e-10'),('3.6404e-6','3.6405e-6')]):
    il,ih=saved(row['gap']);ok=F(l)<=il<=ih<=F(h)
    table_checks.append(dict(e=row['e'],delta=row['delta'],key='gap',outward=ok));assert ok

# Check the independent backend against exact rational arithmetic, including
# negative intervals and scales well below the certified minimum effect.
primitive_count=0
values=[F(-7,3),F(-1,10**35),F(0),F(1,10**35),F(3,7),F(12),F(144)]
for x in values:
    for y in values:
        for v,exact in [(I(x)+I(y),x+y),(I(x)-I(y),x-y),(I(x)*I(y),x*y)]:
            assert F(v.l)<=exact<=F(v.h);primitive_count+=1
        if y>0:
            v=I(x)/I(y);assert F(v.l)<=x/y<=F(v.h);primitive_count+=1
    if x>=0:
        v=I(x).sqrt();assert F(v.l)**2<=x<=F(v.h)**2;primitive_count+=1
for x in [F(0),F(1,100),F(3,10),F(1)]:
    # Independent alternating rational series encloses exp(-x).
    lower=sum(((-x)**k/F(s.factorial(k)) for k in range(102)),F(0))
    upper=sum(((-x)**k/F(s.factorial(k)) for k in range(101)),F(0))
    v=I(-x).exp();assert F(v.l)<=lower<=upper<=F(v.h);primitive_count+=1
result=dict(status='PASS',symbolic=checks,table_checks=table_checks,primitive_tests=primitive_count,pi=PI.encode(),resources=resources(),seconds=time.monotonic()-start)
with Path('review21_checks/algebra_and_tables.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
