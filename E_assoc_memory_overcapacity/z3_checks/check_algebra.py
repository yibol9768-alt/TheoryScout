"""Symbolic identities behind Z3; no numerical evidence is used as proof."""
import os
from pathlib import Path
import json
import sympy as sp

assert all(int(os.environ[k]) <= 2 for k in
    ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'))
cg = Path('/sys/fs/cgroup')/Path(next(x.split(':', 2)[2].strip().lstrip('/')
    for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')))
assert (cg/'memory.max').read_text().strip() == '3221225472'
assert (cg/'memory.swap.max').read_text().strip() == '0'

B,V,r,a,e,z,p,t = sp.symbols('B V r a e z p t', real=True)
q = 1-2*(1-r)*B+2*(1-r)*V
c0,c1 = (1-(1-r)*B)/sp.sqrt(q), (r+(1-r)*B)/sp.sqrt(q)
jac = sp.Matrix([c0,c1]).jacobian([B,V]).det()
assert sp.simplify(jac-(1-r)**2*(1+r)/q**2) == 0
Ba,Va = sp.symbols('Ba Va')
dc1 = sp.diff(c1,B)*Ba+sp.diff(c1,V)*Va
Lam = Ba*(q+r+(1-r)*B)-(r+(1-r)*B)*Va
assert sp.simplify(dc1-(1-r)*Lam/q**sp.Rational(3,2)) == 0
A,Q,Aa,Qa,Ab,Qb = sp.symbols('A Q Aa Qa Ab Qb')
J = 2*Ab*Q-A*Qb+a*(Ab*Qa-Aa*Qb)
cross = (a*Ab)*(2*a*Q+a*a*Qa)-(A+a*Aa)*(a*a*Qb)
assert sp.expand(cross-a*a*J) == 0
M1,M2,M3,M4 = sp.symbols('M1 M2 M3 M4')
D = M1*M3-M2*M2
H = D-e*(M1*M4-M2*M3)
assert sp.expand(M1*(M3-e*M4)-M2*(M2-e*M3)-H) == 0
Da = sp.diff(D,M1)*(-M2)+sp.diff(D,M2)*(-2*M3)+sp.diff(D,M3)*(-3*M4)
assert sp.expand(D+e*Da/3-H) == 0
x,y = sp.symbols('x y', positive=True)
expr = (x*y**3+y*x**3-2*x*x*y*y)/2
assert sp.expand(expr-x*y*(x-y)**2/2) == 0
expr4 = (x*y**4+y*x**4-x*x*y**3-y*y*x**3)/2
assert sp.expand(expr4-x*y*(x+y)*(x-y)**2/2) == 0
print(json.dumps(dict(status='PASS', identities=7,
    memory_max=(cg/'memory.max').read_text().strip(),
    swap_max=(cg/'memory.swap.max').read_text().strip(),
    threads=len(list(Path('/proc/self/task').iterdir())))))
