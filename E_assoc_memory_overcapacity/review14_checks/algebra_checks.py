"""Centered noncommutative perturbation expansion, derived from GD recursion."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
          'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    assert os.environ.get(k) == '1'
import json
from pathlib import Path
import sympy as s

# Each word records centered scatter factors. Only degrees <=2 can affect
# the first d/n coefficient. Distinct independent-task scatter factors have
# zero trace contraction; equal pairs contribute d/n + 1/n.
eta = s.symbols('eta', real=True)
def add(*polys):
    out = {}
    for p in polys:
        for w, c in p.items():
            out[w] = s.expand(out.get(w, 0)+c)
    return out
def scale(p, c):
    return {w:s.expand(v*c) for w,v in p.items()}
def mul(*polys):
    out = {():s.Integer(1)}
    for p in polys:
        new = {}
        for w,c in out.items():
            for z,v in p.items():
                if len(w+z) <= 2:
                    new[w+z] = new.get(w+z, 0)+c*v
        out = {w:s.expand(c) for w,c in new.items()}
    return out
def trace_first(p):
    return s.expand(p.get((0,0),0)+p.get((1,1),0))
identity = {():s.Integer(1)}
s0, s1 = {():s.Integer(1),(0,):s.Integer(1)}, {():s.Integer(1),(1,):s.Integer(1)}
step = add(identity, scale(s1,-eta))
a, b = identity, {}
for _ in range(4):
    a = mul(a,step)
    b = add(mul(b,step),scale(identity,eta))
u = a[()]
v = b[()]
q = s.expand(u*u+v*v)
mean_coefficient = trace_first(a)
# Average row energy from Gaussian output pairing and training-key reuse.
row_energy = mul(a,s0,a,s0)
row_energy = add(row_energy,mul(b,s1,b,s0))
energy_coefficient = trace_first(row_energy)
# An additional Wick contraction between a wrong label and the endpoint
# contributes u^2 d/n; centering deletes only order 1/n.
variance_coefficient = s.expand(energy_coefficient+u*u)
M = mean_coefficient/s.sqrt(q)-u*variance_coefficient/(2*q**s.Rational(3,2))
loss_num = mul(add(s1,identity),a,a)
L = trace_first(loss_num)/2
shift = -L/s.diff(u*u,eta)
mu = u/s.sqrt(q)
Mmatched = M+s.diff(mu,eta)*shift
star = 1-s.Rational(3,4)**s.Rational(1,8)
values = {name:float(expr.subs(eta,star).evalf(18)) for name,expr in
          [('eta_star',star),('mu',mu),('M',M),('M_matched',Mmatched),
           ('mean_coefficient',mean_coefficient/s.sqrt(q)),('w',variance_coefficient/q),
           ('L_Lambda',L),('s_eta',shift)]}
out = dict(method='GD recurrence in centered noncommutative scatter words; independent Gaussian trace contractions',
           expressions={k:str(s.factor(p)) for k,p in
             [('A_first',mean_coefficient),('V_first',variance_coefficient),
              ('loss_first',L)]}, values=values)
cg_rel=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
cg=Path('/sys/fs/cgroup')/cg_rel.lstrip('/')
out['controls']={k:(cg/k).read_text().strip() for k in ('memory.max','memory.swap.max','memory.peak')}
out['threads']=len(list(Path('/proc/self/task').iterdir()))
assert out['controls']['memory.max']==str(3*1024**3)
assert out['controls']['memory.swap.max']=='0'
assert out['threads']<=2
with (Path(__file__).parent/'algebra_results.json').open('x') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(out,ensure_ascii=False,indent=2))
