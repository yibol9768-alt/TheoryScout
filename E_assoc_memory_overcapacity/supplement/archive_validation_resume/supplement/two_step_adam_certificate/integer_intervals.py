import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from fractions import Fraction

P = 160
S = 1 << P

def rat(n, d=1):
    return (n*S//d, -((-n*S)//d))

Z = rat(0)
O = rat(1)

def add(a, b): return (a[0]+b[0], a[1]+b[1])
def neg(a): return (-a[1], -a[0])
def sub(a, b): return add(a, neg(b))
def hull(a, b): return (min(a[0], b[0]), max(a[1], b[1]))
def mul(a, b):
    v = [x*y for x in a for y in b]
    return (min(v)//S, -((-max(v))//S))
def sq(a):
    lo = 0 if a[0] <= 0 <= a[1] else min(a[0]**2, a[1]**2)//S
    return (lo, -((-max(a[0]**2, a[1]**2))//S))
def div(a, b):
    assert b[0] > 0, ('nonpositive denominator', b)
    return (min(x*S//y for x in a for y in b),
            max(-((-x*S)//y) for x in a for y in b))
def sqrt(a):
    assert a[0] >= 0, ('negative sqrt', a)
    lo = math.isqrt(a[0]*S)
    hi = math.isqrt(a[1]*S)
    return (lo, hi+(hi*hi < a[1]*S))
def abmax(a): return max(abs(a[0]), abs(a[1]))
def expneg(x):
    # exp(-x/256): alternating Taylor, even partial sum 60 above, 61 below.
    assert 0 <= x[0] <= x[1] <= 72*S
    y = div(x, rat(256))
    acc = O
    term = O
    for k in range(1, 62):
        term = div(mul(term, y), rat(k))
        acc = sub(acc, term) if k % 2 else add(acc, term)
        if k == 60: upper = acc[1]
    ans = (acc[0], upper)
    for _ in range(8): ans = sq(ans)
    return ans
def atan(x):
    term = x
    acc = Z
    for k in range(122):
        v = div(term, rat(2*k+1))
        acc = sub(acc, v) if k % 2 else add(acc, v)
        if k == 120: upper = acc[1]
        term = mul(term, sq(x))
    return (acc[0], upper)

PI = sub(mul(rat(16), atan(rat(1,5))), mul(rat(4), atan(rat(1,239))))
DEN = sqrt(div(rat(2), PI))
TAIL = rat(1, 1 << 60)
def density(z): return mul(DEN, expneg(div(sq(z), rat(2))))
class Jet:
    """Interval Taylor coefficients f^(k)/k!, including interval input centers."""
    def __init__(self, c): self.c = c
    @staticmethod
    def const(x, n): return Jet([x]+[Z]*n)
    def __add__(self, other): return Jet([add(a,b) for a,b in zip(self.c,other.c)])
    def __neg__(self): return Jet([neg(a) for a in self.c])
    def __sub__(self, other): return self+-other
    def __mul__(self, other):
        return Jet([sum_iv([mul(self.c[j],other.c[k-j]) for j in range(k+1)])
                    for k in range(len(self.c))])
    def square(self):
        out = self*self
        out.c[0] = sq(self.c[0])
        return out
    def scale(self, x): return Jet([mul(a,x) for a in self.c])
    def __truediv__(self, other):
        c = []
        for k,a in enumerate(self.c):
            rem = sum_iv([mul(other.c[j],c[k-j]) for j in range(1,k+1)])
            c.append(div(sub(a,rem),other.c[0]))
        return Jet(c)
    def root(self):
        c = [sqrt(self.c[0])]
        for k in range(1,len(self.c)):
            rem = sum_iv([mul(c[j],c[k-j]) for j in range(1,k)])
            c.append(div(sub(self.c[k],rem),mul(rat(2),c[0])))
        return Jet(c)

def sum_iv(values):
    out = Z
    for v in values: out = add(out,v)
    return out

def sources():
    base=Path(__file__).resolve().parents[1]
    folder=Path(__file__).resolve().parent
    names=['integer_intervals.py','adam_integrals.py','stabilizer_ordering.py','matched_chain.py','moment_history.py','history_reversal.py'] if folder.name=='two_step_adam_certificate' else ['polar_variance.py']
    paths=[folder/name for name in names]+sorted((base/'statements').glob('*'))
    return {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def resources():
    return dict(maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                thread_limits={k:os.environ.get(k) for k in
                  ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
