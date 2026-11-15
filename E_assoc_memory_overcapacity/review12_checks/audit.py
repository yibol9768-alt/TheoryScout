"""Independent review: no imports from y2_checks, no execution of audited code."""
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from environment import resources


class Exact:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = F(lo), F(lo if hi is None else hi)
        assert self.lo <= self.hi
    def __add__(self, other):
        other = ex(other)
        return Exact(self.lo + other.lo, self.hi + other.hi)
    __radd__ = __add__
    def __neg__(self):
        return Exact(-self.hi, -self.lo)
    def __sub__(self, other):
        return self + -ex(other)
    def __rsub__(self, other):
        return ex(other) + -self
    def __mul__(self, other):
        other = ex(other)
        p = [x*y for x in (self.lo, self.hi) for y in (other.lo, other.hi)]
        return Exact(min(p), max(p))
    __rmul__ = __mul__
    def square(self):
        return Exact(0 if self.lo <= 0 <= self.hi else min(self.lo**2, self.hi**2), max(self.lo**2, self.hi**2))
    def floats(self):
        return [float(self.lo), float(self.hi)]


def ex(x):
    return x if isinstance(x, Exact) else Exact(x)


def decode(x):
    assert x['power'] == 160
    return Exact(F(int(x['lo']), 2**160), F(int(x['hi']), 2**160))


def dyadic_range(bounds):
    a, b = map(F, bounds)
    return Exact(F(a.numerator*2**160//a.denominator, 2**160), F(-((-b.numerator*2**160)//b.denominator), 2**160))


def contained(inner, outer):
    assert outer.lo <= inner.lo <= inner.hi <= outer.hi, (inner.floats(), outer.floats())


def stats(a, values):
    A, Q, Aa, Qa, Ae, Qe = values
    B, V = a*A, a.square()*Q
    Ba, Va = A+a*Aa, 2*a*Q+a.square()*Qa
    N = Ba*(1-B+2*V)-B*Va
    J = 2*Ae*Q-A*Qe+a*(Ae*Qa-Aa*Qe)
    q = 1-2*B+2*V
    return dict(N=N, J=J, B=B, V=V, q=q, H=B.square()-F(4,25)*q)


def cover_1d(rows, start, end):
    cursor = F(start)
    for a, b in sorted((F(a), F(b)) for a, b in rows):
        assert a == cursor and b > a, (a, b, cursor)
        cursor = b
    assert cursor == F(end)


def w_sens(z, a, e, betas=(.9,.99)):
    """Run two moments in units of z. Dual arrays are [value, d/da, d/de]."""
    b1, b2 = betas
    # d/z = a*w/z, so its residual divided by z is a*coefficient-1.
    coeff = np.zeros(3)
    m = np.zeros(3)
    v = np.zeros(3)
    for t in (1, 2):
        g = a*coeff
        g[0] -= 1
        g[1] += coeff[0]
        m = b1*m+(1-b1)*g
        g2 = np.array([g[0]**2, 2*g[0]*g[1], 2*g[0]*g[2]])
        v = b2*v+(1-b2)*g2
        mh, vh = m/(1-b1**t), v/(1-b2**t)
        root = math.sqrt(vh[0])
        denom = z*np.array([root, vh[1]/(2*root), vh[2]/(2*root)])
        denom[0] += e
        denom[2] += 1
        direction = mh/denom[0]
        direction[1:] -= mh[0]*denom[1:]/denom[0]**2
        coeff -= direction
    return z*coeff


def moments(a, e, betas=(.9,.99)):
    def integrand(z):
        w, wa, we = w_sens(z, a, e, betas)
        return np.array([z*w, w*w, z*wa, 2*w*wa, z*we, 2*w*we])*math.sqrt(2/math.pi)*math.exp(-z*z/2)
    result, err = quad_vec(integrand, 0, 12, epsabs=3e-12, epsrel=3e-12)
    return result, float(err)


def signals(a, e, betas=(.9,.99)):
    vals, _ = moments(a, e, betas)
    B, V = a*vals[0], a*a*vals[1]
    return (1-B)/math.sqrt(1-2*B+2*V), B/math.sqrt(1-2*B+2*V)


def direct(z, a, e, mode='actual'):
    d = np.zeros_like(z)
    m = np.zeros_like(z)
    v = np.zeros_like(z)
    for t in (1,2):
        g = d-z
        if mode == 'reset':
            m[:] = 0; v[:] = 0
        m = .9*m+.1*g
        v = .99*v+.01*g*g
        mh = m/(1-.9**t) if mode != 'no_bias' else m
        vh = v/(1-.99**t) if mode != 'no_bias' else v
        denominator = np.sqrt(vh+e) if mode == 'eps_inside' else np.sqrt(vh)+e
        d -= a*mh/denominator
    return d


def main():
    watched = [Path('Y2_MULTISTEP.md')] + sorted(Path('y2_checks').glob('*'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched if p.is_file()}
    with open('review12_checks/input_sha256.json', 'x') as f:
        json.dump(hashes, f, indent=2)
    reg, pts, hist = [json.loads(Path('y2_checks/'+n+'.json').read_text()) for n in ('region','points','history')]
    rectangles = []
    minimum = {'N': 1e20, 'J': 1e20}
    for row in reg['leaves']:
        al, ar = map(F,row['a']); el, er = map(F,row['e'])
        assert 0 <= al < ar <= F(1,2) and F(1,10) <= el < er <= 1
        rectangles.append((al,ar,el,er))
        st = stats(dyadic_range(row['a']), list(map(decode,row['integrals'])))
        for name in ('N','J'):
            stored = decode(row['bounds'][name])
            contained(st[name], stored)
            assert stored.lo > 0
            minimum[name] = min(minimum[name], float(stored.lo))
    edges = sorted({v for box in rectangles for v in box[:2]})
    assert edges[0] == 0 and edges[-1] == F(1,2)
    for al, ar in zip(edges[:-1], edges[1:]):
        mid = (al+ar)/2
        cover_1d([(el,er) for l,r,el,er in rectangles if l < mid < r], F(1,10), 1)
    assert sum((ar-al)*(er-el) for al,ar,el,er in rectangles) == F(9,20)
    cover_1d([r['e'] for r in reg['edge']], F(1,10), 1)
    edge_lower = 1e20
    for row in reg['edge']:
        st = stats(Exact(F(1,2)), list(map(decode,row['integrals'])))
        H = st['B'].square()-F(1,4)*st['q']
        assert st['B'].lo > 0 and H.lo > 0
        contained(H, decode(row['H5']))
        edge_lower = min(edge_lower, float(H.lo))
    roots = []
    for row in pts['points']+hist['models']:
        betas = tuple(float(F(b)) for b in row.get('beta',('9/10','99/100')))
        if 'derivative_slabs' in row:
            cover_1d([s['a'] for s in row['derivative_slabs']],0,F(2,5))
            for slab in row['derivative_slabs']:
                st = stats(dyadic_range(slab['a']), list(map(decode, slab['integrals'])))
                assert st['N'].lo > 0
                contained(st['N'], decode(slab['N']))
        for index, endpoint in enumerate(row['endpoints']):
            assert F(endpoint['a']) == F(row['a'][index])
            st = stats(dyadic_range([endpoint['a']]*2), list(map(decode, endpoint['integrals'])))
            assert st['B'].lo > 0
            assert st['H'].hi < 0 if index == 0 else st['H'].lo > 0
        a = sum(map(lambda x: float(F(x)),row['a']))/2
        e = float(F(row['e']))
        c0,c1 = signals(a,e,betas)
        c0_iv = decode(row['c0'])
        assert float(c0_iv.lo) < c0 < float(c0_iv.hi)
        roots.append(dict(beta=betas,e=e,a=a,c0=c0,c1=c1))
    z = np.r_[np.linspace(-12,12,1001), -1e-10, 0, 1e-10]
    recursion_errors = []
    for a,e in ((.5,.1),(.2233914125,.1),(.3806017005,1)):
        zz=np.abs(z); t=a/(zz+e); L=np.sqrt((.99+(1-t)**2)/1.99)
        explicit = a*z*(1/(zz+e)+(1-t/1.9)/(zz*L+e))
        error=float(np.max(np.abs(direct(z,a,e)-explicit)))
        assert error < 1e-13
        recursion_errors.append(dict(a=a,e=e,actual_error=error,variant_max_errors={mode:float(np.max(np.abs(direct(z,a,e,mode)-explicit))) for mode in ('reset','no_bias','eps_inside')}))
    sweep=[]
    for u in (1e-5,.01,.1,.4,.5):
        previous=-math.inf
        for e in (.1,.2,.5,.8,1):
            root=brentq(lambda a: signals(a,e)[1]-u,0,.5,xtol=1e-14)
            c0,c1=signals(root,e)
            vals,err=moments(root,e)
            A,Q,Aa,Qa,Ae,Qe=vals
            B,V=root*A,root*root*Q
            N=(A+root*Aa)*(1-B+2*V)-B*(2*root*Q+root*root*Qa)
            J=2*Ae*Q-A*Qe+root*(Ae*Qa-Aa*Qe)
            slope=root*root*J/(math.sqrt(1-2*B+2*V)*N)
            assert c0 > previous and slope > 0
            previous=c0
            sweep.append(dict(u=u,e=e,a=root,c0=c0,c1=c1,J=J,N=N,matched_slope=slope,quad_error=err))
    result=dict(status='PASS',rectangle_count=len(rectangles),a_strips=len(edges)-1,exact_area='9/20',gap_or_overlap=False,minimum_stored_lower=minimum,minimum_H5=edge_lower,roots=roots,recursion_comparison=recursion_errors,numerical_sweep=sweep,resources=resources())
    assert hashes == {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}
    with open('review12_checks/audit.json','x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('numerical_sweep',)},indent=2),flush=True)


if __name__ == '__main__':
    main()
