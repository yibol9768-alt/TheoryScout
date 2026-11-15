#!/usr/bin/env python3
"""P04 checks. No project imports, API calls, or writes.

certificate: standard-library rational/dyadic interval arithmetic only.
diagnostics: ordinary floating quadrature and finite-array identities.
Run with the resource-limited command in PROOF_EXT_L1L3L4.md.
"""
import argparse
from fractions import Fraction as F
from math import isqrt

BITS = 384
SCALE = 1 << BITS


class I:
    """Closed intervals with outward-rounded integer / 2**BITS endpoints."""
    def __init__(self, x=0, hi=None):
        x = F(x)
        y = x if hi is None else F(hi)
        assert x <= y
        self.lo = (x.numerator * SCALE) // x.denominator
        self.hi = -((-y.numerator * SCALE) // y.denominator)

    @classmethod
    def raw(cls, lo, hi):
        z = object.__new__(cls)
        z.lo, z.hi = lo, hi
        assert lo <= hi
        return z

    @staticmethod
    def cast(x):
        return x if isinstance(x, I) else I(x)

    def __add__(self, y):
        y = I.cast(y)
        return I.raw(self.lo + y.lo, self.hi + y.hi)

    __radd__ = __add__

    def __neg__(self):
        return I.raw(-self.hi, -self.lo)

    def __sub__(self, y):
        return self + -I.cast(y)

    def __rsub__(self, y):
        return I.cast(y) + -self

    def __mul__(self, y):
        y = I.cast(y)
        products = [a*b for a in (self.lo, self.hi)
                    for b in (y.lo, y.hi)]
        return I.raw(min(products)//SCALE, -((-max(products))//SCALE))

    __rmul__ = __mul__

    def reciprocal(self):
        assert not self.lo <= 0 <= self.hi
        return I.raw(SCALE*SCALE//self.hi,
                     -((-SCALE*SCALE)//self.lo))

    def __truediv__(self, y):
        return self * I.cast(y).reciprocal()

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        out, a = I(1), self
        while n:
            if n & 1:
                out = out * a
            a = a * a
            n //= 2
        return out

    def sqrt(self):
        assert self.lo >= 0
        lo = isqrt(self.lo*SCALE)
        hi = isqrt(self.hi*SCALE)
        if hi*hi < self.hi*SCALE:
            hi += 1
        return I.raw(lo, hi)

    def lower(self):
        return F(self.lo, SCALE)

    def upper(self):
        return F(self.hi, SCALE)


def atan_sum(x, degree):
    return sum(((-1)**j * x**(2*j+1)/F(2*j+1)
                for j in range(degree+1)), F(0))


def pi_interval():
    # Machin identity; odd partial sums are lower, even ones upper.
    lower = 16*atan_sum(F(1, 5), 101)-4*atan_sum(F(1, 239), 100)
    upper = 16*atan_sum(F(1, 5), 100)-4*atan_sum(F(1, 239), 101)
    return I(lower, upper)


def exp_negative(s):
    # Taylor's signed integral remainder, valid even before terms decrease.
    assert F(0) <= s <= 72
    x = I(s)
    def poly(n):
        p = I(1)
        for j in range(n, 0, -1):
            p = 1 - x*p/j
        return p
    lower, upper = poly(401), poly(400)
    return I.raw(lower.lo, upper.hi)


def panels(e):
    pending = [(F(0), F(12))]
    result = []
    while pending:
        left, right = pending.pop()
        m, r = (left+right)/2, (right-left)/2
        radius = min((m+e)/2, F(1, 2))
        if r <= radius/4:
            result.append((m, r, radius))
        else:
            pending.append((m, right))
            pending.append((left, m))
    return result


def integrals(e):
    # Simultaneous rigorous enclosure of unnormalized a and q on [0,infinity).
    degree = 48
    total = [I(0), I(0)]
    parts = panels(e)
    for m, r, radius in parts:
        exp_coeff = [exp_negative(m*m/2)]
        exp_coeff.append(-m*exp_coeff[0])
        for j in range(2, degree+1):
            exp_coeff.append((-m*exp_coeff[j-1]-exp_coeff[j-2])/j)
        for k in (1, 2):
            inv = I(m+e).reciprocal()
            rational_coeff = [inv**k]
            for j in range(1, degree+1):
                rational_coeff.append(-rational_coeff[-1]*inv*F(k+j-1, j))
            conv = [sum((exp_coeff[j]*rational_coeff[l-j]
                         for j in range(l+1)), I(0))
                    for l in range(degree+1)]
            value = I(0)
            for l in range(0, degree+1, 2):
                coeff = m*m*conv[l]
                if l >= 1:
                    coeff += 2*m*conv[l-1]
                if l >= 2:
                    coeff += conv[l-2]
                value += coeff * F(2*r**(l+1), l+1)
            # Cauchy's coefficient estimate on |z-m|=radius.
            bound = 2187*(m+radius)**2/(m+e-radius)**k
            ratio = r/radius
            error = 2*r*bound*ratio**(degree+1)/(1-ratio)
            total[k-1] += value + I(-error, error)
    # e**(-72) < 2**(-72); q tail additionally divided by 12.
    total[0] += I(0, F(1, 2**72))
    total[1] += I(0, F(1, 12*2**72))
    return total, len(parts)


def decimal_enclosure(x, digits=22):
    # Display is outward rounded, never converted through a float.
    scale10 = 10**digits
    low = x.lo*scale10//SCALE
    high = -((-x.hi*scale10)//SCALE)
    def fmt(n):
        sign = '-' if n < 0 else ''
        n = abs(n)
        return f'{sign}{n//scale10}.{n%scale10:0{digits}d}'
    return [fmt(low), fmt(high)]


def certificate():
    pi = pi_interval()
    assert pi.lower() > F(8, 3) and pi.upper() < F(32, 9)
    normal = (I(2)/pi).sqrt()
    b = I(8)/(3*pi)
    # Squaring avoids introducing a numerical root solver into the certificate.
    endpoints = [F('0.050156130219'), F('0.050156130220')]
    for index, e in enumerate(endpoints):
        vals, count = integrals(e)
        a, q = normal*vals[0], normal*vals[1]
        assert a.lo > 0 and q.lo > 0
        residual = a*a - b*b*q
        if index == 0:
            assert residual.hi < 0
        else:
            assert residual.lo > 0
        print('certificate_endpoint', str(e), 'panels', count,
              'a^2-b^2*q', decimal_enclosure(residual))
    print('CERTIFIED: 0.050156130219 < e_* < 0.050156130220')
    print('CERTIFIED: rounding e_* to ten digits after decimal gives 0.0501561302')
    print('CERTIFIED: rounding e_* to ten significant digits gives 0.05015613022')


def diagnostics():
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from math import pi, sqrt
    def aq(e):
        k = sqrt(2/pi)
        a = quad(lambda t: k*t*t/(t+e)*np.exp(-t*t/2), 0, np.inf,
                 epsabs=2e-14, epsrel=2e-14)[0]
        q = quad(lambda t: k*(t/(t+e))**2*np.exp(-t*t/2), 0, np.inf,
                 epsabs=2e-14, epsrel=2e-14)[0]
        return a, q
    b = 8/(3*pi)
    root = brentq(lambda e: aq(e)[0]/sqrt(aq(e)[1])-b, .04, .06,
                  xtol=1e-15)
    print('FLOAT_ONLY root', format(root, '.16g'))
    print('FLOAT_ONLY n_star_epsilon_1e-8', (root/1e-8)**2)
    rng = np.random.default_rng(20261106)
    d, n = 11, 37
    x, y = rng.standard_normal((d, n)), rng.standard_normal((d, n))
    z = (y-y.mean(axis=1, keepdims=True)) @ x.T/sqrt(n)
    u = z/(abs(z)+.1)
    h = rng.standard_normal((d, d))
    mask = ~np.eye(n, dtype=bool)
    def stats(mat):
        s = y.T @ mat @ x/d
        off_mean = (s.sum(axis=0)-s.diagonal())/(n-1)
        centered = s-off_mean[None, :]
        return ((s.diagonal()-off_mean).mean(),
                np.mean(centered[mask]**2)), centered
    (mu, vu), cu = stats(u)
    (mh, vh), ch = stats(h)
    tau = .7
    (mt, vt), _ = stats(u+tau*h)
    exact = sqrt(n)/(d*(n-1))*np.sum(u*z)
    np.testing.assert_allclose(mu, exact, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(mt, mu+tau*mh, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(vt, vu+2*tau*np.mean((cu*ch)[mask])+tau*tau*vh,
                               rtol=1e-13, atol=1e-13)
    for gamma in (16, 64, 256):
        t = 1/gamma
        bm, bp = b*(1-t/10-t*t), b*(1-t/10+t*t)
        delta = 52*t**1.5
        assert 1-delta-2*bp*bp*t > 0
        for c in (sqrt(2/pi), .9, 1.):
            center = c/b*(1+(.1-c*c)*t)
            for m in (bm, bp):
                for v in (1-delta, 1+delta):
                    ratio = c*np.sqrt(v)/(m*np.sqrt(1+2*c*c*t))
                    assert abs(ratio-center) <= 60*c/b*t**1.5
    print('Finite-array margin/noise identities and envelope corner checks passed.')
    gamma, c = 16, aq(.1)[0]/sqrt(aq(.1)[1])
    mu = b*(1-1/(10*gamma))
    crossing = (mu*mu*(1+2*c*c/gamma)-c*c)/(c*c-mu*mu)
    print('PREDICTION_CENTER_ONLY gamma16 e0.1 tau_star', sqrt(crossing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['certificate', 'diagnostics', 'all'])
    mode = parser.parse_args().mode
    if mode in ('certificate', 'all'):
        certificate()
    if mode in ('diagnostics', 'all'):
        diagnostics()
