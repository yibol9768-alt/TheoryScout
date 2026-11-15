"""Independent algebra/arithmetic checks, not a statistical certificate."""
from fractions import Fraction as Q
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from polar_tool import I, k_estimate, mu_estimate


def contains(interval, rational):
    assert Q.from_float(float(interval.lo)) <= rational <= Q.from_float(float(interval.hi))


rng = np.random.default_rng(202611063)
for _ in range(1000):
    a, b = [Q(int(rng.integers(-10**6, 10**6)), int(rng.integers(1, 10**6))) for j in range(2)]
    aa, bb = I.rational(a.numerator, a.denominator), I.rational(b.numerator, b.denominator)
    contains(aa+bb, a+b)
    contains(aa*bb, a*b)
    if b > 0:
        contains(aa/bb, a/b)
    if a > 0:
        ss = aa.sqrt()
        assert Q.from_float(float(ss.lo))**2 <= a <= Q.from_float(float(ss.hi))**2


# Independent MP integral check of the closed two-resolvent kernel.
resolvent_checks = []
for g in (16, 64):
    c = 1/g
    a, b = (1-math.sqrt(c))**2, (1+math.sqrt(c))**2
    def mp_expect(f):
        # s=1+c-2 sqrt(c) cos(theta); removes both MP edge roots.
        return quad(lambda theta: 2/math.pi*math.sin(theta)**2 /
                    (1+c-2*math.sqrt(c)*math.cos(theta)) *
                    f(1+c-2*math.sqrt(c)*math.cos(theta)), 0, math.pi,
                    epsabs=2e-13)[0]
    for t, u in ((.01, .03), (.5, 2.), (3., 3.)):
        def param(v):
            p = brentq(lambda p: p*p*(1-c+c*p)/(1-p)-v, 1e-13, 1-1e-13)
            return p, v/p
        p, w = param(t)
        r, v = param(u)
        jj = [mp_expect(lambda s: s**j/((s+w)*(s+v))) for j in (0, 1, 2)]
        kernel_mp = (jj[0]*jj[2]-jj[1]**2)/(p*r*(1-jj[2]))
        d, e = 1-c+c*p, 1-c+c*r
        den = p+r-p*r-c*(1-p)*(1-r)*(p+r)
        kernel_closed = c*(1-p)**2*(1-r)**2/(d*e*p*r*den)
        assert abs(kernel_mp-kernel_closed) < 2e-11
        resolvent_checks.append({"gamma": g, "t": t, "u": u,
                                 "absolute_difference": abs(kernel_mp-kernel_closed)})


# Independent finite matrix mixed-resolvent and square-root checks.
# Conditional on eigenvalues S, the off-diagonal eigenbasis formula avoids
# cancellation of the leading 1 and avoids numerical S inverse.
matrix_rows = []
for g in (16, 64):
    d = 384
    for seed in range(3):
        rg = np.random.default_rng([202611063, g, seed])
        aa = rg.normal(size=(d, g*d))
        ss = aa@aa.T/(g*d)
        s = np.linalg.eigvalsh(ss)
        # H is rotationally invariant, so S may be diagonalized first.
        zz = rg.normal(size=(d, d))
        hh = zz.T@zz/d
        root_s = np.sqrt(s)
        eigen, vec = np.linalg.eigh(root_s[:, None]*hh*root_s[None, :])
        tt = (vec*np.sqrt(np.maximum(eigen, 0)))@vec.T
        weight = (s[:, None]-s[None, :])**2/(s[:, None]*s[None, :])
        sample_k = float(np.sum(weight*tt**2)/(2*d))
        direct = float(np.trace((tt*s[None, :])@tt/s[:, None])/d-np.mean(eigen))
        assert abs(sample_k-direct) < 3e-14
        matrix_rows.append({"gamma": g, "d": d, "seed": seed,
                            "sample_k": sample_k, "limit_k": k_estimate(g),
                            "paired_identity_error": abs(sample_k-direct)})


# First-order coefficient from a separately integrated scalar series.
bb = 8/(3*math.pi)
c = 1/100000
assert abs(k_estimate(100000)/c-(1-bb*bb)) < 2e-6
for g in (16, 32, 64, 128, 256):
    simple = 4/math.pi*quad(lambda u: (1-u*u)*math.sqrt(1-u*u/g), 0, 1)[0]
    assert abs(simple-mu_estimate(g)) < 2e-14


with Path("x3_checks/polar_certificate.json").open() as stream:
    cert = json.load(stream)
for row in cert["rows"]:
    for cross in row["crossings"]:
        ec = cross["e_certificate"]
        assert ec["lower"] <= cross["e_estimate"] <= ec["upper"]
        assert ec["c_at_lower"]["upper"] < ec["target"]["lower"]
        assert ec["c_at_upper"]["lower"] > ec["target"]["upper"]

with Path("x3_checks/audit_results.json").open("x") as stream:
    json.dump({"rational_primitive_checks": 1000,
               "mp_kernel_checks": resolvent_checks, "matrix_checks": matrix_rows,
               "first_coefficient_estimate": k_estimate(100000)/c,
               "first_coefficient_exact": 1-bb*bb,
               "all_crossing_signs_pass": True}, stream, indent=2)
print("PASS: rational arithmetic, MP kernel identities, six matrix identities, asymptotic coefficient, 15 certified crossings")
