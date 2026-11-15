"""Deterministic polar limits and outward numerical enclosures.

Only writes a NEW output file explicitly requested by --output (exclusive create).
Run under the resource-limited command in README.md. No network or API.
The certificate uses outward binary64 interval arithmetic, including quadrature
and truncation errors; scipy tolerances are used ONLY for exploratory estimates.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import json
import math
from pathlib import Path
import time
import os, resource

RAW_EFFICIENCY=[]
RAW_KERNEL=[]
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


import numpy as np
from scipy.integrate import dblquad, quad
from scipy.optimize import brentq


class I:
    """IEEE-754 round-to-nearest primitives, expanded by one adjacent float.

    All arithmetic is explicit; no BLAS, exp, or polynomial black box is used.
    Array reductions use a separate worst-case summation bound.
    """
    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(lo if hi is None else hi, dtype=np.float64)

    @staticmethod
    def rational(n, d=1):
        z = Q(n, d)
        f = float(z)
        fq = Q.from_float(f)
        return I(np.nextafter(f, -np.inf) if fq > z else f,
                 np.nextafter(f, np.inf) if fq < z else f)

    def __add__(self, b):
        b = box(b)
        return I(down(self.lo+b.lo), up(self.hi+b.hi))
    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, b):
        return self+-box(b)

    def __rsub__(self, b):
        return box(b)+-self

    def __mul__(self, b):
        b = box(b)
        v = np.broadcast_arrays(self.lo*b.lo, self.lo*b.hi,
                                self.hi*b.lo, self.hi*b.hi)
        return I(down(np.minimum.reduce(v)), up(np.maximum.reduce(v)))
    __rmul__ = __mul__

    def __truediv__(self, b):
        b = box(b)
        assert np.all(b.lo > 0), (np.min(b.lo), np.min(b.hi))
        return self*I(down(1/b.hi), up(1/b.lo))

    def __rtruediv__(self, b):
        return box(b)/self

    def sqrt(self):
        assert np.all(self.lo >= 0)
        return I(np.maximum(0, down(np.sqrt(self.lo))), up(np.sqrt(self.hi)))

    def power(self, n):
        assert n >= 0
        ans = I(1)
        for _ in range(n):
            ans = ans*self
        return ans

    def sum(self):
        # Standard |fl(sum)-sum| <= gamma_(n-1) sum |x_i|, with u <= eps.
        n = self.lo.size
        eta = up(n*np.finfo(float).eps)
        eta = up(eta/down(1-eta))
        sl, sh = np.sum(self.lo), np.sum(self.hi)
        # Bound absolute sums themselves by n*max|x_i|, avoiding another sum.
        al = up(n*np.max(np.abs(self.lo)))
        ah = up(n*np.max(np.abs(self.hi)))
        return I(down(sl-up(eta*al)), up(sh+up(eta*ah)))

    def out(self):
        assert self.lo.ndim == self.hi.ndim == 0
        l, h = float(self.lo), float(self.hi)
        return {"lower": l, "upper": h, "lower_hex": l.hex(),
                "upper_hex": h.hex(), "width": h-l}


def box(x):
    return x if isinstance(x, I) else I(x)


def down(x):
    return np.nextafter(x, -np.inf)


def up(x):
    return np.nextafter(x, np.inf)


def pi_interval():
    def atan_q(x):
        s = sum((-1)**j*x**(2*j+1)/Q(2*j+1) for j in range(61))
        return s-x**123/Q(123), s
    a, b = atan_q(Q(1, 5)), atan_q(Q(1, 239))
    l, h = 16*a[0]-4*b[1], 16*a[1]-4*b[0]
    return I(I.rational(l.numerator, l.denominator).lo,
             I.rational(h.numerator, h.denominator).hi)


PI = pi_interval()


def mu_interval(gamma, terms=16):
    c = Q(1, gamma)
    s = sum(Q(math.comb(2*j, j), 4**j)*c**j *
            (Q(1, 2*j+1)+Q(1, 2*j+3)-2*c/Q(2*j+5))
            for j in range(terms+1))
    # Absolute tail: B_j <= 1 and all denominators >= 1.
    tail = (2+2*c)*c**(terms+1)/(1-c)
    lower, upper = s-tail, s+tail
    z = I(I.rational(lower.numerator, lower.denominator).lo,
          I.rational(upper.numerator, upper.denominator).hi)
    return 2*z/PI


def k_interval(gamma, n=2048, batch=32):
    """Positive compact-square integral; full interval rectangles, no tolerance.

    n must be a power of two: grid and initial p=1-x^2 endpoints are exact.
    """
    assert n >= 2 and n & (n-1) == 0
    c = I.rational(1, gamma)
    edges = np.arange(n+1, dtype=np.float64)/n
    pl = 1-edges[1:]**2
    ph = 1-edges[:-1]**2
    r = I(pl[None, :], ph[None, :])
    dr = 1-c+c*r
    nr = (2-r)*dr+c*r*(1-r)
    acc = I(0)
    for start in range(0, n, batch):
        p = I(pl[start:start+batch, None], ph[start:start+batch, None])
        dp = 1-c+c*p
        nn = (2-p)*dp+c*p*(1-p)
        base = p+r-p*r
        denom = base-c*(1-p)*(1-r)*(p+r)
        # Exact monotone lower bound D >= (1-c)(p+r-pr).
        base_lo = (I(p.lo)+I(r.lo)-I(p.lo)*I(r.lo)).lo
        better = ((1-c)*I(np.maximum(0, base_lo))).lo
        dl = np.maximum(denom.lo, better)
        norm = (dp*dr).sqrt()
        numerator = p*r*nn*nr
        corner = dl <= 0
        safe = I(np.where(corner, 1, dl), np.where(corner, 1, denom.hi))
        value = numerator/norm/safe
        # pr/D <= min(p,r)/(1-c), valid even at p=r=0.
        cap = I(np.minimum(p.hi, r.hi))*nn*nr/norm/(1-c)
        vl = np.where(corner, 0, np.maximum(0, value.lo))
        vh = np.where(corner, cap.hi, np.minimum(cap.hi, value.hi))
        assert np.all(vh >= vl)
        contribution=I(vl, vh).sum()/I(n*n)
        RAW_KERNEL.append(dict(gamma=gamma,start=start,stop=min(start+batch,n),integral=contribution.out()))
        acc = acc+contribution
    return 4*c*acc/(PI*PI)


def exp_negative(s):
    # On s/256 in [0, 72/256], P_27 <= exp <= P_26, then 8 squarings.
    assert np.min(s.lo) >= 0 and np.max(s.hi) <= 72.0000000001
    z = s/I(256)
    term, total = I(np.ones_like(s.lo)), I(np.ones_like(s.lo))
    for j in range(1, 28):
        term = term*z/I(j)
        total = total+((-term) if j % 2 else term)
        if j == 26:
            upper = total.hi
    ans = I(np.maximum(0, total.lo), upper)
    for _ in range(8):
        ans = ans*ans
    return ans


def efficiency_interval(e_num, e_den=10**9, degree=40):
    """Self-contained analytic Taylor-integral certificate for c(e).

    The Cauchy remainder and half-normal tail are included explicitly.
    """
    e = I.rational(e_num, e_den)
    pending, cells = [(0., 12.)], []
    while pending:
        l, h = pending.pop()
        m, r = (l+h)/2, (h-l)/2
        radius_lower = min(float((I(m)+I(e.lo)).lo)/2, .5)
        if r <= radius_lower/4:
            cells.append((m, r))
        else:
            pending.extend([(l, m), (m, h)])
    mid, rad = np.array(cells).T
    m, r = I(mid), I(rad)
    # Normalized coefficients of exp(-m*x-x^2/2).
    es = [I(np.ones_like(mid)), -m]
    for j in range(2, degree+1):
        es.append((-m*es[-1]-es[-2])/I(j))
    exp0 = exp_negative(m*m/I(2))
    answers = []
    for power in (1, 2):
        bs = [I(1)/(m+e).power(power)]
        for j in range(1, degree+1):
            bs.append(-bs[-1]*I(power+j-1)/I(j)/(m+e))
        coeff = []
        for j in range(degree+1):
            coef = I(np.zeros_like(mid))
            for k in range(j+1):
                coef = coef+es[k]*bs[j-k]
            coeff.append(coef)
        integ = I(np.zeros_like(mid))
        for j in range(0, degree+1, 2):
            val = m*m*coeff[j]
            if j >= 1:
                val = val+2*m*coeff[j-1]
            if j >= 2:
                val = val+coeff[j-2]
            integ = integ+2*val*r.power(j+1)/I(j+1)
        integ = integ*exp0
        # M >= sup_disk |z^2 (z+e)^-power exp(-z^2/2)|.
        bound = I(2187)*(m+I(.5)).power(2)*(I(2)/(m+I(e.lo))).power(power)
        rem = 2*r*bound*I.rational(1, 4**(degree+1))/I.rational(3, 4)
        total = I(down(integ.lo-rem.hi), up(integ.hi+rem.hi)).sum()
        # exp(-72) < 2^-72; second tail <= exp(-72)/12.
        tail = I.rational(1, 2**72*(1 if power == 1 else 12))
        total = I(total.lo, up(total.hi+tail.hi))*(I(2)/PI).sqrt()
        assert total.lo > 0
        answers.append(total)
    RAW_EFFICIENCY.append(dict(e_numerator=e_num,e_denominator=e_den, A=answers[0].out(), Q=answers[1].out(), cells=len(cells)))
    return answers[0]/answers[1].sqrt()


def mu_estimate(g):
    c = 1/g
    return 2/math.pi*quad(lambda u: (1+u*u-2*c*u**4)/math.sqrt(1-c*u*u),
                         0, 1, epsabs=2e-13)[0]


def k_estimate(g):
    c = 1/g
    def f(y, x):
        p, r = 1-x*x, 1-y*y
        dp, dr = 1-c+c*p, 1-c+c*r
        nn = (2-p)*dp+c*p*(1-p)
        nr = (2-r)*dr+c*r*(1-r)
        d = p+r-p*r-c*(1-p)*(1-r)*(p+r)
        return 4*c/math.pi**2*p*r*nn*nr/math.sqrt(dp*dr)/d
    return dblquad(f, 0, 1, lambda x: 0, lambda x: 1,
                   epsabs=2e-12, epsrel=2e-10)[0]


def efficiency_estimate(e):
    density = math.sqrt(2/math.pi)
    a = density*quad(lambda t: t*t/(t+e)*math.exp(-t*t/2), 0, 12,
                     epsabs=2e-13)[0]
    q = density*quad(lambda t: (t/(t+e))**2*math.exp(-t*t/2), 0, 12,
                     epsabs=2e-13)[0]
    return a/math.sqrt(q)


def root_estimate(target):
    return brentq(lambda e: efficiency_estimate(e)-target, 1e-9, 20,
                  xtol=1e-14)


def root_certificate(target):
    den = 10**9
    lo = math.floor(root_estimate(float(target.lo))*den)-2
    hi = math.ceil(root_estimate(float(target.hi))*den)+2
    cl, ch = efficiency_interval(lo, den), efficiency_interval(hi, den)
    assert cl.hi < target.lo and ch.lo > target.hi
    # Decimal rational endpoints retained exactly, independent of JSON floats.
    return {"lower_numerator": lo, "upper_numerator": hi, "denominator": den,
            "lower": lo/den, "upper": hi/den,
            "c_at_lower": cl.out(), "c_at_upper": ch.out(),
            "target": target.out(), "strict_signs": True}


def run(n, gammas):
    started = time.time()
    results = []
    for g in gammas:
        mu, k = mu_interval(g), k_interval(g, n)
        nu = 1+k/I(g)-mu*mu/I(g*g)
        me, ke = mu_estimate(g), k_estimate(g)
        ve = 1+ke/g-me*me/g**2
        assert mu.lo <= me <= mu.hi and k.lo <= ke <= k.hi
        vlo = 1-mu*mu/I(g*g)
        vhi = vlo+8*(1-mu*mu)/I(g*g)/(1-I.rational(1, g)).power(2)
        assert k.lo >= 0 and nu.lo >= vlo.lo and nu.hi <= vhi.hi
        rows = []
        for tau in (0, 1, 2):
            x = tau*tau
            def target(v):
                return mu*((1+x)/(v+x-2*mu*mu/I(g))).sqrt()
            true_target = target(nu)
            cert = root_certificate(true_target)
            ce = me*math.sqrt((1+x)/(ve+x-2*me*me/g))
            ee = root_estimate(ce)
            low_env = root_estimate(me*math.sqrt((1+x)/
                                  (float(vhi.hi)+x-2*me*me/g)))
            high_env = root_estimate(me*math.sqrt((1+x)/
                                   (float(vlo.lo)+x-2*me*me/g)))
            # Outward certificates for the analytic analytic envelope as well.
            env_low = root_certificate(target(vhi))
            env_high = root_certificate(target(vlo))
            old_halfwidth = 52/g**1.5
            old_lower_target = me*math.sqrt((1+x)/(1+old_halfwidth+x-2*me*me/g))
            old_lower = (0. if old_lower_target <= math.sqrt(2/math.pi)
                         else root_estimate(old_lower_target))
            old_upper_denom = 1-old_halfwidth+x-2*me*me/g
            old_upper_target = me*math.sqrt((1+x)/old_upper_denom) if old_upper_denom > 0 else math.inf
            old_upper = (None if old_upper_target >= 1 else root_estimate(old_upper_target))
            rows.append({"tau": tau, "e_estimate": ee, "e_certificate": cert,
                         "analytic_lower_estimate": low_env, "analytic_upper_estimate": high_env,
                         "analytic_lower_certificate": env_low, "analytic_upper_certificate": env_high,
                         "gap_to_lower": ee-low_env, "gap_to_upper": high_env-ee,
                         "old_52_lower_estimate": old_lower, "old_52_upper_estimate": old_upper})
        results.append({"gamma": g, "mu": mu.out(), "k": k.out(), "nu": nu.out(),
                        "mu_estimate": me, "k_estimate": ke, "nu_estimate": ve,
                        "crossings": rows})
        print(json.dumps({"gamma": g, "k": k.out(), "nu": nu.out(),
                          "e_tau0": rows[0]["e_estimate"],
                          "seconds": time.time()-started}), flush=True)
    return {"method": "positive 2D integral, outward IEEE binary64 rectangles; analytic Taylor c(e)",
            "grid": n, "mu_series_terms": 16, "efficiency_Taylor_degree": 40,
            "pi": PI.out(), "rows": results, "elapsed_seconds": time.time()-started,
            "source_hashes": sources(), "raw_efficiency_integrals":RAW_EFFICIENCY, "raw_kernel_batches":RAW_KERNEL,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=4096)
    parser.add_argument("--gammas", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert all(g >= 16 for g in args.gammas)
    out = Path(args.output)
    assert not out.exists()
    result = run(args.grid, args.gammas)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
