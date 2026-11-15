"""Review 10: independent implementation; never imports the submitted programs.

Run only with the resource-limited command recorded in README.md.
Monotonicity, not the submitted rectangle interval expression, encloses k.
Arbitrary-precision interval Taylor integration independently encloses c(e).
"""
import hashlib
import json
import math
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre
from scipy.optimize import brentq

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'review10_checks'
mp.mp.dps = 75
mp.iv.dps = 55
IV = mp.iv.mpf


def limits(x):
    return mp.mpf(x._mpi_[0]), mp.mpf(x._mpi_[1])


def dump_iv(x):
    a, b = limits(x)
    return [mp.nstr(a, 65), mp.nstr(b, 65)]


# Separate lower/upper primitive expressions; all ndarray operations are expanded.
def add(a, b):
    return (np.nextafter(a[0]+b[0], -np.inf),
            np.nextafter(a[1]+b[1], np.inf))


def neg(a):
    return -a[1], -a[0]


def sub(a, b):
    return add(a, neg(b))


def mul(a, b):
    candidates = [a[i]*b[j] for i in (0, 1) for j in (0, 1)]
    return (np.nextafter(np.minimum.reduce(np.broadcast_arrays(*candidates)), -np.inf),
            np.nextafter(np.maximum.reduce(np.broadcast_arrays(*candidates)), np.inf))


def div(a, b):
    assert np.all(b[0] > 0)
    return mul(a, (np.nextafter(1/b[1], -np.inf),
                   np.nextafter(1/b[0], np.inf)))


def root(a):
    assert np.all(a[0] >= 0)
    return (np.nextafter(np.sqrt(a[0]), -np.inf),
            np.nextafter(np.sqrt(a[1]), np.inf))


def pt(x):
    return (x, x)


def endpoint_ratio(p0, r0, lam):
    """p*r/D is increasing in each argument; evaluate only at exact points."""
    p, r, one = pt(p0), pt(r0), pt(1.)
    d = sub(sub(add(p, r), mul(p, r)),
            mul(pt(lam), mul(mul(sub(one, p), sub(one, r)), add(p, r))))
    zero = (p0 == 0) & (r0 == 0)
    safe = (np.where(zero, 1., d[0]), np.where(zero, 1., d[1]))
    q = div(mul(p, r), safe)
    return (np.where(zero, 0., np.maximum(0., q[0])),
            np.where(zero, 0., q[1]))


def endpoint_factor(p0, lam):
    # N(p)/sqrt(d(p)); N'=4*lam-1-4*lam*p < 0 for these loads.
    p, one = pt(p0), pt(1.)
    d = add(pt(1-lam), mul(pt(lam), p))
    n = add(mul(sub(pt(2.), p), d),
            mul(pt(lam), mul(p, sub(one, p))))
    return div(n, root(d))


def monotone_k(gamma, n=4096):
    assert n <= 2**26 and n & (n-1) == 0
    lam = 1./gamma  # loads are powers of two, hence exact
    edges = np.arange(n+1, dtype=float)/n
    low, high = 1-edges[1:]**2, 1-edges[:-1]**2
    f_low = endpoint_factor(high, lam)[0]
    f_high = endpoint_factor(low, lam)[1]
    accum = IV(0)
    for start in range(0, n, 32):
        sl = slice(start, min(start+32, n))
        qlo = endpoint_ratio(low[sl, None], low[None, :], lam)[0]
        qhi = endpoint_ratio(high[sl, None], high[None, :], lam)[1]
        lo = mul(mul(pt(qlo), pt(f_low[sl, None])), pt(f_low[None, :]))[0]
        hi = mul(mul(pt(qhi), pt(f_high[sl, None])), pt(f_high[None, :]))[1]
        # Worst-case reduction error, bounded independently by N*max(abs(x)).
        count = lo.size
        eps = np.finfo(float).eps
        eta = np.nextafter((count*eps)/(1-count*eps), np.inf)
        elo = np.nextafter(eta*np.nextafter(count*np.max(np.abs(lo)), np.inf), np.inf)
        ehi = np.nextafter(eta*np.nextafter(count*np.max(np.abs(hi)), np.inf), np.inf)
        s_lo = np.nextafter(np.sum(lo)-elo, -np.inf)
        s_hi = np.nextafter(np.sum(hi)+ehi, np.inf)
        accum += IV([float(s_lo), float(s_hi)])/(n*n)
    return 4*IV(lam)*accum/(mp.iv.pi**2)


def mu_box(gamma):
    # Binomial sqrt series in the NEW mean integral, integrated exactly.
    lam = IV(1)/gamma
    coeff, result = IV(1), IV(0)
    for j in range(36):
        if j:
            coeff *= IV(2*j-3)/(2*j)
        result += coeff * lam**j * (IV(1)/(2*j+1)-IV(1)/(2*j+3))
    # absolute binomial coefficients <=1; integrated |1-u^2| <=1.
    tail = lam**36/(1-lam)
    return 4*(result+IV([-1,1])*tail)/mp.iv.pi


def efficiency_box(num, den):
    """Independent interval Taylor integral, degree 28, r/R<=1/8.

    Uses interval exp at the center; no submitted coefficient code is executed.
    """
    e = IV(num)/den
    degree = 28
    pending, panels = [(mp.mpf(0), mp.mpf(12))], []
    e_lower = limits(e)[0]
    while pending:
        left, right = pending.pop()
        center, half = (left+right)/2, (right-left)/2
        radius = min((center+e_lower)/2, mp.mpf('0.5'))
        if half*8 <= radius:
            panels.append((center, half, radius))
        else:
            pending.extend([(left, center), (center, right)])
    integrals = []
    for exponent in (1, 2):
        total = IV(0)
        for center, half, radius in panels:
            m, r = IV(center), IV(half)
            exponential = [mp.iv.exp(-m*m/2)]
            exponential.append(-m*exponential[0])
            for j in range(2, degree+1):
                exponential.append((-m*exponential[-1]-exponential[-2])/j)
            inverse = [(-1)**j * math.comb(exponent+j-1,j)/(m+e)**(exponent+j)
                       for j in range(degree+1)]
            # Include z^2 in the rational coefficients FIRST.
            rational = []
            for j in range(degree+1):
                a = m*m*inverse[j]
                if j >= 1:
                    a += 2*m*inverse[j-1]
                if j >= 2:
                    a += inverse[j-2]
                rational.append(a)
            polynomial_integral = IV(0)
            for j in range(0, degree+1, 2):
                cj = sum((rational[k]*exponential[j-k] for k in range(j+1)), IV(0))
                polynomial_integral += cj * 2*r**(j+1)/(j+1)
            # On this disk |exp(-z^2/2)| <= exp(-m^2/2+mR+R^2/2).
            R = IV(radius)
            magnitude = (m+R)**2/(m+e-R)**exponent * mp.iv.exp(-m*m/2+m*R+R*R/2)
            tail = 2*r*magnitude*(r/R)**(degree+1)/(1-r/R)
            total += polynomial_integral + IV([-1,1])*tail
        total += IV([0,1])*mp.iv.exp(-72)/(1 if exponent == 1 else 12)
        integrals.append(total*mp.iv.sqrt(2/mp.iv.pi))
    return integrals[0]/mp.iv.sqrt(integrals[1])


def quadrature(gamma, order):
    nodes, weights = roots_legendre(order)
    x, weights = (nodes+1)/2, weights/2
    p = 1-x*x
    lam = 1./gamma
    d = 1-lam+lam*p
    nf = ((2-p)*d+lam*p*(1-p))/np.sqrt(d)
    r = p[:, None]
    D = r+p-r*p-lam*(1-r)*(1-p)*(r+p)
    integrand = (r*p/D)*nf[:, None]*nf[None,:]
    k = (4*lam/math.pi**2)*np.dot(weights, integrand@weights)
    mu = 4/math.pi*np.dot(weights, (1-x*x)*np.sqrt(1-lam*x*x))
    return float(mu), float(k), float(1+k/gamma-mu*mu/gamma**2)


def scalar_eff(e):
    a = quad(lambda z: z*z/(z+e)*math.exp(-z*z/2), 0, 15, epsabs=2e-14)[0]
    q = quad(lambda z: z*z/(z+e)**2*math.exp(-z*z/2), 0, 15, epsabs=2e-14)[0]
    return (2/math.pi)**.25*a/math.sqrt(q)


def run():
    started = time.time()
    files = [BASE/'X3_POLAR_TOOL.md', BASE/'X3_PRIOR_COMPARISON.md',
             BASE/'X3_POLAR_APPENDIX_PREVIEW.tex', BASE/'paper/appendix/F.tex',
             BASE/'paper/sections/main_results.tex', BASE/'paper/appendix/B.tex',
             BASE/'paper/appendix/C.tex', BASE/'paper/appendix/K_ft_rank.tex']
    files += sorted((BASE/'x3_checks').glob('*'))
    before = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.is_file()}
    submitted = json.loads((BASE/'x3_checks/polar_certificate.json').read_text())
    output = {'input_hashes': before, 'rows': [], 'independent_source': True,
              'versions': {'numpy': np.__version__, 'mpmath': mp.__version__}}
    for row in submitted['rows']:
        g = row['gamma']
        estimates = {str(n): quadrature(g,n) for n in (128,256,512,1024)}
        mub, kb = mu_box(g), monotone_k(g)
        nub = 1+kb/g-mub**2/(g*g)
        me, ke, ve = estimates['1024']
        assert limits(mub)[0] <= me <= limits(mub)[1] or abs(me-float(limits(mub)[0])) < 2e-14
        assert limits(kb)[0] < ke < limits(kb)[1]
        roots = []
        for crossing in row['crossings']:
            tau = crossing['tau']
            cert = crossing['e_certificate']
            den = cert['denominator']
            target = mub*mp.iv.sqrt((1+tau*tau)/(nub+tau*tau-2*mub*mub/g))
            cl = efficiency_box(cert['lower_numerator'],den)
            ch = efficiency_box(cert['upper_numerator'],den)
            # Recheck submitted enclosures, using our independent high-precision boxes.
            certified_cl = cert['c_at_lower']
            certified_ch = cert['c_at_upper']
            contain_lower = (mp.mpf(certified_cl['lower']) <= limits(cl)[0]
                             and limits(cl)[1] <= mp.mpf(certified_cl['upper']))
            contain_upper = (mp.mpf(certified_ch['lower']) <= limits(ch)[0]
                             and limits(ch)[1] <= mp.mpf(certified_ch['upper']))
            margin_low = limits(target)[0]-limits(cl)[1]
            margin_high = limits(ch)[0]-limits(target)[1]
            assert margin_low > 0 and margin_high > 0
            estimate_target = me*math.sqrt((1+tau*tau)/(ve+tau*tau-2*me*me/g))
            root_est = brentq(lambda e: scalar_eff(e)-estimate_target, .001, 1, xtol=1e-14)
            roots.append({'tau':tau, 'root_estimate':root_est,
                          'submitted_endpoints':[cert['lower_numerator'],cert['upper_numerator'],den],
                          'own_target_box':dump_iv(target), 'own_c_lower_box':dump_iv(cl),
                          'own_c_upper_box':dump_iv(ch),
                          'strict_lower_margin':mp.nstr(margin_low,20),
                          'strict_upper_margin':mp.nstr(margin_high,20),
                          'submitted_c_boxes_contain_own':[contain_lower,contain_upper]})
        output['rows'].append({'gamma':g, 'quadrature_estimates':estimates,
                               'independent_mu_box':dump_iv(mub),
                               'independent_k_box':dump_iv(kb),
                               'independent_nu_box':dump_iv(nub), 'roots':roots})
        print(json.dumps({'gamma':g, 'nu':ve, 'e0':roots[0]['root_estimate'],
                          'nu_box':dump_iv(nub), 'seconds':time.time()-started}), flush=True)
        (OUT/'independent_results.json').write_text(json.dumps(output,indent=2)+'\n')
    after = {str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in files if p.is_file()}
    output['input_hashes_unchanged'] = before == after
    output['elapsed_seconds'] = time.time()-started
    (OUT/'independent_results.json').write_text(json.dumps(output,indent=2)+'\n')
    assert before == after


if __name__ == '__main__':
    run()
