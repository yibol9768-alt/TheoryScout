#!/usr/bin/env python3
"""Independent, bounded counterexample searches; stdout only, no APIs or imports of project code.

Run from the project root with the systemd scope documented in PROOF_REVIEW.md.
Numerical nonviolations are not proofs. Exact Fraction checks are identified separately.
"""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import math
import resource
import time
from fractions import Fraction as F
from math import factorial

import numpy as np
from scipy.special import ndtr

START = time.monotonic()
DEADLINE_SECONDS = 240
RNG = np.random.default_rng(20261106)
RESULT = {"seed": 20261106, "threads_requested": 1, "deadline_seconds": DEADLINE_SECONDS}


def tick():
    if time.monotonic() - START > DEADLINE_SECONDS:
        raise TimeoutError("Local review deadline exceeded")


def require_le(left, right, label, tol=2e-9):
    if left > right + tol * max(1.0, abs(left), abs(right)):
        raise AssertionError((label, float(left), float(right)))


def exact_l5():
    count = 0

    def lt(x, y):
        nonlocal count
        x, y = F(x), F(y)
        assert x.numerator * y.denominator < y.numerator * x.denominator
        count += 1

    def arctan(N, x):
        return sum(((-1)**k * x**(2*k+1)/(2*k+1) for k in range(N+1)), F(0))

    def exp_poly(N, s):
        return sum(((-s)**k/factorial(k) for k in range(N+1)), F(0))

    def integral(N, a):
        return sum(((-1)**k*a**(2*k+1)/(2**k*factorial(k)*(2*k+1))
                    for k in range(N+1)), F(0))

    pm, pp = F(3141592653589, 10**12), F(3141592653590, 10**12)
    km, kp = F(3989422804013, 10**13), F(3989422804016, 10**13)
    lt(pm, 16*arctan(11, F(1, 5))-4*arctan(10, F(1, 239)))
    lt(16*arctan(10, F(1, 5))-4*arctan(11, F(1, 239)), pp)
    lt(2*pp*km**2, 1)
    lt(1, 2*pm*kp**2)
    lt(F(9, 8)**21/factorial(21), F(1, 10**18))
    lt(F(3, 2)**43/(2**21*factorial(21)*43), F(1, 10**20))
    intervals = []
    for a, low, high, floor, ceiling in [
        (F(3, 5), F('0.89986274'), F('0.89986275'),
         F('0.809752952916164'), F('0.809752952917884')),
        (F(3, 2), F('0.70865055'), F('0.70865057'),
         F('0.502185610785213'), F('0.502185610788418')),
    ]:
        s = a*a/2
        fm, fp = km*exp_poly(21, s), kp*exp_poly(20, s)
        bm, bp = F(1, 2)-kp*integral(20, a), F(1, 2)-km*integral(21, a)
        for x in (exp_poly(21, s), integral(21, a), fm, bm):
            lt(0, x)
        lt(fm, fp)
        lt(bm, bp)
        lower, upper = 2*fm*fm/bp, 2*fp*fp/bm
        assert lower <= upper
        lt(low*low, lower)
        lt(upper, high*high)
        lt(floor, lower)
        lt(upper, ceiling)
        intervals.append([float(a), float(low), float(high)])
    lt(F('0.84882636315'), 8/(3*pp))
    lt(8/(3*pm), F('0.84882636317'))
    lt(F('0.79788456080'), 2*km)
    lt(2*kp, F('0.79788456081'))
    return {"strict_rational_comparisons": count, "non_strict_comparisons": 2,
            "c2_intervals": intervals}


def covariance_search():
    count, worst_cov, worst_rem = 0, 0.0, 0.0
    for d in (2, 4, 8, 16, 32):
        for delta in (1e-4, .01, .1, .25):
            for _ in range(8):
                tick()
                raw = RNG.normal(size=(d, d))
                raw = (raw+raw.T)/2
                S = np.eye(d)+delta*raw/np.linalg.norm(raw, 2)
                v = np.diag(S)
                R = S/np.sqrt(v[:, None]*v[None, :])
                dr = np.linalg.norm(R-np.eye(d), 2)
                for name in ("sign", "cubic"):
                    if name == "sign":
                        K = 2/np.pi*np.arcsin(np.clip(R, -1, 1))
                        q, Q = 1., 1.
                        L = .5*np.sqrt(2/np.pi)*(.75)**(-1.5)
                        beta = np.full(d, np.sqrt(2/np.pi))
                    else:
                        K = 9*v[:, None]*v[None, :]*S+6*S**3
                        q, Q, L = 15., 15*(1.25)**3, 45*(1.25)**2+3
                        beta = 3*v**1.5
                    error = K-beta[:, None]*beta[None, :]*R
                    np.fill_diagonal(error, 0.)
                    rem = np.linalg.norm(error, 2)
                    require_le(rem, Q*dr**3, "L2 Hermite remainder")
                    lhs = np.linalg.norm(K-q*np.eye(d), 2)
                    rhs = (L+4*Q)*delta
                    require_le(lhs, rhs, "L2 covariance bound")
                    worst_cov = max(worst_cov, lhs/rhs)
                    worst_rem = max(worst_rem, rem/(Q*dr**3))
                    count += 1
    # Exact Gaussian moments for T=sqrt(b)V+sqrt(1-b)epsilon, F=V**k.
    def moment(p):
        return 0 if p % 2 else math.prod(range(1, p, 2))
    regressions, worst_reg = 0, 0.
    for k in (1, 3, 5, 7):
        for b in (0., 1e-8, .001, .1, .9, 1.):
            correction = b*(moment(2*k+2)-moment(2*k)-moment(k+1)**2)
            rhs = b*(math.sqrt(2*moment(4*k))+moment(2*k))
            require_le(abs(correction), rhs, "L2 regression variance correction")
            if rhs:
                worst_reg = max(worst_reg, abs(correction)/rhs)
            regressions += 1
    return {"covariance_cases": count, "max_covariance_bound_ratio": worst_cov,
            "max_remainder_bound_ratio": worst_rem, "regression_cases": regressions,
            "max_regression_bound_ratio": worst_reg}


def sign_search():
    count, worst = 0, 0.
    for d in (3, 7):
        for alpha in (.2, .6, 1.5):
            for delta in (0., 1e-8, .01, .2, .8, 1.5):
                for _ in range(5):
                    tick()
                    raw = RNG.normal(size=(d, d))
                    raw = (raw+raw.T)/2
                    S = np.eye(d)+delta*raw/np.linalg.norm(raw, 2)
                    eps = np.linalg.norm(S-np.eye(d), 2)
                    Z = alpha*RNG.integers(-5, 6, size=(d, d)).astype(float)
                    Z += RNG.choice([0., 1e-10, -1e-10, .1], size=(d, d))
                    U, H = np.zeros_like(Z), np.zeros_like(Z)
                    bad = np.zeros_like(Z, dtype=bool)
                    small_sums = {eta: 0. for eta in (1e-4, .1, 1.)}
                    for t in range(6):
                        old = bad.mean()
                        residual = H-Z
                        forcing = U@(S-np.eye(d))
                        next_U = U-alpha*np.sign(U@S-Z)
                        next_H = H-alpha*np.sign(residual)
                        new_bad = bad | (next_U != next_H)
                        for eta in small_sums:
                            small = np.mean(np.abs(residual) <= eta)
                            require_le(new_bad.mean(), old+small+
                                       np.linalg.norm(forcing, 'fro')**2/(d*d*eta**2),
                                       "L3 first disagreement")
                            if t >= 1:
                                small_sums[eta] += small
                            T = t+1
                            rhs = 4*T*T*alpha**2*(small_sums[eta] +
                                  alpha**2*eps**2/eta**2*sum(j*j for j in range(1, T)))
                            lhs = np.mean((next_U-next_H)**2)
                            require_le(lhs, rhs, "L3 total SignGD error")
                            if rhs:
                                worst = max(worst, lhs/rhs)
                            count += 1
                        U, H, bad = next_U, next_H, new_bad
    return {"inequality_pairs": count, "max_total_error_bound_ratio": worst}


def weights(beta, t):
    if beta == 0:
        return np.r_[np.zeros(t-1), 1.]
    a = beta**np.arange(t-1, -1, -1, dtype=float)
    return a/a.sum()


def adam_search():
    count, worst = 0, 0.
    root_checks = []
    for beta1, beta2 in ((0., 0.), (.9, 0.), (.999, 0.), (0., .999), (.9, .999), (.999, .01)):
        for alpha in (.01, .6, 2.):
            for e in (.001, .05, 1.):
                tick()
                c, gamma = math.sqrt(1-beta2), 1+2*alpha/e
                zgrid = [-1e5, -10., -alpha, -1e-8, 0., 1e-8, alpha, 10., 1e5]
                if alpha > e:
                    zgrid.extend([alpha-e, alpha-e-1e-10, alpha-e+1e-10])
                for z in zgrid:
                    h, history = 0., []
                    for t in range(1, 7):
                        history.append(h-z)
                        s = np.array(history)
                        a, b = weights(beta1, t), weights(beta2, t)

                        def update(r):
                            return np.dot(a, r)/(np.linalg.norm(np.sqrt(b)*r)+e)

                        ref = update(s)
                        J = gamma**(t-1)/c
                        L = (1+J)/e
                        require_le(abs(ref), J, "L3 reachable Adam update")
                        for scale in (1e-10, 1e-5, .01, 1., 1e3, 1e6):
                            r = s+scale*RNG.normal(size=t)
                            lhs = abs(update(r)-ref)
                            rhs = L*np.linalg.norm(r-s)
                            require_le(lhs, rhs, "L3 Adam history perturbation")
                            if rhs:
                                worst = max(worst, lhs/rhs)
                            count += 1
                        lhs = abs(update(np.zeros(t))-ref)
                        require_le(lhs, L*np.linalg.norm(s), "L3 zero second moment")
                        h -= alpha*ref
                        H = e*(gamma**t-1)/(2*c)
                        require_le(abs(h), H, "L3 bounded Adam proxy")
                        if beta1 == .9 and beta2 == 0 and alpha == .6 and e == .05 and t == 2 and z == alpha-e:
                            root_checks.append({"z": z, "current_residual": history[-1],
                                                "corrected_update": ref, "h2": h, "J2": J})
    # This stronger global claim is FALSE, but the submitted proof does not use it.
    global_history = []
    a, b, e = weights(.9, 2), weights(0., 2), .1
    for R in (1., 100., 10000.):
        s, r = np.array([R, 0.]), np.array([R, 1e-8])
        fun = lambda x: np.dot(a, x)/(np.linalg.norm(np.sqrt(b)*x)+e)
        global_history.append([R, abs(fun(r)-fun(s))/np.linalg.norm(r-s)])
    return {"history_perturbation_cases": count, "max_bound_ratio": worst,
            "beta2_zero_current_residual_root": root_checks,
            "counterexample_to_unused_global_lipschitz": global_history}


def gd_and_measurement_search():
    count, worst = 0, 0.
    for eta in (.001, .5, 1., 1.9, 1.999):
        for T in (1, 2, 3, 8):
            for delta in (0., 1e-5, .1, .5):
                d = 7
                raw = RNG.normal(size=(d, d))
                raw = (raw+raw.T)/2
                S = np.eye(d)+delta*raw/np.linalg.norm(raw, 2)
                eps = np.linalg.norm(S-np.eye(d), 2)
                Z, U = RNG.normal(size=(d, d)), np.zeros((d, d))
                Q, power = np.zeros((d, d)), np.eye(d)
                for _ in range(T):
                    U -= eta*(U@S-Z)
                    Q += eta*power
                    power = power@(np.eye(d)-eta*S)
                require_le(np.linalg.norm(U-Z@Q, 'fro'), 0., "GD exact polynomial", 1e-7)
                k = eta*sum((1-eta)**j for j in range(T))
                assert k > 0
                radius = max(abs(1-eta), np.linalg.norm(np.eye(d)-eta*S, 2))
                lip = eta**2*sum(j*radius**(j-1) for j in range(1, T))
                lhs = np.linalg.norm(U-k*Z, 'fro')
                rhs = np.linalg.norm(Z, 'fro')*lip*eps
                require_le(lhs, rhs, "GD proposed repair bound", 1e-7)
                if rhs > 1e-10:
                    worst = max(worst, lhs/rhs)
                count += 1
    # Cross-measurement inequality for matrices adapted to the data.
    measurements = 0
    for _ in range(40):
        d, n = 6, 20
        X, Y = RNG.normal(size=(d, n))/math.sqrt(d), RNG.normal(size=(d, n))/math.sqrt(d)
        S, SY = d/n*(X@X.T), d/n*(Y@Y.T)
        for E in (Y@X.T, np.outer(Y[:, 0], X[:, 0]), RNG.normal(size=(d, d))):
            scores = Y.T@E@X
            lhs = (np.sum(scores**2)-np.sum(np.diag(scores)**2))/(n*(n-1))
            rhs = n/(n-1)*np.linalg.norm(S, 2)*np.linalg.norm(SY, 2)*np.mean(E**2)
            require_le(lhs, rhs, "L3 wrong-candidate pathwise inequality")
            measurements += 1
    return {"gd_cases": count, "max_gd_bound_ratio": worst, "adapted_measurement_cases": measurements}


def rank_search():
    count, worst = 0, 0.
    for n in range(2, 31):
        for _ in range(20):
            W = np.sort(RNG.integers(-3, 4, size=n-1).astype(float)/2)
            unique = np.unique(W)
            delta = max(max(abs(np.mean(W <= z)-ndtr(z)), abs(np.mean(W < z)-ndtr(z)))
                        for z in unique)
            largest_jump = max(np.mean(W == z) for z in unique)
            require_le(largest_jump, 2*delta, "L4 largest jump")
            for truth in (*unique, -.123, .123):
                before, tied = np.sum(W > truth), np.sum(W == truth)
                for offset in range(tied+1):
                    rank = (before+offset)/(n-1)
                    lhs = abs(rank-ndtr(-truth))
                    require_le(lhs, delta, "L4 dependent rank with ties")
                    worst = max(worst, lhs/delta)
                    count += 1
    return {"rank_cases_with_all_tie_positions": count, "max_rank_bound_ratio": worst}


def missing_integrability_example():
    # h(z)=sin(exp(z*z)): bounded C-infinity, but Gaussian |h'| is not integrable.
    # u=exp(z*z), z>0 gives phi(z)|h'(z)| dz = (2*pi)**(-1/2) u**(-1/2)|cos(u)| du.
    # On [2*pi*k-pi/3, 2*pi*k+pi/3], |cos u|>=1/2.
    table = []
    for N in (10, 100, 1000, 10000):
        k = np.arange(1, N+1)
        lo, hi = 2*np.pi*k-np.pi/3, 2*np.pi*k+np.pi/3
        lower = np.sum(2*(hi-lo)/(np.sqrt(hi)+np.sqrt(lo)))/math.sqrt(2*np.pi)
        # Factor 2 here accounts for both positive and negative z.
        table.append({"number_of_intervals_each_half_axis": N,
                      "z_cutoff": math.sqrt(math.log(hi[-1])),
                      "lower_bound_truncated_E_abs_derivative": float(lower)})
    return table


for name, check in (("L5_exact", exact_l5), ("L2_covariance_regression", covariance_search),
                    ("L3_sign", sign_search), ("L3_adam", adam_search),
                    ("GD_and_measurements", gd_and_measurement_search),
                    ("L4_rank", rank_search), ("auxiliary_integrability_counterexample", missing_integrability_example)):
    tick()
    RESULT[name] = check()

RESULT["elapsed_seconds"] = time.monotonic()-START
RESULT["peak_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
RESULT["searched_inequality_violations"] = 0
print(json.dumps(RESULT, indent=2, ensure_ascii=False))
