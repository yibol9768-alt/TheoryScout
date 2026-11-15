"""Independent finite checks for THEOREM_M.md; no project imports or network.

Run only with the resource-limited command in README.md. All output is JSON.
Numerics can refute a finite inequality; passing is not an asymptotic proof.
"""
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import resource
import time
from fractions import Fraction

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent.parent
START = time.monotonic()
RNG = np.random.default_rng(202611062)
OUT = {"seed": 202611062, "checks": {}, "violations": []}


def check(name, lhs, rhs, tol=3e-11):
    if time.monotonic() - START > 400:
        raise RuntimeError("Internal aggregate deadline exceeded")
    item = OUT["checks"].setdefault(name, {"count": 0, "max_resolved_ratio": 0.0,
                                         "near_roundoff_rhs": 0, "max_absolute_excess": 0.0})
    item["count"] += 1
    # Ratios against almost-zero bounds are dominated by double precision noise.
    # Record their count explicitly; validate the scalar GD boundary cases below
    # with exact rational arithmetic instead of interpreting such ratios.
    if rhs > 1e-8:
        item["max_resolved_ratio"] = max(item["max_resolved_ratio"], float(lhs / rhs))
    else:
        item["near_roundoff_rhs"] += 1
    item["max_absolute_excess"] = max(item["max_absolute_excess"], float(lhs-rhs))
    if not (np.isfinite(lhs) and np.isfinite(rhs)) or lhs > rhs + tol * max(1, abs(rhs)):
        OUT["violations"].append({"name": name, "lhs": float(lhs), "rhs": float(rhs)})
        if len(OUT["violations"]) > 20:
            raise AssertionError("Too many violations")


def norm(a):
    return np.linalg.norm(a, 2)


def odd_df(k):
    return math.prod(range(1, k + 1, 2))


def gd_checks():
    for d, eta, T in itertools.product(
        [2, 5, 11], [1e-7, .01, .4, 1., 1.6, 1.9999999], [1, 2, 3, 8, 17]
    ):
        Q, _ = np.linalg.qr(RNG.normal(size=(d, d)))
        for delta in [0., 1e-7, .02, .25, .5]:
            eig = 1 + delta * np.linspace(-1, 1, d)
            S = (Q * eig) @ Q.T
            Z = RNG.normal(size=(d, d)) @ ((Q * np.sqrt(eig)) @ Q.T)
            U = np.zeros_like(Z)
            P = np.zeros_like(S)
            power = np.eye(d)
            A = np.eye(d) - eta * S
            for _ in range(T):
                U += eta * (Z - U @ S)
                P += eta * power
                power = power @ A
            k = 1 - (1 - eta)**T
            R = 1 + 1.5 * eta
            C = eta**2 * sum(j * R**(j-1) for j in range(1, T))
            check("GD matrix polynomial identity", norm(U-Z@P), 0,
                  tol=2e-9*max(1, norm(U), norm(Z@P)))
            check("GD scalar proxy identity", abs(eta*sum((1-eta)**j for j in range(T))-k), 0)
            check("GD operator difference (5.17)", norm(P-k*np.eye(d)), C*norm(S-np.eye(d)))
            check("GD Frobenius path bound", np.linalg.norm(U-k*Z, 'fro')**2,
                  C**2*np.linalg.norm(Z, 'fro')**2*norm(S-np.eye(d))**2,
                  tol=1e-9)
            if T == 1:
                check("GD first step exact", norm(U-eta*Z), 0)
    exact_count = 0
    for eta, T, s in itertools.product(
        [Fraction(1, 10**7), Fraction(1, 100), Fraction(2, 5), Fraction(1),
         Fraction(8, 5), Fraction(19999999, 10**7)],
        [1, 2, 3, 8, 17],
        [Fraction(1, 2), Fraction(9999999, 10**7), Fraction(1),
         Fraction(10000001, 10**7), Fraction(3, 2)]
    ):
        k = 1-(1-eta)**T
        p = eta*sum((1-eta*s)**j for j in range(T))
        R = 1+3*eta/2
        C = eta**2*sum(j*R**(j-1) for j in range(1, T))
        assert k > 0 and abs(p-k) <= C*abs(s-1)
        exact_count += 1
    OUT['GD_exact_rational_scalar_bounds'] = exact_count


def ibp_checks():
    jumps = [(-.8, -1.7), (.4, 2.4), (2., .2)]
    records = []
    for sigma in [.03, .2, .7, 1., 3., 10.]:
        def h(z):
            return z**3-2*z+.3+sum(v for u, v in jumps if z > u)
        cuts = [-np.inf] + [u/sigma for u, _ in jumps] + [np.inf]
        val = sum(quad(lambda g: g*h(sigma*g)*math.exp(-g*g/2)/math.sqrt(2*math.pi)/sigma,
                       l, r, epsabs=1e-10, epsrel=1e-11)[0] for l, r in zip(cuts[:-1], cuts[1:]))
        expected = 3*sigma*sigma-2+sum(v*math.exp(-u*u/(2*sigma*sigma)) /
                    (math.sqrt(2*math.pi)*sigma) for u, v in jumps)
        check("IBP smooth part plus signed jumps (3.4)", abs(val-expected), 0, tol=2e-8)
        records.append({"sigma": sigma, "regression_integral": val, "derivative_plus_atoms": expected})
    OUT["ibp_values"] = records
    # One half-axis of the derivative's absolute integral for sin(exp(z^2)).
    # |cos u| >= 1/2 on [2*pi*k-pi/3, 2*pi*k+pi/3].
    OUT["nonintegrable_derivative_half_axis_lower_bounds"] = {
        str(N): sum((math.sqrt(2*math.pi*k+math.pi/3)-math.sqrt(2*math.pi*k-math.pi/3)) /
                    math.sqrt(2*math.pi) for k in range(1, N+1))
        for N in [10, 100, 1000, 10000]
    }


def gaussian_covariance_checks():
    for d, delta, rep in itertools.product([2, 5, 16], [1e-5, .01, .1, .25], range(7)):
        Q, _ = np.linalg.qr(RNG.normal(size=(d, d)))
        vals = RNG.uniform(-1, 1, d)
        vals /= max(abs(vals))
        S = np.eye(d) + (Q * (delta*vals)) @ Q.T
        v = np.diag(S)
        R = S/np.sqrt(np.outer(v, v))
        np.fill_diagonal(R, 1)
        cov = (2/math.pi)*np.arcsin(np.clip(R, -1, 1))
        L = math.sqrt(2/math.pi)/2*(.75)**(-1.5)
        check("Sign covariance (4.1)", norm(cov-np.eye(d)), (L+4)*delta)
        for m in [1, 3, 5, 7]:
            cov = np.zeros((d, d))
            for k in range(1, m+1, 2):
                factor = math.comb(m, k)**2*math.factorial(k)*odd_df(m-k-1)**2
                cov += factor * np.outer(v**((m-k)/2), v**((m-k)/2)) * S**k
            q = odd_df(2*m-1)
            Qbound = q*1.25**m
            L = q*m*1.25**(m-1)
            if m > 1:
                L += odd_df(m)*(m-1)/2*1.25**((m-3)/2)
            check("Polynomial covariance (4.1)", norm(cov-q*np.eye(d)), (L+4*Qbound)*delta)
    for m, rho in itertools.product([1, 3, 5, 7, 9], [0., 1e-5, .1, .9, 1.]):
        q, fourth = odd_df(2*m-1), odd_df(4*m-1)
        correction = rho**2*(2*m*q-odd_df(m)**2)
        check("Regression variance including centering (4.4)", abs(correction),
              rho**2*(math.sqrt(2*fourth)+q))


def adam_weights(beta, t):
    if beta == 0:
        return np.eye(1, t, t-1)[0]
    weights = (1-beta)*beta**np.arange(t-1, -1, -1)/(1-beta**t)
    return weights


def adam_checks():
    T = 6
    for alpha, e, b1, b2 in itertools.product([.05, .6, 3.], [.001, .1, 2.], [0., .9, .999], [0., .9, .999]):
        c = math.sqrt(1-b2)
        gamma = 1+2*alpha/e
        for z in [0., 1e-10, .001, .05, .55, 1., 10., 1e6]:
            h, hist = 0., []
            for t in range(1, T+1):
                hist.append(h-z)
                s = np.asarray(hist)
                a, b = adam_weights(b1, t), adam_weights(b2, t)
                def update(r):
                    return float(a@r)/(math.sqrt(float(b@(r*r)))+e)
                J = gamma**(t-1)/c
                L = (1+J)/e
                H = e*(gamma**t-1)/(2*c)
                As = update(s)
                check("Adam reachable update bound (5.9)", abs(As), J)
                h -= alpha*As
                check("Adam reachable state bound (5.9)", abs(h), H)
                for scale in [1e-9, 1e-3, 1., 1e5]:
                    r = s + scale*RNG.normal(size=t)
                    check("Adam arbitrary-to-reachable (5.10)", abs(update(r)-As), L*np.linalg.norm(r-s))
    # A reachable residual equal to zero with beta2=0 and beta1>0.
    alpha, e, b1 = .6, .05, .9
    z = alpha-e
    h1 = alpha*z/(z+e)
    s = np.array([-z, h1-z])
    a = adam_weights(b1, 2)
    As = float(a@s)/(abs(s[-1])+e)
    OUT["reachable_zero_current_residual"] = {"alpha": alpha, "e": e, "beta1": b1,
        "beta2": 0, "z": z, "history": s.tolist(), "update_factor": As}


def measurement_checks():
    for d, n in [(2, 5), (3, 11), (5, 21), (8, 45)]:
        for _ in range(8):
            X, Y = RNG.normal(size=(d, n))/math.sqrt(d), RNG.normal(size=(d, n))/math.sqrt(d)
            S, Sy = d/n*X@X.T, d/n*Y@Y.T
            # Nonzero eigenvalues of the frame equal those of its n-by-n Gram matrix.
            frame_norm = float(np.linalg.eigvalsh((X.T@X)*(Y.T@Y))[-1])
            for E in [RNG.normal(size=(d, d)), Y@X.T,
                      np.outer(Y[:, 0], X[:, 0]), Y@np.diag(RNG.uniform(size=n))@X.T]:
                scores = Y.T@E@X
                fro2 = float(np.linalg.norm(E, 'fro')**2)
                check("Adaptive training diagonal (6.1)", np.mean(np.diag(scores)**2), frame_norm/n*fro2)
                off = (np.sum(scores*scores)-np.sum(np.diag(scores)**2))/(n*(n-1))
                check("Adaptive training off diagonal (6.2)", off, n/(n-1)*norm(S)*norm(Sy)*fro2/d**2)


def tie_checks():
    for N in range(1, 7):
        for scores in itertools.combinations_with_replacement([-2., -.2, .7, 2.], N):
            values = np.asarray(scores)
            points = np.unique(values)
            eps = max(max(abs(np.mean(values <= x)-ndtr(x)), abs(np.mean(values < x)-ndtr(x))) for x in points)
            maxjump = max(np.mean(values == x) for x in points)
            check("Maximum CDF jump (8.1)", maxjump, 2*eps)
            for correct in [-3., -2., -.2, 0., .7, 2., 3.]:
                A = int(np.sum(values > correct))
                B = int(np.sum(values == correct))
                for K in range(B+1):
                    rank_fraction = (A+K)/N
                    check("Every legal tied rank (8.1)", abs(rank_fraction-ndtr(-correct)), eps)
    # All four priorities may be iid continuous and independent of data. Let
    # A,B be the independent fair events that each query's correct item wins.
    # I=1 if A else 2 is uniform and independent of data but not of priorities.
    # This satisfies the independence explicitly stated in section 1.3.
    outcomes = []
    for A, B in itertools.product([0, 1], repeat=2):
        I = 1 if A else 2
        selected = A if I == 1 else B
        outcomes.append({'H1': A, 'H2': B, 'I': I, 'H_I': selected,
                         'empirical_recall': (A+B)/2})
    # A concrete open event of the Gaussian model has all scores tied: for
    # d=1,n=2,alpha=1,T=2, X and Y sufficiently near these nonzero values,
    # the real SignGD returns exactly to U2=0.
    X = np.array([[.1, .1]])
    Y = np.array([[.01, .01]])
    S = X@X.T/2
    Z = Y@X.T/math.sqrt(2)
    U1 = np.sign(Z)
    U2 = U1-np.sign(U1@S-Z)
    assert U2.item() == 0 and 0 < Z.item() < S.item()
    OUT['missing_query_tie_independence_counterexample'] = {
        'n': 2, 'q': .5, 'uniform_I': True, 'iid_continuous_priorities_possible': True,
        'outcomes_probability_each': .25, 'outcomes': outcomes,
        'E_H_I': sum(row['H_I'] for row in outcomes)/4,
        'E_empirical_recall': sum(row['empirical_recall'] for row in outcomes)/4,
        'SignGD_open_event_center': {'d': 1, 'n': 2, 'alpha': 1, 'T': 2,
                                   'X': X.tolist(), 'Y': Y.tolist(),
                                   'S': S.item(), 'Z': Z.item(), 'U2': U2.item()}}


def certificate_checks():
    manuscript = (ROOT/'THEOREM_M.md').read_text()
    code = manuscript.split('```python\n', 1)[1].split('```', 1)[0]
    code = code.replace('def lt(x, y):', 'strict_comparisons = 0\ndef lt(x, y):\n    global strict_comparisons\n    strict_comparisons += 1')
    code = code.replace('    assert lower <= upper', '    nonstrict_comparisons += 1\n    assert lower <= upper')
    env = {"nonstrict_comparisons": 0}
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        exec(compile(code, 'THEOREM_M.md:990 (in memory)', 'exec'), env)
    OUT["rational_certificate"] = {"strict": env['strict_comparisons'],
        "nonstrict": env['nonstrict_comparisons'], "output": capture.getvalue().strip()}


for function in [gd_checks, ibp_checks, gaussian_covariance_checks, adam_checks, measurement_checks, tie_checks, certificate_checks]:
    function()
OUT["wall_seconds"] = time.monotonic()-START
OUT["max_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
OUT["thread_environment"] = {name: os.environ.get(name) for name in [
    'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS']}
OUT["input_sha256"] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ['THEOREM_M.md', 'PROOF_REVIEW.md', 'FIX_LOG.md']}
print(json.dumps(OUT, indent=2, ensure_ascii=False, allow_nan=False))
if OUT["violations"]:
    raise SystemExit(1)
