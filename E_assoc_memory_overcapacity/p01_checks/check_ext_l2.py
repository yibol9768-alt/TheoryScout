#!/usr/bin/env python3
"""P0.1/P0.2 diagnostics, not a proof or a replacement for frozen experiments.

Run from the E project with the resource-limited command in PROOF_EXT_L2.md.
All output is a new file under this script's directory; existing files are refused.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "2"
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent


def constants(e):
    if e is None:
        return 1.0, 1.0
    if e == 0:
        return 1.0, math.sqrt(2 / math.pi)
    weight = lambda t: math.sqrt(2 / math.pi) * math.exp(-t*t/2)
    q = quad(lambda t: (t/(t+e))**2 * weight(t), 0, np.inf,
             epsabs=2e-12, epsrel=2e-12)[0]
    a = quad(lambda t: t*t/(t+e) * weight(t), 0, np.inf,
             epsabs=2e-12, epsrel=2e-12)[0]
    return q, a


def transform(z, e):
    if e is None:
        return z
    if e == 0:
        return np.sign(z)
    return z / (np.abs(z) + e)


def stats(score):
    n = score.shape[0]
    diag = np.diag(score)
    b = (score.sum(axis=0) - diag) / (n-1)
    second = ((score*score).sum() - diag @ diag) / (n*(n-1))
    b2 = np.mean(b*b)
    margin = np.mean(diag-b)
    return dict(second=float(second), b2=float(b2),
                variance=float(second-b2), margin=float(margin))


def full_checks(dims, seeds):
    rows = []
    methods = [("linear", None), ("e0.1", .1), ("sign", 0.),
               ("smooth0.03", .03), ("smooth0.003", .003)]
    const = {name: constants(e) for name, e in methods}
    for d in dims:
        for gamma in (2, 4):
            n = gamma*d
            for seed in range(seeds):
                rng = np.random.default_rng(2026110600 + 100000*d + 1000*gamma + seed)
                x = rng.standard_normal((d, n))
                y = rng.standard_normal((d, n))
                z0 = y @ x.T / math.sqrt(n)
                zc = z0 - math.sqrt(n)*np.outer(y.mean(axis=1), x.mean(axis=1))
                sx_norm = np.linalg.eigvalsh(x @ x.T/n)[-1]
                sy_norm = np.linalg.eigvalsh(y @ y.T/n)[-1]
                alpha = n/(n-1)*sx_norm*sy_norm
                sign_u = np.sign(zc)
                sign_score = y.T @ sign_u @ x / d
                for name, e in methods:
                    q, a = const[name]
                    u = transform(zc, e)/math.sqrt(q)
                    score = y.T @ u @ x/d
                    st = stats(score)
                    st.update(d=d, n=n, gamma=gamma, seed=seed, method=name,
                              c=a/math.sqrt(q), target_second=1+2*a*a/(q*gamma),
                              margin_scaled=st["margin"]*math.sqrt(n)/d)
                    error = u-sign_u
                    error_score = score-sign_score
                    measured = stats(error_score)["second"]
                    bound = alpha*np.mean(error*error)
                    assert measured <= bound + 2e-11*max(1., bound)
                    st.update(smoothing_score_mse=measured,
                              smoothing_bound=float(bound),
                              coordinate_mse=float(np.mean(error*error)))
                    u0 = transform(z0, e)/math.sqrt(q)
                    centering_score = y.T @ (u-u0) @ x/d
                    centering_mse = stats(centering_score)["second"]
                    centering_bound = alpha*np.mean((u-u0)**2)
                    assert centering_mse <= centering_bound + 2e-11*max(1., centering_bound)
                    st.update(centering_score_mse=centering_mse,
                              centering_bound=float(centering_bound))
                    exact_margin = math.sqrt(n)/(d*(n-1))*np.sum(u*zc)
                    assert np.isclose(exact_margin, st["margin"], rtol=1e-10, atol=1e-10)
                    rows.append(st)
            print(f"completed d={d}, gamma={gamma}, seeds={seeds}", flush=True)
    return rows


def conditional_rows():
    """The two extra terms, on one fixed X, measured before normalization by q."""
    d, n, count = 128, 512, 16384
    rng = np.random.default_rng(710603)
    x = rng.standard_normal((d, n))
    yrows = rng.standard_normal((count, n))
    zrows = yrows @ x.T/math.sqrt(n)
    xi = x[:, 0]/math.sqrt(d)
    sigma = np.sqrt(np.mean(x*x, axis=1))
    out = []
    for name, e in [("linear", None), ("e0.1", .1)]:
        q, a = constants(e)
        if e is None:
            coeff = np.ones(d)
        else:
            coeff = np.array([quad(lambda t: math.sqrt(2/math.pi)*t*t/(s*t+e)
                                   *math.exp(-t*t/2), 0, np.inf,
                                   epsabs=2e-11)[0] for s in sigma])
        mu = np.sum(x[:, 0]*x[:, 1]*coeff)/math.sqrt(n)
        trows = yrows[:, 1]*(transform(zrows, e) @ xi)
        out.append(dict(method=name, d=d, n=n, row_samples=count,
                        conditional_score_mean_exact=float(mu),
                        conditional_score_mean_approx=float(a*(x[:, 0]@x[:, 1])/math.sqrt(n)),
                        conditional_noise_variance_mc=float(np.var(trows)),
                        conditional_noise_asymptotic=q+a*a*d/n,
                        random_mean_variance_asymptotic=a*a*d/n))
    return out


def noise_checks():
    d, n, draws = 16, 64, 4096
    rng = np.random.default_rng(820603)
    x = rng.standard_normal((d, n))
    y = rng.standard_normal((d, n))
    z = (y-y.mean(axis=1, keepdims=True)) @ x.T/math.sqrt(n)
    u = np.sign(z)
    # Row-major vectorization, w[j,i] = vec(v_j u_i^T)/d.
    w = np.einsum("aj,bi->jiab", y, x).reshape(n, n, d*d)/d
    mask = ~np.eye(n, dtype=bool)
    off = w[mask]
    means = (w.sum(axis=0)-w[np.arange(n), np.arange(n)])/(n-1)
    aa = off.T @ off/(n*(n-1))
    bb = means.T @ means/n
    kk = aa-bb
    uv = u.reshape(-1)
    g = rng.standard_normal((draws, d*d))
    mixed = g @ (kk @ uv)
    quadratic = np.einsum("ij,ij->i", g @ kk, g)
    score_u = y.T @ u @ x/d
    h = g[0].reshape(d, d)
    score_h = y.T @ h @ x/d
    dense_mixed = (stats(score_u+score_h)["variance"]
                   - stats(score_u)["variance"]-stats(score_h)["variance"])/2
    assert np.isclose(dense_mixed, mixed[0], rtol=1e-10, atol=1e-10)
    assert np.isclose(stats(score_h)["variance"], quadratic[0], rtol=1e-10, atol=1e-10)
    norm_bound = np.linalg.norm(x, 2)**2*np.linalg.norm(y, 2)**2/(n*(n-1)*d*d)
    assert np.linalg.eigvalsh(kk)[0] > -1e-11
    assert np.linalg.eigvalsh(kk)[-1] <= norm_bound + 1e-11
    ell = math.sqrt(n)/(d*(n-1))*z.reshape(-1)
    return dict(d=d, n=n, gaussian_draws=draws,
                mixed_variance_exact=float(uv @ kk @ kk @ uv),
                mixed_variance_mc=float(np.var(mixed)),
                quadratic_mean_exact=float(np.trace(kk)),
                quadratic_mean_mc=float(np.mean(quadratic)),
                quadratic_variance_exact=float(2*np.sum(kk*kk)),
                quadratic_variance_mc=float(np.var(quadratic)),
                excluded_diagonal_trace=float(np.sum(w[np.arange(n), np.arange(n)]**2)/(n*(n-1))),
                wrong_mean_square_trace=float(np.trace(bb)),
                margin_variance_exact=float(ell @ ell),
                margin_scaled_variance_exact=float((ell @ ell)*n/(d*d)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output", default="l2_checks_results.json")
    args = parser.parse_args()
    output = ROOT/args.output
    if output.parent.resolve() != ROOT or output.exists():
        raise ValueError("output must be a new direct child of p01_checks")
    start = time.monotonic()
    rows = full_checks(args.dims, args.seeds)
    result = dict(description=__doc__, rows=rows, conditional_rows=conditional_rows(),
                  noise=noise_checks(), elapsed_seconds=time.monotonic()-start,
                  numpy_version=np.__version__, threads=2)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(output), elapsed_seconds=result["elapsed_seconds"],
                          rows=len(rows), noise=result["noise"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
