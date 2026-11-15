"""AA2 numerical diagnostics, not a proof/certificate. Exclusive output writes.

Run only with the resource-limited command in AA2_EXT_RETRIEVAL.md.
No API/network; all paths are inside this project; no existing file is changed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri
from scipy.stats import kstest


def moments(e):
    if e == 0:
        return math.sqrt(2 / math.pi), 1.0
    density = math.sqrt(2 / math.pi)
    a = density * quad(lambda t: t*t/(t+e)*math.exp(-t*t/2),
                       0, np.inf, epsabs=2e-12)[0]
    q = density * quad(lambda t: (t/(t+e))**2*math.exp(-t*t/2),
                       0, np.inf, epsabs=2e-12)[0]
    return a, q


def efficiency(e):
    a, q = moments(e)
    return a / math.sqrt(q)


def inverse_c(c):
    return brentq(lambda e: efficiency(e)-c, 1e-9, 30, xtol=2e-13)


def polar_constants(gamma, order):
    lam = 1/gamma
    mu = 4/math.pi * quad(lambda u: (1-u*u)*math.sqrt(1-lam*u*u),
                         0, 1, epsabs=2e-13)[0]
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x+1)/2, w/2
    p = (1-x*x)[:, None]
    r = p.T
    dp, dr = 1-lam+lam*p, 1-lam+lam*r
    np_, nr = (2-p)*dp+lam*p*(1-p), (2-r)*dr+lam*r*(1-r)
    den = p+r-p*r-lam*(1-p)*(1-r)*(p+r)
    kernel = p*r*np_*nr/(np.sqrt(dp*dr)*den)
    k = 4*lam/math.pi**2 * float(w @ kernel @ w)
    nu = 1+k/gamma-mu*mu/gamma**2
    return mu, k, nu


def formula_checks():
    rows = []
    for gamma in (16, 32, 64, 128, 256, 1024):
        mu, k, nu = polar_constants(gamma, 256)
        low = polar_constants(gamma, 128)
        er = inverse_c(mu)
        for tau in (0., 1., 2.):
            target = mu*math.sqrt((1+tau*tau)/(nu+tau*tau-2*mu*mu/gamma))
            em = inverse_c(target)
            rows.append(dict(gamma=gamma, tau=tau, mu=mu, k=k, nu=nu,
                             quadrature_128_256_maxdiff=max(abs(a-b) for a,b in zip(low,(mu,k,nu))),
                             retrieval_root_absolute=er, msr_root=em, gap=em-er,
                             gamma_times_gap=gamma*(em-er)))
            assert er < em
            # At the relative-budget root both recall formulas agree.
            ca = efficiency(em)
            assert abs(ca/math.sqrt(1+2*ca*ca/gamma+tau*tau)
                       -mu/math.sqrt(nu+tau*tau)) < 2e-11
    estar = inverse_c(8/(3*math.pi))
    h = 1e-5
    cp = (efficiency(estar+h)-efficiency(estar-h))/(2*h)
    return dict(rows=rows, reference_root=estar, reference_cprime=cp,
                leading_gap_coefficient_tau0=(8/(3*math.pi))**3/cp,
                arithmetic="SciPy and Gauss-Legendre floating estimates; not interval certificates")


def empirical_normal_distance(values, sigma):
    z = np.sort(values)
    cdf = ndtr(z/sigma)
    n = len(z)
    return float(max(np.max(np.arange(1,n+1)/n-cdf),
                     np.max(cdf-np.arange(n)/n)))


def simulation(dims, seeds, noise_repeats, alpha, sigma):
    gamma = 16
    mu, _, nu = polar_constants(gamma, 256)
    z = ndtri(1-alpha)
    rows = []
    for d in dims:
        n = gamma*d
        for seed in range(seeds):
            rng = np.random.default_rng(202611060000 + 10000*d + seed)
            x = rng.standard_normal((d,n))
            y = rng.standard_normal((d,n))
            zc = (y@x.T-np.outer(y.sum(axis=1),x.sum(axis=1))/n)/math.sqrt(n)
            left, _, right = np.linalg.svd(zc, full_matrices=False)
            upolar = math.sqrt(d)*(left@right)
            # Same global parameter perturbation is paired across optimizers/budgets.
            noise = [y.T@(rng.standard_normal((d,d))@x)/d for _ in range(noise_repeats)]
            methods = [("polar", None)] + [(f"adam_{e}", e) for e in (.02,.1,.3)]
            for name, e in methods:
                if e is None:
                    u, mean, variance = upolar, mu, nu
                else:
                    a, q = moments(e)
                    u = zc/(np.abs(zc)+e)/math.sqrt(q)
                    mean, variance = a/math.sqrt(q), 1+2*a*a/(q*gamma)
                scores = y.T@(u@x)/d
                diag = scores.diagonal().copy()
                wrong_mean = (scores.sum(axis=0)-diag)/(n-1)
                qoff = (np.square(scores).sum()-float(diag@diag))/(n*(n-1))
                vbar = qoff-float(wrong_mean@wrong_mean)/n
                mbar = float(np.mean(diag-wrong_mean))
                base = dict(d=d,n=n,seed=seed,method=name,
                            mean_coefficient_prediction=mean,
                            margin_coefficient_observed=mbar/math.sqrt(d/gamma),
                            correct_scaled_mse=float(np.mean((diag/math.sqrt(d)-mean/math.sqrt(gamma))**2)),
                            variance_prediction=variance,variance_observed=vbar,
                            unperturbed_recall=float(np.mean(np.count_nonzero(scores>diag[None,:],axis=0)/(n-1)<=alpha)))
                for budget in ("absolute", "relative"):
                    bd = 1 if budget == "absolute" else math.sqrt(max(vbar,1e-100))
                    blim = 1 if budget == "absolute" else math.sqrt(variance)
                    prediction = float(ndtr(mean/(math.sqrt(gamma)*sigma*blim)-z))
                    for rep, hn in enumerate(noise):
                        # Normalized score matrix; division preserves ranks.
                        perturbed = scores/(bd*math.sqrt(d))+sigma*hn
                        correct = perturbed.diagonal().copy()
                        frac = np.count_nonzero(perturbed>correct[None,:],axis=0)/(n-1)
                        # The diagonal itself never counts as strictly larger.
                        recall = float(np.mean(frac<=alpha))
                        cdf_dist = []
                        for i in np.linspace(0,n-1,8,dtype=int):
                            wrong = np.concatenate((perturbed[:i,i],perturbed[i+1:,i]))
                            cdf_dist.append(empirical_normal_distance(wrong,sigma))
                        rows.append(dict(**base,budget=budget,noise_rep=rep,
                                         recall=recall,recall_prediction=prediction,
                                         wrong_cdf_sup_mean=float(np.mean(cdf_dist)),
                                         correct_normal_ks=float(kstest((correct-mean/(math.sqrt(gamma)*blim))/sigma,"norm").statistic)))
            print(json.dumps(dict(progress_d=d,seed=seed)), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dims", type=int, nargs="+", default=[64,128,256])
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--noise-repeats", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output)
    if out.parent.resolve() != Path("aa2_checks").resolve():
        raise ValueError("Output must be directly inside aa2_checks")
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    paths = ["ICML_REVIEW_SIM6.md", "X3_POLAR_TOOL.md",
             "paper/sections/setup.tex", "paper/sections/main_results.tex",
             "paper/appendix/E.tex", "paper/appendix/F.tex", "paper/appendix/J_novelty.tex",
             "aa2_checks/retrieval_check.py"]
    before = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    formulas = formula_checks()
    records = simulation(args.dims,args.seeds,args.noise_repeats,.1,.12)
    summary = []
    for d in args.dims:
        for method in ("polar","adam_0.02","adam_0.1","adam_0.3"):
            for budget in ("absolute","relative"):
                subset = [r for r in records if r["d"]==d and r["method"]==method and r["budget"]==budget]
                per_seed = [np.mean([r["recall"] for r in subset if r["seed"]==s]) for s in range(args.seeds)]
                summary.append(dict(d=d,method=method,budget=budget,
                                    recall=float(np.mean(per_seed)),
                                    dataset_standard_error=float(np.std(per_seed,ddof=1)/math.sqrt(args.seeds)),
                                    prediction=subset[0]["recall_prediction"],
                                    cdf_distance=float(np.mean([r["wrong_cdf_sup_mean"] for r in subset])),
                                    correct_normal_ks=float(np.mean([r["correct_normal_ks"] for r in subset])),
                                    correct_scaled_mse=float(np.mean([r["correct_scaled_mse"] for r in subset])),
                                    variance=float(np.mean([r["variance_observed"] for r in subset])),
                                    unperturbed_recall=float(np.mean([r["unperturbed_recall"] for r in subset]))))
    after = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    result = dict(formulas=formulas,parameters=vars(args),alpha=.1,sigma=.12,gamma=16,
                  summary=summary,records=records,seconds=time.monotonic()-started,
                  inputs_sha256_before=before,inputs_sha256_after=after,
                  inputs_unchanged=before==after,
                  threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")})
    with out.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(dict(output=str(out),seconds=result["seconds"],inputs_unchanged=before==after)),flush=True)


if __name__ == "__main__":
    main()
