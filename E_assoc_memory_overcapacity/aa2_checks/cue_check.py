"""Weak-cue retrieval diagnostics. No API; exclusive writes inside aa2_checks.

Imports only a newly created AA2 helper; PYTHONDONTWRITEBYTECODE must be set.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import kstest
from retrieval_check import moments, inverse_c, polar_constants, empirical_normal_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--dims", nargs="+", type=int, default=[64,128,256])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--queries", type=int, default=2048)
    args = ap.parse_args()
    output = Path(args.output)
    assert output.parent.resolve() == Path("aa2_checks").resolve()
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    source_paths = ["ICML_REVIEW_SIM6.md", "X3_POLAR_TOOL.md",
                    "paper/sections/setup.tex", "paper/sections/main_results.tex",
                    "paper/appendix/A.tex", "paper/appendix/E.tex", "paper/appendix/F.tex",
                    "paper/appendix/J_novelty.tex", "aa2_checks/retrieval_check.py",
                    "aa2_checks/cue_check.py"]
    hashes = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}
    gamma, amplitude, sigma, alpha = 16, 1., .2, .1
    z = ndtri(1-alpha)
    roots = []
    for g in (16,32,64,128,256,1024):
        mu, k, nu = polar_constants(g,256)
        for tau in (0.,1.,2.):
            t = 1+tau*tau
            er = inverse_c(mu)
            eq = inverse_c(mu*math.sqrt(t/(t-mu*mu/g)))
            em = inverse_c(mu*math.sqrt(t/(nu+tau*tau-2*mu*mu/g)))
            assert er < eq < em
            roots.append(dict(gamma=g,tau=tau,mu=mu,k=k,nu=nu,
                              mean_root=er,cue_root=eq,msr_root=em,
                              msr_minus_cue=em-eq,cue_minus_mean=eq-er,
                              gamma_times_msr_minus_cue=g*(em-eq)))
    mu, _, nu = polar_constants(gamma,256)
    records = []
    for d in args.dims:
        n = gamma*d
        for seed in range(args.seeds):
            rng = np.random.default_rng(202611060001+10000*d+seed)
            x, y = rng.standard_normal((d,n)), rng.standard_normal((d,n))
            zc = (y@x.T-np.outer(y.sum(axis=1),x.sum(axis=1))/n)/math.sqrt(n)
            left, _, right = np.linalg.svd(zc,full_matrices=False)
            idx = rng.integers(n,size=args.queries)
            xi = rng.standard_normal((d,args.queries))
            cue = amplitude*x[:,idx]/math.sqrt(d)+sigma*xi
            for method, e in [("polar",None),("adam_0.02",.02),("adam_0.06",.06),
                              ("adam_0.1",.1),("adam_0.3",.3)]:
                if e is None:
                    u, r, ell = math.sqrt(d)*(left@right), mu, 1.
                else:
                    a, q = moments(e)
                    u, r = zc/(np.abs(zc)+e)/math.sqrt(q), a/math.sqrt(q)
                    ell = 1+r*r/gamma
                yt_u = y.T@u
                leverage = np.square(yt_u).sum(axis=1)/d**2
                score = yt_u@cue/d
                correct = score[idx,np.arange(args.queries)]
                fraction = np.count_nonzero(score>correct[None,:],axis=0)/(n-1)
                recall = float(np.mean(fraction<=alpha))
                pred = float(ndtr(amplitude*r/(sigma*math.sqrt(gamma*ell))-z))
                cdf_errors = []
                for col in np.linspace(0,args.queries-1,24,dtype=int):
                    i = idx[col]
                    wrong = np.concatenate((score[:i,col],score[i+1:,col]))
                    cdf_errors.append(empirical_normal_distance(wrong,sigma*math.sqrt(ell)))
                record = dict(d=d,n=n,seed=seed,method=method,recall=recall,prediction=pred,
                              leverage_prediction=ell,leverage_mean=float(np.mean(leverage)),
                              leverage_mse=float(np.mean((leverage-ell)**2)),
                              wrong_cdf_sup=float(np.mean(cdf_errors)),
                              correct_normal_ks=float(kstest((correct-amplitude*r/math.sqrt(gamma))/(sigma*math.sqrt(ell)),"norm").statistic))
                records.append(record)
            print(json.dumps(dict(progress_d=d,seed=seed)),flush=True)
    summary = []
    for d in args.dims:
        for method in ("polar","adam_0.02","adam_0.06","adam_0.1","adam_0.3"):
            rows = [r for r in records if r["d"]==d and r["method"]==method]
            summary.append(dict(d=d,method=method,
                                **{k:float(np.mean([r[k] for r in rows])) for k in
                                   ("recall","prediction","leverage_prediction","leverage_mean",
                                    "leverage_mse","wrong_cdf_sup","correct_normal_ks")},
                                dataset_standard_error=float(np.std([r["recall"] for r in rows],ddof=1)/math.sqrt(args.seeds))))
    paired = []
    for d in args.dims:
        for method in ("adam_0.02","adam_0.06","adam_0.1","adam_0.3"):
            diffs = []
            pred = None
            for seed in range(args.seeds):
                ra = next(r for r in records if r["d"]==d and r["seed"]==seed and r["method"]==method)
                rp = next(r for r in records if r["d"]==d and r["seed"]==seed and r["method"]=="polar")
                diffs.append(ra["recall"]-rp["recall"])
                pred = ra["prediction"]-rp["prediction"]
            paired.append(dict(d=d,method=method,adam_minus_polar=float(np.mean(diffs)),prediction=pred,
                               dataset_standard_error=float(np.std(diffs,ddof=1)/math.sqrt(args.seeds))))
    after = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}
    result = dict(parameters=vars(args),gamma=gamma,amplitude=amplitude,sigma=sigma,alpha=alpha,
                  roots=roots,summary=summary,paired=paired,records=records,
                  seconds=time.monotonic()-started,input_hashes_before=hashes,input_hashes_after=after,
                  inputs_unchanged=hashes==after,
                  note="Floating quadrature and Monte Carlo diagnostics only; no interval or finite-dimension guarantee")
    with output.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(dict(output=str(output),seconds=result["seconds"],inputs_unchanged=hashes==after)),flush=True)


if __name__ == "__main__":
    main()
