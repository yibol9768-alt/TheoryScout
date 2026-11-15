"""独立 V6 审稿核验；仅用 numpy/scipy，无 QED 代码或 API。

运行方式见 run.sh。所有拟合仅为有限规模诊断，不是渐近证明。
"""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, stdtr, zeta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT.parent / "qed_runs/out"
RNG = np.random.default_rng(20261107)
RESULT = {"seed": 20261107, "threads": {k: os.environ.get(k) for k in
          ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}}


def save():
    (HERE / "results.json").write_text(json.dumps(RESULT, indent=2, ensure_ascii=False))


def model(alpha, beta, n, sigma=1.0):
    j = np.arange(1, n + 1, dtype=float)
    lam = j ** (-alpha)
    theta = j ** ((alpha - 1 - beta) / 2)
    tail = float(zeta(1 + beta, n + 1))
    return lam, theta, tail, sigma * sigma + tail


def mean_se(y):
    y = np.asarray(y)
    return float(np.mean(y)), float(np.std(y, ddof=1) / math.sqrt(y.size))


def provenance():
    rows = []
    for k in range(1, 5):
        f = OUT / f"D_optimizer_speedup_phase__v6_L{k}" / "proof.md"
        s = f.read_text()
        statement = re.search(r"\\begin\{problem\}.*?\\end\{problem\}", s, re.S).group()
        local = (ROOT / f"lemmas_v6/L{k}.tex").read_text().strip()
        rows.append({"lemma": k, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                     "same_statement_without_whitespace": re.sub(r"\s+", "", statement) == re.sub(r"\s+", "", local),
                     "summary_exists": (f.parent / "proof_effort_summary.md").exists()})
    RESULT["provenance"] = rows


def conditional_moments():
    rows = []
    nsamp = 80000
    for alpha, beta, n, batch in [(1.5, .7, 4, 1), (2, 1, 13, 2),
                                  (3, .5, 32, 7), (1.2, 2, 64, 3)]:
        lam, theta, tail, s2 = model(alpha, beta, n)
        e = -theta.copy()
        e[::3] *= -.4
        e[1] = 0
        g = np.zeros((nsamp, n))
        for _ in range(batch):
            x = RNG.normal(size=g.shape) * np.sqrt(lam)
            residual = x @ e - RNG.normal(size=nsamp) * math.sqrt(s2)
            g += x * residual[:, None] / batch
        energy = float(np.dot(lam, e * e))
        sign_exact = 2 * stdtr(batch, np.sqrt(batch * lam) * e /
                              np.sqrt(s2 + energy - lam * e * e)) - 1
        eta = .01
        rec = (1 - 2 * eta * lam + eta**2 * (1 + 1 / batch) * lam**2) * e**2
        rec += eta**2 * lam * (s2 + energy) / batch
        for i in sorted(set([0, 1, n - 1])):
            sm, ss = mean_se(np.sign(g[:, i]))
            gm, gs = mean_se((e[i] - eta * g[:, i])**2)
            rows.append({"alpha": alpha, "beta": beta, "n": n, "b": batch, "i": i + 1,
                         "sign_mc": sm, "sign_exact_student_t": float(sign_exact[i]),
                         "sign_z": (sm - sign_exact[i]) / max(ss, 1e-15),
                         "sgd_square_mc": gm, "sgd_square_exact": float(rec[i]),
                         "sgd_z": (gm - rec[i]) / max(gs, 1e-15)})
    RESULT["conditional_moments"] = rows


def polar_integrals(lam, e, s2, i):
    def kernel(t):
        den = 1 + 2 * t * lam
        weight = math.exp(-.5 * float(np.log(den).sum()))
        tilted_s = s2 + float(np.dot(lam / den, e * e))
        return weight / den[i], tilted_s

    # t=y^2 removes the square-root singularity at zero.
    k = lam[i] * quad(lambda t: kernel(t)[0], 0, np.inf,
                     epsabs=1e-10, epsrel=1e-8)[0]
    m = lam[i] * 2 / math.sqrt(math.pi) * quad(
        lambda y: kernel(y*y)[0], 0, np.inf, epsabs=1e-10, epsrel=1e-8)[0]
    slope = lam[i] * 2 * math.sqrt(2) / math.pi * quad(
        lambda y: kernel(y*y)[0] / math.sqrt(kernel(y*y)[1]),
        0, np.inf, epsabs=1e-10, epsrel=1e-8)[0]
    return k, m, slope


def polar_and_marking():
    rows = []
    marked = []
    for alpha, beta, n in [(1.5, 1, 4), (2, .5, 16), (3, .7, 128)]:
        lam, theta, tail, s2 = model(alpha, beta, n)
        e = -.8 * theta
        e[0] = .7
        x = RNG.normal(size=(100000, n)) * np.sqrt(lam)
        norm = np.linalg.norm(x, axis=1)
        u = x / norm[:, None]
        # Conditional mean of residual sign integrates label noise exactly.
        residual_mean = 2 * ndtr((x @ e) / math.sqrt(s2)) - 1
        for i in sorted(set([0, n - 1])):
            k, m, slope = polar_integrals(lam, e, s2, i)
            km, ks = mean_se(u[:, i]**2)
            dm, ds = mean_se(u[:, i] * residual_mean)
            kap = math.sqrt(2 / math.pi) * m / math.sqrt(s2)
            rows.append({"alpha": alpha, "n": n, "i": i + 1,
                         "k_quad": k, "k_mc": km, "k_z": (km-k)/ks,
                         "drift_quad": slope * e[i], "drift_mc": dm,
                         "drift_z": (dm-slope*e[i])/ds,
                         "sector_lower_ratio": slope/(math.sqrt(s2)*kap/math.sqrt(s2+np.dot(lam,e*e))),
                         "sector_upper_ratio": slope/kap})
        # Independently check exact marked sign Q_i V identity (L4 3.3).
        residual = x @ e - RNG.normal(size=x.shape[0]) * math.sqrt(s2)
        us = np.sign(x) * np.sign(residual[:, None])
        eta = .007
        for i in [0, n - 1]:
            sn = s2 + float(np.dot(lam, e*e))
            drift = 2 / math.pi * np.arcsin(np.sqrt(lam)*e / math.sqrt(sn))
            h = e[i]**2
            pi = h - 2*eta*e[i]*drift[i] + eta**2
            ep = e - eta*us
            direct = ep[:, i]**2 * (ep*ep @ lam) / pi
            qm, qs = mean_se(direct)
            exact = float(np.dot(lam,e*e) - 2*eta*(h+eta**2)/pi*np.dot(lam*e,drift)
                          + eta**2*lam.sum() + 4*eta**2*lam[i]*h/pi)
            marked.append({"alpha": alpha, "n": n, "i": i+1,
                           "Q_i_V_mc": qm, "Q_i_V_exact": exact, "z": (qm-exact)/qs})
    RESULT["polar_quadrature"] = rows
    RESULT["marked_sign_identity"] = marked


def sgd_exact(alpha, beta, n, b, eta, horizon):
    lam, theta, tail, s2 = model(alpha, beta, n)
    # Energy coordinates: affine diagonal-plus-rank-one recursion. Binary
    # exponentiation is exact up to float rounding; no SGD surrogate is used.
    q = 1 - 2*eta*lam + eta**2*(1+1/b)*lam**2
    drive = eta**2*lam**2/b
    a = np.diag(q) + drive[:, None] * np.ones((1, n))
    stat = drive/(1-q)
    mass = float(stat.sum())
    vinf = stat*s2/(1-mass)
    y = lam*theta**2 - vinf
    power = horizon
    while power:
        if power & 1:
            y = a @ y
        power >>= 1
        if power:
            a = a @ a
    return tail + float(np.sum(y+vinf)), mass


def envelopes_and_paid_batch():
    rows, comparisons = [], []
    for alpha in [1.5, 2., 3.]:
        beta = .5
        for n in [4, 6, 9, 13, 18, 26, 40, 64, 128, 256]:
            lam, theta, tail, s2 = model(alpha, beta, n)
            for alg, p, d in [("S", alpha/2, float(np.sqrt(lam).sum())),
                              ("G", alpha, 1.), ("P_clock_envelope", alpha, 1.)]:
                eta = .012*n**(-beta)/d
                t = math.ceil(.25*n**p/eta)
                risk = tail + float(np.sum(lam*theta**2*np.exp(-eta*t*np.arange(1,n+1.)**(-p)))) + eta*d
                if alg == "G":
                    risk, _ = sgd_exact(alpha,beta,n,1,eta,t)
                rows.append({"alpha":alpha,"beta":beta,"alg":alg,"n":n,
                             "eta":eta,"T":t,"C":n*t,"risk":risk})
        # Stress cutoff elimination over oversized n, very short clocks,
        # unused budgets, and batches that can grow with n.
        for n in [4, 17, 64, 256]:
            lam, theta, tail, s2 = model(alpha,beta,n)
            p = alpha/2
            for b in [1, 4, 25, 256]:
                for gamma in [.0001, .003, .01]:
                    for t in [1, 7, 301, 10001]:
                        tau = gamma*t
                        m = min(n,tau**(1/p))
                        h = float(np.sum(np.arange(1,math.floor(m)+1.)**(-p)))
                        cost = n*b*t
                        assert gamma/b + 1e-15 >= m**(1+p)/cost
                        # Exact scalar affine lower comparison, independently
                        # evaluated with expm1 for small coefficients.
                        c = 2*math.sqrt(2/math.pi)*gamma*np.sqrt(lam)
                        power = np.exp(t*np.log1p(-c))
                        risk_lb = tail + float(np.sum(lam*(theta**2*power + gamma**2/b*(-np.expm1(t*np.log1p(-c)))/c)))
                        envelope = math.exp(-4*math.sqrt(2/math.pi))/beta*(1+m)**(-beta)
                        envelope += (1-math.exp(-2*math.sqrt(2/math.pi)))/(2*math.sqrt(2/math.pi))*gamma/b*h
                        comparisons.append(risk_lb/envelope)
    RESULT["envelope_grid"] = rows
    RESULT["paid_batch_checks"] = {"count":len(comparisons), "minimum_affine_over_cutoff_lower":min(comparisons)}


def online_simulations():
    rows = []
    paths = 96
    beta = .5
    for alpha in [1.5, 2., 3.]:
        for n in [4, 6, 9, 13]:
            lam, theta, tail, s2 = model(alpha,beta,n)
            for alg in ["S", "G", "P"]:
                p = alpha/2 if alg == "S" else alpha
                d = float(np.sqrt(lam).sum()) if alg == "S" else 1.
                eta = .012*n**(-beta)/d
                tmax = math.ceil(.25*n**p/eta)
                err = np.tile(-theta, (paths,1))
                ratios = []
                checkpoints = {max(1,tmax//4),max(1,tmax//2),tmax}
                for t in range(1,tmax+1):
                    x = RNG.normal(size=err.shape)*np.sqrt(lam)
                    residual = np.sum(x*err,axis=1)-RNG.normal(size=paths)*math.sqrt(s2)
                    if alg == "S":
                        update = np.sign(x)*np.sign(residual[:,None])
                    elif alg == "P":
                        update = x/np.linalg.norm(x,axis=1)[:,None]*np.sign(residual[:,None])
                    else:
                        update = x*residual[:,None]
                    err -= eta*update
                    if t in checkpoints:
                        energy = err*err @ lam
                        den = np.mean(err*err,axis=0)
                        ratios.append(float(np.max(np.mean(err*err*(s2+energy[:,None]),axis=0)/den)))
                pred = tail+err*err @ lam
                mean, se = mean_se(pred)
                row = {"alpha":alpha,"beta":beta,"n":n,"alg":alg,"paths":paths,
                       "eta":eta,"T":tmax,"C":n*tmax,"risk":mean,"risk_se":se,
                       "max_observed_weighted_ratio":max(ratios)}
                if alg == "G":
                    exact,mass=sgd_exact(alpha,beta,n,1,eta,tmax)
                    row.update(exact_risk=exact,mc_z=(mean-exact)/se,convolution_mass=mass)
                rows.append(row)
            print(f"online alpha={alpha} n={n} complete",flush=True)
            RESULT["online_simulations"] = rows
            save()


def diagnostics():
    slopes=[]
    for alpha in [1.5,2.,3.]:
        for alg in ["S","G","P"]:
            rr=[r for r in RESULT["online_simulations"] if r["alpha"]==alpha and r["alg"]==alg]
            slope=float(np.polyfit(np.log([r["C"] for r in rr]),np.log([r["risk"] for r in rr]),1)[0])
            beta=rr[0]["beta"]
            expected=-beta/(beta+2 if alpha<=2 else beta+1+alpha/2) if alg=="S" else -beta/(beta+1+alpha)
            slopes.append({"alpha":alpha,"alg":alg,"finite_size_slope":slope,"asymptotic_power":expected})
    RESULT["online_slopes"]=slopes
    # At alpha=2 log correction predicts -r*(1-1/log C), hence a less
    # negative local slope than the limiting power -r.
    logrows=[]
    for lo,hi in [(16,256),(64,256)]:
        n=np.arange(lo,hi+1,dtype=float)
        h=np.cumsum(1/np.arange(1,257,dtype=float))[lo-1:]
        c=n**2.5*h
        slope=float(np.polyfit(np.log(c),np.log(n**(-.5)),1)[0])
        logrows.append({"n_min":lo,"n_max":hi,"critical_balanced_slope":slope,"limiting_slope":-.2})
    RESULT["critical_log_diagnostics"]=logrows


if __name__ == "__main__":
    started=time.monotonic()
    provenance()
    conditional_moments()
    polar_and_marking()
    envelopes_and_paid_batch()
    save()
    print("identities, quadrature, paid-batch checks complete",flush=True)
    online_simulations()
    diagnostics()
    RESULT["elapsed_seconds"]=time.monotonic()-started
    save()
    print(json.dumps({k:RESULT[k] for k in ["paid_batch_checks","online_slopes","critical_log_diagnostics","elapsed_seconds"]},indent=2),flush=True)
