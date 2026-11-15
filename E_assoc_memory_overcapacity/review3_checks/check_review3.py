"""Independent diagnostics for review 3; all writes stay in this directory.

Finite simulation is a search for counterexamples, never a proof of a limit.
No imports of author check scripts and no network/API calls.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "review3_checks"
SOURCES = ["PROOF_EXT_L2.md", "PROOF_EXT_L1L3L4.md", "THEOREM_M.md",
           "EXT_FORMULATION.md", "P03_LITCHECK.md"]
SOURCES += [f"EXT_lemmas/L{k}.tex" for k in range(1, 5)]
SOURCES += ["EXT_sources/EXT_2611.03633v1.txt", "p04_checks/check_l1_l3_l4.py"]
fingerprints = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in SOURCES}
rng = np.random.default_rng(63317091)
started = time.monotonic()
results = {"seed": 63317091, "input_sha256": fingerprints,
           "thread_environment": {k: os.environ.get(k) for k in
               ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]}}


def stats(mat, x, y):
    d, n = x.shape
    scores = y.T @ mat @ x / d
    diag = scores.diagonal().copy()
    scores[np.diag_indices(n)] = 0
    b = scores.sum(axis=0)/(n-1)
    Q = np.sum(scores*scores)/(n*(n-1))
    B = np.mean(b*b)
    return Q, B, np.mean(diag-b)


def aq(e):
    k = math.sqrt(2/math.pi)
    a = quad(lambda t: k*t*t/(t+e)*math.exp(-t*t/2), 0, np.inf,
             epsabs=1e-12)[0]
    q = quad(lambda t: k*(t/(t+e))**2*math.exp(-t*t/2), 0, np.inf,
             epsabs=1e-12)[0]
    return a, q


def arrays():
    records = []
    max_measurement_ratio = 0.
    max_margin_identity_error = 0.
    a, q = aq(.1)
    c_adam = a/math.sqrt(q)
    smooth = []
    for gamma in (1.25, 2., 4.):
        for d in (32, 64, 128, 256):
            n = round(gamma*d)
            accum = {name: [] for name in ("linear", "adam_0.1", "sign")}
            for rep in range(12):
                x, y = rng.normal(size=(d, n)), rng.normal(size=(d, n))
                z0 = y @ x.T/math.sqrt(n)
                z = z0-math.sqrt(n)*np.outer(y.mean(axis=1), x.mean(axis=1))
                sx = np.linalg.norm(x, 2)**2/n
                sy = np.linalg.norm(y, 2)**2/n
                Ad = n/(n-1)*sx*sy
                for name, mat, c in [("linear", z, 1.),
                        ("adam_0.1", z/(abs(z)+.1)/math.sqrt(q), c_adam),
                        ("sign", np.sign(z), math.sqrt(2/math.pi))]:
                    Q, B, margin = stats(mat, x, y)
                    exact = math.sqrt(n)/(d*(n-1))*np.sum(mat*z)
                    max_margin_identity_error = max(max_margin_identity_error,
                                                     abs(margin-exact))
                    assert abs(margin-exact) < 2e-12
                    accum[name].append((Q, B, margin/(d/math.sqrt(n))))
                if rep < 3:
                    for delta in (.1, .01, .001, .0001):
                        _, qd = aq(delta)
                        err = z/(abs(z)+delta)/math.sqrt(qd)-np.sign(z)
                        QE = stats(err, x, y)[0]
                        D = np.mean(err*err)
                        ratio = QE/(Ad*D) if D else 0.
                        max_measurement_ratio = max(max_measurement_ratio, ratio)
                        assert ratio <= 1+1e-12
                        smooth.append({"gamma": gamma, "d": d,
                                       "delta": delta, "Q_error": QE, "D": D})
                    err = (z/(abs(z)+.1)-z0/(abs(z0)+.1))/math.sqrt(q)
                    QE = stats(err, x, y)[0]
                    assert QE <= Ad*np.mean(err*err)+1e-12
            for name, values in accum.items():
                val = np.array(values)
                c = {"linear": 1., "adam_0.1": c_adam,
                     "sign": math.sqrt(2/math.pi)}[name]
                records.append({"gamma": gamma, "d": d, "h": name,
                    "target_Q": 1+2*c*c/gamma, "target_margin": c,
                    "mean_Q_B_margin": val.mean(axis=0).tolist(),
                    "sd_Q_B_margin": val.std(axis=0, ddof=1).tolist()})
    return {"records": records, "smoothing": smooth,
            "max_measurement_ratio": max_measurement_ratio,
            "max_margin_identity_error": max_margin_identity_error}


def noise():
    d, n = 6, 11
    x, y = rng.normal(size=(d, n)), rng.normal(size=(d, n))
    z = (y-y.mean(axis=1, keepdims=True)) @ x.T/math.sqrt(n)
    K = np.zeros((d*d, d*d))
    for i in range(n):
        ts = np.array([np.kron(x[:, i], y[:, j])/d for j in range(n) if j != i])
        ts -= ts.mean(axis=0)
        K += ts.T @ ts/(n*(n-1))
    ev, vec = np.linalg.eigh(K)
    # A deliberately data-dependent matrix aligned with the maximal eigenvector.
    w = d*vec[:, -1]
    V = w.reshape((d, d), order="F")
    gv = rng.normal(size=(20000, d*d))
    cross = gv @ (K @ w)
    hv = np.einsum("bi,ij,bj->b", gv, K, gv, optimize=True)
    scale = np.linalg.norm(x, 2)**2*np.linalg.norm(y, 2)**2/(n*(n-1)*d*d)
    assert ev[0] > -1e-12 and ev[-1] <= scale+1e-12
    assert abs(stats(V, x, y)[0]-stats(V, x, y)[1]-w@K@w) < 1e-12
    H = gv[0].reshape((d, d), order="F")
    tau = .7
    f = lambda M: stats(M, x, y)[0]-stats(M, x, y)[1]
    assert abs(f(V+tau*H)-f(V)-2*tau*(w@K@gv[0])-tau*tau*f(H)) < 1e-12
    return {"largest_eigenvalue": float(ev[-1]), "spectral_upper": scale,
            "trace_K": float(np.trace(K)),
            "cross_variance_exact": float(w@K@K@w),
            "cross_variance_MC": float(np.var(cross)),
            "noise_variance_exact": float(2*np.trace(K@K)),
            "noise_variance_MC": float(np.var(hv))}


def envelope():
    b = 8/(3*math.pi)
    max_scaled_error = 0.
    # Include all rectangle corners, then random interior points.
    for _ in range(20000):
        t = 1/float(np.exp(rng.uniform(math.log(16), math.log(1e7))))
        delta = 52*t**1.5
        c = rng.uniform(math.sqrt(2/math.pi), 1)
        x = float(np.exp(rng.uniform(-15, 15)))
        bm, bp = b*(1-t/10-t*t), b*(1-t/10+t*t)
        cm = bm*math.sqrt((1+x)/(1+delta+x-2*bm*bm*t))
        cp = bp*math.sqrt((1+x)/(1-delta+x-2*bp*bp*t))
        for mu in (bm, bp, rng.uniform(bm, bp)):
            for nu in (1-delta, 1+delta, rng.uniform(1-delta, 1+delta)):
                den = nu+x-2*mu*mu*t
                assert den > 0
                crit = mu*math.sqrt((1+x)/den)
                assert cm-1e-12 <= crit <= cp+1e-12
                Delta = c*c*(nu+x)-mu*mu*(1+2*c*c*t+x)
                assert (Delta > 0) == (c > crit)
                R = c*math.sqrt(nu)/(mu*math.sqrt(1+2*c*c*t))
                center = c/b*(1+(.1-c*c)*t)
                scaled = abs(R-center)/((c/b)*t**1.5)
                max_scaled_error = max(max_scaled_error, scaled)
                assert scaled < 60
    gamma = 16
    # Use the exact margin integral, rather than its truncated prediction center.
    mu = quad(lambda p: (math.sqrt(1-p/gamma)*(1+p)/math.sqrt(p)
            + math.sqrt(p)*(1-p)/(gamma*math.sqrt(1-p/gamma)))/math.pi,
              0, 1, epsabs=1e-12)[0]
    delta = 52/gamma**1.5
    a, q = aq(.1)
    c = a/math.sqrt(q)
    examples = []
    for nu in (1-delta, 1+delta):
        examples.append({"nu": nu, "mu_exact": mu,
            "delta_no_noise": c*c*nu-mu*mu*(1+2*c*c/gamma),
            "formal_c_star": mu/math.sqrt(nu-2*mu*mu/gamma)})
    # Logical counterexample to existence from the envelope alone:
    # nu_d alternates between these two admissible values. This is not a
    # counterexample to the actual polar model.
    assert examples[0]["delta_no_noise"] < 0 < examples[1]["delta_no_noise"]
    return {"max_scaled_error_for_bound_60": max_scaled_error,
            "gamma16_same_margin_opposite_admissible_orderings": examples,
            "envelope_alone_does_not_imply_limit": True}


def root():
    mp.mp.dps = 65
    k, b = mp.sqrt(2/mp.pi), 8/(3*mp.pi)
    endpoints = []
    for e in map(mp.mpf, ["0.050156130219", "0.050156130220"]):
        a = k*mp.quad(lambda t: t*t/(t+e)*mp.exp(-t*t/2), [0, e, 1, mp.inf])
        q = k*mp.quad(lambda t: (t/(t+e))**2*mp.exp(-t*t/2), [0, e, 1, mp.inf])
        endpoints.append({"e": str(e), "D_float65": str(a*a-b*b*q)})
    assert mp.mpf(endpoints[0]["D_float65"]) < 0 < mp.mpf(endpoints[1]["D_float65"])
    return {"warning": "Independent high precision quadrature, not interval certification",
            "endpoints": endpoints}


for name, fn in [("root", root), ("noise", noise), ("envelope", envelope), ("arrays", arrays)]:
    results[name] = fn()
    print(name, "completed", round(time.monotonic()-started, 3), "seconds", flush=True)
results["source_hashes_unchanged"] = all(
    hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == h
    for name, h in fingerprints.items())
assert results["source_hashes_unchanged"]
results["elapsed_seconds"] = time.monotonic()-started
with (OUT/"results.json").open("x") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("All finite diagnostic assertions passed; no claim of an asymptotic proof.", flush=True)
