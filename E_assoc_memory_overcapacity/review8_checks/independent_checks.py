"""Independent review: no imports or execution of w9_checks code.

Run only under the resource command recorded in README.md.
All new artifacts stay in this directory. Existing inputs are read only.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from pathlib import Path
from fractions import Fraction
import itertools
import hashlib
import json
import time
import resource
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import t as student
from scipy.fft import dct

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
HERE = Path(os.environ.get("REVIEW8_OUTPUT_DIR", str(BASE))).resolve()
assert HERE.is_relative_to(BASE), "Review outputs must stay within review8_checks/"
HERE.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
ALPHA, ETA, EPS = .55, .45, .17
B1, B2, STEPS = .8, .93, 4
KAPPA, NQUERY, NSEED = .5, 384, 12
LAM = np.array([.2, 1.8])
METHODS = ["gd1", "gd4", "sign1", "sign4", "adam4"]


def save(name, obj):
    with (HERE / name).open("x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def fingerprint():
    paths = [ROOT / "W9_EXTENSION.md", ROOT / "THEOREM_M.md"]
    paths.extend(sorted(p for p in (ROOT / "w9_checks").rglob("*") if p.is_file()))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def evolve(source, right_multiply):
    # Independent optimizer implementation, with all snapshots retained explicitly.
    state = {name: np.zeros_like(source) for name in ("gd", "sign", "adam")}
    moment, variance = np.zeros_like(source), np.zeros_like(source)
    endpoints = {}
    for step in range(1, STEPS + 1):
        state["gd"] += ETA * (source - right_multiply(state["gd"]))
        state["sign"] += ALPHA * np.sign(source - right_multiply(state["sign"]))
        gradient = right_multiply(state["adam"]) - source
        moment = B1 * moment + (1 - B1) * gradient
        variance = B2 * variance + (1 - B2) * gradient ** 2
        state["adam"] -= ALPHA * moment / (1 - B1 ** step) / (
            np.sqrt(variance / (1 - B2 ** step)) + EPS)
        if step == 1:
            endpoints["gd1"] = state["gd"].copy()
            endpoints["sign1"] = state["sign"].copy()
    endpoints.update({"gd4": state["gd"], "sign4": state["sign"], "adam4": state["adam"]})
    return endpoints


def exact_sign_moments(lam):
    # Integrate the final piecewise constant function by propagating its intervals.
    intervals = [(-np.inf, np.inf, 0.)]
    for _ in range(STEPS):
        new = []
        for lo, hi, value in intervals:
            threshold = lam * value
            if lo < min(hi, threshold):
                new.append((lo, min(hi, threshold), value - ALPHA))
            if max(lo, threshold) < hi:
                new.append((max(lo, threshold), hi, value + ALPHA))
        intervals = new
    sd = np.sqrt(lam)
    phi = lambda z: np.exp(-z*z/2) / np.sqrt(2*np.pi)
    a = sum(value * sd * (phi(lo/sd) - phi(hi/sd)) for lo, hi, value in intervals)
    q = sum(value**2 * (ndtr(hi/sd) - ndtr(lo/sd)) for lo, hi, value in intervals)
    return a, q


def adam_scalar(z, lam):
    u = m = v = 0.
    for step in range(1, STEPS + 1):
        g = lam*u - z
        m = B1*m + (1-B1)*g
        v = B2*v + (1-B2)*g*g
        u -= ALPHA*m/(1-B1**step)/(np.sqrt(v/(1-B2**step)) + EPS)
    return u


def constants():
    result = {}
    for method in METHODS:
        if method.startswith("gd"):
            k = 1 if method == "gd1" else STEPS
            g = 1 - (1 - ETA * LAM)**k
            a, q = np.mean(g), np.mean(g*g)
        elif method == "sign1":
            a, q = ALPHA*np.sqrt(2/np.pi)*np.mean(np.sqrt(LAM)), ALPHA**2
        elif method == "sign4":
            moments = [exact_sign_moments(lam) for lam in LAM]
            a = np.mean([p[0] for p in moments])
            q = np.mean([lam*p[1] for lam, p in zip(LAM, moments)])
        else:
            moments = []
            for lam in LAM:
                density = lambda z: np.exp(-z*z/(2*lam))/np.sqrt(2*np.pi*lam)
                aa = 2*quad(lambda z: z*adam_scalar(z, lam)*density(z), 0, np.inf,
                            epsabs=1e-11, epsrel=1e-10, limit=400)[0]
                qq = 2*quad(lambda z: adam_scalar(z, lam)**2*density(z), 0, np.inf,
                            epsabs=1e-11, epsrel=1e-10, limit=400)[0]
                moments.append((aa, qq))
            a = np.mean([p[0] for p in moments])
            q = np.mean([lam*p[1] for lam, p in zip(LAM, moments)])
        result[method] = prediction(a, q)
    rho = .8
    result["block_sign1"] = prediction(ALPHA*np.sqrt(2/np.pi),
        ALPHA**2*(1 + 2*rho*np.arcsin(rho)/np.pi))
    return result


def prediction(a, q):
    c = a/np.sqrt(q)
    return dict(A=float(a), Q=float(q), C=float(c), mean=float(c/np.sqrt(KAPPA)),
                recall=float(ndtr(c/np.sqrt(KAPPA)-ndtri(.9))),
                rank=float(ndtr(-c/np.sqrt(2*KAPPA))))


def skew(rng, shape):
    # Mean zero, variance one, nonzero third moment, bounded support.
    return np.where(rng.random(shape) < 1/3, np.sqrt(2.), -1/np.sqrt(2.))


def one_dataset(case, d, seed, preds):
    n = round(KAPPA*d*d)
    rng = np.random.default_rng(np.random.SeedSequence([8117903, d, seed,
        ["diagonal_gaussian", "diagonal_skew_keys", "block_skew_keys", "dense_skew_both"].index(case)]))
    if case == "diagonal_gaussian":
        x = rng.standard_normal((d, n))
    else:
        x = skew(rng, (d, n))
    lam = np.tile(LAM, d//2)
    if case == "block_skew_keys":
        root = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
        root = root @ np.diag(np.sqrt(LAM)) @ root.T
        x = np.einsum("ij,bjn->bin", root, x.reshape(d//2, 2, n)).reshape(d, n)
        sigma = np.kron(np.eye(d//2), root @ root.T)
    elif case == "dense_skew_both":
        orth = dct(np.eye(d), norm="ortho", axis=0)
        root = (orth*np.sqrt(lam)) @ orth.T
        x = root @ x
        sigma = (orth*lam) @ orth.T
    else:
        x *= np.sqrt(lam)[:, None]
        sigma = np.diag(lam)
    x /= np.sqrt(d)
    y = (skew(rng, (d, n)) if case == "dense_skew_both" else rng.standard_normal((d, n))) / np.sqrt(d)
    s = d/n*(x @ x.T)
    source = d/np.sqrt(n)*(y @ x.T)
    if case == "dense_skew_both":
        # Only GD is asserted here. Do not interpret nonlinear endpoints as predictions.
        u = np.zeros_like(source)
        uh = np.zeros_like(source)
        actual, reference = {}, {}
        for step in range(1, STEPS+1):
            u += ETA*(source-u @ s)
            uh += ETA*(source-uh @ sigma)
            if step in (1, STEPS):
                name = "gd1" if step == 1 else "gd4"
                actual[name], reference[name] = u.copy(), uh.copy()
    elif case == "block_skew_keys":
        actual = {"sign1": ALPHA*np.sign(source)}
        reference = actual
    else:
        actual = evolve(source, lambda u: u @ s)
        reference = evolve(source, lambda u: u @ sigma)
    query = rng.choice(n, NQUERY, replace=False)
    rows = []
    for name, weights in actual.items():
        p = preds["block_sign1" if case == "block_skew_keys" else name]
        scores = (y.T @ (weights @ x[:, query])) / np.sqrt(p["Q"])
        correct = scores[query, np.arange(NQUERY)].copy()
        greater = np.count_nonzero(scores > correct[None, :], axis=0)
        equal = np.count_nonzero(scores == correct[None, :], axis=0)-1
        # A legal uniform tie rank, rather than a deterministic midrank.
        rank = 1 + greater + np.array([rng.integers(int(k)+1) for k in equal])
        scores[query, np.arange(NQUERY)] = np.nan
        ks = []
        for col in range(16):
            wrong = np.sort(scores[:, col][~np.isnan(scores[:, col])])
            normal = ndtr(wrong)
            ks.append(float(max(np.max(np.arange(1, n)/ (n-1)-normal),
                                np.max(normal-np.arange(n-1)/(n-1)))))
        difference = weights-reference[name]
        measurement = (y.T @ (difference @ x[:, query])) / np.sqrt(p["Q"])
        diagonal_error = measurement[query, np.arange(NQUERY)].copy()
        measurement[query, np.arange(NQUERY)] = 0.
        rows.append(dict(case=case, d=d, n=n, seed=seed, method=name,
            correct_mean=float(np.mean(correct)), correct_var=float(np.var(correct, ddof=1)),
            wrong_var=float(np.nanvar(scores)), recall=float(np.mean(rank <= np.ceil(.1*n))),
            rank=float(np.mean((rank-1)/(n-1))), ks_mean=float(np.mean(ks)), ks_max=float(np.max(ks)),
            tied_queries=int(np.count_nonzero(equal)), tie_total=int(np.sum(equal)),
            frobenius=float(np.mean(difference**2)), correct_error=float(np.mean(diagonal_error**2)),
            wrong_error=float(np.sum(measurement**2)/((n-1)*NQUERY)), prediction=p))
    return rows


def exact_row_certificate():
    # Rational enumeration with a three-point law (not W9's two-point audit).
    # Z in {-3,0,1}: probabilities 1/12,2/3,1/4; mean 0, variance 1, kurtosis 7.
    support = [Fraction(-3), Fraction(0), Fraction(1)]
    probabilities = [Fraction(1,12), Fraction(2,3), Fraction(1,4)]
    weights = [Fraction(2,7), Fraction(-1,5), Fraction(1,4), Fraction(3,11)]
    means = [Fraction(0)]*3
    second = [[Fraction(0)]*3 for _ in range(3)]
    for inds in itertools.product(range(3), repeat=4):
        p = np.prod([probabilities[k] for k in inds])
        z = [support[k] for k in inds]
        f = sum(a*b for a,b in zip(weights,z))
        tf = [z[k]*f for k in range(3)]
        for j in range(3):
            means[j] += p*tf[j]
            for k in range(3):
                second[j][k] += p*tf[j]*tf[k]
    mu4 = sum(p*z**4 for p,z in zip(probabilities,support))
    norm2 = sum(t*t for t in weights)
    assert means == weights[:3]
    covariance = []
    for j in range(3):
        covariance.append([])
        for k in range(3):
            empirical = second[j][k]-means[j]*means[k]
            predicted = (norm2 if j == k else 0) + weights[j]*weights[k]
            if j == k:
                predicted += (mu4-3)*weights[j]**2
            assert empirical == predicted
            covariance[j].append(str(empirical))
    return dict(law="P(-3)=1/12,P(0)=2/3,P(1)=1/4", third_moment=-2,
                fourth_moment=int(mu4), states=81, exact_identity=True, covariance=covariance)


def tie_counterexample():
    # Open Gaussian-support event around X=I, Y=.1*ones; Sign2 and Sign4 are zero.
    x, y = np.eye(2), np.full((2,2), .1)
    s = x @ x.T
    source = np.sqrt(2)*(y @ x.T)
    u, states = np.zeros((2,2)), []
    for _ in range(4):
        u += ALPHA*np.sign(source-u @ s)
        states.append(u.tolist())
    assert np.count_nonzero(u) == 0
    return dict(d=2,n=2,alpha=ALPHA,X=x.tolist(),Y=y.tolist(),states=states,
        strict_margin=float(min(np.min(source),np.min(ALPHA-source))),
        all_candidate_scores_zero=True,positive_probability_under_gaussian=True)


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["case"],r["d"],r["method"]), []).append(r)
    result = []
    for (case,d,method), rr in sorted(groups.items()):
        out = dict(case=case,d=d,method=method,seeds=len(rr),prediction=rr[0]["prediction"])
        for key in ("recall","rank","correct_mean","correct_var","wrong_var","ks_mean","ks_max",
                    "frobenius","correct_error","wrong_error"):
            vals = np.array([r[key] for r in rr])
            out[key] = dict(mean=float(vals.mean()), half95=float(student.ppf(.975,len(rr)-1)*vals.std(ddof=1)/np.sqrt(len(rr))))
        out["tie_total"] = sum(r["tie_total"] for r in rr)
        result.append(out)
    return result


def main():
    observed_threads = len(list(Path("/proc/self/task").iterdir()))
    assert observed_threads <= 2
    initial = fingerprint()
    save("input_hashes_before.json", initial)
    save("exact_certificates.json", dict(row_identity=exact_row_certificate(),gaussian_tie_counterexample=tie_counterexample()))
    preds = constants()
    save("predictions.json", preds)
    all_rows = []
    cases = ["diagonal_gaussian", "diagonal_skew_keys", "block_skew_keys", "dense_skew_both"]
    with (HERE/"rows.jsonl").open("x") as f:
        for d in (64,128,256):
            for seed in range(NSEED):
                for case in cases:
                    rows = one_dataset(case,d,seed,preds)
                    all_rows.extend(rows)
                    for row in rows:
                        f.write(json.dumps(row)+"\n")
                    f.flush()
                    observed_threads = max(observed_threads, len(list(Path("/proc/self/task").iterdir())))
                    assert observed_threads <= 2
                print(json.dumps(dict(d=d,seed=seed,elapsed=round(time.monotonic()-START,2))), flush=True)
    save("summary.json",summarize(all_rows))
    final = fingerprint()
    assert final == initial, "Existing review inputs changed during the run"
    save("input_hashes_after.json",final)
    metadata = dict(seconds=time.monotonic()-START,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=os.sys.executable,numpy=np.__version__,kappa=KAPPA,dimensions=[64,128,256],
        queries=NQUERY,seeds=NSEED,alpha=ALPHA,eta=ETA,epsilon=EPS,beta1=B1,beta2=B2,steps=STEPS,
        eigenvalues=LAM.tolist(),rng_seed_base=8117903,datasets=3*NSEED*len(cases),rows=len(all_rows),
        observed_process_threads_max=observed_threads,
        thread_limits={k:os.environ[k] for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
        inputs_unchanged=True,external_code_imports=False)
    save("metadata.json",metadata)
    print(json.dumps(metadata),flush=True)


if __name__ == "__main__":
    main()
