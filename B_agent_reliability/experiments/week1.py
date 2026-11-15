"""Offline preregistered Week 1 experiments. See preregistration_week1.md.

Run with run_week1.sh. Raw data are JSON or streamed JSON.gz; no network access.
"""
import os
os.environ.update(OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  MPLCONFIGDIR="/tmp/theory-week1-mpl")
import resource
resource.setrlimit(resource.RLIMIT_AS, (3_500_000_000, 3_500_000_000))
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time
import numpy as np
import torch
torch.set_num_threads(3)
from scipy.special import digamma, expit, gammaln, ndtri
from scipy.linalg import solve_banded
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import gamma as gamma_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "week1_results"
RAW = OUT / "raw"
FIG = OUT / "figures"
for directory in (OUT, RAW, FIG):
    directory.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
STAGE_START = START
STAGE_LIMIT = 480


def clean(obj):
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def save(path, obj):
    path = Path(path)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rng_for(*keys):
    return np.random.default_rng(np.random.SeedSequence(keys))


def checkpoint():
    if time.monotonic() - STAGE_START > STAGE_LIMIT:
        raise TimeoutError("Stage hard limit reached; partial JSON remains on disk")


def h(k, rho):
    if k == 1:
        return 1.0
    return 1.0 / sum(rho ** j for j in range(k))


def w(k, rho):
    return k - 1.0 if abs(rho - 1) < 1e-10 else (1 + rho) * (k * h(k, rho) - 1) / (1 - rho)


def solve_chain(ratios):
    p = 1 / (1 + np.asarray(ratios, dtype=float))
    d = 1 - p
    m = len(p)
    band = np.zeros((3, m))
    band[1] = 1
    band[0, 1:] = -p[:-1]
    band[2, :-1] = -d[1:]
    rhs = np.zeros((m, 2))
    rhs[-1, 0] = p[-1]
    rhs[:, 1] = 1
    ans = solve_banded((1, 1), band, rhs)
    return float(ans[0, 0]), float(ans[0, 1])


def simulate(rng, k, ratios, n, eta=0.0):
    """Every internal +/-1 transition is drawn; no closed-form marks."""
    ratios = np.asarray(ratios)
    state = np.ones(n, dtype=np.int32)
    steps = np.zeros(n, dtype=np.int32)
    bypass = rng.random(n) < eta
    state[bypass] = k
    U = np.zeros(k - 1, dtype=np.int64)
    D = U.copy()
    active = np.flatnonzero(~bypass)
    while active.size:
        old = state[active]
        up = rng.random(active.size) < 1 / (1 + ratios[old - 1])
        U += np.bincount(old[up] - 1, minlength=k - 1)
        D += np.bincount(old[~up] - 1, minlength=k - 1)
        state[active] += 2 * up.astype(np.int32) - 1
        steps[active] += 1
        active = active[(state[active] > 0) & (state[active] < k)]
    return state == k, steps, U, D


def wilson(success, n, z):
    p = success / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0., float(center - half)), min(1., float(center + half))]


def fragment_stats(fail, steps, U, D, rng, z):
    n = len(steps)
    joint = np.bincount(steps * 2 + fail.astype(int))
    idx = np.flatnonzero(joint)
    hist = joint[idx]
    boot = rng.multinomial(n, hist / n, size=200) @ (idx // 2) / n
    return dict(n=n, failures=int(fail.sum()), failure_rate=float(fail.mean()),
                failure_wilson=wilson(int(fail.sum()), n, z),
                mean_work=float(steps.mean()), work_se=float(steps.std(ddof=1) / np.sqrt(n)),
                work_bootstrap_ci=np.quantile(boot, [.025, .975]),
                work_p95=float(np.quantile(steps, .95)), work_p99=float(np.quantile(steps, .99)),
                over_100=float(np.mean(steps > 100)),
                budget_success=float(np.mean((~fail) & (steps <= 100))),
                up=U, down=D, joint_histogram=[[int(i // 2), int(i % 2), int(c)] for i, c in zip(idx, hist)])


def adam(fun, initial, max_steps=1500):
    x = np.asarray(initial, dtype=float).copy()
    m = np.zeros_like(x)
    v = m.copy()
    old = math.inf
    stable = 0
    best = (math.inf, x.copy())
    for step in range(1, max_steps + 1):
        val, grad = fun(x)
        if val < best[0]:
            best = (float(val), x.copy())
        stable = stable + 1 if abs(val - old) < 1e-7 else 0
        if stable >= 50:
            break
        old = val
        m = .9 * m + .1 * grad
        v = .999 * v + .001 * grad * grad
        x -= .02 * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
        x = np.clip(x, -25, 25)
    return dict(loss=best[0], log_parameters=best[1], steps=step, converged=stable >= 50)


def nb_fit(counts):
    hist = np.bincount(counts)
    M = np.arange(len(hist))
    freq = hist / len(counts)
    def objective(x):
        alpha, beta = np.exp(x)
        logp = (gammaln(M + alpha) - gammaln(alpha) - gammaln(M + 1)
                + alpha * (np.log(beta) - np.log(beta + 100))
                + M * (np.log(100) - np.log(beta + 100)))
        da = alpha * (digamma(M + alpha) - digamma(alpha) + np.log(beta / (beta + 100)))
        db = alpha - beta * (alpha + M) / (beta + 100)
        return -float(freq @ logp), -np.array([freq @ da, freq @ db])
    trials = [adam(objective, np.log(x)) for x in ((.5, 5), (1, 10), (2, 20))]
    best = min(trials, key=lambda r: r["loss"])
    mu, var = counts.mean(), counts.var()
    mm_alpha = mu * mu / (var - mu) if var > mu else math.nan
    return dict(alpha=float(np.exp(best["log_parameters"][0])), beta=float(np.exp(best["log_parameters"][1])),
                moment_alpha=mm_alpha, moment_beta=100 * mm_alpha / mu, trials=trials,
                count_histogram=hist)


def log_curve(model, x, pars):
    if model == "exponential":
        logp = -np.exp(pars[0]) * x
        deriv = logp[:, None]
    elif model == "gamma":
        c, alpha = np.exp(pars)
        xx = c * x
        logp = -alpha * np.log1p(xx)
        deriv = np.column_stack((-alpha * xx / (1 + xx), logp))
    else:
        logH, logb = pars
        b = np.exp(logb)
        z = b * (np.log(x) - logH)
        logp = -np.logaddexp(0, z)
        deriv = np.column_stack((b * expit(z), -z * expit(z)))
    return np.minimum(logp, -1e-12), deriv


def fit_curve(model, x, y, max_steps=1000):
    x, y = np.asarray(x), np.asarray(y)
    def objective(pars):
        logp, deriv = log_curve(model, x, pars)
        p = np.exp(logp)
        log1mp = np.log(-np.expm1(logp))
        val = -np.mean(y * logp + (1 - y) * log1mp)
        coeff = (p - y) / (-np.expm1(logp))
        return float(val), np.mean(coeff[:, None] * deriv, axis=0)
    center = np.median(x)
    if model == "exponential":
        starts = [[np.log(v / center)] for v in (.2, .7, 2.)]
    elif model == "gamma":
        starts = [np.log([1 / center, a]) for a in (.5, 1., 2.)]
    else:
        starts = [[np.log(center), np.log(b)] for b in (.5, 1., 2.)]
    trials = [adam(objective, initial, max_steps) for initial in starts]
    best = min(trials, key=lambda r: r["loss"])
    return dict(model=model, **best, trials=trials)


def e1():
    rows, calibrations, tests = [], [], []
    z = float(ndtri(1 - .01 / (2 * 96)))
    t0 = time.monotonic()
    simulate(rng_for(10), 16, [1.] * 15, 1000)
    pilot_seconds = time.monotonic() - t0
    n = 20_000
    for seed in (11, 23, 37):
        for ri, rho in enumerate((.5, 1., 1.5, 2.)):
            for k in (2, 4, 8, 12, 16):
                checkpoint()
                fail, steps, U, D = simulate(rng_for(seed, 1, ri, k), k, [rho] * (k - 1), n)
                rec = fragment_stats(fail, steps, U, D, rng_for(seed, 101, ri, k), z)
                sh, sw = solve_chain([rho] * (k - 1))
                rec.update(seed=seed, rho=rho, k=k, exact_h=h(k, rho), exact_w=w(k, rho),
                           linear_h=sh, linear_w=sw,
                           probability_pass=rec["failure_wilson"][0] <= h(k, rho) <= rec["failure_wilson"][1],
                           work_pass=abs(rec["mean_work"] - w(k, rho)) <= max(.03 * w(k, rho), 4 * rec["work_se"]))
                rows.append(rec)
            fail, steps, U, D = simulate(rng_for(seed, 2, ri), 3, [rho] * 2, n)
            rh = float(D.sum() / U.sum()) if U.sum() else None
            M = int(rng_for(seed, 3, ri).poisson(1000))
            cal = fragment_stats(fail, steps, U, D, rng_for(seed, 102, ri), z)
            cal.update(seed=seed, true_rho=rho, rho_hat=rh, theta_hat=M / 10000, M=M, T=10000)
            calibrations.append(cal)
            for k in (4, 8, 12):
                fail, steps, U, D = simulate(rng_for(seed, 4, ri, k), k, [rho] * (k - 1), n)
                rec = fragment_stats(fail, steps, U, D, rng_for(seed, 104, ri, k), z)
                predh, predw = h(k, rh), w(k, rh)
                rec.update(seed=seed, rho=rho, k=k, rho_hat=rh, predicted_h=predh, predicted_w=predw,
                           exact_h=h(k, rho), exact_w=w(k, rho), rare_failure=rec["failures"] < 30,
                           prediction_probability_pass=rec["failure_wilson"][0] <= predh <= rec["failure_wilson"][1],
                           prediction_work_pass=abs(rec["mean_work"] - predw) <= max(.03 * predw, 4 * rec["work_se"]))
                tests.append(rec)
    res = dict(pilot_seconds=pilot_seconds, n_per_cell=n, z=z, family_size=96, formula=rows,
               calibration=calibrations, heldout=tests,
               probability_passes=sum(r["probability_pass"] for r in rows),
               work_passes=sum(r["work_pass"] for r in rows),
               extrap_probability_passes=sum(r["prediction_probability_pass"] for r in tests),
               extrap_work_passes=sum(r["prediction_work_pass"] for r in tests))
    save(OUT / "E1.json", res)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for rho in (.5, 1., 1.5, 2.):
        rr = [r for r in rows if r["rho"] == rho and r["seed"] == 11]
        kk = [r["k"] for r in rr]
        ax[0].semilogy(kk, [r["exact_h"] for r in rr], label=f"rho={rho}")
        ax[0].scatter(kk, [max(r["failure_rate"], 1e-6) for r in rr], s=15)
        ax[1].plot(kk, [r["mean_work"] for r in rr], "o-", label=f"rho={rho}")
    ax[0].set(xlabel="K", ylabel="fragment failure probability")
    ax[1].set(xlabel="K", ylabel="mean internal events")
    for a in ax:
        a.legend()
    fig.tight_layout(); fig.savefig(FIG / "E1.png", dpi=150); plt.close(fig)
    return res


def stream_matrix(path, metadata, matrix):
    """Bounded memory JSON serialization; row i keeps task identity across L."""
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as f:
        prefix = json.dumps(clean(metadata), separators=(",", ":"), allow_nan=False)
        f.write(prefix[:-1] + ',"rows":[')
        for i, row in enumerate(matrix):
            if i:
                f.write(",")
            f.write(json.dumps(row.tolist(), separators=(",", ":")))
        f.write("]}\n")


def task_bootstrap(counts, seed):
    # Exact n-out-of-n bootstrap of task rows, in blocks of 5 replicates.
    n = len(counts)
    rng = rng_for(seed)
    vals = counts.astype(np.float64)
    ans = np.empty((200, counts.shape[1]))
    for start in range(0, 200, 5):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=5)
        ans[start:start + 5] = weights @ vals / (8 * n)
    return ans


def task_cost(rng, theta, k, L, rho=2.):
    n = len(theta)
    exposure = np.zeros(n)
    work = np.zeros(n)
    success = np.zeros(n, dtype=bool)
    alive = np.ones(n, dtype=bool)
    while alive.any():
        ids = np.flatnonzero(alive)
        delta = rng.exponential(1 / theta[ids])
        advance = np.minimum(delta, L - exposure[ids])
        work[ids] += advance
        exposure[ids] += advance
        finished = exposure[ids] >= L
        success[ids[finished]] = True
        alive[ids[finished]] = False
        ids = ids[~finished]
        if len(ids):
            fail, steps, _, _ = simulate(rng, k, [rho] * (k - 1), len(ids))
            work[ids] += steps
            alive[ids[fail]] = False
    return dict(n=n, L=L, success=float(success.mean()), budget_success=float(np.mean(success & (work <= 5 * L))),
                over_budget=float(np.mean(work > 5 * L)), mean_work_over_L=float(work.mean() / L),
                p95_work_over_L=float(np.quantile(work / L, .95)), p99_work_over_L=float(np.quantile(work / L, .99)),
                work=work, successes=success.astype(int))


def e2():
    lengths = np.logspace(1, 6, 21)
    summaries, families, costs = [], [], []
    for ai, alpha in enumerate((.5, 1., 2.)):
        beta = 10 * alpha
        rng = rng_for(2201, ai)
        theta = rng.gamma(alpha, 1 / beta, 100_000)
        M = rng.poisson(100 * theta)
        fit = nb_fit(M)
        families.append(dict(alpha=alpha, beta=beta, fit=fit))
        stream_matrix(RAW / f"E2_tasks_a{alpha}.json.gz", dict(columns=["theta", "calibration_count"], seed=[2201, ai]), np.column_stack((theta, M)))
        configs = [("fixed", k) for k in (4, 8, 12)] + [("log", a) for a in (.5, 1., 2.)]
        for ci, (kind, cap) in enumerate(configs):
            ks = np.full(21, int(cap)) if kind == "fixed" else np.maximum(2, np.ceil(cap * np.log(lengths))).astype(int)
            for ei, eta in enumerate((0., .001, .01)):
                checkpoint()
                g = np.array([eta + (1 - eta) * h(int(k), 2.) for k in ks])
                counts = np.empty((100_000, 21), dtype=np.uint8)
                rng = rng_for(2202, ai, ci, ei)
                for li, L in enumerate(lengths):
                    prob = np.exp(-theta * g[li] * L)
                    counts[:, li] = rng.binomial(8, prob)
                observed = counts.mean(axis=0) / 8
                boot = task_bootstrap(counts, 2250 + ai)
                exact = (1 + lengths * g / beta) ** (-alpha)
                frozen = (1 + lengths * g / fit["beta"]) ** (-fit["alpha"])
                models = {}
                for model in ("exponential", "loglogistic", "gamma"):
                    fitted = fit_curve(model, lengths[:11], observed[:11])
                    pred = np.exp(log_curve(model, lengths, fitted["log_parameters"])[0])
                    # Every individual run is binary: E[(Y-p)^2] = y(1-2p)+p^2.
                    brier = observed * (1 - 2 * pred) + pred * pred
                    fitted.update(prediction=pred, heldout_brier=float(brier[11:].mean()))
                    models[model] = fitted
                mask = (np.arange(21) >= 11) & (counts.sum(axis=0) >= 100) & ((800_000 - counts.sum(axis=0)) >= 100)
                failure_slope = kind == "log" and cap == 2 and eta == 0
                quantity = 1 - observed if failure_slope else observed
                qexact = 1 - exact if failure_slope else exact
                slope = exact_slope = slope_ci = None
                if mask.sum() >= 3:
                    xx = np.log(lengths[mask]); centered = xx - xx.mean()
                    denom = centered @ centered
                    slope = float(-centered @ np.log(quantity[mask]) / denom)
                    exact_slope = float(-centered @ np.log(qexact[mask]) / denom)
                    qb = 1 - boot[:, mask] if failure_slope else boot[:, mask]
                    sb = -(np.log(np.maximum(qb, 1e-12)) @ centered) / denom
                    slope_ci = np.quantile(sb, [.025, .975])
                rec = dict(alpha=alpha, beta=beta, kind=kind, capacity=cap, eta=eta, ks=ks, lengths=lengths,
                           observed=observed, exact=exact, frozen_gamma_poisson=frozen,
                           task_bootstrap_ci=np.quantile(boot, [.025, .975], axis=0).T,
                           task_se=counts.std(axis=0, ddof=1) / (8 * np.sqrt(len(counts))),
                           heldout_max_abs_error=float(np.max(np.abs(observed[11:] - frozen[11:]))),
                           frozen_heldout_brier=float(np.mean(observed[11:] * (1 - 2 * frozen[11:]) + frozen[11:] ** 2)),
                           curve_models=models, tail_quantity="failure" if failure_slope else "success",
                           tail_point_indices=np.flatnonzero(mask), tail_exponent=slope,
                           finite_length_exact_exponent=exact_slope, tail_exponent_ci=slope_ci)
                if kind == "fixed":
                    gh = g[0]
                    H50 = beta * (2 ** (1 / alpha) - 1) / gh
                    H80 = beta * (.8 ** (-1 / alpha) - 1) / gh
                    xx = gh * lengths / beta
                    curve_b = alpha * xx / ((1 + xx) * (-np.expm1(-alpha * np.log1p(xx))))
                    rec.update(H50=H50, H80=H80, H80_over_H50=H80 / H50, local_logit_slope=curve_b,
                               logit_slope_range=[float(curve_b.min()), float(curve_b.max())])
                    if eta == 0:
                        diag = task_cost(rng_for(2260, ai, ci), theta[:2000], int(cap), H50)
                        save(RAW / f"E2_cost_a{alpha}_K{cap}.json", diag)
                        costs.append(dict(alpha=alpha, k=cap, **{key: val for key, val in diag.items() if key not in ("work", "successes")}))
                rawname = f"E2_a{alpha}_{kind}{cap}_eta{eta}.json.gz"
                stream_matrix(RAW / rawname, dict(seed=[2202, ai, ci, ei], bootstrap_seed=2250 + ai,
                                                 lengths=lengths, repeat_count=8, task_file=f"E2_tasks_a{alpha}.json.gz"), counts)
                rec["raw_file"] = rawname
                summaries.append(rec)
                save(OUT / "E2_partial.json", dict(families=families, configurations=summaries, cost_diagnostics=costs))
                print(f"E2 alpha={alpha} {kind}={cap} eta={eta} done", flush=True)
                del counts, boot
                gc.collect()
        del theta, M
    res = dict(families=families, configurations=summaries, cost_diagnostics=costs)
    save(OUT / "E2.json", res)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a in (.5, 1., 2.):
        rr = next(r for r in summaries if r["alpha"] == 1 and r["kind"] == "log" and r["capacity"] == a and r["eta"] == 0)
        ax[0].loglog(lengths, rr["exact"], label=f"a={a}")
        ax[0].scatter(lengths, rr["observed"], s=8)
    for eta in (0., .001, .01):
        rr = [r for r in summaries if r["alpha"] == 1 and r["kind"] == "fixed" and r["eta"] == eta]
        ax[1].semilogy([r["capacity"] for r in rr], [r["H50"] for r in rr], "o-", label=f"eta={eta}")
    ax[0].set(xlabel="L", ylabel="success probability")
    ax[1].set(xlabel="K", ylabel="H50")
    for a in ax: a.legend()
    fig.tight_layout(); fig.savefig(FIG / "E2.png", dpi=150); plt.close(fig)
    return res


def episode_law(k, rho=2.):
    """Exact joint law of (tau, endpoint) by transient propagation, not test data."""
    p = 1 / (1 + rho); d = 1 - p
    transient = np.zeros(k + 1); transient[1] = 1
    recover, fail = [0.], [0.]
    for step in range(1, 10001):
        recover.append(float(transient[1] * d))
        fail.append(float(transient[k - 1] * p))
        nxt = np.zeros(k + 1)
        nxt[2:k] += transient[1:k - 1] * p
        nxt[1:k - 1] += transient[2:k] * d
        transient = nxt
        if transient.sum() < 1e-13:
            break
    probs = np.column_stack((recover, fail)).ravel()
    probs /= probs.sum()
    return probs, float(transient.sum())


def nonregen_tasks(rng, k, theta, maxL, n=10000):
    law, tail = episode_law(k)
    cdf = np.cumsum(law); cdf[-1] = 1
    Z = np.zeros(n); exposure = np.zeros(n); death = np.full(n, maxL)
    internal = np.zeros(n); overfrag = np.zeros(n, dtype=bool)
    alive = np.ones(n, dtype=bool)
    while alive.any():
        ids = np.flatnonzero(alive)
        delta = rng.exponential(1 / (theta * (1 + .5 * Z[ids])))
        exposure[ids] += delta
        stop = exposure[ids] >= maxL
        alive[ids[stop]] = False
        ids = ids[~stop]
        if len(ids):
            draw = np.searchsorted(cdf, rng.random(len(ids)))
            tau, failed = draw // 2, draw % 2 == 1
            internal[ids] += tau
            overfrag[ids] |= tau > 100
            Z[ids] = .8 * Z[ids] + (tau > 4)
            death[ids[failed]] = exposure[ids[failed]]
            alive[ids[failed]] = False
    return death, internal, overfrag, tail


def rejected_probability(pred, observed, n):
    se = math.sqrt(observed * (1 - observed) / n)
    sufficient = n * min(observed, 1 - observed) >= 30
    return bool(sufficient and abs(pred - observed) > max(.03, 5 * se)), se


def e3():
    results = []
    for env in (1, 2, 3):
        eta = .01 if env == 3 else 0.
        fail, steps, U, D = simulate(rng_for(3301, env), 3, [2., 2.], 30000, eta)
        rh = float(D.sum() / U.sum())
        eh = float(np.mean(steps == 0))
        calibration = fragment_stats(fail, steps, U, D, rng_for(3302, env), ndtri(.975))
        calibration.update(rho_hat=rh, eta_hat_from_zero_events=eh,
                           rho_by_state=np.divide(D, U, out=np.full(2, np.nan), where=U > 0),
                           unobserved_states="n>=3 not identifiable from K=3",
                           eta_without_event_counts="not separately identified from absorption alone")
        # Clean calibration for seed intensity; contamination is absent initially.
        intervals = rng_for(3303, env).exponential(10., 30000)
        th = float(1 / intervals.mean())
        calibration["theta_hat_clean"] = th
        rows = []
        for k in (6, 10, 16):
            checkpoint()
            ratios = [2. if env != 1 or n <= 3 else .4 for n in range(1, k)]
            fail, steps, U, D = simulate(rng_for(3310, env, k), k, ratios, 30000, eta)
            rec = fragment_stats(fail, steps, U, D, rng_for(3311, env, k), ndtri(.975))
            predh = h(k, rh)
            expg = eh + (1 - eh) * predh
            exacth, _ = solve_chain(ratios)
            trueg = eta + (1 - eta) * exacth
            L = math.log(2) / (th * predh)
            reject_h, se_h = rejected_probability(predh, rec["failure_rate"], 30000)
            task_n = 10000
            if env == 2:
                death, internal, fragover, tail = nonregen_tasks(rng_for(3320, env, k), k, .1, 2 * L, task_n)
                success = death > L
                grid = np.linspace(0, 2 * L, 401)
                surv = np.array([(death > x).mean() for x in grid])
                crossing = np.flatnonzero(surv <= .5)
                Hinterval = [float(grid[crossing[0] - 1]), float(grid[crossing[0]])] if len(crossing) else [2 * L, None]
                trueH = float(np.median(death))
                # Work at L cannot use work accumulated after L; only report joint
                # death law here. Full task costs are in independent E2 diagnostics.
                save(RAW / f"E3_nonregen_K{k}.json", dict(death_exposure=death, censoring_horizon=2 * L,
                                                          internal_work_to_death_or_censor=internal,
                                                          fragment_over100_to_death_or_censor=fragover.astype(int),
                                                          joint_law_tail=tail, seed=[3320, env, k]))
                nonregen = dict(H50_interval=Hinterval, survival_grid=grid, observed_survival=surv,
                               joint_law_tail=tail, H50_estimate=trueH)
            else:
                task_rng = rng_for(3320, env, k)
                success = task_rng.random(task_n) < math.exp(-.1 * trueg * L)
                trueH = math.log(2) / (.1 * trueg)
                nonregen = None
                save(RAW / f"E3_tasks_env{env}_K{k}.json", dict(seed=[3320, env, k], L=L, successes=success.astype(int)))
            obs = float(success.mean())
            task_reject, se = rejected_probability(.5, obs, task_n)
            explicit_pred = math.exp(-th * expg * L)
            explicit_reject, _ = rejected_probability(explicit_pred, obs, task_n)
            ratioH = max(L / trueH, trueH / L)
            rec.update(k=k, predicted_h_constant=predh, predicted_g_explicit=expg,
                       exact_failure_probability=trueg, fragment_rejected=reject_h, fragment_se=se_h,
                       L=L, task_n=task_n, predicted_task_success_constant=.5,
                       predicted_task_success_explicit=explicit_pred, observed_task_success=obs, task_se=se,
                       task_rejected_constant=task_reject, task_rejected_explicit=explicit_reject,
                       H50_constant=L, H50_true_or_MC=trueH, H50_ratio=ratioH,
                       H50_rejected=ratioH > 2, mechanism_rejected=reject_h or task_reject or ratioH > 2,
                       state_model_prediction=None, state_model_status="unidentifiable for n>=3",
                       nonregeneration=nonregen)
            rows.append(rec)
        # Sequential intervals diagnose residual history dependence independently.
        history = None
        if env == 2:
            rng = rng_for(3330)
            Z = 0.; zz = []; gaps = []
            # Independent episodes reset physical state; carry only pollution.
            law, _ = episode_law(3); cdf = np.cumsum(law); cdf[-1] = 1
            for i in range(30000):
                zz.append(Z); gaps.append(rng.exponential(10 / (1 + .5 * Z)))
                tau = np.searchsorted(cdf, rng.random()) // 2
                Z = .8 * Z + (tau > 4)
            zz, gaps = np.asarray(zz), np.asarray(gaps)
            low, high = zz <= np.quantile(zz, .25), zz >= np.quantile(zz, .75)
            history = dict(clean_theta_hat=th, overall_rate=float(1 / gaps.mean()),
                           low_Z_rate=float(1 / gaps[low].mean()), high_Z_rate=float(1 / gaps[high].mean()),
                           gap_Z_correlation=float(np.corrcoef(zz, gaps)[0, 1]))
            stream_matrix(RAW / "E3_nonregen_intervals.json.gz", dict(columns=["Z", "gap"], seed=[3330]), np.column_stack((zz, gaps)))
        results.append(dict(environment=env, calibration=calibration, tests=rows, history=history))
    res = dict(environments=results, mandatory_negative_control_pass=all(r["mechanism_rejected"] for r in results[0]["tests"]))
    save(OUT / "E3.json", res)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, env in zip(axes, results):
        rr = env["tests"]
        ax.plot([r["k"] for r in rr], [r["observed_task_success"] for r in rr], "o-", label="observed")
        ax.axhline(.5, ls="--", color="black", label="frozen constant")
        ax.plot([r["k"] for r in rr], [r["predicted_task_success_explicit"] for r in rr], "x--", label="explicit bypass")
        ax.set(title=f"Environment {env['environment']}", xlabel="K", ylabel="success at predicted H50", ylim=(-.03, 1))
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "E3.png", dpi=150); plt.close(fig)
    return res


def ece(y, p):
    bins = np.minimum((p * 10).astype(int), 9)
    return float(sum(np.mean(bins == b) * abs(y[bins == b].mean() - p[bins == b].mean())
                     for b in range(10) if np.any(bins == b)))


def coverage_integral(L, N, alpha, beta, g=1.):
    # Rescale by Gamma quantile u=theta*beta and split around transition.
    def integrand(u):
        if u <= 0:
            return 0.
        s = math.exp(-u * g * L / beta)
        cov = 1. if s >= 1 else -math.expm1(N * math.log1p(-s))
        return cov * math.exp((alpha - 1) * math.log(u) - u - math.lgamma(alpha))
    transition = beta * math.log(max(N, 2)) / (g * max(L, 1e-20))
    cut = min(40., max(1e-8, transition))
    return quad(integrand, 0, cut, epsabs=1e-9)[0] + quad(integrand, cut, np.inf, epsabs=1e-9)[0]


def e4(e2res):
    source = ROOT / "data/eval-analysis-public/reports/time-horizon-1-1/data/raw/runs.jsonl"
    groups = {}; fields = set(); versions = set(); row_count = 0; skipped = 0
    with source.open() as stream:
        for line in stream:
            row_count += 1
            r = json.loads(line); fields.update(r)
            if r.get("alias") == "human" or not r.get("human_minutes") or r["human_minutes"] <= 0 or r.get("score_binarized") not in (0, 1):
                skipped += 1; continue
            alias = r["alias"]; task = r["task_id"]
            table = groups.setdefault(alias, {})
            versions.add(str(r.get("task_version")))
            if task not in table:
                table[task] = dict(task_id=task, family=r.get("task_family") or task, t=float(r["human_minutes"]), n=0, s=0,
                                   human_minutes_inconsistent=False)
            rr = table[task]
            rr["human_minutes_inconsistent"] |= rr["t"] != float(r["human_minutes"])
            rr["n"] += 1; rr["s"] += int(r["score_binarized"])
    eligible = [name for name, table in groups.items() if len(table) >= 40 and
                sum(r["s"] for r in table.values()) >= 10 and sum(r["n"] - r["s"] for r in table.values()) >= 10]
    selected = sorted(eligible)[:3]
    models = []
    for alias in selected:
        checkpoint()
        tasks = sorted(groups[alias].values(), key=lambda r: r["task_id"])
        save(RAW / f"E4_tasks_model{len(models)}.json", dict(alias=alias, tasks=tasks))
        x = np.array([r["t"] for r in tasks]); y = np.array([r["s"] / r["n"] for r in tasks])
        fam = np.array([r["family"] for r in tasks]); uf = np.unique(fam)
        shuffled = rng_for(4401).permutation(uf)
        mapping = {f: i % 5 for i, f in enumerate(shuffled)}
        folds = np.array([mapping[f] for f in fam])
        fits, predictions, loss = {}, {}, {}
        for model in ("exponential", "loglogistic", "gamma"):
            pred = np.zeros(len(tasks)); details = []
            for fold in range(5):
                fit = fit_curve(model, x[folds != fold], y[folds != fold])
                pred[folds == fold] = np.exp(log_curve(model, x[folds == fold], fit["log_parameters"])[0])
                details.append(fit)
            pred = np.clip(pred, 1e-12, 1 - 1e-12)
            losses = -y * np.log(pred) - (1 - y) * np.log1p(-pred)
            fits[model] = dict(folds=details, cv_log_loss=float(losses.mean()), ECE=ece(y, pred))
            predictions[model] = pred; loss[model] = losses
        boots = {name: [] for name in fits}; diffs = []; eces = {name: [] for name in fits}
        rng = rng_for(4402, len(models))
        for b in range(200):
            pick = rng.choice(uf, len(uf), replace=True)
            ids = np.concatenate([np.flatnonzero(fam == f) for f in pick])
            for model in fits:
                boots[model].append(loss[model][ids].mean())
                eces[model].append(ece(y[ids], predictions[model][ids]))
            diffs.append(np.mean(loss["gamma"][ids] - loss["loglogistic"][ids]))
        for model in fits:
            fits[model].update(loss_family_bootstrap_ci=np.quantile(boots[model], [.025, .975]),
                               ECE_family_bootstrap_ci=np.quantile(eces[model], [.025, .975]))
        rec = dict(alias=alias, tasks=len(tasks), families=len(uf), runs=sum(t["n"] for t in tasks),
                   successes=sum(t["s"] for t in tasks), failures=sum(t["n"] - t["s"] for t in tasks),
                   inconsistent_human_minutes=sum(t["human_minutes_inconsistent"] for t in tasks),
                   fit_results=fits, gamma_minus_loglogistic=float(np.mean(loss["gamma"] - loss["loglogistic"])),
                   gamma_minus_loglogistic_ci=np.quantile(diffs, [.025, .975]))
        save(RAW / f"E4_predictions_model{len(models)}.json", dict(alias=alias, fold=folds, task_id=[t["task_id"] for t in tasks],
                                                                  predictions=predictions, per_task_losses=loss))
        models.append(rec)
    retries = []
    lengths = np.logspace(1, 6, 21)
    for ai, alpha in enumerate((.5, 1., 2.)):
        beta = alpha * 10
        theta = rng_for(2201, ai).gamma(alpha, 1 / beta, 100_000)
        for N in (1, 2, 4, 8, 16, 32, 64, 128):
            checkpoint()
            mc = np.zeros(len(lengths))
            mc_se = np.zeros(len(lengths))
            for li, L in enumerate(lengths):
                single = np.exp(-theta * L)
                log_fail = np.log1p(-np.minimum(single, 1 - 1e-16))
                vals = -np.expm1(N * log_fail)
                mc[li] = vals.mean()
                mc_se[li] = vals.std(ddof=1) / np.sqrt(len(theta))
            exact = np.array([coverage_integral(float(L), N, alpha, beta) for L in lengths])
            baseline = -np.expm1(N * np.log1p(-(1 + lengths / beta) ** (-alpha)))
            H = brentq(lambda L: coverage_integral(L, N, alpha, beta) - .5, 1e-4, 1e6)
            asymptote = 1 / gamma_dist.ppf(.5, alpha, scale=1 / beta)
            retries.append(dict(alpha=alpha, beta=beta, N=N, lengths=lengths, shared_task_MC=mc,
                                shared_task_SE=mc_se,
                                integral=exact, redrawn_task_baseline=baseline, H50=H,
                                H50_over_logN=H / math.log(N) if N > 1 else None,
                                asymptotic_H50_over_logN=asymptote, max_MC_integral_abs=float(np.max(np.abs(mc - exact)))))
        del vals, theta
    res = dict(data=dict(path=str(source.relative_to(ROOT)), sha256=sha(source), rows=row_count, skipped=skipped,
                         fields=sorted(fields), task_versions=sorted(versions), snapshot="time-horizon-1-1",
                         license="README refers to LICENSE; LICENSE absent from local snapshot; not verified",
                         network_used=False, eligible_models=sorted(eligible), selection="first 3 aliases in lexical order"),
               models=models, retries=retries)
    save(OUT / "E4.json", res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [r["alias"].split(" (")[0] for r in models]
    for i, model in enumerate(("exponential", "gamma", "loglogistic")):
        axes[0].bar(np.arange(len(models)) + (i - 1) * .25, [r["fit_results"][model]["cv_log_loss"] for r in models], width=.25, label=model)
    axes[0].set_xticks(np.arange(len(models)), names, rotation=15, fontsize=7)
    axes[0].set(ylabel="family CV task-average log loss")
    rr = [r for r in retries if r["alpha"] == 1]
    axes[1].plot([r["N"] for r in rr if r["N"] > 1], [r["H50_over_logN"] for r in rr if r["N"] > 1], "o-", label="exact H50/log N")
    axes[1].axhline(rr[0]["asymptotic_H50_over_logN"], ls="--", label="asymptotic limit")
    axes[1].set(xscale="log", xlabel="N", ylabel="H50/log N (alpha=1, g=1)")
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "E4.png", dpi=150); plt.close(fig)
    return res


def main():
    global STAGE_START, STAGE_LIMIT
    manifest = dict(start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), interpreter=sys.executable,
                    python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                    torch_threads=torch.get_num_threads(), torch_device="cpu", cuda_initialized=torch.cuda.is_initialized(),
                    env={key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    RLIMIT_AS_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
                    formulation_sha256=sha(ROOT / "FORMULATION.md"), preregistration_sha256=sha(ROOT / "experiments/preregistration_week1.md"),
                    code_sha256=sha(__file__), stage_times={}, status="running")
    save(OUT / "manifest.json", manifest)
    stages = [("E1", 480, e1), ("E2", 480, e2), ("E3", 900, e3)]
    try:
        result2 = None
        for name, limit, function in stages:
            STAGE_START = time.monotonic(); STAGE_LIMIT = limit
            signal.alarm(limit)
            print(name + " started", flush=True)
            res = function()
            if name == "E2": result2 = res
            manifest["stage_times"][name] = time.monotonic() - STAGE_START
            save(OUT / "manifest.json", manifest)
            print(name + " completed", flush=True)
            if name != "E2": del res
            gc.collect()
        STAGE_START = time.monotonic(); STAGE_LIMIT = 1080
        signal.alarm(1080)
        print("E4 started", flush=True)
        e4(result2)
        manifest["stage_times"]["E4"] = time.monotonic() - STAGE_START
        signal.alarm(0)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - START
        manifest["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        manifest["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save(OUT / "manifest.json", manifest)


def resume_e4():
    """Resume unchanged completed stages after documented E4 memory repair."""
    from datetime import datetime, timezone
    global STAGE_START, STAGE_LIMIT
    manifest = json.loads((OUT / "manifest_initial_attempt.json").read_text())
    manifest["initial_attempt_status"] = manifest.pop("status")
    manifest["initial_attempt_error"] = manifest.pop("error", None)
    manifest["initial_attempt_code_sha256"] = manifest["code_sha256"]
    manifest["code_sha256"] = sha(__file__)
    manifest["amendment_sha256"] = sha(ROOT / "experiments/amendment_week1.md")
    manifest["initial_attempt_active_seconds"] = manifest["elapsed_seconds"]
    STAGE_START = time.monotonic(); STAGE_LIMIT = 1080
    signal.alarm(1080)
    try:
        e4(None)
        manifest["stage_times"]["E4"] = time.monotonic() - STAGE_START
        manifest["E4_initial_attempt_seconds"] = manifest["initial_attempt_active_seconds"] - sum(manifest["stage_times"][name] for name in ("E1", "E2", "E3"))
        manifest["E4_all_attempts_seconds"] = manifest["stage_times"]["E4"] + manifest["E4_initial_attempt_seconds"]
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        signal.alarm(0)
        manifest["elapsed_seconds"] = (datetime.now(timezone.utc) - datetime.strptime(manifest["start_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)).total_seconds()
        manifest["active_compute_seconds"] = manifest["initial_attempt_active_seconds"] + time.monotonic() - STAGE_START
        manifest["max_rss_mib"] = max(manifest["max_rss_mib"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        manifest["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save(OUT / "manifest.json", manifest)


if __name__ == "__main__":
    if "--resume-e4" in sys.argv:
        resume_e4()
    else:
        main()
