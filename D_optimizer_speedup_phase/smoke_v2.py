#!/usr/bin/env python3
"""Offline, single-process checks for FORMULATION_V2; NumPy only.

Large C values are virtual budgets in exact risk evaluations, not executed
training. Finite-grid frontiers are upper bounds, not certified global optima.
"""
import os

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np


def envelope(alpha, beta, nu, chi, z):
    lines = [(1.0, beta + 1), (1 + chi + z, alpha / 2),
             (1 + chi + (1 - nu / 2) * z, alpha)]
    candidates = [1.0]
    for j, (a, b) in enumerate(lines):
        for c, d in lines[j + 1:]:
            if abs(a - c) > 1e-14:
                u = (d - b) / (a - c)
                if u >= 1:
                    candidates.append(u)
    u = min(candidates, key=lambda x: (max(a * x + b for a, b in lines), x))
    return max(a * u + b for a, b in lines), u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="smoke_v2_results.json")
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    started = time.monotonic()
    max_n = 65536
    idx = np.arange(1, max_n + 1, dtype=np.float64)
    weights = idx ** -2
    tails = math.pi ** 2 / 6 - np.cumsum(weights)
    rng = np.random.default_rng(20261105)
    # All scenarios have beta=1, so the infinite tail is known exactly.
    configs = [
        dict(name="stat", alpha=1.5, nu=1., chi=0., z=0., eta=.5),
        dict(name="lr_decay", alpha=1.5, nu=1., chi=1., z=0., eta=.5),
        dict(name="nu1_b1", alpha=3., nu=1., chi=0., z=0., eta=.5),
        dict(name="nu1_growing_batch", alpha=3., nu=1., chi=0., z=1., eta=.5),
        dict(name="nu8_resolving_batch", alpha=3., nu=8., chi=0., z=.375, eta=.5),
        dict(name="oversize", alpha=16., nu=8., chi=0., z=1., eta=.5),
        dict(name="baseline_eta025", alpha=3., nu=1., chi=0., z=0., eta=.25, gd=True),
        dict(name="ridge_eta035", alpha=3., nu=1., chi=0., z=0., eta=.35),
    ]

    def guard():
        if time.monotonic() - started > args.max_seconds:
            raise TimeoutError("Internal wall-time cap reached")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if rss >= 3_000_000_000:
            raise MemoryError("3.0 GB soft RSS cap reached")

    def rate(cfg, n, b):
        lam = idx[:n] ** -cfg["alpha"]
        if cfg.get("gd"):
            return cfg["eta"] * n ** -cfg["chi"] * lam
        return cfg["eta"] * n ** -cfg["chi"] * lam / np.sqrt(lam + b ** -cfg["nu"])

    def risk(cfg, n, b, t):
        a = rate(cfg, n, b)
        rem = float(t) * np.log1p(-a)
        filt = -np.expm1(rem)
        bias = float(np.dot(weights[:n], np.exp(2 * rem)))
        variance = float(np.dot(filt, filt)) / (b * t)
        tail = float(tails[n - 1])
        return tail + bias + variance, tail, bias, variance, a

    # Six independently specified small iteration checks and blocked MC checks.
    exact_checks = []
    for j, cfg in enumerate(configs[:6]):
        n, b, t = 24, 3 + j, 47 + 3 * j
        total, tail, _, _, a = risk(cfg, n, b, t)
        filt = -np.expm1(t * np.log1p(-a))
        target = np.sqrt(weights[:n])
        zbar = target + rng.normal(size=n) / math.sqrt(b * t)
        est = np.zeros(n)
        for _ in range(t):
            est += a * (zbar - est)
        iteration_error = float(np.max(np.abs(est - filt * zbar)))
        samples = []
        for _ in range(16):
            err = -(1 - filt) * target + filt * rng.normal(size=(256, n)) / math.sqrt(b * t)
            samples.extend((tail + np.sum(err ** 2, axis=1)).tolist())
        se = float(np.std(samples, ddof=1) / math.sqrt(len(samples)))
        zscore = abs(float(np.mean(samples)) - total) / se
        exact_checks.append(dict(name=cfg["name"], iteration_max_error=iteration_error,
                                 mc_error_in_se=zscore, passed=iteration_error < 1e-10 and zscore < 4))

    size_grid = np.unique(np.rint(np.geomspace(1, max_n, 192)).astype(int))

    def record(cfg, c, n, b):
        # c, n, b and t remain arbitrary-precision integers for budget division.
        t = c // (n * b)
        if t < 1:
            return None
        total, tail, bias, variance, a = risk(cfg, n, b, t)
        ta = float(t) * a
        if ta[0] < 1:
            h = 1.0
        elif ta[-1] >= 1:
            h = float(n)
        else:
            # Monotone interpolation of the continuous log-rate root.
            h = math.exp(float(np.interp(0., np.log(ta)[::-1], np.log(idx[:n])[::-1])))
        return dict(C=c, n=n, b=b, T=t, risk=total, A=tail, B=bias, V=variance,
                    h=h, Ta_n=float(ta[-1]), n_over_h=n/h,
                    size_boundary=n in (1, max_n))

    def curve(cfg, free_batch=False):
        exponents = range(24, 85, 4) if cfg["name"] == "oversize" else range(12, 43, 2)
        records = []
        for e in exponents:
            guard()
            c = 2 ** e

            def candidates(n):
                if free_batch:
                    # Deliberately finite subset, not a global optimization claim.
                    choices = {1, 2, 4, 16, 64, 256, 1024, 4096}
                    knee = max(1, int(math.ceil(n ** (cfg["alpha"] / cfg["nu"]))))
                    choices.update({max(1, knee // 2), knee, knee * 2})
                else:
                    choices = {int(math.ceil(n ** cfg["z"]))}
                return [r for b in sorted(choices) if (r := record(cfg, c, n, b)) is not None]

            best = min((r for n in size_grid for r in candidates(int(n))), key=lambda r: r["risk"])
            j = int(np.searchsorted(size_grid, best["n"]))
            lo = size_grid[max(0, j-1)]
            hi = size_grid[min(len(size_grid)-1, j+1)]
            local = np.unique(np.rint(np.geomspace(lo, hi, 33)).astype(int))
            best = min([best] + [r for n in local for r in candidates(int(n))], key=lambda r: r["risk"])
            if records and records[-1]["risk"] < best["risk"]:
                best = dict(records[-1], C=c, budget_not_exhausted=True)
            records.append(best)
        return records

    curves, summaries = {}, []
    for cfg in configs:
        guard()
        records = curve(cfg)
        curves[cfg["name"]] = records
        e, u = envelope(cfg["alpha"], 1., cfg["nu"], cfg["chi"], cfg["z"])
        if cfg.get("gd"):
            e, u = 1 + max(2., cfg["alpha"] + cfg["chi"]), 1.
        selected = [r for r in records[-5:] if not r["size_boundary"]]

        def slope(key):
            if len(selected) < 4:
                return None
            return float(np.polyfit([math.log(r["C"]) for r in selected],
                                    np.log([r[key] for r in selected]), 1)[0])

        observed = slope("risk")
        summary = dict(name=cfg["name"], predicted_E=e, predicted_u=u,
                       predicted_k=1/e, observed_k=None if observed is None else -observed,
                       observed_n_slope=slope("n"), observed_h_slope=slope("h"),
                       last_n=records[-1]["n"], last_h=records[-1]["h"],
                       last_boundary=records[-1]["size_boundary"],
                       last_Ta_n=records[-1]["Ta_n"],
                       last_V_over_A=records[-1]["V"] / records[-1]["A"],
                       slope_status="unassessed" if observed is None else
                       ("within_smoke_tolerance" if abs(-observed - 1/e) < .02 else "needs_more_scale_or_review"))
        g = cfg["chi"] + max(cfg["alpha"] / 2 + cfg["z"],
                              cfg["alpha"] + (1 - cfg["nu"] / 2) * cfg["z"])
        if g < 2 and not cfg.get("gd"):
            kstat = 1.5 * 2 ** (1/3)
            summary["last_stat_constant_relative_error"] = records[-1]["risk"] * records[-1]["C"] ** (1/3) / kstat - 1
        summaries.append(summary)

    cfg = configs[2]
    free = curve(cfg, free_batch=True)
    curves["nu1_free_batch_subset"] = free
    free_ratio = [f["risk"] / r["risk"] for f, r in zip(free, curves["nu1_b1"])]

    # Independent local derivative checks, away from integer ceilings.
    alpha, nu, chi, i, n, b = 3., 1.3, .4, 7., 128., 17.
    def log_rate(log_i, log_n, log_b):
        lam = math.exp(-alpha * log_i)
        rho = math.exp(-nu * log_b)
        return math.log(.5) - chi * log_n + math.log(lam) - .5 * math.log(lam + rho)
    logs = [math.log(i), math.log(n), math.log(b)]
    numerical = []
    for axis in range(3):
        left, right = list(logs), list(logs)
        left[axis] -= 1e-5
        right[axis] += 1e-5
        numerical.append((log_rate(*right) - log_rate(*left)) / 2e-5)
    lam, rho = i ** -alpha, b ** -nu
    expected = [-alpha + alpha / 2 * lam / (lam + rho), -chi,
                nu / 2 * rho / (lam + rho)]
    derivative_error = max(abs(a-b) for a, b in zip(numerical, expected))

    # Compare sharp optimization constants through same-budget loss ratios;
    # equal-loss interpolation would require a sufficiently dense cost grid.
    eta_ratio_estimate = (curves["baseline_eta025"][-1]["risk"] /
                          curves["ridge_eta035"][-1]["risk"]) ** 4
    report = dict(scope="Gaussian-sequence ridge proxy only; finite-grid upper bounds; no actual Muon training",
                  beta=1., sigma2=1., max_n=max_n, size_grid_count=len(size_grid),
                  exact_checks=exact_checks, frontier_summaries=summaries,
                  nu1_free_subset_last_risk_ratio_to_b1=free_ratio[-1],
                  nu1_free_subset_last_b=free[-1]["b"],
                  nu1_free_subset_slope=-float(np.polyfit([math.log(r["C"]) for r in free[-5:]],
                                                                       np.log([r["risk"] for r in free[-5:]]), 1)[0]),
                  local_derivative_max_error=derivative_error,
                  eta_ratio_asymptotic_estimate=eta_ratio_estimate,
                  eta_ratio_prediction=1.4, curves=curves,
                  elapsed_seconds=time.monotonic()-started,
                  peak_rss_bytes_linux=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "curves"}, ensure_ascii=False, indent=2))
    if not all(r["passed"] for r in exact_checks) or derivative_error > 1e-8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
