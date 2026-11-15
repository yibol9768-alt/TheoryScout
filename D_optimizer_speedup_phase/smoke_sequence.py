#!/usr/bin/env python3
"""Offline CPU checks of the Gaussian-sequence proxy; no model/API calls.

Requires NumPy only. This evaluates a finite grid of full-budget allocations,
not the exact finite-budget infimum over all integers or a Muon experiment.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="smoke_sequence_results.json")
    parser.add_argument("--max-seconds", type=float, default=90)
    args = parser.parse_args()
    start = time.monotonic()
    max_n, beta, eta, sigma2 = 32768, 1.0, 0.5, 1.0
    indices = np.arange(1, max_n + 1, dtype=np.float64)
    weights = indices ** (-1.0 - beta)
    # beta=1: zeta(2) is known exactly, so no ambient-dimension floor.
    tails = math.pi**2 / 6.0 - np.cumsum(weights)
    rng = np.random.default_rng(20261105)

    def check_time():
        if time.monotonic() - start > args.max_seconds:
            raise TimeoutError("Sequence check exceeded its wall-clock cap")

    def risk(n, t, p, chi=0.0, noise=sigma2):
        rate = eta * n ** (-chi) * indices[:n] ** (-p)
        log_remaining = t * np.log1p(-rate)
        filt = -np.expm1(log_remaining)
        a = float(tails[n - 1])
        b = float(np.dot(weights[:n], np.exp(2 * log_remaining)))
        v = float(noise / t * np.dot(filt, filt))
        return a + b + v, a, b, v

    checks = []
    for alpha in (1.5, 3.0, 6.0):
        for q in (0.0, 0.5):
            p = alpha * (1 - q)
            n, t = 64, 256
            filt = -np.expm1(t * np.log1p(-eta * indices[:n] ** (-p)))
            # Whitened coordinate errors, including independent observation noise.
            errors = -(1 - filt) * np.sqrt(weights[:n])
            errors = errors + filt * math.sqrt(sigma2 / t) * rng.normal(size=(4096, n))
            sampled_risks = tails[n - 1] + np.sum(errors * errors, axis=1)
            expected = risk(n, t, p)[0]
            se = float(np.std(sampled_risks, ddof=1) / math.sqrt(len(sampled_risks)))
            standardized_error = abs(float(np.mean(sampled_risks)) - expected) / se
            checks.append({"alpha": alpha, "q": q, "expected": expected,
                           "mc_mean": float(np.mean(sampled_risks)),
                           "standard_error": se, "error_in_se": standardized_error,
                           "pass_4se": standardized_error < 4})

    n, t, p = 16, 31, 1.5
    lam = indices[:n] ** (-3.0)
    theta_star = np.sqrt(weights[:n] / lam)
    zbar = theta_star + np.sqrt(sigma2 / (t * lam)) * rng.normal(size=n)
    theta = np.zeros(n)
    for _ in range(t):
        theta -= eta * indices[:n] ** (-p) * (theta - zbar)
    closed = -np.expm1(t * np.log1p(-eta * indices[:n] ** (-p))) * zbar
    iteration_error = float(np.max(np.abs(theta - closed)))

    budgets = [2**e for e in range(14, 37, 2)]
    size_grid = np.unique(np.rint(np.geomspace(1, max_n, 128)).astype(int))

    def frontier(p, chi=0.0, noise=sigma2):
        records = []
        for c in budgets:
            check_time()
            candidates = [int(n) for n in size_grid if n <= c]

            def record(n):
                t = c // n
                total, a, b, v = risk(n, t, p, chi=chi, noise=noise)
                return {"C": c, "n": n, "T": t, "risk": total,
                        "A": a, "B": b, "V": v,
                        "z": eta * t / n ** (p + chi)}

            best = min((record(n) for n in candidates), key=lambda x: x["risk"])
            grid_index = candidates.index(best["n"])
            left = candidates[max(0, grid_index - 1)]
            right = candidates[min(len(candidates) - 1, grid_index + 1)]
            nearby = range(left, right + 1)
            best = min([best] + [record(n) for n in nearby], key=lambda x: x["risk"])
            best["size_grid_boundary"] = best["n"] in (1, max_n)
            # Retain already available allocations if finite-time noise is nonmonotone.
            if records and records[-1]["risk"] < best["risk"]:
                best = dict(records[-1], C=c, budget_not_exhausted=True)
            records.append(best)
        return records

    curves, summaries = {}, []
    for alpha in (1.5, 2.0, 3.0, 6.0):
        pair = {}
        for q in (0.0, 0.5):
            p = alpha * (1 - q)
            records = frontier(p)
            curves[f"alpha={alpha},q={q}"] = records
            pair[q] = records
            last = [r for r in records[-4:] if not r["size_grid_boundary"]]
            observed_k = None
            if len(last) >= 3:
                observed_k = -float(np.polyfit(np.log([r["C"] for r in last]),
                                               np.log([r["risk"] for r in last]), 1)[0])
            summaries.append({"alpha": alpha, "q": q,
                              "predicted_k": beta / (1 + max(p, beta + 1)),
                              "last4_observed_k": observed_k,
                              "last_z": records[-1]["z"],
                              "last_V_over_A": records[-1]["V"] / records[-1]["A"],
                              "last_boundary": records[-1]["size_grid_boundary"]})
        # Invert only within the measured baseline loss range, never extrapolate.
        baseline = pair[0.0]
        log_losses = np.log([r["risk"] for r in baseline])[::-1]
        log_costs = np.log([r["C"] for r in baseline])[::-1]
        for r in pair[0.5]:
            target = math.log(r["risk"])
            if log_losses[0] <= target <= log_losses[-1]:
                r["equal_loss_compute_ratio"] = math.exp(float(np.interp(
                    target, log_losses, log_costs))) / r["C"]
            else:
                r["equal_loss_compute_ratio"] = None

    curves["alpha=3,q=0.5,chi=1"] = frontier(1.5, chi=1.0)
    curves["alpha=1.5,q=0.5,sigma2=4"] = frontier(0.75, noise=4.0)
    # Dimensionless sharp-constant comparisons in the strict statistical phase.
    kstat = beta / (beta + 2)
    constant_stat = (beta + 2) / (2 * beta) * (2 * sigma2) ** kstat
    for summary in summaries:
        p = summary["alpha"] * (1 - summary["q"])
        if p < beta + 1:
            r = curves[f"alpha={summary['alpha']},q={summary['q']}"][-1]
            summary["stat_constant_relative_error"] = (
                r["risk"] * r["C"]**kstat / constant_stat - 1)

    # Log-coordinate quadrature for the critical/optimization sharp constants.
    # x=exp(-y), y in [0,40]; the omitted J tail is at most exp(-40).
    y = np.linspace(0.0, 40.0, 8193)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    sharp_constants = {}
    for p in (2.0, 3.0, 6.0):
        check_time()
        k = beta / (1 + max(p, beta + 1))

        def objective(log_z):
            z = math.exp(log_z)
            s = z * np.exp(p * y)
            h = 1 / beta + integrate(np.exp(beta * y - 2 * s), y)
            j = integrate(np.exp(-y) * (-np.expm1(-s))**2, y)
            variance = sigma2 * eta * j / z if p == beta + 1 else 0
            return float((1 / eta)**k * z**k * (h + variance))

        # Coarse search followed by local refinement; no uniqueness is presumed.
        log_grid = np.linspace(-10.0, 5.0, 301)
        best_index = min(range(len(log_grid)), key=lambda j: objective(log_grid[j]))
        lo = log_grid[max(0, best_index - 1)]
        hi = log_grid[min(len(log_grid) - 1, best_index + 1)]
        for _ in range(40):
            a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if objective(a) < objective(b):
                hi = b
            else:
                lo = a
        log_z = (lo + hi) / 2
        sharp_constants[str(p)] = {"z_min": math.exp(log_z), "K": objective(log_z),
                                  "numerical_quadrature_only": True}
        for summary in summaries:
            if summary["alpha"] * (1 - summary["q"]) == p:
                r = curves[f"alpha={summary['alpha']},q={summary['q']}"][-1]
                summary["sharp_constant_relative_error"] = (
                    r["risk"] * r["C"]**k / objective(log_z) - 1)

    intervention = curves["alpha=3,q=0.5,chi=1"][-4:]
    shifted_slope = -float(np.polyfit(np.log([r["C"] for r in intervention]),
                                      np.log([r["risk"] for r in intervention]), 1)[0])
    noise_base = curves["alpha=1.5,q=0.5"][-1]
    noise_high = curves["alpha=1.5,q=0.5,sigma2=4"][-1]
    interventions = {"chi1_predicted_k": 2 / 7, "chi1_observed_k": shifted_slope,
                     "noise4_n_ratio": noise_high["n"] / noise_base["n"],
                     "noise4_predicted_n_ratio": 4**(-1 / 3),
                     "noise4_risk_ratio": noise_high["risk"] / noise_base["risk"],
                     "noise4_predicted_risk_ratio": 4**(1 / 3)}

    import resource
    report = {"scope": "Proxy only; finite grid; not a Muon or LLM validation",
              "parameters": {"beta": beta, "sigma2": sigma2, "eta": eta,
                             "max_n": max_n, "size_grid_count": len(size_grid)},
              "iteration_max_abs_error": iteration_error,
              "iteration_pass": iteration_error < 1e-10,
              "monte_carlo": checks, "frontier_summaries": summaries,
              "sharp_constants": sharp_constants, "interventions": interventions,
              "curves": curves, "elapsed_seconds": time.monotonic() - start,
              "peak_rss_mb_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "curves"},
                     ensure_ascii=False, indent=2))
    if not report["iteration_pass"] or not all(c["pass_4se"] for c in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
