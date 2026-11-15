"""CPU-only algebra checks; this is not evidence about natural-language agents.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/sanity_check.py
Only NumPy and the Python standard library are needed. No network or model APIs.
"""

import itertools
import json
import math
from pathlib import Path
import time

import numpy as np


def gain(q, indices, kappa):
    if len(indices) == 0:
        return 0.0
    return float(q[list(indices)].sum()) / (1.0 + kappa * (len(indices) - 1))


def prefix_opt(q, budget, kappa):
    order = np.argsort(-q, kind="stable")
    scores = np.r_[0.0, np.cumsum(q[order[:budget]]) /
                   (1.0 + kappa * np.arange(budget))]
    m = int(np.argmax(scores))
    return float(scores[m]), order[:m]


def main():
    start = time.perf_counter()
    rng = np.random.default_rng(20261105)
    max_opt_gap = 0.0
    cases = 0
    for n in (3, 6, 9):
        for _ in range(6):
            q = rng.dirichlet(np.ones(n))
            for kappa in (0.0, 0.001, 0.03, 0.5):
                for budget in range(n + 1):
                    best = max(gain(q, s, kappa)
                               for m in range(budget + 1)
                               for s in itertools.combinations(range(n), m))
                    computed, _ = prefix_opt(q, budget, kappa)
                    max_opt_gap = max(max_opt_gap, abs(best - computed))
                    assert abs(best - computed) < 1e-12
                    cases += 1

    phase_cases = 0
    for groups, chunks in ((4, 3), (64, 8)):
        for kappa in (0.0, 0.001, 0.003, 0.01, 0.1):
            alpha_c = ((1.0 - kappa + kappa * groups) /
                       (1.0 - kappa + kappa * groups * chunks))
            for alpha in np.linspace(1.0 / chunks, 0.99, 19):
                q = np.full((groups, chunks), (1 - alpha) / (groups * (chunks - 1)))
                q[:, 0] = alpha / groups
                q = q.ravel()
                for budget in sorted({1, groups // 2, groups, groups * 2,
                                      groups * chunks}):
                    if budget <= groups or alpha < alpha_c:
                        expected_m = min(budget, groups * chunks)
                    else:
                        expected_m = groups
                    best, _ = prefix_opt(q, budget, kappa)
                    fixed_q = np.sort(q)[::-1]
                    predicted = fixed_q[:expected_m].sum() / (
                        1 + kappa * (expected_m - 1))
                    assert abs(best - predicted) < 1e-12
                    phase_cases += 1

    # Directly sample all competing Gumbels, rather than assuming a Bernoulli hit.
    calibration = []
    trials = 20000
    for m in (1, 8, 64, 512):
        for kappa in (0.001, 0.003, 0.01):
            hits = 0
            for offset in range(0, trials, 128):
                count = min(128, trials - offset)
                scores = rng.gumbel(size=(count, m))
                scores[:, 0] -= math.log(kappa)
                hits += int(np.count_nonzero(np.argmax(scores, axis=1) == 0))
            expected = 1 / (1 + kappa * (m - 1))
            empirical = hits / trials
            standard_error = math.sqrt(expected * (1 - expected) / trials)
            assert abs(empirical - expected) <= 5 * standard_error + 1 / trials
            calibration.append({"m": m, "kappa": kappa, "expected": expected,
                                "empirical": empirical,
                                "absolute_error": abs(empirical - expected)})

    result = {
        "scope": "algebra and correctly specified Gumbel channel only; no agent validation",
        "seed": 20261105,
        "exhaustive_optimum_cases": cases,
        "max_optimum_gain_gap": max_opt_gap,
        "homogeneous_phase_cases": phase_cases,
        "gumbel_trials_per_cell": trials,
        "gumbel_calibration": calibration,
        "elapsed_seconds": time.perf_counter() - start,
    }
    path = Path(__file__).with_name("sanity_results.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
