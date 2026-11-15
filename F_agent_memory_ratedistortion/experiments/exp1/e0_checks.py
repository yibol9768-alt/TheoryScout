"""Local proof checks for FORMULATION_V2; no network, model, or API calls.

These checks validate finite formulas, not claims about natural-language agents.
"""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import itertools
import json
import math
from pathlib import Path
import resource
import time

import numpy as np


def gain(p, policy, u, loads, signal):
    return float(np.dot(p, u[policy])) / (signal + loads[policy].sum())


def budget_dp(p, costs, u, loads, budget, signal, tolerance=1e-11):
    """Bisection of max(N-t W)-t signal, with a final traceback."""
    def solve(t, traceback=False):
        values = np.full(budget + 1, -np.inf)
        values[0] = 0.0
        records = []
        for probability in p:
            updated = np.full_like(values, -np.inf)
            choices = np.full(budget + 1, -1, dtype=np.int8)
            for mode, cost in enumerate(costs):
                if cost > budget:
                    continue
                candidate = values[:budget + 1 - cost] + (
                    probability * u[mode] - t * loads[mode])
                target = updated[cost:]
                better = candidate > target
                target[better] = candidate[better]
                choices[cost:][better] = mode
            values = updated
            if traceback:
                records.append(choices)
        used = int(np.argmax(values))
        phi = float(values[used] - t * signal)
        if not traceback:
            return phi
        policy = []
        for choices in reversed(records):
            mode = int(choices[used])
            assert mode >= 0
            policy.append(mode)
            used -= int(costs[mode])
        return np.array(policy[::-1], dtype=int)

    lower, upper = 0.0, 1.0
    for _ in range(math.ceil(math.log2(1.0 / tolerance))):
        middle = (lower + upper) / 2
        if solve(middle) >= 0:
            lower = middle
        else:
            upper = middle
    policy = solve(lower, traceback=True)
    return gain(p, policy, u, loads, signal), policy


def fractional_vertices(p, costs, u, loads, budget, signal):
    """Enumerate every vertex: integral or one group mixing two modes."""
    groups = len(p)
    best = (-1.0, None)
    for policy in itertools.product(range(3), repeat=groups):
        policy = np.array(policy)
        if costs[policy].sum() <= budget:
            value = gain(p, policy, u, loads, signal)
            if value > best[0]:
                best = (value, np.eye(3)[policy])
    for mixed in range(groups):
        others = [g for g in range(groups) if g != mixed]
        for modes in itertools.product(range(3), repeat=groups - 1):
            matrix = np.zeros((groups, 3))
            matrix[others, modes] = 1
            remaining = budget - float((matrix @ costs).sum())
            for low, high in itertools.combinations(range(3), 2):
                fraction = (remaining - costs[low]) / (costs[high] - costs[low])
                if not 0 < fraction < 1:
                    continue
                matrix[mixed, low] = 1 - fraction
                matrix[mixed, high] = fraction
                value = float(np.dot(p, matrix @ u)) / (
                    signal + float((matrix @ loads).sum()))
                if value > best[0]:
                    best = (value, matrix.copy())
                matrix[mixed, :] = 0
    return best


def main():
    start = time.perf_counter()
    # Decimal bytes: below the requested 3.5 GB even if interpreted strictly.
    memory_cap = 1_900_000_000
    previous = resource.getrlimit(resource.RLIMIT_AS)
    hard = previous[1]
    resource.setrlimit(resource.RLIMIT_AS, (
        memory_cap if hard == resource.RLIM_INFINITY else min(memory_cap, hard), hard))
    rng = np.random.default_rng(20261106)
    report = {"seed": 20261106, "scope": "finite mathematical checks only; no agent evidence"}
    max_dp_gap = 0.0
    max_rounding_gap = 0.0
    dp_cases = fractional_cases = threshold_cases = 0
    for groups in (2, 4, 5):
        for _ in range(4):
            p = rng.dirichlet(np.ones(groups))
            k = int(rng.integers(2, 5))
            signal = float(rng.uniform(0.2, 12))
            ws = float(rng.uniform(0.3, 0.9))
            wr = float(rng.uniform(1, 1.5))
            vs, vr = float(rng.uniform(0.2, 0.8)), 1.0
            costs = np.array([0, 1, k])
            loads = np.array([0.0, ws, k * wr])
            u = np.array([0.0, vs * (signal + ws), vr * (signal + wr)])
            for budget in sorted({0, 1, groups, k * groups // 2, k * groups}):
                brute = max(gain(p, np.array(z), u, loads, signal)
                            for z in itertools.product(range(3), repeat=groups)
                            if costs[list(z)].sum() <= budget)
                actual, policy = budget_dp(p, costs, u, loads, budget, signal)
                gap = abs(actual - brute)
                max_dp_gap = max(max_dp_gap, gap)
                assert gap < 1e-10
                assert costs[policy].sum() <= budget
                dp_cases += 1

                optimum, matrix = fractional_vertices(p, costs, u, loads, budget, signal)
                assert optimum + 1e-12 >= brute
                mixed = np.flatnonzero((matrix > 1e-10).sum(axis=1) > 1)
                assert len(mixed) <= 1
                rounded = np.argmax(matrix, axis=1)
                for g in mixed:
                    rounded[g] = int(np.flatnonzero(matrix[g] > 1e-10)[0])
                rounded_value = gain(p, rounded, u, loads, signal)
                bound = float(p.max() * u[2] / (signal + (matrix @ loads).sum()))
                assert costs[rounded].sum() <= budget
                assert optimum - rounded_value <= bound + 1e-11
                max_rounding_gap = max(max_rounding_gap, optimum - rounded_value)
                fractional_cases += 1

                # LP dual support certificate, including its single fractional group.
                mu_lower, mu_upper = 0.0, math.inf
                for g in range(groups):
                    for z in np.flatnonzero(matrix[g] > 1e-10):
                        for other in range(3):
                            intercept = p[g] * (u[z] - u[other]) - optimum * (
                                loads[z] - loads[other])
                            difference = costs[z] - costs[other]
                            if difference > 0:
                                mu_upper = min(mu_upper, intercept / difference)
                            elif difference < 0:
                                mu_lower = max(mu_lower, intercept / difference)
                            else:
                                assert intercept >= -1e-10
                assert mu_lower <= mu_upper + 1e-10
                if float((matrix @ costs).sum()) < budget - 1e-10:
                    assert mu_lower <= 1e-10 and mu_upper >= -1e-10
                threshold_cases += 1

    phase_cases = 0
    max_phase_gap = 0.0
    integer_scarcity_example = None
    for groups in (2, 5, 16):
        for k in (2, 3, 5):
            for signal in (0.1, 3.0, 31.0):
                vr = 0.8
                critical = vr * (signal + groups) / (signal + k * groups)
                for vs in (vr / k, (vr / k + critical) / 2, critical,
                           (critical + vr) / 2, vr):
                    for budget in range(k * groups + 1):
                        def value(n, s):
                            return ((signal + 1) * (vs * s + vr * n) /
                                    (groups * (signal + s + k * n)))
                        brute = max(value(n, s)
                                    for n in range(groups + 1)
                                    for s in range(groups + 1 - n)
                                    if s + k * n <= budget)
                        if budget <= groups:
                            predicted = value(0, budget)
                        else:
                            seam = (budget - groups) / (k - 1)
                            candidates = {0, math.floor(seam), math.ceil(seam)}
                            candidates = [n for n in candidates
                                          if 0 <= n <= min(groups, budget // k)]
                            predicted = max(value(n, min(groups - n, budget - k * n))
                                            for n in candidates)
                        max_phase_gap = max(max_phase_gap, abs(brute - predicted))
                        assert abs(brute - predicted) < 1e-12
                        if budget >= groups and (budget - groups) % (k - 1) == 0:
                            upgrades = (budget - groups) // (k - 1)
                            n = upgrades if vs < critical else 0
                            assert abs(brute - value(n, groups - n)) < 1e-12
                        phase_cases += 1
    # Extra budget can be insufficient for a complete upgrade.
    groups, k, budget, signal, vs, vr = 5, 5, 6, 31.0, 0.4, 1.0
    options = [(n, min(groups - n, budget - k * n))
               for n in range(budget // k + 1)]
    values = [(signal + 1) * (vs * s + vr * n) /
              (groups * (signal + s + k * n)) for n, s in options]
    assert options[int(np.argmax(values))] == (0, groups)
    integer_scarcity_example = {"G": groups, "k": k, "b": budget,
                               "signal": signal, "vs": vs, "vr": vr,
                               "best_raw_summary_counts": [0, groups]}

    calibration = []
    for signal, ws, wr, ns, nr in ((3, 1, 1, 3, 2), (7, 0.5, 1.3, 2, 3)):
        weights = np.r_[np.full(ns, ws), np.full(4 * nr, wr)]
        for target in (0, len(weights) - 1):
            means = np.log(weights)
            means[target] = math.log(signal + weights[target])
            trials, hits = 30000, 0
            for offset in range(0, trials, 128):
                batch = min(128, trials - offset)
                scores = rng.gumbel(size=(batch, len(weights))) + means
                hits += int((scores.argmax(axis=1) == target).sum())
            expected = (signal + weights[target]) / (signal + weights.sum())
            error = abs(hits / trials - expected)
            se = math.sqrt(expected * (1 - expected) / trials)
            assert error <= 5 * se + 1 / trials
            calibration.append({"signal": signal, "target": target,
                                "expected": expected, "empirical": hits / trials,
                                "absolute_error": error})

    lower, upper = 2 / 3, 1.0
    for _ in range(80):
        a = (lower + upper) / 2
        h = -a * math.log2(a) - (1 - a) * math.log2((1 - a) / 2)
        if h > 1:
            lower = a
        else:
            upper = a
    report.update({"budget_dp_cases": dp_cases, "max_dp_gain_gap": max_dp_gap,
                   "fractional_vertex_cases": fractional_cases,
                   "dual_support_certificate_cases": threshold_cases,
                   "max_observed_rounding_gain_gap": max_rounding_gap,
                   "uniform_integer_phase_cases": phase_cases,
                   "max_uniform_phase_gain_gap": max_phase_gap,
                   "integer_scarcity_example": integer_scarcity_example,
                   "gumbel_trials_per_cell": 30000, "gumbel_checks": calibration,
                   "equal_entropy_three_group_pmax": a,
                   "elapsed_seconds": time.perf_counter() - start,
                   "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    output = Path(__file__).with_name("checks_v2_results.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
