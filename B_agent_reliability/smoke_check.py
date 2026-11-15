"""Small mathematical checks; not evidence about an actual language agent.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 smoke_check.py
Requires only Python and NumPy. Writes smoke_results.json beside this file.
"""

import json
import math
from pathlib import Path
import resource
import time

import numpy as np


def h_closed(k, rho):
    return 1.0 / sum(rho**j for j in range(k))


def w_closed(k, rho):
    if rho == 1.0:
        return float(k - 1)
    return (1 + rho) * (k * h_closed(k, rho) - 1) / (1 - rho)


def solve_chain(ratios):
    ratios = np.asarray(ratios, dtype=float)
    p = 1 / (1 + ratios)
    d = 1 - p
    m = len(p)
    matrix = np.eye(m)
    for i in range(m):
        if i + 1 < m:
            matrix[i, i + 1] = -p[i]
        if i > 0:
            matrix[i, i - 1] = -d[i]
    exits = np.zeros(m)
    exits[-1] = p[-1]
    return float(np.linalg.solve(matrix, exits)[0]), float(
        np.linalg.solve(matrix, np.ones(m))[0]
    )


def simulate_cascades(rng, k, ratios, count):
    """Simulate actual jumps, without sampling closed-form absorption marks."""
    state = np.ones(count, dtype=np.int32)
    steps = np.zeros(count, dtype=np.int32)
    up_count = 0
    down_count = 0
    while True:
        active = np.flatnonzero((state > 0) & (state < k))
        if not len(active):
            break
        p = 1 / (1 + np.asarray(ratios)[state[active] - 1])
        up = rng.random(len(active)) < p
        up_count += int(up.sum())
        down_count += int((~up).sum())
        state[active] += np.where(up, 1, -1)
        steps[active] += 1
    return {
        "failure_rate": float(np.mean(state == k)),
        "mean_work": float(steps.mean()),
        "work_se": float(steps.std(ddof=1) / math.sqrt(count)),
        "work_p99": float(np.quantile(steps, 0.99)),
        "up": up_count,
        "down": down_count,
        "count": count,
    }


def main():
    started = time.perf_counter()
    rng = np.random.default_rng(20261105)
    report = {"scope": "formula and simulator checks only; no real agent data"}

    max_h_relative_error = 0.0
    max_w_relative_error = 0.0
    for rho in (0.5, 1.0, 1.5, 2.0):
        for k in (2, 4, 8, 12, 16, 32):
            h, w = solve_chain([rho] * (k - 1))
            max_h_relative_error = max(
                max_h_relative_error, abs(h / h_closed(k, rho) - 1)
            )
            max_w_relative_error = max(
                max_w_relative_error, abs(w / w_closed(k, rho) - 1)
            )
    assert max_h_relative_error < 1e-9
    assert max_w_relative_error < 1e-9
    report["linear_system_checks"] = {
        "cases": 24,
        "max_h_relative_error": max_h_relative_error,
        "max_w_relative_error": max_w_relative_error,
    }

    calibration = simulate_cascades(rng, 3, [2.0, 2.0], 100_000)
    rho_hat = calibration["down"] / calibration["up"]
    tests = []
    for k in (4, 8, 12):
        measured = simulate_cascades(rng, k, [2.0] * (k - 1), 100_000)
        prediction = h_closed(k, rho_hat)
        actual_probability = h_closed(k, 2.0)
        se = math.sqrt(actual_probability * (1 - actual_probability) / 100_000)
        # Formula vs simulator; separately report the frozen-parameter prediction.
        assert abs(measured["failure_rate"] - actual_probability) < 5 * se
        assert abs(measured["mean_work"] - w_closed(k, 2.0)) < (
            5 * measured["work_se"]
        )
        tests.append({"k": k, "predicted_h_from_calibration": prediction,
                      "true_h": actual_probability, **measured})
    report["short_capacity_calibration"] = {"rho_hat": rho_hat, **calibration}
    report["held_out_capacities"] = tests

    ratios = [2.0 if n <= 3 else 0.4 for n in range(1, 12)]
    h_inhomogeneous, _ = solve_chain(ratios)
    cumulative_product = 1.0
    denominator = 1.0
    for ratio in ratios:
        cumulative_product *= ratio
        denominator += cumulative_product
    assert abs(h_inhomogeneous - 1 / denominator) < 1e-12
    negative = simulate_cascades(rng, 12, ratios, 100_000)
    h_wrong = h_closed(12, rho_hat)
    negative_se = math.sqrt(
        negative["failure_rate"] * (1 - negative["failure_rate"]) / 100_000
    )
    rejected = abs(negative["failure_rate"] - h_wrong) > max(0.03, 5 * negative_se)
    assert rejected, "Known state-dependent negative control was not rejected"
    report["state_dependent_negative_control"] = {
        "constant_model_prediction": h_wrong,
        "state_dependent_exact_h": h_inhomogeneous,
        "constant_model_rejected": bool(rejected), **negative,
    }

    mixture = []
    for alpha in (0.5, 1.0, 2.0):
        beta = 10 * alpha
        theta = rng.gamma(alpha, 1 / beta, 100_000)
        max_z = 0.0
        for x in (0.1, 1.0, 10.0, 100.0):
            values = np.exp(-x * theta)
            expected = (1 + x / beta) ** (-alpha)
            se = values.std(ddof=1) / math.sqrt(len(values))
            z = abs(float(values.mean()) - expected) / se
            max_z = max(max_z, z)
            assert z < 5
        mixture.append({"alpha": alpha, "beta": beta, "max_mc_z": max_z})
    report["gamma_laplace_checks"] = mixture

    phase = []
    for a in (0.5, 1.0, 2.0):
        lengths = [math.exp((k - 0.25) / a) for k in (60, 100)]
        quantities = []
        for length in lengths:
            k = math.ceil(a * math.log(length))
            x = length * h_closed(k, 2.0) / 10
            log_s = -math.log1p(x)
            # For a=2 test failure rather than success, avoiding cancellation.
            quantities.append(math.exp(log_s) if a < 1.5 else -math.expm1(log_s))
        fitted = -math.log(quantities[1] / quantities[0]) / math.log(
            lengths[1] / lengths[0]
        )
        predicted = abs(1 - a * math.log(2))
        assert abs(fitted - predicted) < 1e-5
        phase.append({"a": a, "quantity": "success" if a < 1.5 else "failure",
                      "predicted_exponent": predicted, "numerical_exponent": fitted})
    report["phase_checks_large_length_aligned_rounding"] = phase
    report["elapsed_seconds"] = time.perf_counter() - started
    report["max_rss_mib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report["all_checks_passed"] = True
    output = Path(__file__).resolve().parent / "smoke_results.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
