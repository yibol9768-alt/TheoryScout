"""Small CPU checks for FORMULATION.md; not the proposed training study.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 smoke_checks.py
No files are modified.  Monte Carlo output is diagnostic, not a proof.
"""

import json
import math
import resource
import time

import numpy as np


def benchmark_attention():
    """Check analytic gradients and time 50 fixed-batch steps, not a study."""
    rng = np.random.default_rng(11)
    batch, heads, dim, width, length = 32, 4, 8, 32, 8
    keys = rng.normal(size=(batch, length, width))
    labels = rng.integers(length, size=batch)
    x = keys[np.arange(batch), labels].copy() + 0.1 * rng.normal(size=(batch, width))
    x *= math.sqrt(width) / np.linalg.norm(x, axis=-1, keepdims=True)
    keys *= math.sqrt(width) / np.linalg.norm(keys, axis=-1, keepdims=True)
    target = np.full((batch, length), 0.1 / length)
    target[np.arange(batch), labels] += 0.9
    wq = rng.normal(size=(heads * dim, width)) * 0.1 / math.sqrt(width)
    wk = rng.normal(size=wq.shape) * 0.1 / math.sqrt(width)

    def loss_grad(qweight, kweight):
        q = (x @ qweight.T).reshape(batch, heads, dim)
        k = (keys.reshape(batch * length, width) @ kweight.T)
        k = k.reshape(batch, length, heads, dim).transpose(0, 2, 1, 3)
        logits = np.einsum("bhd,bhnd->bhn", q, k) / math.sqrt(dim)
        logits -= logits.max(axis=-1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=-1, keepdims=True)
        loss = -np.sum(target[:, None, :] * np.log(prob)) / (batch * heads)
        error = (prob - target[:, None, :]) / (batch * heads)
        dq = np.einsum("bhn,bhnd->bhd", error, k) / math.sqrt(dim)
        dk = error[:, :, :, None] * q[:, :, None, :] / math.sqrt(dim)
        gq = dq.reshape(batch, heads * dim).T @ x
        gk = (dk.transpose(0, 2, 1, 3).reshape(batch * length, heads * dim).T
              @ keys.reshape(batch * length, width))
        return float(loss), gq, gk

    initial_loss, gq, gk = loss_grad(wq, wk)
    max_abs_error = 0.0
    for weight, grad in [(wq, gq), (wk, gk)]:
        for ix in [(0, 0), (13, 7), (31, 31)]:
            original, eps = weight[ix], 1e-6
            weight[ix] = original + eps
            plus = loss_grad(wq, wk)[0]
            weight[ix] = original - eps
            minus = loss_grad(wq, wk)[0]
            weight[ix] = original
            max_abs_error = max(max_abs_error, abs((plus - minus) / (2 * eps) - grad[ix]))
    assert max_abs_error < 1e-8
    start = time.monotonic()
    for _ in range(50):
        _, gq, gk = loss_grad(wq, wk)
        for weight, grad in [(wq, gq), (wk, gk)]:
            u, singular, vt = np.linalg.svd(grad, full_matrices=False)
            keep = singular > 1e-12 * singular[0]
            weight *= 0.997
            weight -= 0.03 * (u[:, keep] @ vt[keep, :])
    elapsed = time.monotonic() - start
    return {
        "gradient_finite_difference_max_abs_error": max_abs_error,
        "fixed_batch_steps": 50,
        "initial_loss": initial_loss,
        "final_fixed_batch_loss": loss_grad(wq, wk)[0],
        "seconds_excluding_probe_measurement": elapsed,
        "scope": "kernel calibration only; not E1 or E2 results",
    }


def main():
    start = time.monotonic()
    rng = np.random.default_rng(20261105)
    eta, lam, dh = 0.02, 0.2, 16
    c = 1 - eta * lam
    v = eta**2 / (1 - c**2)
    assert math.isclose(v, eta / (lam * (2 - eta * lam)))
    mstar = math.sqrt(eta * lam / 2)
    assert math.isclose(mstar**2 / lam**2, (1 - mstar**2) * v)

    # Direct double covariance sum versus the correlated-noise closed form.
    corr_c, rho = 0.8, 0.6
    indices = np.arange(160)
    weights = corr_c**indices
    covariance = rho**np.abs(indices[:, None] - indices[None, :])
    direct_variance = eta**2 * (weights @ covariance @ weights)
    correlated_variance = (eta**2 / (1 - corr_c**2)
                           * (1 + corr_c * rho) / (1 - corr_c * rho))
    assert math.isclose(direct_variance, correlated_variance, rel_tol=1e-12)

    # Exact finite-time diagonal retrieval recurrence and its sign direction.
    p = np.array([0.8, 0.15, 0.05])
    a = np.array([0.1, 0.3, 0.7])
    b = np.array([0.2, 0.4, 0.9])
    g = -p * b / (math.sqrt(3) * (1 + np.exp(a * b / math.sqrt(3))))
    u, _, vt = np.linalg.svd(np.diag(g), full_matrices=False)
    assert np.allclose(u @ vt, np.diag(np.sign(g)))
    a0 = a.copy()
    for _ in range(100):
        a = c * a + eta
    assert np.allclose(a, c**100 * a0 + (1 - c**100) / lam)

    # Mean-square identity, including the phase-boundary bias.
    reps, steps = 16000, 1500
    biases = np.array([0.0, mstar, 0.3, 1.0])
    states = np.zeros((len(biases), reps))
    for _ in range(steps):
        noise = 2 * (rng.random(states.shape) < (1 + biases[:, None]) / 2) - 1
        states = c * states + eta * noise
    mu = biases * (1 - c**steps) / lam
    expected = mu**2 + (1 - biases**2) * v * (1 - c ** (2 * steps))
    observed = np.mean(states**2, axis=1)
    relative = np.abs(observed - expected) / expected
    # A broad diagnostic guard; exact formulas are proved in the document.
    assert np.all(relative < 0.06), (expected, observed)

    # A fresh projected process, not a post-hoc clipping of linear samples.
    radius, horizon, delta, burn = 2.5, 2500, 0.05, 142
    rn = math.sqrt(2 * v * math.log(2 * horizon / delta))
    assert c**burn * radius + rn < radius
    projected = np.full(4096, radius)
    ever_after_burn = np.zeros(projected.size, dtype=bool)
    early_events = 0
    for n in range(1, horizon + 1):
        signs = rng.integers(0, 2, projected.size) * 2 - 1
        candidate = c * projected + eta * signs
        events = np.abs(candidate) > radius
        if n < burn:
            early_events += int(events.sum())
        else:
            ever_after_burn |= events
        projected = np.clip(candidate, -radius, radius)

    # Observe recurrence in a deliberately less concentrated configuration.
    eta2, lam2, radius2 = 0.2, 0.5, 1.0
    c2 = 1 - eta2 * lam2
    run_length = next(
        k for k in range(1, 1000)
        if c2**k < (1 - lam2 * radius2) / (1 + lam2 * radius2)
    )
    recurrent = np.zeros(128)
    recurrence_counts = np.zeros(128, dtype=np.int64)
    for _ in range(20000):
        candidate = c2 * recurrent + eta2 * (rng.integers(0, 2, 128) * 2 - 1)
        recurrence_counts += np.abs(candidate) > radius2
        recurrent = np.clip(candidate, -radius2, radius2)

    # Sharp constructions for the two RMS budgets.
    for heads in [1, 2, 4, 8]:
        d = 8
        q = np.zeros((heads, d))
        q[0, 0] = math.sqrt(heads * d)
        assert np.isclose(np.max(np.sum(q * q, axis=1)) / math.sqrt(d),
                          heads * math.sqrt(d))
        q[0, 0] = math.sqrt(d)
        assert np.isclose(np.max(np.sum(q * q, axis=1)) / math.sqrt(d),
                          math.sqrt(d))

    calibration = benchmark_attention()
    print(json.dumps({
        "seed": 20261105,
        "variance_floor": v,
        "logit_floor": v / math.sqrt(dh),
        "critical_bias": mstar,
        "correlated_variance_direct_sum": direct_variance,
        "correlated_variance_closed_form": correlated_variance,
        "moment_expected": expected.tolist(),
        "moment_observed": observed.tolist(),
        "moment_relative_errors": relative.tolist(),
        "clip_r_N": rn,
        "clip_envelope_at_142": c**burn * radius + rn,
        "clip_radius": radius,
        "early_clip_events": early_events,
        "fraction_any_clip_142_to_2500": float(ever_after_burn.mean()),
        "recurrence_forcing_run_length": run_length,
        "recurrence_events_min_median_max": [
            int(recurrence_counts.min()), float(np.median(recurrence_counts)),
            int(recurrence_counts.max())],
        "attention_kernel_calibration": calibration,
        "peak_RSS_MiB_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "elapsed_seconds": time.monotonic() - start,
    }, indent=2))


if __name__ == "__main__":
    main()
