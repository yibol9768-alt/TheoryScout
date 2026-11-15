"""Small numerical checks, not the full proposed experiment. NumPy only."""
import json
import time
from pathlib import Path
import numpy as np

V0 = np.array([[0, 0], [1, -1], [2, 0], [1, -9 / 8],
               [-1, 1 / 2], [-1, 1 / 2], [-1, 1 / 2]], dtype=float)
target = np.array([1 / 3] * 3 + [0.] * 4)
labels = np.array([1.] * 3 + [-1.] * 4)


def nce(w, v):
    s = v @ w
    p = np.exp(s - s.max())
    p /= p.sum()
    a = p - target
    loss = s.max() + np.log(np.exp(s - s.max()).sum()) - s[:3].mean()
    return loss, a @ v, a[:, None] * w[None, :]


def gap(w, v):
    s = v @ w
    return float(s[:3].min() - s[3:].max())


def main():
    start = time.perf_counter()
    star = np.array([np.log(2), 0.])
    _, grad, _ = nce(star, V0)
    w = np.zeros(2)
    for _ in range(6000):
        w -= .05 * nce(w, V0)[1]
    fixed = {"gradient_at_star": grad.tolist(), "w_gd": w.tolist(),
             "distance_to_star": float(np.linalg.norm(w - star)),
             "gap": gap(w, V0), "excess": nce(w, V0)[0] - np.log(3)}
    # Bias is an explicit third coordinate; neither norm nor temperature is fitted.
    x = np.column_stack((V0, np.ones(7)))
    theta = np.zeros(3)
    for _ in range(40000):
        z = labels * (x @ theta)
        theta += .1 * x.T @ (labels * np.exp(-np.logaddexp(0, z))) / 7
    sig = {"theta": theta.tolist(), "gap": gap(theta[:2], V0),
           "min_signed_logit": float((labels * (x @ theta)).min())}
    # Euler approximations of the two-timescale gradient flow, warm start at star.
    # First passage is a diagnostic, not a theorem about an upper bound.
    runs = []
    for eps in [.04, .02, .01, .005, .0025]:
        w, v = star.copy(), V0.copy()
        hit = None
        h = .05
        max_steps = int(5 / (eps * h))
        for step in range(max_steps):
            _, gw, gv = nce(w, v)
            w, v = w - h * gw, v - eps * h * gv
            if gap(w, v) > 1e-4:
                hit = (step + 1) * h
                break
        runs.append({"epsilon": eps, "hit_time": hit,
                     "epsilon_times_hit": None if hit is None else eps * hit,
                     "final_gap": gap(w, v), "steps": step + 1})
    # Exact two-positive formula and optimum by a deterministic scalar bisection.
    temp = []
    for r in [2, 8, 32]:
        for ratio in [.01, .05, .2]:
            lo, hi = 0., 50.
            for _ in range(80):
                b = (lo + hi) / 2
                if ratio * np.sinh(b * ratio) < (r / 2) * np.exp(-b):
                    lo = b
                else:
                    hi = b
            beta = (lo + hi) / 2
            temp.append({"r": r, "delta_over_g": ratio, "beta_g": beta,
                         "optimal_excess": float(np.log(np.cosh(beta * ratio)
                                                         + r / 2 * np.exp(-beta)))})
    result = {"fixed_nce": fixed, "fixed_sigmoid": sig, "slow_documents": runs,
              "temperature": temp, "elapsed_seconds": time.perf_counter() - start,
              "note": "Deterministic numerical sanity checks, not proof or full sweep."}
    out = Path(__file__).with_name("core_results.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
