"""Postprocess saved task rows for preregistered H ratios and curve slopes.

No new data, model fitting, threshold or hypothesis selection. Bootstrap seeds
and the 200 task-cluster replicates are identical to the original analysis.
"""
import os
os.environ.update(OMP_NUM_THREADS="3", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import resource
resource.setrlimit(resource.RLIMIT_AS, (3_500_000_000, 3_500_000_000))
import gzip
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "week1_results"
e2 = json.loads((OUT / "E2.json").read_text())


def horizon(values, target, lengths):
    values = np.asarray(values)
    crossing = np.flatnonzero(values <= target)
    if not len(crossing) or crossing[0] == 0:
        return None
    right = crossing[0]; left = right - 1
    logit = lambda x: np.log(x / (1 - x))
    frac = (logit(target) - logit(values[left])) / (logit(values[right]) - logit(values[left]))
    return float(np.exp(np.log(lengths[left]) * (1 - frac) + np.log(lengths[right]) * frac))


rows = []
for r in e2["configurations"]:
    if r["kind"] != "fixed":
        continue
    lengths = np.array(r["lengths"])
    obs, exact = np.array(r["observed"]), np.array(r["exact"])
    H80 = horizon(obs, .8, lengths); H50 = horizon(obs, .5, lengths)
    H80_grid = horizon(exact, .8, lengths); H50_grid = horizon(exact, .5, lengths)
    with gzip.open(OUT / "raw" / r["raw_file"], "rt") as f:
        raw = json.load(f)
    counts = np.array(raw.pop("rows"), dtype=np.float64)
    del raw
    rng = np.random.default_rng(np.random.SeedSequence([2250 + (0 if r["alpha"] == .5 else 1 if r["alpha"] == 1 else 2)]))
    n = len(counts); boot = np.zeros((200,21))
    for start in range(0, 200, 5):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=5)
        boot[start:start+5] = weights @ counts / (8*n)
    bs80 = [horizon(b, .8, lengths) for b in boot]
    bs50 = [horizon(b, .5, lengths) for b in boot]
    ratios = [a / b for a,b in zip(bs80,bs50) if a is not None and b is not None]
    valid = (obs * 800000 >= 100) & ((1 - obs) * 800000 >= 100)
    valid[[0,-1]] = False
    logit = lambda x: np.log(np.clip(x, 1e-12, 1-1e-12) / (1-np.clip(x, 1e-12, 1-1e-12)))
    slopes = -np.gradient(logit(obs), np.log(lengths))
    exact_slopes = -np.gradient(logit(exact), np.log(lengths))
    record = dict(alpha=r["alpha"], k=r["capacity"], eta=r["eta"],
                  observed_H80=H80, observed_H50=H50, observed_H80_over_H50=H80/H50,
                  H80_bootstrap_ci=np.quantile(bs80,[.025,.975]).tolist(),
                  H50_bootstrap_ci=np.quantile(bs50,[.025,.975]).tolist(),
                  ratio_bootstrap_ci=np.quantile(ratios,[.025,.975]).tolist(),
                  exact_formula_H50=r["H50"], exact_formula_ratio=r["H80_over_H50"],
                  exact_grid_H50=H50_grid, exact_grid_ratio=H80_grid/H50_grid,
                  finite_difference_logit_slope_observed=slopes.tolist(),
                  finite_difference_logit_slope_exact=exact_slopes.tolist(), valid_indices=np.flatnonzero(valid).tolist(),
                  interpolation="linear logit(S) against log L inside first target crossing; no extrapolation")
    rows.append(record)
    del counts, weights, boot

result = dict(configurations=rows,
              note="Derived only from archived outputs; fills H80/H50 and logit curvature measurements. Does not change preregistered predictions or decisions.",
              elapsed_scope="postprocessing, not an additional generated-data experiment",
              max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(OUT / "E2_derived.json").write_text(json.dumps(result, indent=2)+"\n")
print(f"Derived {len(rows)} fixed-capacity configurations")
