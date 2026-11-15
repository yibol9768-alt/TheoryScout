"""Summarize the frozen small check; all files stay in this directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr, ndtri


def load(name):
    with (ROOT / name).open() as stream:
        return list(csv.DictReader(stream))


summary, rows, cdfs = load("summary.csv"), load("rows.csv"), load("rank_cdf.csv")
predictions = json.loads((ROOT / "predictions.json").read_text())
meta = json.loads((ROOT / "metadata.json").read_text())
methods = ["GD", "Adam_1.0", "Adam_0.1", "Sign"]
colors = dict(zip(methods, ["#2459a6", "#26937b", "#d08b16", "#be4b62"]))
lines = ["# W7 numerical check", "", "Descriptive checks, without significance or equivalence claims.", "",
         "| d | mean absolute correct-mean error | mean abs variance error | mean wrong-CDF KS | mean rank-grid max error | max rank-grid error | mean abs cross-covariance error |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
aggregate = []
for d in [32, 64, 128]:
    select = [row for row in summary if int(row["d"]) == d]
    mean_error = np.mean([abs(float(row["correct_mean"])-float(row["predicted_mu"])) for row in select])
    var_error = np.mean([abs(float(row["correct_variance"])-1) for row in select])
    wrong = np.mean([float(row["wrong_cdf_ks"]) for row in select])
    rank = np.mean([float(row["rank_grid_max_error"]) for row in select])
    rank_max = max(float(row["rank_grid_max_error"]) for row in select)
    cross = np.mean([abs(float(row["correct_cross_cov"])-float(row["predicted_cross_cov"])) for row in select])
    lines.append(f"| {d} | {mean_error:.6f} | {var_error:.6f} | {wrong:.6f} | {rank:.6f} | {rank_max:.6f} | {cross:.6f} |")
    aggregate.append(dict(d=d, mean_error=float(mean_error), variance_error=float(var_error),
                          wrong_cdf_ks=float(wrong), rank_grid_error=float(rank),
                          max_rank_grid_error=float(rank_max), cross_cov_error=float(cross)))
lines += ["", "Matched population retrieval (all six key structures represented in CSVs):", "",
          "| r | target c1 | method | c0 | predicted old Recall@10% | observed old Recall@10%, d=128 | observed new Recall@10%, d=128 | predicted new Recall@10% |",
          "|---:|---:|:---|---:|---:|---:|---:|---:|"]
for case in ["shared_zero", "shared_positive", "shared_negative"]:
    r = {"shared_zero": 0., "shared_positive": .5, "shared_negative": -.5}[case]
    for method in methods:
        pp = predictions[case][method]
        old = next(row for row in summary if row["case"] == case and int(row["d"]) == 128 and row["method"] == method and row["task"] == "0")
        new = next(row for row in summary if row["case"] == case and int(row["d"]) == 128 and row["method"] == method and row["task"] == "1")
        lines.append(f"| {r:g} | {pp['target']:.3f} | {method} | {pp['c0']:.6f} | {float(old['predicted_recall_at_10pct']):.6f} | {float(old['recall_at_10pct']):.6f} | {float(new['recall_at_10pct']):.6f} | {float(new['predicted_recall_at_10pct']):.6f} |")
lines += ["", f"Run: {meta['dataset_count']} independent datasets, {meta['method_task_records']} method/task records; "
          f"{meta['seconds']:.3f} seconds, peak RSS {meta['peak_rss_mib']:.3f} MiB.",
          "Every dataset uses the actual residual Z1-Z0 S1, not the coordinate proxy residual.",
          "Each plotted CDF pools 4 seeds and 256 distinct queries per seed. Queries within a dataset are dependent.",
          "Population matching does not force finite-d new recalls to agree exactly. Small method gaps cannot be resolved by this sample size.",
          "The wrong-CDF KS uses 8 queries per seed and all n-1 wrong candidates per query.",
          "The rank error is the maximum over the nine declared quantiles, not a continuum supremum."]
(ROOT / "results.md").write_text("\n".join(lines) + "\n")
(ROOT / "aggregate.json").write_text(json.dumps(aggregate, indent=2))

fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey=True)
for col, case in enumerate(["shared_zero", "shared_positive", "shared_negative"]):
    for task in [0, 1]:
        ax = axes[task, col]
        for method in methods:
            select = [row for row in cdfs if row["case"] == case and row["d"] == "128" and row["method"] == method and row["task"] == str(task)]
            xx = sorted(set(float(row["tau"]) for row in select))
            yy = [np.mean([float(row["observed"]) for row in select if float(row["tau"]) == x]) for x in xx]
            mu = predictions[case][method][f"c{task}"] / np.sqrt(.5)
            grid = np.linspace(.001, .999, 400)
            ax.plot(grid, ndtr(mu + ndtri(grid)), color=colors[method], lw=1.25,
                    label=method.replace("_", " "))
            ax.scatter(xx, yy, color=colors[method], s=17, marker="o", alpha=.75)
        ax.set_title(f"{case}; task {task}", fontsize=10)
        ax.grid(alpha=.2)
        if col == 0:
            ax.set_ylabel("P(normalized rank <= tau)")
        if task == 1:
            ax.set_xlabel("tau")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Actual one-step FT: curves = rank law; dots = d=128 simulation\nNew-task population retrieval is matched across methods", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "rank_cdf.png", dpi=160)
fig.savefig(ROOT / "rank_cdf.pdf")
plt.close(fig)
print("\n".join(lines))
