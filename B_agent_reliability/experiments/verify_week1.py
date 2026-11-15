"""Audit saved outputs; no new stochastic experiment or model selection."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/week1_results"
checks = []


def check(condition, description):
    if not condition:
        raise AssertionError(description)
    checks.append(description)


def read(name):
    return json.loads((OUT / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read("manifest.json")
check(manifest["status"] == "complete", "all four stages completed")
check(manifest["interpreter"] == "/home/liuyibo/venvs/theory/bin/python", "requested interpreter")
check(manifest["torch_threads"] == 3 and manifest["env"]["OMP_NUM_THREADS"] == "3", "three requested threads")
check(not manifest["cuda_initialized"], "CUDA was not initialized")
check(manifest["RLIMIT_AS_bytes"] == 3_500_000_000, "3.5 GB address space hard limit")
check(manifest["max_rss_mib"] * 1024 ** 2 < 3_500_000_000, "peak RSS below 3.5 GB")
check(manifest["elapsed_seconds"] < 3300, "total experimental wall time below 55 minutes")
for name, limit in (("E1", 480), ("E2", 480), ("E3", 900), ("E4", 1080)):
    check(manifest["stage_times"][name] < limit, f"{name} stage within preregistered hard limit")
check(digest(ROOT / "FORMULATION.md") == manifest["formulation_sha256"], "original formulation unchanged")
check(digest(ROOT / "experiments/preregistration_week1.md") == manifest["preregistration_sha256"], "frozen implementation protocol unchanged")
check(digest(ROOT / "experiments/week1.py") == manifest["code_sha256"], "executed experiment code unchanged")
if "initial_attempt_code_sha256" in manifest:
    check(digest(ROOT / "experiments/week1_initial_attempt.py") == manifest["initial_attempt_code_sha256"], "original E1-E3 executed code preserved")
    check(digest(ROOT / "experiments/amendment_week1.md") == manifest["amendment_sha256"], "memory amendment preserved")
    check(manifest["E4_all_attempts_seconds"] < 1080, "E4 combined attempts within hard limit")

e1 = read("E1.json")
check(len(e1["formula"]) == 60 and len(e1["heldout"]) == 36 and len(e1["calibration"]) == 12, "E1 full sample grid")
for r in e1["formula"] + e1["calibration"] + e1["heldout"]:
    hist = np.array(r["joint_histogram"])
    n = int(hist[:, 2].sum())
    work = int(hist[:, 0] @ hist[:, 2])
    failures = int(hist[:, 1] @ hist[:, 2])
    check(n == 20000 and failures == r["failures"], "E1 raw histogram counts")
    check(work == sum(r["up"]) + sum(r["down"]), "E1 jump conservation for work")
    k = r.get("k", 3)
    check(sum(r["up"]) - sum(r["down"]) == k * failures - n, "E1 endpoint conservation")

e2 = read("E2.json")
check(len(e2["configurations"]) == 54, "E2 all alpha/capacity/bypass configurations")
for r in e2["configurations"]:
    with gzip.open(OUT / "raw" / r["raw_file"], "rt") as f:
        raw = json.load(f)
    rows = np.asarray(raw["rows"], dtype=np.uint8)
    check(rows.shape == (100000, 21) and rows.max() <= 8, "E2 task counts and repeated-run bounds")
    check(np.allclose(rows.mean(axis=0) / 8, r["observed"], rtol=0, atol=1e-14), "E2 aggregate matches raw task JSON")
    check(np.allclose(rows.std(axis=0, ddof=1) / (8 * np.sqrt(len(rows))), r["task_se"], atol=1e-14), "E2 standard errors use task clusters")
    del raw, rows

e3 = read("E3.json")
if (OUT / "E2_derived.json").exists():
    derived=read("E2_derived.json")["configurations"]
    check(len(derived)==27, "E2 empirical horizon and curvature measurements completed")
    for row in derived:
        check(0 < row["observed_H80"] < row["observed_H50"], "E2 observed horizons have correct order")
        check(row["H50_bootstrap_ci"][0] <= row["H50_bootstrap_ci"][1], "E2 task bootstrap horizon interval order")
check(e3["mandatory_negative_control_pass"], "E3 required state-dependent negative control rejected")
for env in e3["environments"]:
    check(len(env["tests"]) == 3, "E3 all held-out capacities")
    for r in env["tests"]:
        check(r["state_model_prediction"] is None, "E3 inaccessible states never filled from test data")
        if env["environment"] == 2:
            check(r["nonregeneration"]["joint_law_tail"] < 1e-12, "E3 exact episode acceleration negligible truncation tail")

e4 = read("E4.json")
check(not e4["data"]["network_used"], "E4 local-data-only execution")
check(len(e4["models"]) == 3 and len(e4["retries"]) == 24, "E4 all public-model and retry configurations")
for i, r in enumerate(e4["models"]):
    tasks = read(f"raw/E4_tasks_model{i}.json")["tasks"]
    predictions = read(f"raw/E4_predictions_model{i}.json")
    family_folds = {}
    for task, fold in zip(tasks, predictions["fold"]):
        family_folds.setdefault(task["family"], set()).add(fold)
    check(all(len(v) == 1 for v in family_folds.values()), "E4 family separation between folds")
    check(r["tasks"] >= 40 and min(r["successes"], r["failures"]) >= 10, "E4 eligibility thresholds")

for name in ("E1", "E2", "E3", "E4"):
    with Image.open(OUT / "figures" / (name + ".png")) as im:
        im.verify()
    check(True, f"{name} PNG decodes")

result = dict(status="pass", checks=len(checks), distinct_checks=sorted(set(checks)))
(OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
