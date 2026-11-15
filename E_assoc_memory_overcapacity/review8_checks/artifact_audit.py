"""Read-only independent reconciliation of delivered W9 tables and review runs."""
import os
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
P = Path(os.environ.get("REVIEW8_OUTPUT_DIR", str(BASE))).resolve()
assert P.is_relative_to(BASE)
W = ROOT/"w9_checks"
raw = [json.loads(line) for line in (W/"formal_rows.jsonl").read_text().splitlines()]
focused = [json.loads(line) for line in (W/"focus_rows.jsonl").read_text().splitlines()]
groups = defaultdict(list)
for r in raw:
    groups[(r["case"],r["d"],r["method"])].append(r)
table = list(csv.DictReader((W/"summary.csv").open()))
max_discrepancy = 0.
metrics = ["recall10","correct_mean","correct_var","wrong_var","cdf_grid_error","rank_mean",
           "frob_error","correct_error2","wrong_error2"]
for r in table:
    observations = groups[(r["case"],int(r["d"]),r["method"])]
    assert len(observations)==8
    for metric in metrics:
        values = np.array([obs[metric] for obs in observations])
        calculated = (values.mean(),t.ppf(.975,7)*values.std(ddof=1)/np.sqrt(8))
        max_discrepancy = max(max_discrepancy,abs(calculated[0]-float(r[metric])),
                              abs(calculated[1]-float(r[metric+"_half95"])))
assert len(raw)==600 and len(groups)==75 and len(table)==75
assert len(focused)==72
focus_summary = json.loads((W/"focus_summary.json").read_text())
for record in focus_summary["summary"]:
    vals = np.array([r["recall"] for r in focused if r["case"]==record["case"]])
    assert len(vals)==24
    max_discrepancy = max(max_discrepancy,abs(vals.mean()-record["mean"]),
        abs(t.ppf(.975,23)*vals.std(ddof=1)/np.sqrt(24)-record["half95"]))
paired = json.loads((W/"paired_contrasts.json").read_text())
for r in paired:
    left = {v["seed"]:v["recall10"] for v in groups[(r["left"],r["d"],r["method"])]}
    right = {v["seed"]:v["recall10"] for v in groups[(r["right"],r["d"],r["method"])]}
    diff = np.array([left[k]-right[k] for k in sorted(left)])
    max_discrepancy = max(max_discrepancy, abs(diff.mean()-r["delta"]),
                         abs(t.ppf(.975,7)*diff.std(ddof=1)/np.sqrt(8)-r["half95"]))
for r in focus_summary["contrasts"]:
    left = {v["seed"]:v["recall"] for v in focused if v["case"]==r["left"]}
    right = {v["seed"]:v["recall"] for v in focused if v["case"]==r["right"]}
    diff = np.array([left[k]-right[k] for k in sorted(left)])
    max_discrepancy = max(max_discrepancy, abs(diff.mean()-r["delta"]),
                         abs(t.ppf(.975,23)*diff.std(ddof=1)/np.sqrt(24)-r["half95"]))
assert max_discrepancy < 1e-12
assert all(r["frob_error"]==r["correct_error2"]==r["wrong_error2"]==0
           for r in raw if r["method"] in ("gd1","sign1"))
old_audit = json.loads((W/"audit.json").read_text())
sources_match = all(hashlib.sha256((W/name).read_bytes()).hexdigest()==expected
                   for name,expected in old_audit["source_hashes"].items())
assert sources_match
before = json.loads((P/"input_hashes_before.json").read_text())
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
           for name,value in before.items())
out = dict(formal_rows=len(raw),formal_groups=len(groups),focus_rows=len(focused),
           paired_contrasts=len(paired),focused_contrasts=len(focus_summary["contrasts"]),
           maximum_table_discrepancy=max_discrepancy,source_hashes_match=sources_match,
           first_steps_equal_proxy=True,review_inputs_unchanged=True,
           observed_process_threads=len(list(Path("/proc/self/task").iterdir())),
           limitation="Original outputs do not record ties; zero finite-sample tie probability is false.")
assert out["observed_process_threads"] <= 2
with (P/"artifact_audit.json").open("x") as f:
    json.dump(out,f,indent=2)
    f.write("\n")

review = json.loads((P/"summary.json").read_text())
meta = json.loads((P/"metadata.json").read_text())
lines = ["# 独立模拟汇总", "", "区间为 12 个独立数据集种子的逐项 t₁₁ 95% 区间半宽。",
         "", "|情形|d|算法|C|预测召回|实际召回 ± 半宽|正确均值 / 预测|正确方差|错误方差|逐查询 KS 均值|",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in review:
    p = r["prediction"]
    lines.append(f"|{r['case']}|{r['d']}|{r['method']}|{p['C']:.5f}|{p['recall']:.5f}|"
        f"{r['recall']['mean']:.5f} ± {r['recall']['half95']:.5f}|"
        f"{r['correct_mean']['mean']:.4f} / {p['mean']:.4f}|"
        f"{r['correct_var']['mean']:.4f}|{r['wrong_var']['mean']:.4f}|{r['ks_mean']['mean']:.4f}|")
lines += ["", "|情形|d|算法|Frobenius 均方误差|正确测量误差²|错误测量误差²|",
          "|---|---:|---|---:|---:|---:|"]
for r in review:
    if r["method"] in ("gd4","sign4","adam4"):
        lines.append(f"|{r['case']}|{r['d']}|{r['method']}|{r['frobenius']['mean']:.6g}|"
                     f"{r['correct_error']['mean']:.6g}|{r['wrong_error']['mean']:.6g}|")
lines += ["",f"主体耗时 {meta['seconds']:.2f} 秒；峰值 RSS {meta['maxrss_kib']/1024:.2f} MiB。",
          f"测试端点的 ties 总数：{sum(r['tie_total'] for r in review)}。该观测不能证明 ties 概率为零。",
          "预测未从训练数据拟合；Sign4 用精确分段高斯积分，Adam4 用独立一维数值积分。",
          "KS 按每数据集前 16 个抽取的查询逐一计算，再汇总；它是有限维诊断。"]
with (P/"SUMMARY.md").open("x") as f:
    f.write("\n".join(lines)+"\n")
print(json.dumps(out,ensure_ascii=False))
print("\n".join(line for line in lines if "|256|" in line))
