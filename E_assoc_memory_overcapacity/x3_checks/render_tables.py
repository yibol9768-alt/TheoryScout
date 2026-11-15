"""Render certificate outputs; refuses to overwrite."""
import json
import math
from pathlib import Path

data = json.loads(Path("x3_checks/polar_certificate.json").read_text())
md = ["数值估计与可验证包围分开列出。区间端点向外显示。", "",
      "| γ | μ（估计） | k（估计） | ν（估计） | ν 的包围 |",
      "|---:|---:|---:|---:|---:|"]
for r in data["rows"]:
    lo = math.floor(r["nu"]["lower"]*10**12)/10**12
    hi = math.ceil(r["nu"]["upper"]*10**12)/10**12
    md.append(f'|{r["gamma"]}|{r["mu_estimate"]:.10f}|{r["k_estimate"]:.10f}|'
              f'{r["nu_estimate"]:.10f}|[{lo:.12f}, {hi:.12f}]|')
md.extend(["", "| γ | τ | e*（估计） | e* 的严格符号包围 | W35 包络带（约） | e*−下端 | 上端−e* |",
           "|---:|---:|---:|---:|---:|---:|---:|"])
tex = [r"\begin{center}", r"\begin{tabular}{rrrrr}", r"\hline",
       r"$\gamma$ & $\tau$ & Estimate of $e_*$ & Certified enclosure & W35 envelope\\",
       r"\hline"]
for r in data["rows"]:
    for c in r["crossings"]:
        e = c["e_certificate"]
        lo, hi = e["lower_numerator"]/e["denominator"], e["upper_numerator"]/e["denominator"]
        md.append(f'|{r["gamma"]}|{c["tau"]}|{c["e_estimate"]:.9f}|[{lo:.9f}, {hi:.9f}]|'
                  f'[{c["w35_lower_estimate"]:.7f}, {c["w35_upper_estimate"]:.7f}]|'
                  f'{c["gap_to_lower"]:.8f}|{c["gap_to_upper"]:.8f}|')
        tex.append(f'{r["gamma"]} & {c["tau"]} & {c["e_estimate"]:.9f} & '
                   f'$[{lo:.9f},{hi:.9f}]$ & '
                   f'$[{c["w35_lower_estimate"]:.7f},{c["w35_upper_estimate"]:.7f}]$'+r'\\')
tex.extend([r"\hline", r"\end{tabular}", r"\end{center}",
            "The estimate and W35 columns are descriptive rounded values; only the enclosure column has certified decimal endpoints."])
md.extend(["", "更早的对称方差包络 1±52γ^(−3/2)，τ=0：", "",
           "| γ | 旧包络得到的 e 带（约） | W35 带宽 | 本次 e* 包围宽度 | 缩窄倍数 |",
           "|---:|---:|---:|---:|---:|"])
for r in data["rows"]:
    c = r["crossings"][0]
    old_hi = "∞（无有限充分上端）" if c["old_52_upper_estimate"] is None else f'{c["old_52_upper_estimate"]:.7f}'
    width = c["w35_upper_estimate"]-c["w35_lower_estimate"]
    cw = c["e_certificate"]["upper"]-c["e_certificate"]["lower"]
    md.append(f'|{r["gamma"]}|[{c["old_52_lower_estimate"]:.7f}, {old_hi}]|{width:.8f}|{cw:.9f}|{width/cw:.1f}|')
with Path("x3_checks/tables.md").open("x") as f:
    f.write("\n".join(md)+"\n")
with Path("x3_checks/tables.tex").open("x") as f:
    f.write("\n".join(tex)+"\n")
print("\n".join(md))
