"""Produce exportable figures and concise tables; does not refit results."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
os.environ["MPLCONFIGDIR"]=str(HERE/"mplconfig")
os.environ["XDG_CACHE_HOME"]=str(HERE/"cache")
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import json
import resource
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator


def main():
    start=time.monotonic()
    a=json.loads((HERE/"analysis.json").read_text())
    p=json.loads((HERE/"formal/proxy.json").read_text())
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    labels={"stat_M":"stat M (1/3)","lr_decay":"LR transfer (2/7)",
            "nu1_b1":"nu=1,b=1 (1/4)","nu1_z1":"nu=1,z=1 (2/9)",
            "nu8_z038":"nu=8,z=3/8 (1/3)"}
    for name,label in labels.items():
        rows=p["curves"][name]
        axs[0].loglog([r["C"] for r in rows],[r["risk"] for r in rows],label=label)
    axs[0].set(xlabel="Virtual cost C = nbT",ylabel="Exact expected risk",title="Proxy frontiers (finite grid)")
    axs[0].legend(fontsize=8)
    rows=p["curves"]["oversize"]
    C=np.array([float(r["C"]) for r in rows]);R=np.array([r["risk"] for r in rows])
    axs[1].loglog(C,R,"o-",label="measured")
    axs[1].loglog(C,R[-1]*(C/C[-1])**(-1/12),"--",label="slope -1/12")
    axs[1].loglog(C,R[-1]*(C/C[-1])**(-1/14),":",label="forced u=1: -1/14")
    axs[1].set(xlabel="Virtual cost C",ylabel="Exact expected risk",title="Oversized-model core test")
    axs[1].legend(fontsize=8)
    for key,exp in (("n",1/6),("h",1/12)):
        v=np.array([r[key] for r in rows])
        axs[2].loglog(C,v,"o-",label=key)
        axs[2].loglog(C,v[-1]*(C/C[-1])**exp,"--",label=f"slope {exp:.4f}")
    axs[2].set(xlabel="Virtual cost C",ylabel="Dimension",title="Model size vs learned dimension")
    axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(HERE/"proxy_frontiers.png",dpi=180)
    fig.savefig(HERE/"proxy_frontiers.pdf")
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for alpha,color in ((1.5,"C0"),(3.,"C1")):
        rs=[r for r in a["state_response"] if r["alpha"]==alpha and r["algorithm"]=="muon0"]
        b=np.array([r["b"] for r in rs])
        axs[0].errorbar(b,[r["fit_rate1"] for r in rs],
                        yerr=[1.96*r["fit_se1"] for r in rs],color=color,fmt="o-",label=f"alpha={alpha}, fit E")
        axs[0].errorbar(b,[r["holdout_rate1"] for r in rs],
                        yerr=[1.96*r["holdout_se1"] for r in rs],color=color,fmt="s--",label=f"alpha={alpha}, holdout E/4")
        axs[1].loglog(b,[r["fit_noise_trace"] for r in rs],"o-",color=color,label=f"alpha={alpha}, E")
        axs[1].loglog(b,[r["holdout_noise_trace"] for r in rs],"s--",color=color,label=f"alpha={alpha}, E/4")
    axs[0].set(xscale="log",xlabel="Batch size b",ylabel="Unit-residual first-mode drift",
               title="Muon: fixed ridge cannot absorb state change")
    axs[1].set(xlabel="Batch size b",ylabel="Trace of update noise covariance",
               title="Noise changes with residual state")
    for ax in axs:
        ax.set_xticks([4,16,64],labels=["4","16","64"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(HERE/"bridge_holdout.png",dpi=180)
    fig.savefig(HERE/"bridge_holdout.pdf")
    plt.close(fig)
    # Compact numeric tables keep all seeds and preselected intervals visible.
    lines=["# 补充数值表（由 analysis.json 原值生成）","","## 风险、尺寸与已学维数","",
           "|配置|k|n斜率|h斜率|末n|末h|Ta_n|B/A|V/A|",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in p["summaries"]:
        r=s["last"]
        lines.append(f'|{s["name"]}|{s["observed_k"]:.9f}|{s["n_slope"]:.9f}|{s["h_slope"]:.9f}|{r["n"]}|{r["h"]:.6f}|{r["Ta_n"]:.6g}|{s["last_B_over_A"]:.6g}|{s["last_V_over_A"]:.6g}|')
    lines.extend(["","## 在线矩阵 nD≤2²⁰ 的有限网格前沿（全部种子）","",
                  "|α|协议|算法|seed0|seed1|seed2|均值|样本SD|选中尺寸|选中batch|选中步数|",
                  "|---:|---|---|---:|---:|---:|---:|---:|---|---|---|"])
    for s in a["shared_compute_frontiers"]:
        rs=s["seeds"]
        lines.append(f'|{s["alpha"]}|{s["protocol"]}|{s["algorithm"]}|'+
                     "|".join(f'{r["risk"]:.7f}' for r in rs)+f'|{s["mean"]:.7f}|{s["sd"]:.7f}|'+
                     "|".join(",".join(str(r[k]) for r in rs) for k in ("n","b","T"))+"|")
    lines.extend(["","## 尺寸转移对相同步数末点风险的影响","",
                  "每行对三种 batch、全部三个种子的 decay/fixed 比取均值；不是计算最优加速比。","",
                  "|α|算法|n|均值|最小|最大|","|---:|---|---:|---:|---:|---:|"])
    for s in a["transfer"]:
        lines.append(f'|{s["alpha"]}|{s["algorithm"]}|{s["n"]}|{s["decay_to_fixed_mean"]:.6f}|{s["min"]:.6f}|{s["max"]:.6f}|')
    lines.extend(["","## 调参锁定学习率","","|α|batch|SGD|oracle|Muon μ=0|Muon μ=.9|",
                  "|---:|---:|---:|---:|---:|---:|"])
    for alpha in (1.5,3.):
        for b in (4,16,64):
            vals={r["algorithm"]:r["eta"] for r in a["locked_learning_rates"] if r["alpha"]==alpha and r["b"]==b}
            lines.append(f'|{alpha}|{b}|'+"|".join(str(vals[k]) for k in ("sgd","oracle","muon0","muon09"))+"|")
    (HERE/"TABLES.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status="complete",elapsed_seconds=time.monotonic()-start,
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)))


if __name__=="__main__":
    main()
