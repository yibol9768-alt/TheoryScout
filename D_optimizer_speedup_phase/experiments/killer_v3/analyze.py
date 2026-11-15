"""Independent saved-array audit and fixed preregistered decisions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"experiments/killer_v3"


def write(path,x):
    path.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")


def analyze(mode):
    start=time.monotonic()
    out=BASE/mode
    data=json.loads((out/"data.json").read_text())
    manifest=json.loads((out/"manifest.json").read_text())
    records=data["records"]
    p0=data["p0"]
    p0pass=all(p0[k]<=v for k,v in {"one_step_max_abs":1e-10,"quadrature_max_abs":1e-8,"sum_k_max_abs":1e-8,"one_dimensional_max_abs":1e-8}.items())
    local=[r for r in records if r["kind"]=="local"]
    path=[r for r in records if r["kind"]=="path"]
    controls=[r for r in records if r["kind"]=="control"]
    traj=[r for r in records if r["kind"]=="trajectory"]
    def kernel_summary(items):
        fails=[{"key":r["key"],"algorithm":r["algorithm"],"alpha":r["alpha"],"n":r["n"],"details":r["details"],
                "drift_pass":r["drift_pass"],"noise_pass":r["noise_pass"],"probe_pass":r["probe_pass"]}
               for r in items if not all(r[k] for k in ("drift_pass","noise_pass","probe_pass"))]
        return {"count":len(items),"failures":fails,"pass":not fails,
                "max_drift_error_over_limit":max(r["drift_error"]/r["drift_limit"] for r in items),
                "noise_ratio_range":[min(r["noise_ratio"] for r in items),max(r["noise_ratio"] for r in items)],
                "max_noise_error_over_limit":max(abs(r["noise_estimate"]-r["noise_prediction"])/r["noise_limit"] for r in items)}
    ks,ps=kernel_summary(local),kernel_summary(path)
    control_fail=[r["key"] for r in controls if not all(r[k] for k in ("fresh_pass","pooled_pass","ratio_pass","theory_order_pass"))]
    strong=[r for r in controls if r["a"]==.2 and r["T"]==2048]
    divergence=True if mode=="smoke" else bool(strong and strong[0]["ratio"]-6*strong[0]["ratio_se"]>100)
    bad_loc=[{"config_key":r["config_key"],"algorithm":r["algorithm"],"t":r["t"]} for r in traj if r["localization_lower_max"]>16]
    unsure_loc=[{"config_key":r["config_key"],"algorithm":r["algorithm"],"t":r["t"]} for r in traj
                if r["localization_lower_max"]<=16<r["localization_upper_max"]]
    risk_fail=[{"config_key":r["config_key"],"algorithm":r["algorithm"],"t":r["t"]} for r in traj if not r["risk_sandwich_pass"]]
    domain_pass=all(r["stepsize_spectral"]<=1 and (r["algorithm"]!="sign" or r["eta_H"]<=.25) for r in traj)
    p4="fail" if bad_loc or risk_fail or not domain_pass else ("inconclusive" if unsure_loc else "pass")
    expected={"local":2,"path":2,"trajectory_algorithms":2,"controls":2} if mode=="smoke" else {"local":54,"path":24,"trajectory_algorithms":36,"controls":12}
    complete=manifest["counts"]==expected and manifest["status"]=="complete"
    resource_pass=("MemoryMax=2147483648" in manifest["scope_properties"] and "MemorySwapMax=0" in manifest["scope_properties"]
                   and all(v=="1" for v in manifest["threads"].values()) and manifest["peak_rss_bytes"]<2*1024**3
                   and manifest["executable"]=="/home/liuyibo/venvs/theory/bin/python")
    subprocess.run(["sha256sum","-c",str(BASE/"FROZEN.sha256")],cwd=ROOT,check=True,capture_output=True)
    # Recompute the substantive risk/localization quantities from saved full checkpoint states.
    audit={"represented_max_abs":0.,"decomposition_max_abs":0.,"localization_max_abs":0.,"controls_max_abs":0.,"array_files":0}
    for key in sorted(set(r["config_key"] for r in traj)):
        with np.load(out/f"arrays/trajectory_{key}.npz") as arr:
            states=arr["states"]
            n=int(arr["n"])
            alpha=float(arr["alpha"])
            lam=np.arange(1,n+1,dtype=float)**(-alpha)
            sig=float(arr["sigma_n"])
            checks=arr["checkpoints"].tolist()
            for r in (z for z in traj if z["config_key"]==key):
                e=states[checks.index(r["t"]),["sign","polar"].index(r["algorithm"])]
                v=np.sum(e*e*lam,axis=1)
                audit["represented_max_abs"]=max(audit["represented_max_abs"],abs(float(v.mean())-r["represented_mean"]))
                decomp=np.dot(lam,e.mean(axis=0)**2+e.var(axis=0,ddof=0))
                audit["decomposition_max_abs"]=max(audit["decomposition_max_abs"],abs(float(decomp)-r["represented_mean"]))
                ratio=np.sum(e*e*(sig*sig+v[:,None]),axis=0)/np.sum(e*e,axis=0)
                audit["localization_max_abs"]=max(audit["localization_max_abs"],float(np.max(abs(ratio-r["localization_ratio"]))))
        audit["array_files"]+=1
    for r in controls:
        with np.load(out/f"arrays/control_{r['key']}.npz") as arr:
            audit["controls_max_abs"]=max(audit["controls_max_abs"],abs(float(np.mean(arr["fresh"]**2))-r["fresh_estimate"]),abs(float(np.mean(arr["pooled"]**2))-r["pooled_estimate"]))
        audit["array_files"]+=1
    audit_pass=all(v<1e-10 for k,v in audit.items() if k.endswith("_max_abs"))
    gates={"P0":"pass" if p0pass and audit_pass else "fail","P1":"pass" if ks["pass"] else "fail",
           "P2":"pass" if not control_fail and divergence else "fail","P3":"pass" if ps["pass"] else "fail","P4":p4,
           "P5":"pass" if complete and resource_pass else "incomplete"}
    gram=[r["gram_ablation"] for r in local+path if r["algorithm"]=="polar"]
    summary={"mode":mode,"gates":gates,"p0":p0,"audit":audit,"local":ks,"path":ps,
             "controls":{"count":len(controls),"failures":control_fail,"strong_separation":strong,"divergence_pass":divergence,
                         "max_relative_ratio_error":max(abs(r["ratio"]/r["ratio_prediction"]-1) for r in controls)},
             "localization":{"failed":bad_loc,"inconclusive":unsure_loc,"risk_failures":risk_fail,"domain_pass":domain_pass,
                             "checkpoint_algorithm_count":len(traj),"max_estimate":max(max(r["localization_ratio"]) for r in traj),
                             "max_upper6se":max(r["localization_upper_max"] for r in traj),
                             "max_eta_H_sign":max(r["eta_H"] for r in traj if r["algorithm"]=="sign"),
                             "max_spectral_stepsize":max(r["stepsize_spectral"] for r in traj)},
             "gram_ablation":{"count":len(gram),"exact_over_gram_noise_range":[min(r["exact_over_gram_noise"] for r in gram),max(r["exact_over_gram_noise"] for r in gram)],
                              "drift_relative_error_range":[min(r["drift_relative_error"] for r in gram),max(r["drift_relative_error"] for r in gram)]},
             "frontier_exponents":"not tested / inconclusive","prereg_sha256":manifest["prereg_sha256"]}
    write(out/"analysis.json",summary)
    lines=["# V3 全部门槛及时间干预", "", "| Gate | Decision |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k,v in gates.items()]
    lines += ["", "| a | T | predicted fresh/pooled | observed | delta SE |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['a']} | {r['T']} | {r['ratio_prediction']:.6g} | {r['ratio']:.6g} | {r['ratio_se']:.6g} |" for r in controls]
    lines += ["", "| kind | algorithm | alpha | n | key | drift error/limit | noise actual/exact | pass |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in local+path:
        passed=all(r[k] for k in ("drift_pass","noise_pass","probe_pass"))
        lines.append(f"| {r['kind']} | {r['algorithm']} | {r['alpha']} | {r['n']} | {r['key']} | {r['drift_error']/r['drift_limit']:.6g} | {r['noise_ratio']:.6g} | {passed} |")
    lines += ["", "| alpha | n | eta | algorithm | t | represented risk | SE | lower | upper | max LOC + 6SE |", "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in traj:
        lines.append(f"| {r['alpha']} | {r['n']} | {r['eta']} | {r['algorithm']} | {r['t']} | {r['represented_mean']:.6g} | {r['represented_se']:.6g} | {r['reference_lower']['represented']:.6g} | {r['reference_upper']['represented']:.6g} | {r['localization_upper_max']:.6g} |")
    (out/"TABLES.md").write_text("\n".join(lines)+"\n")
    if mode=="formal":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
        for a in (.01,.05,.2):
            group=[r for r in controls if r["a"]==a]
            axes[0].plot([r["T"] for r in group],[r["ratio_prediction"] for r in group],label=f"a={a}")
            axes[0].errorbar([r["T"] for r in group],[r["ratio"] for r in group],yerr=[6*r["ratio_se"] for r in group],fmt="o",ms=3,capsize=2)
        axes[0].set(xscale="log",yscale="log",xlabel="T (same D=T)",ylabel="fresh / pooled variance",title="Temporal-noise intervention")
        axes[0].legend(fontsize=8)
        for algo,color in (("sign","C0"),("polar","C1")):
            rows=[r for r in local+path if r["algorithm"]==algo]
            axes[1].scatter(np.arange(len(rows)),[r["drift_error"]/r["drift_limit"] for r in rows],s=12,label=algo,color=color)
        axes[1].axhline(1,color="k",ls="--",lw=1)
        axes[1].set(xlabel="All local + path states per algorithm",ylabel="drift error / preregistered limit",title="Exact kernel; no fitted gain")
        axes[1].legend(fontsize=8)
        for a in (1.5,2.,3.):
            vals=[max(r["localization_upper_max"] for r in traj if r["alpha"]==a and r["n"]==n) for n in (16,64,256)]
            axes[2].plot((16,64,256),vals,"o-",label=f"alpha={a}")
        axes[2].axhline(16,color="k",ls="--",label="candidate M=16")
        axes[2].set(xscale="log",xlabel="n",ylabel="max coordinate LOC estimate + 6 SE",title="Finite-window localization diagnostic")
        axes[2].legend(fontsize=8)
        fig.savefig(BASE/"killer_v3.png",dpi=180)
        fig.savefig(BASE/"killer_v3.pdf")
        plt.close(fig)
    am={"mode":mode,"elapsed_seconds":time.monotonic()-start,"end_utc":datetime.now(timezone.utc).isoformat(),
        "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"executable":sys.executable,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"scope":os.environ.get("KILLER_SCOPE")}
    props=subprocess.run(["systemctl","--user","show",os.environ["KILLER_SCOPE"],"-p","MemoryMax","-p","MemorySwapMax"],capture_output=True,text=True,check=True).stdout
    am["scope_properties"]=props
    write(out/"analysis_manifest.json",am)
    print(json.dumps({"gates":gates,"audit":audit,"local":ks,"path":ps,"localization":summary["localization"],"analysis_seconds":am["elapsed_seconds"]},indent=2))
    return all(v=="pass" for v in gates.values())


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["smoke","formal"])
    args=parser.parse_args()
    okay=analyze(args.mode)
    if args.mode=="smoke" and not okay:
        raise SystemExit("Smoke failed: formal run must not start")
