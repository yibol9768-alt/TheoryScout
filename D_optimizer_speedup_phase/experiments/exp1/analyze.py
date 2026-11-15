"""Post-run auditing, using the already frozen fit intervals and thresholds."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE))
from matrix import Task


def main():
    start=time.monotonic()
    out=HERE/"formal"
    def read(name):
        return json.loads((out/(name+".json")).read_text())
    manifest=read("manifest");p=read("proxy");m=read("matrix")
    cal=read("calibration");loc=read("local")
    assert manifest["status"]=="complete" and m["complete"]
    assert hashlib.sha256((ROOT/"PREREG_EXP1.md").read_bytes()).hexdigest()==manifest["prereg_sha256"]
    for f,digest in manifest["source_sha256"].items():
        assert hashlib.sha256((HERE/f).read_bytes()).hexdigest()==digest
    assert len(cal["records"])==96 and len(m["records"])==432
    assert len({(r["seed"],r["n"],r["alpha"],r["b"],r["algorithm"],r["protocol"]) for r in m["records"]})==432
    assert {r["seed"] for r in m["records"]}=={0,1,2}
    feasible=0;root_errors=[]
    for name,rows in p["curves"].items():
        for i,r in enumerate(rows):
            assert r["actual_C"]==r["n"]*r["b"]*r["T"]<=r["C"]
            assert abs(r["risk"]-(r["A"]+r["B"]+r["V"]))<1e-14
            assert r["A"]>0 and r["B"]>=0 and r["V"]>=0
            assert 1<=r["h"]<=r["n"]*(1+1e-12)
            if i:
                assert r["risk"]<=rows[i-1]["risk"]
            feasible+=1
            # Independent bisection of the continuous root, not its quadratic formula.
            if name=="oversize" and r["Ta_1"]>=1 and r["Ta_n"]<1:
                lo,hi=1.,float(r["n"])
                for _ in range(80):
                    mid=(lo+hi)/2
                    lam=mid**-16
                    ta=.5*float(r["T"])*lam/math.sqrt(lam+r["b"]**-8)
                    if ta>1: lo=mid
                    else: hi=mid
                root_errors.append(abs((lo+hi)/2/r["h"]-1))
    for r in m["records"]+cal["records"]:
        assert all(pt["D"]==r["b"]*pt["T"] and pt["nD"]==r["n"]*pt["D"] for pt in r["rows"])
        assert r["rows"][-1]["T"]==r["steps"]
    invariance=[]
    for r in loc["records"]:
        if r["basis"]!="oracle":continue
        paired=next(q for q in loc["records"] if q["basis"]=="rotated" and
                    all(q[k]==r[k] for k in ("alpha","snapshot","b","algorithm")))
        invariance.append(float(np.max(np.abs(np.array(r["rates"])-paired["rates"])) ))
    assert max(invariance)<1e-12
    # At n=32 the two transfer protocols must coincide under common random numbers.
    transfer=[]
    for alpha in (1.5,3.):
        for alg in ("sgd","oracle","muon0","muon09"):
            for n in (16,32,64):
                ratios=[]
                for seed in (0,1,2):
                    for b in (4,16,64):
                        pair={r["protocol"]:r for r in m["records"] if
                              (r["alpha"],r["algorithm"],r["n"],r["seed"],r["b"])==(alpha,alg,n,seed,b)}
                        ratio=pair["decay"]["rows"][-1]["risk"]/pair["fixed"]["rows"][-1]["risk"]
                        if n==32:assert ratio==1.
                        ratios.append(ratio)
                transfer.append(dict(alpha=alpha,algorithm=alg,n=n,decay_to_fixed_mean=float(np.mean(ratios)),
                                     min=float(min(ratios)),max=float(max(ratios))))
    # Noise formula is a secondary analytic diagnostic; not a new pass criterion.
    task=Task();noise=[];positive=[]
    for r in loc["records"]:
        if r["basis"]!="oracle":continue
        prob=task.problem(32,r["alpha"],"oracle")
        E=-prob["teacher"] if r["snapshot"]=="fit" else -.25*prob["teacher"]
        if r["snapshot"]=="concentrated":
            E=np.zeros_like(E)
            E[-1]=np.linalg.norm(prob["teacher"])*task.v[31]
        lam=prob["lam"]
        residual_output=E.T@(lam[:,None]*E)+prob["tail_root"]@prob["tail_root"].T+np.eye(4)/4
        if r["algorithm"] in ("sgd","oracle"):
            scale=np.ones(32) if r["algorithm"]=="sgd" else 1/np.sqrt(lam+1/r["b"])
            theoretical=float((np.sum(scale**2*lam)*np.trace(residual_output)+
                               np.sum((scale[:,None]*lam[:,None]*E)**2))/r["b"])
            noise.append(dict(alpha=r["alpha"],b=r["b"],snapshot=r["snapshot"],algorithm=r["algorithm"],
                              measured_trace=r["noise_covariance_trace"],theoretical_trace=theoretical,
                              measured_to_theory=r["noise_covariance_trace"]/theoretical))
            if r["snapshot"]!="concentrated":
                rates=np.array(r["rates"]); se=np.array(r["se"])
                expected=lam*scale
                positive.append(dict(alpha=r["alpha"],b=r["b"],snapshot=r["snapshot"],algorithm=r["algorithm"],
                    first_rate=float(rates[0]),first_expected=float(expected[0]),first_se=float(se[0]),
                    outside_95_count=int(np.sum(np.abs(rates-expected)>1.96*se)),coordinate_count=32))
    states=[]
    for alpha in (1.5,3.):
        for b in (4,16,64):
            for alg in ("sgd","oracle","muon0"):
                pair={r["snapshot"]:r for r in loc["records"] if (r["basis"],r["alpha"],r["b"],r["algorithm"])==
                      ("oracle",alpha,b,alg)}
                f,h=pair["fit"],pair["holdout"]
                vf,vh=f["rates"][0],h["rates"][0]
                sf,sh=f["se"][0],h["se"][0]
                combined=math.sqrt(sf*sf+sh*sh)
                states.append(dict(alpha=alpha,b=b,algorithm=alg,fit_rate1=vf,holdout_rate1=vh,
                                   fit_se1=sf,holdout_se1=sh,ratio=vh/vf,
                                   difference_in_combined_se=(vh-vf)/combined,
                                   fit_noise_trace=f["noise_covariance_trace"],
                                   holdout_noise_trace=h["noise_covariance_trace"],
                                   fit_orthogonal_fraction=f["orthogonal_fraction"],
                                   holdout_orthogonal_fraction=h["orthogonal_fraction"]))
    costs=[]
    for alg in ("sgd","oracle","muon0","muon09"):
        rs=[r for r in m["records"] if r["algorithm"]==alg]
        total=sum(r["elapsed_seconds"] for r in rs)
        costs.append(dict(algorithm=alg,configurations=len(rs),training_wall_sum=total,
                          sampling=sum(r["sampling_seconds"] for r in rs),
                          gradient=sum(r["gradient_seconds"] for r in rs),
                          svd=sum(r["svd_seconds"] for r in rs),svd_fraction=sum(r["svd_seconds"] for r in rs)/total))
    # Shared sample-processing budget 2^20, finite grid only, each seed kept.
    frontiers=json.loads((out/"matrix_frontiers.json").read_text())
    shared=[]
    for alpha in (1.5,3.):
        for protocol in ("fixed","decay"):
            for alg in ("sgd","oracle","muon0","muon09"):
                seedrisks=[]
                for seed in (0,1,2):
                    rows=next(r["rows"] for r in frontiers if
                              (r["seed"],r["alpha"],r["protocol"],r["algorithm"],r["cost"])==
                              (seed,alpha,protocol,alg,"nD"))
                    best=min((r for r in rows if r["nD"]<=2**20),key=lambda r:r["risk"])
                    seedrisks.append(dict(seed=seed,risk=best["risk"],n=best["n"],b=best["b"],T=best["T"],
                                          risk_over_capacity=best["risk"]/best["capacity_tail"]))
                shared.append(dict(alpha=alpha,protocol=protocol,algorithm=alg,budget=2**20,
                                   seeds=seedrisks,mean=float(np.mean([r["risk"] for r in seedrisks])),
                                   sd=float(np.std([r["risk"] for r in seedrisks],ddof=1))))
    result=dict(audit=dict(passed=True,feasible_proxy_points=feasible,
        max_oversize_h_relative_bisection_error=max(root_errors),
        max_rotation_rate_discrepancy=max(invariance),calibration_count=96,training_count=432,
        all_three_seeds_complete=True,prereg_and_executed_source_hashes_unchanged=True),
        manifest=manifest,checks=p["checks"],proxy_summaries=p["summaries"],
        equal_loss_summary={name:dict(fit_C=r["fit_C"],slope=r["observed_S_slope"],
            last=r["rows"][-1] if r["rows"] else None) for name,r in p["equal_loss"].items()},
        eta_aux=p["eta_constant_same_budget_auxiliary"],local_fits=loc["fits"],
        state_response=states,noise_theory_secondary=noise,known_drift_controls=positive,
        training_costs=costs,locked_learning_rates=cal["locked"],
        calibration_samples=cal["total_samples"],calibration_svd_seconds=cal["total_svd_seconds"],
        calibration_training_seconds=sum(r["elapsed_seconds"] for r in cal["records"]),
        formal_training_samples=m["total_samples"],formal_training_svd_seconds=m["total_svd_seconds"],
        transfer=transfer,shared_compute_frontiers=shared,
        elapsed_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (HERE/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in
          ("noise_theory_secondary","known_drift_controls","transfer","proxy_summaries","manifest")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
