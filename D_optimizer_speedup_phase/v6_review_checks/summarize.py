"""仅整理独立结果；不重跑实验。"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.special import zeta

p=Path(__file__).resolve().parent
d=json.loads((p/"results.json").read_text())
e=json.loads((p/"extra_results.json").read_text())
out={"inputs_sha256":{name:hashlib.sha256((p.parent/name).read_bytes()).hexdigest()
                      for name in ["V6_INTEGRATED_THEOREM.md","V6_STATUS.md"]}}
out["maximum_absolute_mc_z"]={
    "sign_drift":max(abs(r["sign_z"]) for r in d["conditional_moments"]),
    "sgd_one_step":max(abs(r["sgd_z"]) for r in d["conditional_moments"]),
    "polar_k":max(abs(r["k_z"]) for r in d["polar_quadrature"]),
    "polar_drift":max(abs(r["drift_z"]) for r in d["polar_quadrature"]),
    "marked_sign":max(abs(r["z"]) for r in d["marked_sign_identity"]),
    "sgd_full_run":max(abs(r["mc_z"]) for r in d["online_simulations"] if r["alg"]=="G")}
out["sector_ratios"]={
    "lower_min":min(r["sector_lower_ratio"] for r in d["polar_quadrature"]),
    "upper_max":max(r["sector_upper_ratio"] for r in d["polar_quadrature"])}
out["maximum_observed_weighted_ratio"]=max(r["max_observed_weighted_ratio"] for r in d["online_simulations"] if r["alg"] in ["S","P"])
out["L3_literal_negative_step_counterexample"]={
    "alpha":2,"beta":1,"sigma":1,"n":4,"b":1,"eta":-1,
    "q_first_coordinate":5,"r_t_first_coordinate_lower_bound":"5**t",
    "conclusion":"eta<=eta_0 for every positive eta_0, but sup_t E[V_t]=infinity"}
prob_minus=(1+2/math.pi*math.asin(1/math.sqrt(.001**2+float(zeta(20,1)))))/2
out["L1_analytic_ratio_lower_bound"]=(.001**2+float(zeta(20,5))+
    (prob_minus*3.3**4+(1-prob_minus)*1.3**4)/(prob_minus*3.3**2+(1-prob_minus)*1.3**2))
out["sgd_exact_slopes"]=[]
for alpha in [1.5,2.,3.]:
    for nmin in [4,64]:
        rr=[r for r in d["envelope_grid"] if r["alpha"]==alpha and r["alg"]=="G" and r["n"]>=nmin]
        slope=float(np.polyfit(np.log([r["C"] for r in rr]),np.log([r["risk"] for r in rr]),1)[0])
        out["sgd_exact_slopes"].append({"alpha":alpha,"n_min":nmin,"n_max":256,"slope":slope})
(p/"summary.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps(out,indent=2,ensure_ascii=False))
