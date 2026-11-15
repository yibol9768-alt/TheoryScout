"""读取本轮输出，并独立核对散布表示与原训练分数的代数一致性。"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import json
from pathlib import Path
import time
import resource
import numpy as np

resource.setrlimit(resource.RLIMIT_CPU, (540, 550))
start=time.process_time()
here=Path(__file__).resolve().parent
data={m:json.loads((here/f"AC0_{m}_result.json").read_text()) for m in "abc"}
summary={"resources":{m:{key:data[m][key] for key in ["cpu_seconds","wall_seconds","max_rss_KiB"]} for m in data}}
summary["a_relative_error_mean"]=[{"d":d,"T":t,"mean":float(np.mean([r["relative_squared_error"] for r in data["a"]["actual_S_trajectory"] if r["d"]==d and r["T"]==t]))} for d in [32,64,128] for t in [2,3,32,33,128,129]]
summary["a_parity_identity"]=data["a"]["parity_identity"]
summary["a_adam_envelope"]=data["a"]["adam_existing_constant_upper_envelope"]
summary["b_population_moments"]=data["b"]["population_moments"]
summary["b_center_cells"]=[r for r in data["b"]["exact_scatter_rank_smoke"] if abs(r["shift_actual"])<.01]
summary["b_extremes"]=data["b"]["sphere_extreme_smoke"]
summary["c_monotonicity"]=[{"a_over_e":ratio,"cells":len(rows),**{key:sum(r[key] for r in rows) for key in ["F_increasing","K_e_increasing","K_beta1_decreasing","K_beta2_has_both_slope_signs"]}} for ratio in [.01,.05,.125,.25,.5,1.] for rows in [[r for r in data["c"]["structural_grid"] if r["a_over_e"]==ratio]]]
summary["c_reversals"]=data["c"]["known_reversal_recomputed"]
rng=np.random.default_rng(20261106084)
d,n=12,240
P=2*d
b=np.tril(rng.standard_normal((P,P)),-1)
np.fill_diagonal(b,np.sqrt(rng.chisquare(n-np.arange(P))))
F=b/np.sqrt(n)
right=np.linalg.qr(rng.standard_normal((n,P)),mode="reduced")[0]
XY=np.sqrt(n/d)*F@right.T
X,Y=XY[:d],XY[d:]
S=d/n*X@X.T; Z=d/np.sqrt(n)*Y@X.T
U=np.zeros((d,d))
for _ in range(2): U += .6*np.sign(Z-U@S)
Un=U/np.linalg.norm(U)
errs=[]
for i in range(12):
    scores=d*Y.T@Un@X[:,i]
    Vreal=(np.sum(scores*scores)-scores[i]**2)/(n-1)
    u=right[i]; x=F[:d]@u; y=F[d:]@u; ux=Un@x
    D=n*y@ux
    V=n*n/(n-1)*(ux@(F[d:]@F[d:].T)@ux-(D/n)**2)
    errs.append(max(abs(scores[i]-D),abs(Vreal-V)))
summary["b_direct_training_identity_max_absolute_error"]=max(errs)
summary["summary_and_identity_cpu_seconds"]=time.process_time()-start
with (here/"AC0_summary.json").open("x") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
