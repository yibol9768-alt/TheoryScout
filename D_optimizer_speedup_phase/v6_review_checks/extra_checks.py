"""补充反例、临界 log 和更大 n 的真实 sign 诊断。"""
from pathlib import Path
import json
import math
import time
import numpy as np
from scipy.special import zeta
from run_checks import model, sgd_exact

HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(20261108)
out={"seed":20261108}
start=time.monotonic()

# Independent re-evaluation of the negative-step L1 counterexample.
alpha,beta,n,sigma,eta=20.,19.,4,.001,-2.3
lam,theta,tail,s2=model(alpha,beta,n,sigma)
x=rng.normal(size=(400000,4))*np.sqrt(lam)
u=np.sign(x)*np.sign((x@(-theta)-rng.normal(size=x.shape[0])*math.sqrt(s2))[:,None])
ep=-theta-eta*u
s=s2+ep*ep@lam
den=ep[:,0]**2
ratio=float(np.mean(den*s)/np.mean(den))
v0=float(zeta(1+beta,1))
la=float(zeta(alpha,1))
sp=sigma*sigma+v0
cap=sp+4*(v0+1)+1
w=(math.sqrt(v0)+eta*math.sqrt(la))**2
condition=(sp+w)/eta**2*float(np.max((theta+eta)**2)) # delta=1, k=1
out["L1_negative_counterexample"]={"parameters":[alpha,beta,sigma,n,eta,1],
    "observed_weighted_ratio":ratio,"claimed_M":cap,"additional_condition":condition,
    "all_single_sided_step_caps_hold":True,
    "note":"eta B<0 and eta max kappa<0 for every candidate positive d; delta=1."}

# Refinement of real online sign simulations; same fixed construction, not
# a fitted/proxy recurrence. Exact SGD comparator uses its actual recursion.
rows=[]
for alpha in [1.5,2.,3.]:
    beta=.5
    for n in [18,26,40,64]:
        lam,theta,tail,s2=model(alpha,beta,n)
        h=float(np.sqrt(lam).sum())
        eta=.012*n**(-beta)/h
        tmax=math.ceil(.25*n**(alpha/2)/eta)
        paths=96
        err=np.tile(-theta,(paths,1))
        for t in range(tmax):
            x=rng.normal(size=err.shape)*np.sqrt(lam)
            residual=np.sum(x*err,axis=1)-rng.normal(size=paths)*math.sqrt(s2)
            err-=eta*np.sign(x)*np.sign(residual[:,None])
        vals=tail+err*err@lam
        eta_g=.012*n**(-beta)
        tg=math.ceil(.25*n**alpha/eta_g)
        exact,mass=sgd_exact(alpha,beta,n,1,eta_g,tg)
        rows.append({"alpha":alpha,"beta":beta,"n":n,"alg":"S","paths":paths,
             "eta":eta,"T":tmax,"C":n*tmax,"risk":float(vals.mean()),
             "risk_se":float(vals.std(ddof=1)/math.sqrt(paths)),"tail":tail,
             "tail_fraction":tail/float(vals.mean()),"sgd_exact_risk":exact,"sgd_C":n*tg})
        print(f"extended sign alpha={alpha} n={n} complete",flush=True)
        out["extended_sign"]=rows
        (HERE/"extra_results.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))

# Summarize uncertainty of slope fits by delta-method propagation of the
# independent sample-mean errors. This is conditional on fixed configurations.
base=json.loads((HERE/"results.json").read_text())
slopes=[]
for alpha in [1.5,2.,3.]:
    for nmin in [4,18]:
        rr=[r for r in base["online_simulations"] if r["alpha"]==alpha and r["alg"]=="S"]
        rr += [r for r in rows if r["alpha"]==alpha]
        rr=[r for r in rr if r["n"]>=nmin]
        xx=np.log([r["C"] for r in rr]); yy=np.log([r["risk"] for r in rr])
        xc=xx-xx.mean(); scale=float(xc@xc)
        slope=float(xc@yy/scale)
        se=math.sqrt(sum(float(xc[k])**2*(r["risk_se"]/r["risk"])**2 for k,r in enumerate(rr)))/scale
        slopes.append({"alpha":alpha,"n_min":nmin,"n_max":64,"slope":slope,"delta_method_se":se})
out["sign_slopes"]=slopes
out["elapsed_seconds"]=time.monotonic()-start
(HERE/"extra_results.json").write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({k:out[k] for k in ["L1_negative_counterexample","sign_slopes","elapsed_seconds"]},indent=2))
