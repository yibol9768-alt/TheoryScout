"""AC0 选题冒烟；仅本项目，独占创建输出，不调用外部服务。"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ac0_checks"


def emit(name, obj):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def existing():
    result = {}
    # Protect all root documents, paper sources, and the cited AA1/AA2/Z2B checks.
    # Do not scan 157k existing experiment NPZ files or the TeX font caches.
    paths = list(ROOT.glob("*.md"))
    paths += [p for p in (ROOT / "paper").glob("**/*")
              if ".cache" not in p.parts and "build" not in p.parts
              and p.suffix in {".tex", ".bib", ".py", ".md"}]
    for dirname in ["aa1_checks", "aa2_checks", "z2b_checks"]:
        paths += list((ROOT / dirname).glob("*.py"))
        paths += list((ROOT / dirname).glob("*.json"))
    for p in sorted(set(paths)):
        if p.is_file() and not p.name.startswith("AC0_"):
            digest = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    digest.update(block)
            result[str(p.relative_to(ROOT))] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["baseline", "a", "b", "c", "audit"])
    mode = parser.parse_args().mode
    resource.setrlimit(resource.RLIMIT_CPU, (540, 550))
    start_wall, start_cpu = time.monotonic(), time.process_time()
    if mode == "baseline":
        emit("AC0_existing_before.json", existing())
        print("baseline saved", flush=True)
        return
    if mode == "audit":
        before = json.loads((OUT / "AC0_existing_before.json").read_text())
        after = existing()
        result = {"existing_files": len(before), "modified": [p for p in before if p in after and before[p] != after[p]],
                  "deleted": sorted(before.keys() - after.keys()), "unexpected_new": sorted(after.keys() - before.keys())}
        emit("AC0_integrity.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import ndtr, ndtri
    rng = np.random.default_rng({"a": 20261106081, "b": 20261106082, "c": 20261106083}[mode])
    result = {"mode": mode, "seed": {"a": 20261106081, "b": 20261106082, "c": 20261106083}[mode],
              "numpy": np.__version__, "thread_limits": {key:os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}, "evidence": "float64 smoke, not a proof or interval certificate"}

    def endpoint(z, a, e, b1=.9, b2=.99):
        t = a / (z + e)
        ll = np.sqrt((b2 + (1-t)**2) / (1+b2))
        return a*z*(1/(z+e)+(1-t/(1+b1))/(z*ll+e))

    def derivative(z, a, e, b1, b2, index):
        args = [a, e, b1, b2]
        args[index] += 1e-24j
        return np.imag(endpoint(z, *args)) / 1e-24

    if mode == "a":
        alpha = .6
        z = np.linspace(-8, 8, 200001)
        h = np.zeros_like(z)
        parity_error = []
        for t in range(1, 130):
            h += alpha*np.sign(z-h)
            if t in [16, 17, 32, 33, 64, 65, 128, 129]:
                offset = t % 2
                target = alpha*(2*np.floor((z/alpha-offset)/2+.5)+offset)
                mask = (np.abs(z) < (t-1)*alpha) & (np.abs(np.mod(z/alpha-offset+1, 2)) > 1e-8)
                parity_error.append({"T": t, "max_error_away_from_lattice": float(np.max(np.abs(h[mask]-target[mask])))})
        result["parity_identity"] = parity_error
        rows = []
        for d in [32, 64, 128]:
            for seed_rep in range(3):
                n = d*d
                xx = rng.standard_normal((d,n))
                S = xx @ xx.T/n
                vals, vecs = np.linalg.eigh(S)
                Z = rng.standard_normal((d,d)) @ ((vecs*np.sqrt(vals)) @ vecs.T)
                U, H = np.zeros_like(Z), np.zeros_like(Z)
                for t in range(1,130):
                    U += alpha*np.sign(Z-U@S)
                    H += alpha*np.sign(Z-H)
                    if t in [2, 3, 8, 9, 32, 33, 128, 129]:
                        rows.append({"d": d, "rep": seed_rep, "T": t,
                                     "relative_squared_error": float(np.mean((U-H)**2)/np.mean(H**2)),
                                     "q_proxy": float(np.mean(H**2))})
        result["actual_S_trajectory"] = rows
        bounds = []
        logA = -math.inf
        for t in range(1,65):
            # AA1 (5) grows at least as a product of (1+2 alpha L_t), using its last history term.
            # Record L_t and a stable upper bound for log A_t using sqrt(t)*max history.
            a, e, b2 = .3, 1., .99
            lam, c0 = 1+2*a/e, math.sqrt(1-b2)
            Hprev = e*(lam**(t-1)-1)/(2*c0)
            Lt = (1+lam**(t-1)/c0)/e
            if t == 1:
                logA = math.log(a*Lt*(Hprev+2))
            else:
                logA = float(np.logaddexp(logA, math.log(a*Lt*math.sqrt(t)) +
                             np.logaddexp(math.log(2)+logA, math.log(Hprev+2))))
            if t in [2,4,8,16,32,64]:
                bounds.append({"T":t, "log10_upper_A_T":logA/math.log(10)})
        result["adam_existing_constant_upper_envelope"] = bounds
        result["desktop_bound"] = "Sign AA1 L=O_alpha(T^(13/6) d^(-1/6)); T=d^nu, 0<nu<1/13 suffices after choosing a vanishing failure budget. No claim for polynomial-time Adam."
    elif mode == "b":
        # Exact Wishart scatter + exact conditional spherical training ranks: no independent-probe replacement.
        methods = ["GD3", "Sign2", "Adam2"]
        def scalar(z, method):
            if method == "GD3": return (1-.3**3)*z
            if method == "Sign2": return 1.2*np.sign(z)*(np.abs(z)>.6)
            return np.sign(z)*endpoint(np.abs(z), .3, 1.)
        moments = {}
        for method in methods:
            opts = {"points":[.6]} if method == "Sign2" else {}
            aa = 2*quad(lambda z:z*scalar(z,method)*np.exp(-z*z/2)/np.sqrt(2*np.pi),0,12,epsabs=1e-11,**opts)[0]
            qq = 2*quad(lambda z:scalar(z,method)**2*np.exp(-z*z/2)/np.sqrt(2*np.pi),0,12,epsabs=1e-11,**opts)[0]
            moments[method] = {"a":aa,"q":qq,"c":aa/math.sqrt(qq)}
        result["population_moments"] = moments
        rows=[]
        for d in [64,128,256]:
            for method in methods:
                c = moments[method]["c"]
                for shift in [-.75,0.,.75]:
                    nn = brentq(lambda nn:c*d/np.sqrt(nn)-ndtri(1-1/(nn-1))-shift,2*d+1,10*d*d)
                    n = int(round(nn)); p=2*d; Q=64; B=8
                    recalls={1:[],5:[]}; centers=[]; scales=[]; gaps={1:[],5:[]}
                    for rep in range(B):
                        bart = np.tril(rng.standard_normal((p,p)),-1)
                        np.fill_diagonal(bart,np.sqrt(rng.chisquare(n-np.arange(p))))
                        F=bart/np.sqrt(n); C=F@F.T
                        S=C[:d,:d]; Z=np.sqrt(n)*C[d:,:d]
                        U=np.zeros((d,d))
                        if method == "GD3":
                            for _ in range(3): U += .7*(Z-U@S)
                        elif method == "Sign2":
                            for _ in range(2): U += .6*np.sign(Z-U@S)
                        else:
                            m=np.zeros_like(U); v=np.zeros_like(U)
                            for t in range(1,3):
                                rr=U@S-Z; m=.9*m+.1*rr; v=.99*v+.01*rr*rr
                                U -= .3*(m/(1-.9**t))/(np.sqrt(v/(1-.99**t))+1.)
                        Un=U/np.linalg.norm(U)
                        centers.append(float(np.sum(Un*C[d:,:d])))
                        g=rng.standard_normal((p,Q)); rad=np.sum(g*g,axis=0)+rng.chisquare(n-p,size=Q)
                        x=F[:d]@g; y=F[d:]@g; ux=Un@x
                        D=n*np.sum(y*ux,axis=0)/rad
                        V=(n*n/(n-1))*(np.sum(ux*(C[d:,d:]@ux),axis=0)/rad-(D/n)**2)
                        assert np.min(V)>0
                        scales.extend(np.sqrt(V).tolist())
                        gg=rng.standard_normal((Q,n-1)); norms=np.sqrt(np.mean(gg*gg,axis=1))
                        part=np.partition(gg,[-5,-1],axis=1)
                        for L in [1,5]:
                            T=part[:,-L]/norms
                            recalls[L].append(float(np.mean(D>=T*np.sqrt(V))))
                            gaps[L].extend((T-ndtri(1-1/(n-1))).tolist())
                    rows.append({"d":d,"method":method,"n":n,"shift_actual":float(c*d/np.sqrt(n)-ndtri(1-1/(n-1))),
                                 "limit_prediction":float(ndtr(shift)),"outer_scatter_count":B,"queries_per_scatter":Q,
                                 "recall":{str(L):{"mean":float(np.mean(recalls[L])),"outer_SE":float(np.std(recalls[L],ddof=1)/np.sqrt(B)),
                                                    "mean_sphere_threshold_minus_b":float(np.mean(gaps[L]))} for L in recalls},
                                 "mean_conditional_center_minus_population":float(np.mean(centers)-c*d/np.sqrt(n)),
                                 "mean_wrong_scale":float(np.mean(scales))})
                    print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        result["exact_scatter_rank_smoke"] = rows
        # Pure extreme-value check at larger n; top-L thresholds approach the same b on the O(1) correct-score scale.
        extreme=[]
        for n in [1000,10000,100000]:
            vals={1:[],5:[]}
            for rep in range(128):
                g=rng.standard_normal(n-1); radius=np.sqrt(np.mean(g*g)); part=np.partition(g,[-5,-1])
                for L in vals: vals[L].append(float(part[-L]/radius-ndtri(1-1/(n-1))))
            extreme.append({"n":n,"threshold_gaps":{str(L):{"mean":float(np.mean(v)),"sd":float(np.std(v,ddof=1))} for L,v in vals.items()}})
        result["sphere_extreme_smoke"] = extreme
    else:
        # Structural covariance criterion for exact two-step endpoints.
        rows=[]
        x=np.geomspace(1e-4,1e4,1201)
        for ratio in [.01,.05,.125,.25,.5,1.]:
            for b1 in [0.,.5,.9,.99]:
                for b2 in [0.,.5,.9,.99]:
                    # ratio <= .5 has positive residuals. Larger values test failure beyond the intended region.
                    if b2==0 and ratio>=1: continue
                    a,e=ratio,1.
                    dd=endpoint(x,a,e,b1,b2)
                    da=derivative(x,a,e,b1,b2,0)
                    F=x/dd
                    kk={label:derivative(x,a,e,b1,b2,idx)/da for label,idx in [("e",1),("beta1",2),("beta2",3)]}
                    row={"a_over_e":ratio,"beta1":b1,"beta2":b2,"min_d_a":float(np.min(da)),
                         "F_increasing":bool(np.all(np.diff(F)>-1e-9)),
                         "K_e_increasing":bool(np.all(np.diff(kk["e"])>-1e-9)),
                         "K_beta1_decreasing":bool(np.all(np.diff(kk["beta1"])<1e-9)),
                         "K_beta2_has_both_slope_signs":bool(np.any(np.diff(kk["beta2"])>1e-9) and np.any(np.diff(kk["beta2"])<-1e-9))}
                    rows.append(row)
        result["structural_grid"] = rows
        # Independent scalar quadrature of the known target-dependent reversal, with first-root scanning.
        z,w=np.polynomial.legendre.leggauss(300)
        z=(z+1)*6; w=w*6*np.sqrt(2/np.pi)*np.exp(-z*z/2)
        def matched(r,e,b1,b2,delta):
            def vals(a):
                dd=endpoint(z,a,e,b1,b2); B=np.sum(w*z*dd); V=np.sum(w*dd*dd)
                q=1-2*(1-r)*B+2*(1-r)*V
                return (1-(1-r)*B)/np.sqrt(q),(r+(1-r)*B)/np.sqrt(q)
            grid=np.linspace(0,max(2.,e),1001)
            y=[vals(a)[1]-r-delta for a in grid]
            crossings=[j for j in range(1,len(grid)) if y[j]>=0 and y[j-1]<0]
            if not crossings: return None
            j=crossings[0]; root=brentq(lambda a:vals(a)[1]-r-delta,grid[j-1],grid[j],xtol=1e-14)
            da=np.array([vals(a)[1] for a in np.linspace(0,root,129)])
            return {"a":root,"c0":float(vals(root)[0]),"preceding_grid_c1_increasing":bool(np.all(np.diff(da)>0))}
        contrasts=[]
        for e,delta in [(1.05,.01),(1.10,.01),(1.10,.25),(.5,.25),(2.,.25)]:
            low=matched(.25,e,.9,0.,delta); high=matched(.25,e,.9,.99,delta)
            contrasts.append({"e":e,"delta":delta,"beta2_0":low,"beta2_099":high,"delta_c0":high["c0"]-low["c0"]})
        result["known_reversal_recomputed"] = contrasts
        result["covariance_identity_desktop"] = "B_theta V_a-B_a V_theta=2 M^2 Cov_mu(z/d, d_theta/d_a), M=E[d d_a], dmu=d d_a varrho dz/M; requires d>0,d_a>0 on z>0. This is an exact identity, not an asserted global sign."

    result["wall_seconds"]=time.monotonic()-start_wall
    result["cpu_seconds"]=time.process_time()-start_cpu
    result["max_rss_KiB"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    emit(f"AC0_{mode}_result.json",result)
    print(json.dumps({k:result[k] for k in ["mode","wall_seconds","cpu_seconds","max_rss_KiB"]}),flush=True)


if __name__ == "__main__":
    main()
