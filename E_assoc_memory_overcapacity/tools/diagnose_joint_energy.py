"""Frozen exploratory diagnostic of the joint training score and column energy.

Freeze first; run only with an unchanged source/design. No fitted parameters,
full-dataset recall claims, or confirmatory confidence intervals.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.special import ndtr, ndtri

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("GD3", "Sign2", "Adam2")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def train(s, z, method):
    u = np.zeros_like(z)
    if method == "GD3":
        for _ in range(3):
            u += .7*(z-u@s)
    elif method == "Sign2":
        for _ in range(2):
            u -= .6*np.sign(u@s-z)
    else:
        first, second = np.zeros_like(z), np.zeros_like(z)
        for t in (1, 2):
            residual = u@s-z
            first = .9*first+.1*residual
            second = .99*second+.01*residual**2
            u -= .3*(first/(1-.9**t))/(np.sqrt(second/(1-.99**t))+1.)
    return u


def freeze(destination):
    archive = ROOT/"ac2_checks/full/AC2_results.json"
    design = dict(version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Exploratory conditional-scatter mechanism diagnosis; not confirmatory",
                  seed=202611071827, dimensions=[256,512], methods=list(METHODS),
                  s=0, lists=[1,5], scatters=4, queries=8192, thresholds=8192,
                  threshold_block=128, permutations=16, wall_limit_seconds=480,
                  cpu_limit_seconds=600, threads=2,
                  archived_loads=[{k:r[k] for k in ("d","n","method")}
                                  for r in json.loads(archive.read_text())["rows"] if r["d"] in (256,512)],
                  source_sha256=digest(Path(__file__)), archive_sha256=digest(archive),
                  efficiency_sha256=digest(ROOT/"ac2_checks/full/AC2_design.json"),
                  inference="Descriptive scatter means and SD only; shared threshold bank induces dependence",
                  path=["theory_center", "actual_center", "correct_scale", "actual_D_unit_energy",
                        "actual_D_mean_energy", "permuted_energy", "joint_energy"],
                  missingness="Stop on failure/budget; retain partial records; never redraw or select cells")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination/"design.json", design)
    print("Frozen",destination,"before all training and threshold draws",flush=True)


def seed_rng(base, d, method, rep, purpose):
    return np.random.default_rng(np.random.SeedSequence([base,d,method,rep,purpose]))


def cdf_average(thresholds, values):
    return float(np.searchsorted(thresholds, values, side="right").mean()/len(thresholds))


def verify_geometry():
    rng = np.random.default_rng(202611071826)
    d,n=8,80
    x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
    xy=np.vstack((x,y));c=d/n*xy@xy.T;f=np.linalg.cholesky(c)
    q=np.linalg.solve(f,math.sqrt(d/n)*xy)
    errors=[]
    for method in METHODS:
        u=train(d/n*x@x.T,d/math.sqrt(n)*y@x.T,method)
        b=f[d:].T@u@f[:d]/np.linalg.norm(u)
        scores=d/np.linalg.norm(u)*(y.T@u@x)
        correct=np.diag(scores).copy()
        forward=n*np.sum(q*(b@q),axis=0)
        energy=(n*n*np.sum((b@q)**2,axis=0)-forward**2)/(n-1)
        expected=(np.sum(scores**2,axis=0)-correct**2)/(n-1)
        error=max(float(np.max(abs(forward-correct))),float(np.max(abs(energy-expected))))
        if error>1e-10:
            raise RuntimeError("Single-query exact geometry failed")
        errors.append(dict(method=method,max_error=error))
    return errors


def run(destination):
    design=json.loads((destination/"design.json").read_text())
    if digest(Path(__file__))!=design["source_sha256"]:
        raise RuntimeError("Source changed after freeze")
    if digest(ROOT/"ac2_checks/full/AC2_results.json")!=design["archive_sha256"]:
        raise RuntimeError("Archived load input changed after freeze")
    if digest(ROOT/"ac2_checks/full/AC2_design.json")!=design["efficiency_sha256"]:
        raise RuntimeError("Archived efficiencies changed after freeze")
    resource.setrlimit(resource.RLIMIT_CPU,(design["cpu_limit_seconds"],design["cpu_limit_seconds"]+5))
    write_json(destination/"STARTED.json",dict(time_utc=datetime.now(timezone.utc).isoformat(),
                                              geometry=verify_geometry()))
    start=time.monotonic()
    old_design=json.loads((ROOT/"ac2_checks/full/AC2_design.json").read_text())
    rows=[]
    with (destination/"scatters.jsonl").open("x",buffering=1) as stream:
        for cell in design["archived_loads"]:
            d,n,method=cell["d"],cell["n"],cell["method"]
            method_index=METHODS.index(method)
            m=n-1;count=design["thresholds"]
            rng=seed_rng(design["seed"],d,method_index,0,1)
            bank=np.empty((count,2));normal_bank=np.empty_like(bank)
            for offset in range(0,count,design["threshold_block"]):
                size=min(design["threshold_block"],count-offset)
                g=rng.normal(size=(size,m))
                radius=np.sqrt(np.mean(g*g,axis=1))
                selected=np.partition(g,[m-5,m-1],axis=1)[:,[m-1,m-5]]
                bank[offset:offset+size]=selected/radius[:,None]
                normal_bank[offset:offset+size]=selected
            np.savez(destination/f"thresholds_{d}_{method}.npz",sphere=bank,normal=normal_bank)
            banks=[np.sort(bank[:,j]) for j in range(2)]
            for rep in range(design["scatters"]):
                if time.monotonic()-start>design["wall_limit_seconds"]:
                    raise TimeoutError("Frozen wall budget reached; partial rows retained")
                rng=seed_rng(design["seed"],d,method_index,rep,2)
                x,y=rng.normal(size=(2,d,n))/math.sqrt(d)
                s=d/n*x@x.T;z=d/math.sqrt(n)*y@x.T
                u=train(s,z,method);unorm=float(np.linalg.norm(u))
                if unorm==0:
                    raise RuntimeError("Zero endpoint; recorded as failure without redraw")
                xy=np.vstack((x,y));c=d/n*xy@xy.T;f=np.linalg.cholesky(c)
                b=f[d:].T@u@f[:d]/unorm
                k=(b+b.T)/2;mu=float(np.trace(b))
                sigma2=2*n/(n+2)*(float(np.sum(k*k))-mu*mu/n)
                ev=(n*float(np.sum(b*b))-mu*mu-sigma2)/m
                if sigma2<=0 or ev<=0:
                    raise RuntimeError("Invalid exact conditional moments")
                query_rng=seed_rng(design["seed"],d,method_index,rep,3)
                g=query_rng.normal(size=(2*d,design["queries"]))
                w=g/np.sqrt(np.sum(g*g,axis=0)+query_rng.chisquare(n-2*d,size=design["queries"]))
                bw=b@w;correct=n*np.sum(w*bw,axis=0)
                energy=(n*n*np.sum(bw*bw,axis=0)-correct*correct)/m
                if np.any(energy<=0):
                    raise RuntimeError("Nonpositive energy; retain failure, do not discard/redraw")
                np.savez(destination/f"joint_{d}_{method}_{rep}.npz",D=correct,V=energy,
                         mu=mu,sigma2=sigma2,expected_V=ev)
                mu0=old_design["methods"][method]["c"]*d/math.sqrt(n)
                endpoints=[]
                permutation_rng=seed_rng(design["seed"],d,method_index,rep,4)
                permutations=[permutation_rng.permutation(len(correct)) for _ in range(design["permutations"])]
                for j,length in enumerate(design["lists"]):
                    thresholds=banks[j]
                    path=dict(theory_center=float(ndtr(mu0-thresholds).mean()),
                              actual_center=float(ndtr(mu-thresholds).mean()),
                              correct_scale=float(ndtr((mu-thresholds)/math.sqrt(sigma2)).mean()),
                              actual_D_unit_energy=cdf_average(thresholds,correct),
                              actual_D_mean_energy=cdf_average(thresholds,correct/math.sqrt(ev)),
                              permuted_energy=float(np.mean([cdf_average(thresholds,correct/np.sqrt(energy[p])) for p in permutations])),
                              joint_energy=cdf_average(thresholds,correct/np.sqrt(energy)))
                    endpoints.append(dict(L=length,path=path,
                                          radial_matched_difference=float(ndtr(mu0-bank[:,j]).mean()-ndtr(mu0-normal_bank[:,j]).mean())))
                row=dict(d=d,n=n,method=method,rep=rep,mu=mu,mu0=mu0,sigma2=sigma2,
                         expected_V=ev,mean_sample_V=float(energy.mean()),
                         correlation_D_V=float(np.corrcoef(correct,energy)[0,1]),
                         conditional_D_mean=float(correct.mean()),conditional_D_variance=float(correct.var(ddof=1)),
                         endpoints=endpoints)
                rows.append(row);stream.write(json.dumps(row)+"\n")
                print(f"{d} {method} scatter {rep+1}/{design['scatters']} {time.monotonic()-start:.1f}s",flush=True)
                del x,y,xy,c,f,b,k,g,w,bw,u,s,z
    summary=[]
    for cell in design["archived_loads"]:
        selected=[r for r in rows if all(r[key]==cell[key] for key in ("d","n","method"))]
        for j,length in enumerate(design["lists"]):
            summary.append(dict(**cell,L=length,scatters=len(selected),
                                path_mean={key:float(np.mean([r["endpoints"][j]["path"][key] for r in selected])) for key in design["path"]},
                                mean_V=float(np.mean([r["expected_V"] for r in selected])),
                                mean_correlation_D_V=float(np.mean([r["correlation_D_V"] for r in selected]))))
    if (digest(Path(__file__))!=design["source_sha256"]
        or digest(ROOT/"ac2_checks/full/AC2_results.json")!=design["archive_sha256"]
        or digest(ROOT/"ac2_checks/full/AC2_design.json")!=design["efficiency_sha256"]):
        raise RuntimeError("Frozen input changed during run")
    write_json(destination/"results.json",dict(scope=design["scope"],summary=summary,
               scatters=len(rows),wall_seconds=time.monotonic()-start,
               max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               source_and_archive_hashes_unchanged=True))
    write_json(destination/"FINISHED.json",dict(time_utc=datetime.now(timezone.utc).isoformat(),
               files={p.name:digest(p) for p in sorted(destination.iterdir()) if p.is_file()}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("freeze","run"))
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    (freeze if args.action=="freeze" else run)(args.out)
