"""Frozen, CPU-only V3 experiment. No local-project imports or network calls."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/killer_v3"
MASTER = 2026110603
ALGOS = ("sign", "polar")


def utc():
    return datetime.now(timezone.utc).isoformat()


def clean(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def dump(path, x):
    path.write_text(json.dumps(x, default=clean, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng(phase, *indices):
    return np.random.default_rng(np.random.SeedSequence([MASTER, phase, *indices]))


def model(alpha, n):
    i = np.arange(1, n + 1, dtype=float)
    lam = i ** (-alpha)
    teacher = i ** ((alpha - 2) / 2)
    tail = float(polygamma(1, n + 1))
    return lam, teacher, tail, math.sqrt(1 + tail)


def polar(lam, e, sig, eps=2e-10):
    n = len(lam)

    def integrand(s):
        t = s * s
        lt = lam / (1 + 2 * t * lam)
        z = np.exp(-.5 * np.log1p(2 * t * lam).sum())
        base = z * lt
        drift = 2 * math.sqrt(2) / math.pi * base * e / math.sqrt(sig * sig + np.dot(lt, e * e))
        k = 2 * s * base
        slope = 2 * math.sqrt(2) / (math.pi * sig) * base
        return np.concatenate((drift, k, slope))

    value, err = quad_vec(integrand, 0, np.inf, epsabs=eps, epsrel=eps, limit=400)
    if not np.isfinite(value).all():
        raise ArithmeticError("nonfinite quadrature")
    return value[:n], value[n:2*n], value[2*n:], float(err)


def sign_kernel(lam, e, sig):
    d = 2 / math.pi * np.arcsin(np.sqrt(lam) * e / math.sqrt(sig * sig + np.dot(lam, e * e)))
    return d, np.ones(len(lam)), 2 / (math.pi * sig) * np.sqrt(lam)


def probes(n, key):
    p = np.zeros((8, n))
    for j, idx in enumerate((0, 1, n // 2, n - 1)):
        p[j, idx] = 1
    p[4:] = rng(9, key).choice([-1., 1.], size=(4, n)) / math.sqrt(n)
    return p


def reference(lam, teacher, eta, t, k, slope, c):
    rate = 2 * c * eta * slope
    if np.any(rate > 1) or np.any(rate <= 0):
        raise ArithmeticError("reference outside stable stepsize domain")
    logq = np.log1p(-rate)
    q = np.exp(t * logq)
    bias = float(np.dot(lam * teacher**2, q))
    noise = float(np.dot(lam, eta**2 * k * (-np.expm1(t * logq)) / rate))
    return {"bias": bias, "noise": noise, "represented": bias + noise}


class Experiment:
    def __init__(self, args):
        self.args = args
        self.start = time.monotonic()
        self.out = BASE / args.mode
        self.out.mkdir(exist_ok=False)
        (self.out / "arrays").mkdir()
        self.records = []
        self.counter = {"local": 0, "path": 0, "trajectory_algorithms": 0, "controls": 0}
        self.cost = {"training_samples": 0, "diagnostic_samples": 0, "scalar_observations": 0,
                     "training_nD_per_algorithm": 0, "diagnostic_nD": 0}
        self.p0 = {"one_step_max_abs": 0., "quadrature_max_abs": 0., "sum_k_max_abs": 0.,
                   "one_dimensional_max_abs": 0.}
        self.coeff = {}
        self.path_states = []
        self.times = {}
        self.manifest = {"mode": args.mode, "start_utc": utc(), "status": "running", "executable": sys.executable,
                         "python_version": sys.version, "master_seed": MASTER, "pid": os.getpid(),
                         "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                         "prereg_sha256": sha(ROOT / "PREREG_KILLER_V3.md"), "source_sha256": sha(Path(__file__)),
                         "frozen_manifest_sha256": sha(BASE / "FROZEN.sha256"), "max_seconds": args.max_seconds,
                         "device": "CPU", "project_imports": [], "network_calls": 0}
        scope = os.environ.get("KILLER_SCOPE", "")
        props = subprocess.run(["systemctl", "--user", "show", scope, "-p", "MemoryMax", "-p", "MemorySwapMax", "-p", "ControlGroup", "-p", "ActiveState"],
                               text=True, capture_output=True, check=True).stdout
        (self.out / "scope_properties.txt").write_text(props)
        self.manifest["scope"] = scope
        self.manifest["scope_properties"] = props
        if "MemoryMax=2147483648" not in props or "MemorySwapMax=0" not in props:
            raise RuntimeError("Required memory limits not verified")
        if any(self.manifest["threads"][k] != "1" for k in self.manifest["threads"]):
            raise RuntimeError("Thread environment must be one")
        subprocess.run(["sha256sum", "-c", str(BASE / "FROZEN.sha256")], cwd=ROOT, check=True,
                       stdout=(self.out / "freeze_check.txt").open("w"))
        dump(self.out / "manifest.json", self.manifest)

    def deadline(self):
        if time.monotonic() - self.start > self.args.max_seconds:
            raise TimeoutError("preregistered internal computation deadline")

    def add(self, record):
        self.records.append(record)
        with (self.out / "events.jsonl").open("a") as f:
            f.write(json.dumps(record, default=clean, allow_nan=False) + "\n")

    def coefficients(self, alpha, n):
        key = (alpha, n)
        if key not in self.coeff:
            lam, teacher, tail, sig = model(alpha, n)
            lo = polar(lam, -teacher, sig)
            hi = polar(lam, -teacher, sig, 2e-12)
            self.p0["quadrature_max_abs"] = max(self.p0["quadrature_max_abs"],
                                              max(float(np.max(np.abs(a-b))) for a, b in zip(lo[:3], hi[:3])))
            self.p0["sum_k_max_abs"] = max(self.p0["sum_k_max_abs"], abs(lo[1].sum() - 1))
            self.coeff[key] = {"sign": (np.ones(n), sign_kernel(lam, -teacher, sig)[2]), "polar": (lo[1], lo[2])}
        return self.coeff[key]

    def measure(self, alpha, n, e, algo, kind, key, details):
        self.deadline()
        lam, teacher, tail, sig = model(alpha, n)
        if algo == "sign":
            d, k, slope = sign_kernel(lam, e, sig)
            quadrature_error = 0.
        else:
            d, k, slope, quadrature_error = polar(lam, e, sig)
            hi = polar(lam, e, sig, 2e-12)
            self.p0["quadrature_max_abs"] = max(self.p0["quadrature_max_abs"], max(float(np.max(np.abs(a-b))) for a, b in zip((d,k,slope), hi[:3])))
            self.p0["sum_k_max_abs"] = max(self.p0["sum_k_max_abs"], abs(k.sum()-1))
        count = 1024 if self.args.mode == "smoke" else 16384
        gen = rng(1 if kind == "local" else 3, key, 0 if kind == "local" else ALGOS.index(algo))
        probe = probes(n, key)
        su = np.zeros(n)
        su2 = np.zeros(n)
        sp = np.zeros(8)
        sp2 = np.zeros(8)
        sn = sn2 = 0.
        block_means = []
        v = float(np.dot(lam, e*e))
        for begin in range(0, count, 512):
            self.deadline()
            m = min(512, count-begin)
            x = gen.standard_normal((m,n)) * np.sqrt(lam)
            residual = x @ e - gen.standard_normal(m) * sig
            if algo == "sign":
                u = np.sign(x) * np.sign(residual)[:,None]
            else:
                u = x / np.linalg.norm(x, axis=1)[:,None] * np.sign(residual)[:,None]
            su += u.sum(axis=0)
            su2 += (u*u).sum(axis=0)
            z = (u @ probe.T)**2
            sp += z.sum(axis=0)
            sp2 += (z*z).sum(axis=0)
            noise_score = ((u-d)**2) @ lam
            sn += noise_score.sum()
            sn2 += np.dot(noise_score, noise_score)
            block_means.append(u.mean(axis=0))
            empirical_next = ((e-.016*u)**2) @ lam
            identity = v - 2*.016*(u @ (lam*e)) + .016**2*((u*u) @ lam)
            self.p0["one_step_max_abs"] = max(self.p0["one_step_max_abs"], float(np.max(np.abs(empirical_next-identity))))
        mean = su/count
        var = np.maximum((su2-count*mean**2)/(count-1), 0)
        drift_error = math.sqrt(float(np.dot(lam,(mean-d)**2)))
        drift_se = math.sqrt(float(np.dot(lam,var))/count)
        dnorm = math.sqrt(float(np.dot(lam,d*d)))
        exact_noise = float(np.dot(lam,k-d*d))
        noise_mean = sn/count
        noise_se = math.sqrt(max((sn2-count*noise_mean**2)/(count-1),0)/count)
        pmean = sp/count
        pse = np.sqrt(np.maximum((sp2-count*pmean**2)/(count-1),0)/count)
        ppred = (probe**2) @ k
        p_error = np.abs(pmean-ppred)
        p_limit = .03*ppred+6*pse
        record = {"kind": kind, "algorithm": algo, "alpha": alpha, "n": n, "N": count, "key": key,
                  "details": details, "state": e, "mean_u": mean, "variance_u": var, "second_u": su2/count,
                  "exact_d": d, "exact_k": k, "slope": slope, "block_mean_u": block_means,
                  "quadrature_error_estimate": quadrature_error, "drift_error": drift_error, "drift_se": drift_se,
                  "drift_norm": dnorm, "drift_limit": .03*dnorm+6*drift_se,
                  "noise_prediction": exact_noise, "noise_estimate": noise_mean, "noise_se": noise_se,
                  "noise_ratio": noise_mean/exact_noise, "noise_limit": .08*exact_noise+6*noise_se,
                  "probe_prediction": ppred, "probe_estimate": pmean, "probe_se": pse,
                  "drift_pass": drift_error <= .03*dnorm+6*drift_se,
                  "noise_pass": exact_noise>1e-12 and abs(noise_mean-exact_noise)<=.08*exact_noise+6*noise_se,
                  "probe_pass": bool(np.all(p_error<=p_limit))}
        if algo == "polar":
            a = lam*e
            mg = 2*np.dot(a,a)+lam.sum()*(sig*sig+v)
            dhat = a/math.sqrt(mg)
            what = (2*np.dot(lam**3,e*e)+np.dot(lam,lam)*(sig*sig+v))/mg
            gram_noise = what-float(np.dot(lam,dhat*dhat))
            record["gram_ablation"] = {"drift_relative_error": math.sqrt(float(np.dot(lam,(dhat-d)**2)))/max(dnorm,1e-30),
                                       "predicted_noise": gram_noise, "exact_over_gram_noise": exact_noise/gram_noise}
        self.add(record)
        self.counter[kind] += 1
        self.cost["diagnostic_samples"] += count
        self.cost["diagnostic_nD"] += n*count

    def local(self):
        if self.args.mode == "smoke":
            configs = [(1.5,16,.25)]
        else:
            configs = [(a,n,s) for a in (1.5,2.,3.) for n in (16,64,256) for s in (1.,.25,.0625)]
        for key, (alpha,n,scale) in enumerate(configs):
            lam, teacher, tail, sig = model(alpha,n)
            self.coefficients(alpha,n)
            for algo in ALGOS:
                self.measure(alpha,n,-scale*teacher,algo,"local",key,{"amplitude":scale})
        one = polar(np.ones(1),np.array([.75]),1.)
        truth = 2/math.pi*math.atan(.75)
        self.p0["one_dimensional_max_abs"] = max(abs(one[0][0]-truth),abs(one[1][0]-1))

    def trajectories(self):
        smoke = self.args.mode == "smoke"
        configs = [(1.5,16,.016)] if smoke else [(a,n,eta) for a in (1.5,2.,3.) for n in (16,64,256) for eta in (.004,.016)]
        repeats = 16 if smoke else 256
        tmax = 32 if smoke else 2048
        checkpoints = [0,16,32] if smoke else [0,64,256,1024,2048]
        for key,(alpha,n,eta) in enumerate(configs):
            self.deadline()
            lam, teacher, tail, sig = model(alpha,n)
            cf = self.coefficients(alpha,n)
            e = np.broadcast_to(-teacher,(2,repeats,n)).copy()
            gen = rng(2,key)
            saved = []
            for t in range(tmax+1):
                if t%32==0:
                    self.deadline()
                if t in checkpoints:
                    saved.append(e.copy())
                    for ai,algo in enumerate(ALGOS):
                        sq = e[ai]**2
                        v = sq@lam
                        mean_sq = sq.mean(axis=0)
                        mratio = (sq*(sig*sig+v[:,None])).mean(axis=0)/mean_sq
                        centered = sq*(sig*sig+v[:,None]-mratio)
                        mse = centered.std(axis=0,ddof=1)/math.sqrt(repeats)/mean_sq
                        k,slope = cf[algo]
                        lower = reference(lam,teacher,eta,t,k,slope,1.)
                        upper = reference(lam,teacher,eta,t,k,slope,sig/4)
                        vse = float(v.std(ddof=1)/math.sqrt(repeats))
                        vmean = float(v.mean())
                        bias = float(np.dot(lam,e[ai].mean(axis=0)**2))
                        variance = float(np.dot(lam,e[ai].var(axis=0,ddof=0)))
                        self.add({"kind":"trajectory", "algorithm":algo, "config_key":key, "alpha":alpha,"n":n,"eta":eta,"t":t,"repeats":repeats,
                                  "tail":tail,"sigma_n":sig,"represented_mean":vmean,"represented_se":vse,"total_mean":tail+vmean,
                                  "empirical_squared_mean":bias,"empirical_centered_variance":variance,"mean_square":mean_sq,
                                  "localization_ratio":mratio,"localization_se":mse,
                                  "localization_upper_max":float(np.max(mratio+6*mse)),
                                  "localization_lower_max":float(np.max(mratio-6*mse)),
                                  "reference_lower":lower,"reference_upper":upper,
                                  "risk_sandwich_pass":lower["represented"]-6*vse-1e-12<=vmean<=upper["represented"]+6*vse+1e-12,
                                  "stepsize_spectral":float(2*eta*max(slope)),"eta_H":float(eta*np.sqrt(lam).sum()),
                                  "arrays_file":f"arrays/trajectory_{key}.npz"})
                        selected = (smoke and t==tmax) or (not smoke and alpha in (1.5,3.) and n in (64,256) and eta==.016 and t in (64,256,2048))
                        if selected:
                            self.path_states.append((alpha,n,e[ai,0].copy(),algo,{"config_key":key,"replicate":0,"t":t,"eta":eta}))
                if t==tmax:
                    break
                x = gen.standard_normal((repeats,n))*np.sqrt(lam)
                eps = gen.standard_normal(repeats)*sig
                for ai,algo in enumerate(ALGOS):
                    residual = np.einsum("ij,ij->i",x,e[ai])-eps
                    if algo=="sign":
                        u = np.sign(x)*np.sign(residual)[:,None]
                    else:
                        u = x/np.linalg.norm(x,axis=1)[:,None]*np.sign(residual)[:,None]
                    e[ai] -= eta*u
            np.savez_compressed(self.out/f"arrays/trajectory_{key}.npz",states=np.array(saved),checkpoints=checkpoints,
                                alpha=alpha,n=n,eta=eta,tail=tail,sigma_n=sig)
            self.counter["trajectory_algorithms"] += 2
            self.cost["training_samples"] += 2*repeats*tmax
            self.cost["training_nD_per_algorithm"] += 2*n*repeats*tmax
            print(f"trajectory {key+1}/{len(configs)} alpha={alpha} n={n} eta={eta} complete",flush=True)

    def path(self):
        for key,(alpha,n,e,algo,details) in enumerate(self.path_states):
            self.measure(alpha,n,e,algo,"path",key,details)

    def controls(self):
        smoke = self.args.mode == "smoke"
        configs = [(.05,t) for t in (32,128)] if smoke else [(a,t) for a in (.01,.05,.2) for t in (32,128,512,2048)]
        count = 512 if smoke else 8192
        for key,(a,tmax) in enumerate(configs):
            self.deadline()
            gen = rng(4,key)
            fresh = np.zeros(count)
            total = np.zeros(count)
            for t in range(tmax):
                eps = gen.standard_normal(count)
                fresh = (1-a)*fresh+a*eps
                total += eps
            pooled = (-math.expm1(tmax*math.log1p(-a)))*total/tmax
            f2,p2 = fresh**2,pooled**2
            fm,pm = f2.mean(),p2.mean()
            fs,ps = f2.std(ddof=1)/math.sqrt(count),p2.std(ddof=1)/math.sqrt(count)
            fq = a/(2-a)*(-math.expm1(2*tmax*math.log1p(-a)))
            pq = (-math.expm1(tmax*math.log1p(-a)))**2/tmax
            ratio = fm/pm
            rs = ((f2-ratio*p2)/pm).std(ddof=1)/math.sqrt(count)
            pred = fq/pq
            self.add({"kind":"control","key":key,"a":a,"T":tmax,"N":count,"fresh_estimate":fm,"pooled_estimate":pm,
                      "fresh_se":fs,"pooled_se":ps,"fresh_prediction":fq,"pooled_prediction":pq,
                      "ratio":ratio,"ratio_se":rs,"ratio_prediction":pred,
                      "fresh_pass":abs(fm-fq)<=.02*fq+6*fs,"pooled_pass":abs(pm-pq)<=.02*pq+6*ps,
                      "ratio_pass":abs(ratio-pred)<=.03*pred+6*rs,"theory_order_pass":fq>=pq-1e-14})
            np.savez_compressed(self.out/f"arrays/control_{key}.npz",fresh=fresh,pooled=pooled)
            self.counter["controls"] += 1
            self.cost["scalar_observations"] += count*tmax

    def run(self):
        try:
            for name in ("local","controls","trajectories","path"):
                t = time.monotonic()
                getattr(self,name)()
                self.times[name] = time.monotonic()-t
            self.manifest["status"] = "complete"
            dump(self.out/"data.json",{"p0":self.p0,"records":self.records})
        except BaseException as exc:
            self.manifest["status"] = "incomplete"
            self.manifest["error"] = repr(exc)
            raise
        finally:
            self.manifest.update({"end_utc":utc(),"elapsed_seconds":time.monotonic()-self.start,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                  "counts":self.counter,"cost":self.cost,"stage_seconds":self.times,"p0":self.p0})
            dump(self.out/"manifest.json",self.manifest)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["smoke","formal"])
    parser.add_argument("--max-seconds",type=float,required=True)
    args=parser.parse_args()
    Experiment(args).run()
