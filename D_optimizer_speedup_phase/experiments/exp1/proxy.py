"""Exact Gaussian sequence risk; no online-noise substitution."""
import math
import time

import numpy as np


CONFIGS = [
    dict(name="stat_M", alpha=1.5, nu=1., chi=0., z=0., eta=.5, k=1/3),
    dict(name="lr_decay", alpha=1.5, nu=1., chi=1., z=0., eta=.5, k=2/7),
    dict(name="nu1_b1", alpha=3., nu=1., chi=0., z=0., eta=.5, k=1/4),
    dict(name="nu1_z1", alpha=3., nu=1., chi=0., z=1., eta=.5, k=2/9),
    dict(name="nu8_z038", alpha=3., nu=8., chi=0., z=.375, eta=.5, k=1/3),
    dict(name="oversize", alpha=16., nu=8., chi=0., z=1., eta=.5, k=1/12),
    dict(name="stat_B", alpha=1.5, nu=1., chi=0., z=0., eta=.5, k=1/3, gd=True),
    dict(name="baseline_eta025", alpha=3., nu=1., chi=0., z=0., eta=.25, k=1/4, gd=True),
    dict(name="ridge_eta035", alpha=3., nu=1., chi=0., z=0., eta=.35, k=1/4),
    dict(name="nu1_free", alpha=3., nu=1., chi=0., z=0., eta=.5, k=1/4, free=True),
    dict(name="nu2_free", alpha=3., nu=2., chi=0., z=0., eta=.5, k=1/4, free=True),
]


def ceil_root_power(n, p, q):
    x = max(1, int(n ** (p / q)))
    target = n ** p
    while x ** q < target:
        x += 1
    while x > 1 and (x - 1) ** q >= target:
        x -= 1
    return x


class Proxy:
    def __init__(self):
        self.idx = np.arange(1, 65537, dtype=np.float64)
        self.w = self.idx ** -2
        self.tail = math.pi ** 2 / 6 - np.cumsum(self.w)
        il = np.arange(1, 65537, dtype=np.longdouble)
        pi = np.arccos(np.longdouble(-1))
        tl = pi ** 2 / 6 - np.cumsum(il ** -2)
        x = self.idx + 1
        em = 1/x + .5/x**2 + 1/(6*x**3) - 1/(30*x**5) + 1/(42*x**7)
        mask = self.idx >= 64
        self.tail_check = {
            "max_absolute_float64_vs_longdouble": float(np.max(np.abs(self.tail - tl))),
            "max_relative_float64_vs_longdouble": float(np.max(np.abs((self.tail - tl) / tl))),
            "max_relative_float64_vs_euler_maclaurin_n_ge_64":
                float(np.max(np.abs((self.tail[mask] - em[mask]) / em[mask]))),
            "minimum_infinite_tail": float(self.tail[-1]),
        }

    def rate(self, c, n, b):
        lam = self.idx[:n] ** -c["alpha"]
        a = c["eta"] * n ** -c["chi"] * lam
        return a if c.get("gd") else a / np.sqrt(lam + b ** -c["nu"])

    def risk(self, c, n, b, t, a=None):
        if a is None:
            a = self.rate(c, n, b)
        rem = float(t) * np.log1p(-a)
        filt = -np.expm1(rem)
        A = float(self.tail[n-1])
        B = float(np.dot(self.w[:n], np.exp(2 * rem)))
        V = float(np.dot(filt, filt)) / float(b*t)
        return A+B+V, A, B, V

    def record(self, c, C, n, b, t, a=None):
        if a is None:
            a = self.rate(c, n, b)
        R, A, B, V = self.risk(c, n, b, t, a)
        ta1, tan = float(t)*a[0], float(t)*a[-1]
        if ta1 < 1:
            h = 1.
        elif tan >= 1:
            h = float(n)
        else:
            s = c["eta"] * float(t) * n ** -c["chi"]
            lam = 1/s if c.get("gd") else (1+math.sqrt(1+4*s*s*b**-c["nu"]))/(2*s*s)
            h = lam ** (-1/c["alpha"])
        return dict(C=C, actual_C=n*b*t, n=n, b=b, T=t, risk=R, A=A, B=B, V=V,
                    h=h, Ta_n=tan, Ta_1=ta1, n_over_h=n/h,
                    size_boundary=n in (1,65536), batch_lower_boundary=b==1,
                    batch_upper_boundary=False, budget_exhaustion_fraction=n*b*t/C)


def checks(p, smoke=False):
    rng = np.random.default_rng(20261105)
    out = []
    for j, c in enumerate(CONFIGS[:1 if smoke else 6]):
        n,b,t = 24,3+j,47+3*j
        a = p.rate(c,n,b)
        filt = -np.expm1(t*np.log1p(-a))
        target = np.sqrt(p.w[:n])
        zbar = target + rng.normal(size=n)/math.sqrt(b*t)
        explicit = np.zeros(n)
        for _ in range(t):
            explicit += a*(zbar-explicit)
        error = float(np.max(np.abs(explicit-filt*zbar)))
        risk_error = abs(float(p.tail[n-1]+np.sum((explicit-target)**2)) -
                         float(p.tail[n-1]+np.sum((filt*zbar-target)**2)))
        samples=[]
        for _ in range(16):
            e = -(1-filt)*target + filt*rng.normal(size=(256,n))/math.sqrt(b*t)
            samples.extend((p.tail[n-1]+np.sum(e*e,axis=1)).tolist())
        exact = p.risk(c,n,b,t)[0]
        se = float(np.std(samples,ddof=1)/64)
        z = abs(float(np.mean(samples))-exact)/se
        out.append(dict(name=c["name"], n=n,b=b,T=t, iteration_error=error,
                        explicit_risk_error=risk_error, mc_repeats=4096,
                        exact_risk=exact,mc_mean=float(np.mean(samples)),mc_se=se,mc_z=z,
                        passed=max(error,risk_error)<1e-10 and z<4))
    return out


def slope(rows,key):
    return float(np.polyfit([math.log(r["C"]) for r in rows],
                            [math.log(r[key]) for r in rows],1)[0])


def summarize(c,rows):
    # A contiguous suffix: never skip an intervening boundary point.
    selected=[]
    for r in reversed(rows):
        if r["size_boundary"] or r["batch_upper_boundary"]:
            break
        selected.append(r)
        if len(selected)==5:
            break
    selected.reverse()
    s=dict(name=c["name"],predicted_k=c["k"],fit_count=len(selected),
           fit_C=[r["C"] for r in selected],last=rows[-1])
    if len(selected)>=4:
        s.update(observed_k=-slope(selected,"risk"),n_slope=slope(selected,"n"),
                 h_slope=slope(selected,"h"))
        s["risk_pass"]=abs(s["observed_k"]-c["k"])<.02
        if c["name"]=="oversize":
            s["joint_pass"]=(s["risk_pass"] and abs(s["n_slope"]-1/6)<.03 and
                             abs(s["h_slope"]-1/12)<.03 and
                             selected[-1]["n_over_h"]>selected[0]["n_over_h"])
    else:
        s["risk_pass"]=None
    s["last_V_over_A"]=rows[-1]["V"]/rows[-1]["A"]
    s["last_B_over_A"]=rows[-1]["B"]/rows[-1]["A"]
    s["last_stat_constant_relative_error"]=rows[-1]["risk"]*rows[-1]["C"]**(1/3)/(1.5*2**(1/3))-1
    return s


def curve(p,c,guard):
    coarse=np.unique(np.rint(np.geomspace(1,65536,192)).astype(int))
    rows=[]
    exponents=range(24,85,4) if c["name"]=="oversize" else range(12,43)
    for e in exponents:
        guard()
        C=2**e
        def candidates(n):
            if c.get("free"):
                knee=ceil_root_power(n,3,int(c["nu"]))
                bs=sorted({1,2,4,16,64,256,1024,4096,max(1,knee//2),knee,2*knee})
            else:
                bs=[ceil_root_power(n,3,8) if c["z"]==.375 else (n if c["z"]==1 else 1)]
            best=None
            for b in bs:
                tmax=C//(n*b)
                if tmax<1:
                    continue
                a=p.rate(c,n,b)
                for t in sorted({max(1,tmax//4),max(1,tmax//2),tmax}):
                    r=p.record(c,C,n,b,t,a)
                    r["batch_upper_boundary"]=bool(c.get("free") and b==bs[-1])
                    if best is None or r["risk"]<best["risk"]:
                        best=r
            return best
        trial=[r for n in coarse if (r:=candidates(int(n))) is not None]
        best=min(trial,key=lambda x:x["risk"])
        j=int(np.searchsorted(coarse,best["n"]))
        lo,hi=int(coarse[max(0,j-1)]),int(coarse[min(len(coarse)-1,j+1)])
        for n in np.unique(np.rint(np.geomspace(lo,hi,33)).astype(int)):
            r=candidates(int(n))
            if r is not None and r["risk"]<best["risk"]:
                best=r
        if rows and rows[-1]["risk"]<best["risk"]:
            best=dict(rows[-1],C=C)
            best["budget_exhaustion_fraction"]=best["actual_C"]/C
        if rows:
            dl=math.log(C/rows[-1]["C"])
            best["finite_difference"]={k:math.log(best[k]/rows[-1][k])/dl
                                        for k in ("risk","n","h","b")}
        rows.append(best)
    return rows


def equal_loss(B,M):
    # Deduplicate flat plateaus, choosing the cheapest cost at each attained loss.
    pairs={}
    for b in B:
        pairs[b["risk"]]=min(pairs.get(b["risk"],math.inf),b["actual_C"])
    br=sorted(pairs)
    bx=np.log(br)
    by=np.log([pairs[r] for r in br])
    out=[]
    for m in M:
        if br[0]<=m["risk"]<=br[-1]:
            cb=math.exp(float(np.interp(math.log(m["risk"]),bx,by)))
            out.append(dict(C=m["actual_C"],requested_C=m["C"],risk=m["risk"],
                            C_B_interpolated=cb,S=cb/m["actual_C"],
                            size_boundary=m["size_boundary"]))
    suffix=[]
    for r in reversed(out):
        if r["size_boundary"]:
            break
        suffix.append(r)
        if len(suffix)==5:
            break
    suffix.reverse()
    return dict(rows=out,fit_C=[r["C"] for r in suffix],
                observed_S_slope=slope(suffix,"S") if len(suffix)>=4 else None,
                no_extrapolation=True)


def run(ctx,smoke=False):
    start=time.monotonic()
    p=Proxy()
    out=dict(checks=checks(p,smoke),tail_check=p.tail_check,curves={},summaries=[])
    if not all(c["passed"] for c in out["checks"]):
        ctx.save("proxy",out)
        raise AssertionError("explicit/MC check failed; all results retained")
    if smoke:
        return out
    for c in CONFIGS:
        ctx.guard(start+480)
        rows=curve(p,c,lambda:ctx.guard(start+480))
        out["curves"][c["name"]]=rows
        out["summaries"].append(summarize(c,rows))
        ctx.save("proxy",out)
        ctx.event("proxy_curve",name=c["name"],summary=out["summaries"][-1])
    out["equal_loss"]={}
    for name,baseline in [("stat_M","stat_B"),("lr_decay","stat_B"),
                          ("nu1_z1","baseline_eta025"),("nu8_z038","baseline_eta025"),
                          ("ridge_eta035","baseline_eta025")]:
        out["equal_loss"][name]=equal_loss(out["curves"][baseline],out["curves"][name])
    out["eta_constant_same_budget_auxiliary"]=(out["curves"]["baseline_eta025"][-1]["risk"]/
                                               out["curves"]["ridge_eta035"][-1]["risk"])**4
    alpha,nu,chi,i,n,b=3.,1.3,.4,7.,128.,17.
    def lograte(logs):
        ii,nn,bb=np.exp(logs)
        lam=ii**-alpha
        return math.log(.5*nn**-chi*lam/math.sqrt(lam+bb**-nu))
    logs=np.log([i,n,b]); numerical=[]
    for axis in range(3):
        d=np.zeros(3);d[axis]=1e-5
        numerical.append((lograte(logs+d)-lograte(logs-d))/2e-5)
    lam,rho=i**-alpha,b**-nu
    expected=[-alpha+alpha/2*lam/(lam+rho),-chi,nu/2*rho/(lam+rho)]
    out["local_derivative"]={"observed":numerical,"expected":expected,
                             "max_error":max(abs(x-y) for x,y in zip(numerical,expected))}
    out["free_batch_analytic_lower_bound"]="For nu<=2, b H^(alpha/2) and b^(1-nu/2) H^alpha are nondecreasing in b>=1; inf max >= H^alpha. Together with n>=H and D>=H^(beta+1), C>=H^[1+max(beta+1,chi+alpha)]. Finite grids alone are not a lower bound."
    out["elapsed_seconds"]=time.monotonic()-start
    ctx.save("proxy",out)
    return out
