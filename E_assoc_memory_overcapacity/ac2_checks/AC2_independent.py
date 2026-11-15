"""AC2: independent full-data ranks, corrected column energy, total root brackets.

No historical implementation is imported. All outputs use exclusive creation.
"""
import os
import json
import time
import resource
import hashlib
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.integrate import quad
from scipy.stats import binom, beta

PROJECT = Path(__file__).resolve().parent.parent
HERE = Path(os.environ.get("AC2_OUTPUT_DIR",str(Path(__file__).resolve().parent)))
SEED = 2026110802
METHODS = ("GD3", "Sign2", "Adam2")
DESIGN = {64: 96, 128: 64, 256: 48, 512: 32}

def save(name, obj):
    with (HERE / name).open("x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def train(S, Z, method):
    U = np.zeros_like(Z)
    if method == "GD3":
        for t in range(3):
            U = U + .7 * (Z - U @ S)
    elif method == "Sign2":
        for t in range(2):
            U = U - .6 * np.sign(U @ S - Z)
    else:
        m, v = np.zeros_like(U), np.zeros_like(U)
        for t in range(1, 3):
            r = U @ S - Z
            m = .9 * m + .1 * r
            v = .99 * v + .01 * r * r
            U = U - .3 * (m / (1 - .9**t)) / (np.sqrt(v / (1 - .99**t)) + 1.)
    return U

def efficiency(method):
    def f(g, kind):
        h = float(train(np.ones((1, 1)), np.array([[g]]), method)[0, 0])
        return 2 * np.exp(-g*g/2) / np.sqrt(2*np.pi) * (g*h if kind == 1 else h*h)
    cuts = (0., .6, np.inf) if method == "Sign2" else (0., np.inf)
    a, q = [sum(quad(lambda g: f(g, kind), x, y, epsabs=1e-12)[0]
                for x, y in zip(cuts[:-1], cuts[1:])) for kind in (1, 2)]
    return dict(a=a, q=q, c=a/np.sqrt(q))

def load(d, c):
    lo, hi = 4, 2*d*d
    def f(n):
        return c*d/np.sqrt(n) - ndtri(1-1/(n-1))
    while hi-lo > 1:
        mid = (hi+lo)//2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    n = min((lo, hi), key=lambda v: abs(f(v)))
    return n, f(n)

def ranks_all(X, Y, U, query_block=256, candidate_block=2048):
    """Every original query, exact top-1/top-5; stop only after 5 strict exceedances."""
    n = X.shape[1]
    W = U @ X
    correct = np.sum(Y*W, axis=0)
    capped = np.zeros(n, dtype=np.int64)
    for start in range(0, n, query_block):
        ids = np.arange(start, min(start+query_block, n))
        counts = np.zeros(len(ids), dtype=np.int64)
        for j in range(0, n, candidate_block):
            active = np.flatnonzero(counts < 5)
            if not len(active):
                break
            qids = ids[active]
            scores = Y[:, j:j+candidate_block].T @ W[:, qids]
            # Exclude the diagonal explicitly: two BLAS summations may round differently.
            overlap = (qids >= j) & (qids < min(j+candidate_block, n))
            scores[qids[overlap]-j, np.flatnonzero(overlap)] = -np.inf
            counts[active] += np.count_nonzero(scores > correct[qids], axis=0)
        capped[ids] = np.minimum(counts, 5)
    return capped, correct

def root_bracket(lam, hx, px, jj, threshold, n):
    """Total measurable [lower,upper]; exceptional branch is conservatively [0,1]."""
    k = n-1 + threshold*threshold
    H = k*(1-hx) - threshold*threshold
    if not (0 <= hx < 1 and threshold > 0 and H > 0):
        return 0., 1., True
    root = (-k*np.sqrt(1-hx)*lam + threshold*np.sqrt(k*lam*lam+H*(px+jj))) / H
    p = float(ndtr(-root))
    return p, p, False

def initial_hashes():
    names = ("AC1_TOPL_PROOF.md", "AC1_RESULTS.md", "AC1_NOVELTY.md",
             "PROOF_REVIEW22.md", "ICML_REVIEW_SIM8.md", "AA1_QUANT_RANK.md")
    files = [PROJECT/n for n in names]
    files += sorted(p for p in (PROJECT/"paper").rglob("*") if p.is_file())
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def audits(rng, moments):
    # Pointwise full-array identity: Au is column energy, A^T u is not.
    identities = []
    for method in METHODS:
        d, n = 10, 100
        X, Y = rng.normal(size=(2, d, n)) / np.sqrt(d)
        S, Z = d/n*(X@X.T), d/np.sqrt(n)*(Y@X.T)
        C = d/n*np.vstack((X,Y))@np.vstack((X,Y)).T
        F = np.linalg.cholesky(C)
        Q = np.linalg.solve(F, np.sqrt(d/n)*np.vstack((X,Y)))
        U = train(S,Z,method)
        B = F[d:].T@U@F[:d]/np.linalg.norm(U)
        scores = d/np.linalg.norm(U)*(Y.T@U@X)
        D = np.diag(scores).copy()
        exact = (np.sum(scores*scores,axis=0)-D*D)/(n-1)
        forward = (n*n*np.sum((B@Q)**2,axis=0)-D*D)/(n-1)
        transposed = (n*n*np.sum((B.T@Q)**2,axis=0)-D*D)/(n-1)
        K = (B+B.T)/2
        pencils = []
        for t,s in ((.3,-.2),(1.,1.),(-2.,.1)):
            pencils.append(float(np.max(np.abs(np.linalg.eigvalsh(t*K+s*B.T@B)-
                                                 np.linalg.eigvalsh(t*K+s*B@B.T)))))
        capped, _ = ranks_all(X,Y,U,17,29)
        np.fill_diagonal(scores, -np.inf)
        direct_counts = np.count_nonzero(scores>D[None,:],axis=0)
        assert np.array_equal(capped,np.minimum(direct_counts,5))
        identities.append(dict(method=method, max_Au_error=float(np.max(abs(forward-exact))),
                               max_ATu_error=float(np.max(abs(transposed-exact))),
                               pencil_errors=pencils, rank_block_match=True))
    # Corrected high-dimensional audit, with actual Au including both block components.
    energy = []
    for d in DESIGN:
        for method in METHODS:
            n, s = load(d,moments[method]["c"])
            b = ndtri(1-1/(n-1))
            for rep in range(12):
                X,Y = rng.normal(size=(2,d,n))/np.sqrt(d)
                S,Z = d/n*(X@X.T),d/np.sqrt(n)*(Y@X.T)
                U = train(S,Z,method)
                C = d/n*np.vstack((X,Y))@np.vstack((X,Y)).T
                F = np.linalg.cholesky(C)
                unorm = np.linalg.norm(U)
                B = F[d:].T@U@F[:d]/unorm
                mu = float(np.trace(B))
                g = rng.normal(size=(2*d,256))
                u = g/np.sqrt(np.sum(g*g,axis=0)+rng.chisquare(n-2*d,size=256))
                Bu, BTu = B@u, B.T@u
                D = n*np.sum(u*Bu,axis=0)
                V = (n*n*np.sum(Bu*Bu,axis=0)-D*D)/(n-1)
                Vt = (n*n*np.sum(BTu*BTu,axis=0)-D*D)/(n-1)
                assert V.min() > -1e-10
                delta = np.max(abs(np.linalg.eigvalsh(C)-1))
                R = np.linalg.norm(U,2)/unorm
                energy.append(dict(d=d,n=n,method=method,rep=rep,b_delta=float(b*delta),
                    b_R=float(b*R),energy_RMS=float(np.sqrt(np.mean((b*(np.sqrt(V)-1))**2))),
                    transpose_energy_RMS=float(np.sqrt(np.mean((b*(np.sqrt(Vt)-1))**2))),
                    centered_D_SD=float(D.std(ddof=1)),mu=mu))
    # Exceptional cases are retained, including negative thresholds and nonpositive H.
    root_cases=[]
    for args in ((2.,.1,50.,900.,3.,1000),(2.,.1,50.,900.,-1.,1000),
                 (2.,.999,50.,10.,3.,100), (0.,1.,2.,0.,1.,10)):
        lower,upper,bad = root_bracket(*args)
        assert 0<=lower<=upper<=1
        root_cases.append(dict(arguments=args, lower=lower,upper=upper,exception=bad))
    # Independent direct G experiment on valid branch.
    g=rng.normal(size=200000)
    jj=rng.chisquare(983,size=len(g))
    lam,hx,px,t,n = 2.,.1,50.,3.,1000
    q=lam+np.sqrt(1-hx)*g
    event=q>=t*np.sqrt((px+g*g+jj-q*q)/(n-1))
    k=n-1+t*t; H=k*(1-hx)-t*t
    root=(-k*np.sqrt(1-hx)*lam+t*np.sqrt(k*lam*lam+H*(px+jj)))/H
    difference=event-ndtr(-root)
    return dict(pointwise_identities=identities,corrected_energy_rows=energy,
                root_cases=root_cases,root_direct_difference=float(difference.mean()),
                root_direct_SE=float(difference.std(ddof=1)/np.sqrt(len(g))),
                negative_T5_m279=float(binom.cdf(4,279,.5)))

if __name__ == "__main__":
    wall,cpu=time.time(),time.process_time()
    before=initial_hashes()
    save("AC2_inputs_sha256.json",before)
    moments={m:efficiency(m) for m in METHODS}
    save("AC2_design.json",dict(seed=SEED,dimensions_datasets=DESIGN,methods=moments,s=0,
        lists=[1,5],queries="all n original training queries; exact capped exceedance count",
        independent_unit="complete X,Y dataset; no within-dataset resampling",
        variance_interval="descriptive bootstrap over complete datasets, B=2000; no normality assertion"))
    rng=np.random.default_rng(SEED)
    save("AC2_audit.json",audits(rng,moments))
    print("audits complete",flush=True)
    summaries=[]
    with (HERE/"AC2_datasets.jsonl").open("x",buffering=1) as out:
        for d, reps in DESIGN.items():
            for method in METHODS:
                n,seff=load(d,moments[method]["c"])
                rows=[]
                for rep in range(reps):
                    X,Y=rng.normal(size=(2,d,n))/np.sqrt(d)
                    S,Z=d/n*(X@X.T),d/np.sqrt(n)*(Y@X.T)
                    U=train(S,Z,method)
                    cap,_=ranks_all(X,Y,U)
                    vals=np.array([np.mean(cap==0),np.mean(cap<5)])
                    row=dict(d=d,n=n,method=method,rep=rep,s_eff=seff,
                             recall=vals.tolist(),counts=[int(np.sum(cap==0)),int(np.sum(cap<5))])
                    rows.append(vals)
                    out.write(json.dumps(row)+"\n")
                    if (rep+1)%8==0:
                        print(json.dumps(dict(d=d,method=method,done=rep+1,total=reps,
                                              wall_seconds=time.time()-wall)),flush=True)
                rows=np.array(rows)
                variances=rows.var(axis=0,ddof=1)
                boot=rows[rng.integers(0,reps,size=(2000,reps))].var(axis=1,ddof=1)
                p=rows.mean(axis=0)
                # Unbiased estimate of Var(dataset recall), not a conditional-query group variance.
                summaries.append(dict(d=d,n=n,method=method,datasets=reps,s_eff=seff,
                    mean=p.tolist(),SE=(rows.std(axis=0,ddof=1)/np.sqrt(reps)).tolist(),
                    dataset_variance=variances.tolist(),bootstrap_variance_95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),
                    n_times_variance=(n*variances).tolist(),
                    pair_covariance=((n*variances-p*(1-p))/(n-1)).tolist()))
                print(json.dumps(summaries[-1]),flush=True)
    cg=Path("/proc/self/cgroup").read_text().strip().split(":")[-1]
    cgp=Path("/sys/fs/cgroup")/cg.lstrip("/")
    resources=dict(wall_seconds=time.time()-wall,cpu_seconds=time.process_time()-cpu,
        max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        process_threads=next(x for x in Path("/proc/self/status").read_text().splitlines() if x.startswith("Threads:")))
    for key in ("memory.max","memory.swap.max","memory.peak"):
        try: resources[key]=(cgp/key).read_text().strip()
        except OSError: pass
    after=initial_hashes()
    save("AC2_integrity.json",dict(checked_files=len(before),
        changed=[k for k in before if before[k]!=after.get(k)],after=after))
    save("AC2_results.json",dict(rows=summaries,resources=resources))
    print(json.dumps(resources),flush=True)
