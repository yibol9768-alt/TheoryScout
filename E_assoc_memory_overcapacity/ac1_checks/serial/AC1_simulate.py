"""独立实现：真实有限步训练 + 精确残差/球面阈值积分；不导入历史代码。"""
import os, json, time, resource, csv
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri, digamma
from scipy.integrate import quad
from scipy.linalg import solve_triangular

ROOT = Path(__file__).resolve().parent
SEED = 2026110701
DS = [64, 128, 256, 512]
SS = [-.75, 0., .75]
B, Q = 128, 64
METHODS = ['GD3', 'Sign2', 'Adam2']

def save(name, x):
    with (ROOT / name).open('x') as f:
        json.dump(x, f, indent=2, ensure_ascii=False)

def endpoint(S, Z, method):
    U = np.zeros_like(Z)
    if method == 'GD3':
        for _ in range(3):
            U += .7 * (Z - U @ S)
    elif method == 'Sign2':
        for _ in range(2):
            U += .6 * np.sign(Z - U @ S)
    else:
        m, v = np.zeros_like(Z), np.zeros_like(Z)
        for t in range(1, 3):
            r = U @ S - Z
            m = .9*m + .1*r
            v = .99*v + .01*r*r
            U -= .3 * (m/(1-.9**t))/(np.sqrt(v/(1-.99**t)) + 1.)
    return U

def scalar(g, method):
    return endpoint(np.eye(1), np.asarray(g).reshape(-1, 1), method).ravel()

def moments(method):
    cuts = [0, .6, np.inf] if method == 'Sign2' else [0, np.inf]
    def integ(power):
        def f(z):
            h = scalar([z], method)[0]
            return 2*np.exp(-z*z/2)/np.sqrt(2*np.pi)*(z*h if power == 1 else h*h)
        return sum(quad(f, a, b, epsabs=2e-12, epsrel=2e-12)[0] for a,b in zip(cuts[:-1], cuts[1:]))
    a, q = integ(1), integ(2)
    return {'a': a, 'q': q, 'c': a/np.sqrt(q)}

def load(d, c, s):
    # 单调方程，最终以整数 n 的实际 s_eff 为准。
    lo, hi = 4, max(8, d*d*2)
    def f(n):
        return c*d/np.sqrt(n)-ndtri(1-1/(n-1))-s
    while hi-lo > 1:
        mid = (lo+hi)//2
        if f(mid)>0: lo=mid
        else: hi=mid
    n = min([lo,hi], key=lambda k: abs(f(k)))
    return n, f(n)+s

def bartlett(rng, d, n):
    A = np.tril(rng.standard_normal((d,d)), -1)
    A[np.diag_indices(d)] = np.sqrt(rng.chisquare(n-np.arange(d)))
    return A/np.sqrt(n)

def thresholds(rng, m, q):
    # 同一向量给出 top-1/top-5 及径向归一化，绝不独立抽半径。
    g = rng.standard_normal((q,m))
    radius = np.sqrt(np.mean(g*g,axis=1))
    p = np.partition(g, [-5,-1], axis=1)
    return np.stack((p[:,-1]/radius, p[:,-5]/radius), axis=1)

def integral(lam, hx, px, J, T):
    n = J[1]  # J=(chi-square draws,n)
    j = J[0][:,None]
    k = n-1+T*T
    H = k*(1-hx[:,None])-T*T
    valid = (T>0)&(H>0)
    if not valid.all():
        raise RuntimeError('积分例外事件，必须另给上下界，不能静默删样本')
    root = (-k*np.sqrt(1-hx[:,None])*lam[:,None]
            +T*np.sqrt(k*lam[:,None]**2+H*(px[:,None]+j)))/H
    return ndtr(-root)

def one_group(rng, d, n, method, diagnostic=False):
    L = bartlett(rng,d,n)
    S = L@L.T
    Z = rng.standard_normal((d,d))@L.T
    U = endpoint(S,Z,method)
    g = rng.standard_normal((d,Q))
    norm2 = np.sum(g*g,axis=0)+rng.chisquare(n-d,Q)
    w = np.sqrt(n/d)*g/np.sqrt(norm2)
    x = L@w
    t = U@x
    normt = np.sqrt(np.sum(t*t,axis=0))
    # Z S^{-1} x = Z L^{-T} w；三角解保持实际 S。
    zsx = Z@solve_triangular(L.T,w,lower=False)
    lam = np.sum(zsx*t,axis=0)*np.sqrt(d/n)/normt
    hx = np.sum(g*g,axis=0)/norm2
    projected = solve_triangular(L,Z.T@t,lower=True)
    px = np.sum(projected*projected,axis=0)/(normt*normt)
    T = thresholds(rng,n-1,Q)
    J = rng.chisquare(n-d-1,Q)
    p = integral(lam,hx,px,(J,n),T)
    b = ndtri(1-1/(n-1))
    # 三种说明性近似，均非已证明的有限维误差界。
    mom = MOM[method]
    mu = mom['c']*d/np.sqrt(n)
    sphere_gauss = ndtr(mu-T)
    shift = (-digamma(np.array([1.,5.])))/b
    shift_pred = ndtr(mu-b-shift)
    # 保留实际训练的 lambda,hx,px，J 用条件均值，删去根式的 O(1/n) 项。
    scale = np.sqrt((px+n-d-1)/n)
    residual_smooth = ndtr((lam[:,None]-T*scale[:,None])/np.sqrt(1-hx[:,None]))
    diag = {}
    if diagnostic:
        ev, V = np.linalg.eigh(S)
        G = (Z@V/np.sqrt(ev))@V.T
        Href = scalar(G.ravel(),method).reshape(d,d)
        eps = np.linalg.norm(U-Href)/d
        diag = {'b_eps':float(b*eps), 'eps':float(eps),
                'b_R':float(b*np.linalg.norm(U,2)/np.linalg.norm(U)),
                'mu_S_Z':float(np.sum(U*Z)/(np.sqrt(n)*np.linalg.norm(U))),
                'center_error':float(np.sum(U*Z)/(np.sqrt(n)*np.linalg.norm(U))-mu)}
    return {'p':p.mean(axis=0).tolist(), 'sphere_gauss':sphere_gauss.mean(axis=0).tolist(),
            'shift_pred':shift_pred.tolist(), 'residual_smooth':residual_smooth.mean(axis=0).tolist(),
            'T_minus_b':(T-b).mean(axis=0).tolist(), 'lambda_mean':float(lam.mean()),
            'hx_mean':float(hx.mean()), 'px_over_n_mean':float(px.mean()/n), 'diagnostic':diag}

def audits(rng):
    # 完整数组直接排名与独立残差积分：各数据集作为一组。
    result = {}
    for method in METHODS:
        d, n, reps, q = 16, 150, 768, 12
        direct, smooth = [], []
        for _ in range(reps):
            X, Y = rng.standard_normal((2,d,n))/np.sqrt(d)
            S, Z = d/n*(X@X.T), d/np.sqrt(n)*(Y@X.T)
            U = endpoint(S,Z,method)
            ids = rng.choice(n,q,replace=False)
            x, t = X[:,ids], U@X[:,ids]
            scores = Y.T@t
            ranks = 1+np.sum(scores>scores[ids,np.arange(q)],axis=0)
            direct.append([np.mean(ranks<=k) for k in [1,5]])
            L = np.linalg.cholesky(S)
            w = solve_triangular(L,x,lower=True)
            zsx = Z@solve_triangular(L.T,w,lower=False)
            normt = np.sqrt(np.sum(t*t,axis=0))
            lam = np.sum(zsx*t,axis=0)*np.sqrt(d/n)/normt
            hx = d/n*np.sum(w*w,axis=0)
            proj = solve_triangular(L,Z.T@t,lower=True)
            px = np.sum(proj*proj,axis=0)/normt**2
            T = thresholds(rng,n-1,q)
            smooth.append(integral(lam,hx,px,(rng.chisquare(n-d-1,q),n),T).mean(axis=0))
        diff = np.asarray(direct)-smooth
        result[method] = {'d':d,'n':n,'datasets':reps,'queries':q,
                          'direct':np.mean(direct,axis=0).tolist(),'integral':np.mean(smooth,axis=0).tolist(),
                          'paired_difference':diff.mean(axis=0).tolist(),
                          'dataset_SE':(diff.std(axis=0,ddof=1)/np.sqrt(reps)).tolist()}
    # 固定参数根式与直接 G 事件的独立抽查。
    roots=[]
    for lam,hx,px,n,T in [(2.,.1,50.,1000,3.),(-.4,.05,30.,500,2.),(3.,.3,80.,400,2.8)]:
        q=200000
        jj=rng.chisquare(n-16-1,q)
        gg=rng.standard_normal(q)
        qg=lam+np.sqrt(1-hx)*gg
        energy=(px+gg*gg+jj-qg*qg)/(n-1)
        event=qg>=T*np.sqrt(np.maximum(energy,0))
        pred=integral(np.full(q,lam),np.full(q,hx),np.full(q,px),(jj,n),np.full((q,1),T))[:,0]
        diff=event-pred
        roots.append({'difference':float(diff.mean()),'SE':float(diff.std(ddof=1)/np.sqrt(q))})
    return {'full_array':result,'root_checks':roots}

if __name__=='__main__':
    wall, cpu = time.time(), time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU,(1700,1750))
    MOM = {method:moments(method) for method in METHODS}
    design={'seed':SEED,'d':DS,'s':SS,'B':B,'Q':Q,'moments':MOM,
            'methods':{'GD3':{'T':3,'eta':.7},'Sign2':{'T':2,'alpha':.6},
                       'Adam2':{'T':2,'alpha':.3,'e':1.,'beta1':.9,'beta2':.99}},
            'independent_unit':'dataset (S,Z); conditional queries within group',
            'root_exceptions':'raise, no sample deletion','arithmetic':'float64'}
    save('AC1_design.json',design)
    rng = np.random.default_rng(SEED)
    save('AC1_audit.json',audits(rng))
    rows=[]
    with (ROOT/'AC1_groups.jsonl').open('x',buffering=1) as f:
        for d in DS:
            for method in METHODS:
                for s in SS:
                    n,seff=load(d,MOM[method]['c'],s)
                    groups=[]
                    for j in range(B):
                        out=one_group(rng,d,n,method,j<8)
                        f.write(json.dumps({'d':d,'method':method,'s':s,'n':n,'group':j,**out})+'\n')
                        groups.append(out)
                    row={'d':d,'method':method,'s':s,'n':n,'s_eff':seff,'limit':float(ndtr(seff))}
                    for field in ['p','sphere_gauss','shift_pred','residual_smooth','T_minus_b']:
                        v=np.array([g[field] for g in groups])
                        row[field]=v.mean(axis=0).tolist()
                        row[field+'_SE']=(v.std(axis=0,ddof=1)/np.sqrt(B)).tolist()
                    row['diagnostic_first_8']=[g['diagnostic'] for g in groups[:8]]
                    rows.append(row)
                    print(json.dumps({k:row[k] for k in ['d','method','s','n','p','p_SE']}),flush=True)
    cg=Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    cgp=Path('/sys/fs/cgroup')/cg.lstrip('/')
    resources={'wall_seconds':time.time()-wall,'cpu_seconds':time.process_time()-cpu,
               'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'env':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
    for key in ['memory.max','memory.swap.max','memory.peak']:
        try: resources[key]=(cgp/key).read_text().strip()
        except OSError: pass
    save('AC1_simulation.json',{'design':design,'rows':rows,'resources':resources})
    print(json.dumps(resources),flush=True)
