"""Independent review: raw Adam recursion, tangent propagation, and real matrices.
No imports from the submission. All output is exclusive-create.
"""
import os, json, hashlib, time, resource
from pathlib import Path
import numpy as np
from scipy.special import roots_legendre
from scipy.optimize import brentq
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'review17_checks'

def resources():
    c = Path('/sys/fs/cgroup') / next(x.split(':',2)[2].strip().lstrip('/') for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    r = {s:(c/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak','memory.events']}
    r['threads'] = len(list(Path('/proc/self/task').iterdir()))
    r['rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert r['memory.max']=='3221225472' and r['memory.swap.max']=='0' and r['threads']<=2
    return r

def hashes():
    # Audited sources only: unrelated experiment outputs may be changing concurrently.
    selected=['Z3_ADAM_HISTORY.md','Y2_MULTISTEP.md','ICML_REVIEW_SIM5.md',
              'paper/appendix/L_multistep.tex','paper/appendix/K_ft_rank.tex',
              'paper/sections/fine_tuning.tex']
    files=[ROOT/s for s in selected]+list((ROOT/'z3_checks').glob('*'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}

def save(name,obj):
    with (OUT/name).open('x') as f: json.dump(obj,f,indent=2)

# Differentiation follows m_t and v_t before bias correction, not endpoint (2).
def raw(z,a,e,b1,b2):
    d=np.zeros_like(z); m=d.copy(); v=d.copy()
    dd=np.zeros((3,len(z))); dm=dd.copy(); dv=dd.copy()
    one=np.zeros_like(dd); one[0]=1
    for t in (1,2):
        g=d-z; dg=dd
        nm=b1*m+(1-b1)*g
        ndm=b1*dm+(1-b1)*dg; ndm[1]+=m-g
        nv=b2*v+(1-b2)*g*g
        ndv=b2*dv+(1-b2)*2*g*dg; ndv[2]+=v-g*g
        m,v,dm,dv=nm,nv,ndm,ndv
        mh=m/(1-b1**t); dmh=dm/(1-b1**t); dmh[1]+=m*t*b1**(t-1)/(1-b1**t)**2
        vh=v/(1-b2**t); dvh=dv/(1-b2**t); dvh[2]+=v*t*b2**(t-1)/(1-b2**t)**2
        den=np.sqrt(vh)+e; dden=dvh/(2*np.sqrt(vh))
        dd=dd-one*mh/den-a*(dmh/den-mh*dden/den**2)
        d=d-a*mh/den
    return d,dd

xx,ww=roots_legendre(400); zz=6*(xx+1); weight=6*ww*np.sqrt(2/np.pi)*np.exp(-zz**2/2)

def signals(a,e,b1,b2,r):
    d,dt=raw(zz,a,e,b1,b2)
    B=np.dot(weight,zz*d); V=np.dot(weight,d*d)
    Bt=dt@(weight*zz); Vt=dt@(weight*2*d)
    s=1-r; q=1+2*s*(V-B); qt=2*s*(Vt-Bt)
    nums=np.array([1-s*B,r+s*B]); nt=np.array([-s*Bt,s*Bt])
    val=nums/np.sqrt(q); deriv=nt/np.sqrt(q)-nums[:,None]*qt/(2*q**1.5)
    return val,deriv

def match(r,u,e,b1,b2):
    # First crossing search; this is numerical evidence, not a root certificate.
    hi=1e-8
    while signals(hi,e,b1,b2,r)[0][1]<u:
        hi*=1.3
        if hi>20: raise ValueError('no first crossing')
    a=brentq(lambda x:signals(x,e,b1,b2,r)[0][1]-u,0,hi,xtol=5e-16)
    c,g=signals(a,e,b1,b2,r)
    deriv=g[0,1:]-g[0,0]*g[1,1:]/g[1,0]
    return dict(r=r,u=u,e=e,beta=[b1,b2],a=a,c0=c[0],c1=c[1],matched_derivatives=deriv.tolist())

def moments(e):
    if e>=.01:
        return [float(np.dot(weight,zz**2/(zz+e)**j)) for j in range(1,5)]
    # Log coordinate resolves z~e for the small-e asymptotics.
    return [quad(lambda t:np.sqrt(2/np.pi)*np.exp(3*t-np.exp(2*t)/2-j*np.logaddexp(t,np.log(e))),np.log(e)-40,4,epsabs=2e-12,epsrel=2e-13,limit=400)[0] for j in range(1,5)]

def discr(e):
    m1,m2,m3,m4=moments(e); D=m1*m3-m2*m2; H=D-e*(m1*m4-m2*m3)
    return D,H

def matrix_run(d,seed):
    # Actual training arrays, shared keys r_x=1, r_y=.25, n/d^2~1/8.
    r=.25; u=.5; e=.5; n=d*d//8; rng=np.random.default_rng(seed)
    X=rng.normal(size=(d,n))/np.sqrt(d)
    Y0=rng.normal(size=(d,n))/np.sqrt(d)
    Y1=rng.normal(size=(d,n))/np.sqrt(d)
    Y1*=np.sqrt(1-r*r); Y1+=r*Y0
    S=(d/n)*(X@X.T); Z0=(d/np.sqrt(n))*(Y0@X.T); Z1=(d/np.sqrt(n))*(Y1@X.T)
    ids=rng.choice(n,size=512,replace=False)
    rows=[]
    for b1 in [0.,.9]:
        b2=.99; pop=match(r,u,e,b1,b2); a=pop['a']; alpha=a*np.sqrt(2*(1-r)); eps=e*np.sqrt(2*(1-r))
        U=Z0.copy(); m=np.zeros_like(U); v=np.zeros_like(U)
        for t in [1,2]:
            g=U@S-Z1; m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
            U-=alpha*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
        q=np.mean(U*U)
        c=[float(np.mean(U*Z)/np.sqrt(q)) for Z in [Z0,Z1]]
        recall=[]; normalized_rank=[]
        # All wrong candidates are actual training values. Batch queries for memory.
        for Y in [Y0,Y1]:
            ranks=[]
            for start in range(0,len(ids),64):
                ii=ids[start:start+64]; scores=Y.T@(U@X[:,ii]); correct=scores[ii,np.arange(len(ii))]
                ranks.extend(np.sum(scores>correct[None,:],axis=0).tolist())
            ranks=np.asarray(ranks)
            recall.append(float(np.mean(ranks<np.ceil(.1*n))))
            normalized_rank.append(float(np.mean(ranks/(n-1))))
        rows.append(dict(beta=[b1,b2],population=pop,empirical_c=c,recall_top_fraction_01=recall,mean_normalized_rank=normalized_rank,q=q))
    return dict(d=d,n=n,seed=seed,rx=1,ry=r,query_count=len(ids),rows=rows,
                delta_empirical_c0=rows[1]['empirical_c'][0]-rows[0]['empirical_c'][0],
                delta_old_recall=rows[1]['recall_top_fraction_01'][0]-rows[0]['recall_top_fraction_01'][0])

if __name__=='__main__':
    start=time.monotonic(); resources()
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else 'numeric'
    if mode=='manifest': save('input_hashes.json',hashes())
    elif mode=='numeric':
        rows=[]
        for r,delta in [(-.5,.01),(.25,.01),(.5,.01),(.25,.25)]:
            for e in [.5,1.05,1.1,2]:
                for b1,b2 in [(0,0),(0,.99),(.9,0),(.9,.99),(.999,.999)]:
                    rows.append(match(r,r+delta,e,b1,b2))
        scaled=[]
        for r,e in [(-.5,.5),(.25,.5),(.5,2),(.25,1.05),(.25,1.1)]:
            D,H=discr(e); M1=moments(e)[0]; K=(1-r)/(2*(1-r*r)**3*M1**4)
            pred=[-K*D/(1+.4)**2,K*H/(1+.7)**2]
            for delta in [.02,.01,.005,.0025]:
                x=match(r,r+delta,e,.4,.7)
                scaled.append(dict(r=r,e=e,delta=delta,predicted_coefficients=pred,ratios=(np.array(x['matched_derivatives'])/delta**3/pred).tolist()))
        small=[]
        for e in [1e-4,1e-8,1e-12,1e-13]:
            D,H=discr(e); small.append(dict(e=e,D=D,H=H,ratio=H/D,default_gain_coefficient=H*.99/1.99-D*.9/1.9))
        threshold=brentq(lambda e:discr(e)[1],1.05,1.1,xtol=1e-13)
        obj=dict(rows=rows,scaled_derivatives=scaled,small_e=small,threshold=threshold,resources=resources(),seconds=time.monotonic()-start)
        save('numerical.json',obj); print(json.dumps(dict(threshold=threshold,small_e=small,scaled=scaled[-4:],resources=obj['resources'],seconds=obj['seconds']),indent=2))
    elif mode=='matrix':
        ans=[]
        for d,seed in [(512,17001),(768,17002)]:
            x=matrix_run(d,seed); ans.append(x); print(json.dumps(x),flush=True)
        save('matrix.json',dict(results=ans,resources=resources(),seconds=time.monotonic()-start))
    elif mode=='verify_manifest':
        before=json.loads((OUT/'input_hashes.json').read_text()); after=hashes()
        changed=[p for p,h in before.items() if after.get(p)!=h]
        save('input_integrity.json',dict(preexisting_count=len(before),changed=changed,new_outside_allowed=[p for p in after if p not in before],resources=resources()))
        assert not changed
