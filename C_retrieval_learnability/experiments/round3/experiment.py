"""Round 3: reuse frozen round-2 objectives, projection and Adam exactly."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'round2'))
import core as r2
from core import np, os, resource, time, copy, hashlib, json
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MODES = ['nce', 'sigmoid_zero', 'sigmoid_row', 'sigmoid_lift']
save, digest = r2.save, r2.digest


def init(d, seed, mode):
    A, pos, neg, p = r2.init(16, 3, d, seed)
    if mode == 'sigmoid_row':
        p['b'] = np.zeros(len(A))
    return A, pos, neg, p


def obj(p, A, pos, neg, mode):
    return r2.objective(p, A, pos, neg, 'nce' if mode=='nce' else 'sigmoid',
                        'train', 'row' if mode=='sigmoid_row' else 'zero')


def calibrate(p, g):
    # Round-2 calibration expects every trainable group to have a rate.
    rates = {}
    for key in ('U', 'V'):
        direction = g[key]/(np.abs(g[key])+1e-8)
        def rms(rate):
            return float(np.sqrt(np.mean((r2.normalize(p[key]-rate*direction)-p[key])**2)))
        lo, hi = 0., .02
        for _ in range(60):
            if rms(hi)>=.001: break
            hi *= 2
        else: raise ValueError('cannot bracket calibration')
        for _ in range(60):
            mid=(lo+hi)/2
            if rms(mid)<.001: lo=mid
            else: hi=mid
        rates[key]=(lo+hi)/2
    rates['a']=.1*rates['U']
    if 'b' in p: rates['b']=rates['U']
    q=copy.deepcopy(p)
    r2.Adam(q,rates).update(q,g)
    moves=r2.displacement(p,q)
    assert all(abs(moves[k]['rms']/.001-1)<1e-6 for k in ('U','V'))
    return rates,moves


def metrics(p,A,arrays=False):
    m=r2.metrics(p,A,arrays)
    S=p['U']@p['V'].T
    E=S+(p['b'][:,None]/np.exp(p['a'][0]) if 'b' in p else 0.)
    signed=np.where(A,E,-E)
    z=signed.min(1)
    m.update(effective_sign_margin=float(z.min()), effective_sign_row_fraction=float((z>0).mean()),
             effective_sign_element_fraction=float((signed>0).mean()),
             robust_all_sign=bool(z.min()>1e-10), all_rank=bool(m['min_gap']>0), all_sign=bool(z.min()>0))
    if arrays: m['effective_zeta']=z.tolist()
    if 'b' in p:
        m['bias_min']=float(p['b'].min()); m['bias_max']=float(p['b'].max())
    return m


def lift(p,A):
    gamma,_,S=r2.gaps(p,A)
    assert gamma.min()>0
    theta=(np.where(A,S,np.inf).min(1)+np.where(~A,S,-np.inf).max(1))/2
    scale=np.sqrt(2*(1+theta**2))
    U=np.column_stack((p['U'],-theta))/np.sqrt(1+theta[:,None]**2)
    V=np.column_stack((p['V'],np.ones(len(p['V']))))/np.sqrt(2)
    q=dict(U=U,V=V,a=np.array([np.log(16.)]))
    L=U@V.T
    threshold_z=np.where(A,S-theta[:,None],theta[:,None]-S).min(1)
    lifted_z=np.where(A,L,-L).min(1)
    errors=dict(threshold_identity=float(np.max(abs(threshold_z-gamma/2))),
                lift_score_identity=float(np.max(abs(L-(S-theta[:,None])/scale[:,None]))),
                lift_margin_identity=float(np.max(abs(lifted_z-gamma/(2*scale)))),
                norm=float(max(np.max(abs(np.linalg.norm(U,axis=1)-1)),np.max(abs(np.linalg.norm(V,axis=1)-1)))))
    assert max(errors.values())<1e-12 and lifted_z.min()>0
    assert np.array_equal(np.argsort(S,axis=1),np.argsort(L,axis=1))
    return q,dict(errors=errors,min_gamma=float(gamma.min()),min_threshold_margin=float(threshold_z.min()),
                  min_lift_margin=float(lifted_z.min()),all_sign=True,ordering_preserved=True,
                  theta=theta.tolist())


def train(d,seed,mode,T,deadline,p_override=None):
    start=time.time()
    A,pos,neg,p=init(d,seed,mode)
    if p_override is not None: p=copy.deepcopy(p_override)
    fingerprint=hashlib.sha256(p['U'].tobytes()+p['V'].tobytes()).hexdigest()
    _,g=obj(p,A,pos,neg,mode)
    rates,moves=calibrate(p,g)
    opt=r2.Adam(p,rates)
    check=set(r2.checkpoints(T))
    initial=metrics(p,A,True)
    history=[]; last=None; status='completed'
    hit_rank=0 if initial['all_rank'] else None
    hit_sign=0 if initial['all_sign'] else None
    for t in range(T+1):
        if (not all(np.isfinite(v).all() for v in p.values()) or not np.log(1e-8)<=p['a'][0]<=np.log(1e4)):
            status='numerical_or_temperature_failure'; break
        if t%1000==0 and time.time()>deadline:
            status='budget_truncated'; break
        if t%1000==0 or t in check:
            met=metrics(p,A)
            if met['all_rank'] and hit_rank is None: hit_rank=t
            if met['all_sign'] and hit_sign is None: hit_sign=t
            if t in check:
                history.append(dict(step=t,loss=obj(p,A,pos,neg,mode)[0],last_actual_displacement=last,**met))
        if t==T: break
        val,g=obj(p,A,pos,neg,mode)
        if not np.isfinite(val) or not all(np.isfinite(v).all() for v in g.values()):
            status='nonfinite_gradient_or_loss'; break
        capture=t+1 in check
        prev=copy.deepcopy(p) if capture else None
        opt.update(p,g,.5*(1+np.cos(np.pi*t/T)))
        if capture: last=r2.displacement(prev,p)
    result=dict(config=dict(n=16,k=3,d=d,actual_d=p['U'].shape[1],seed=seed,mode=mode),
                status=status,steps_requested=T,steps_completed=t,initial_fingerprint=fingerprint,
                calibration=dict(rates=rates,actual=moves),initial=initial,final=metrics(p,A,True),
                history=history,first_all_rank_step=hit_rank,first_all_sign_step=hit_sign,
                elapsed_seconds=time.time()-start,memory=r2.memory())
    return result,p
