"""Frozen decision rules, used for the holdout gate as well as final reporting."""
import experiment as c

def completed(r): return r['status']=='completed' and r['steps']==r['requested']
def success(r): return completed(r) and r['final']['all_sign']
def robust(r): return completed(r) and r['final']['robust_sign']
def lookup(records,d,stage,variant):
    return sorted([r for r in records if r['config']['d']==d and r['config']['stage']==stage and r['config']['variant']==variant],key=lambda r:r['config']['seed'])
def count(rs,fun=success): return sum(bool(fun(r)) for r in rs)
def flips(a,b):
    assert len(a)==len(b)
    return dict(gain=sum(not success(x) and success(y) for x,y in zip(a,b)),reverse=sum(success(x) and not success(y) for x,y in zip(a,b)))
def strong(a,b):
    f=flips(a,b); return f['gain']>=3 and f['reverse']<=1

def assess(records):
    out={}; cats=[]; repeat=[]
    for d in (6,7):
        a={v:lookup(records,d,'A',v) for v in ('ZL','ZM','SL','SM')}
        if any(len(v)!=5 for v in a.values()): return dict(complete=False,gate=False)
        cnt={v:count(rs) for v,rs in a.items()}
        rep=cnt['ZL']<=1 and cnt['SM']>=4; repeat.append(rep)
        cat='other'
        if cnt['SL']>=4 and cnt['ZM']<=1 and strong(a['ZL'],a['SL']) and strong(a['ZM'],a['SM']): cat='initialization'
        elif cnt['ZM']>=4 and cnt['SL']<=1 and strong(a['ZL'],a['ZM']) and strong(a['SL'],a['SM']): cat='learning_rate'
        elif cnt['SL']>=4 and cnt['ZM']>=4 and strong(a['ZL'],a['SL']) and strong(a['ZL'],a['ZM']): cat='each_sufficient'
        elif cnt['SL']<=1 and cnt['ZM']<=1 and strong(a['SL'],a['SM']) and strong(a['ZM'],a['SM']): cat='joint_required'
        cats.append(cat)
        effects={}
        for metric in ('min_zeta','rank_fraction'):
            v={k:c.np.array([r['final'][metric] for r in rs]) for k,rs in a.items()}
            effects[metric]={name:arr.tolist() for name,arr in dict(init_legacy=v['SL']-v['ZL'],init_matched=v['SM']-v['ZM'],rate_zero=v['ZM']-v['ZL'],rate_star=v['SM']-v['SL'],interaction=v['SM']-v['SL']-v['ZM']+v['ZL']).items()}
        out[f'A_d{d}']=dict(counts=cnt,repeat=rep,category=cat,effects=effects,
            flips={f'{x}_to_{y}':flips(a[x],a[y]) for x,y in [('ZL','SL'),('ZM','SM'),('ZL','ZM'),('SL','SM')]})
    out['H_A0']=all(repeat); out['H_A']=all(repeat) and cats[0]==cats[1] and cats[0]!='other'
    b={v:lookup(records,7,'B',v) for v in ('b_original','theta_original','b_stop80','theta_stop80','b_cosine_rate','theta_cosine_rate')}
    if any(len(v)!=5 for v in b.values()): return dict(complete=False,gate=False)
    base=b['b_original']
    out['H_B0']=count(base,lambda r:r['late_loss'])>=4 and count(base)<=1 and count(b['theta_original'])>=4
    for path,hyp in [('stop80','H_B1'),('cosine_rate','H_B2')]:
        alt=b['b_'+path]; delta=c.np.array([r['final']['min_zeta']-q['final']['min_zeta'] for q,r in zip(base,alt)])
        detail=dict(final=count(alt),robust=count(alt,robust),flips=flips(base,alt),late_loss=count(alt,lambda r:r['late_loss']),margin_improved=int((delta>0).sum()),margin_delta=delta.tolist(),mean_margin_delta=float(delta.mean()),theta_final=count(b['theta_'+path]))
        out[hyp+'_detail']=detail
        out[hyp]=bool(out['H_B0'] and detail['final']>=4 and detail['robust']>=4 and strong(base,alt) and detail['late_loss']<=1 and detail['margin_improved']>=4 and detail['mean_margin_delta']>=1e-4 and detail['theta_final']>=4)
    orig=lookup(records,6,'B','b_original')
    out['H_B6']=any(count(lookup(records,6,'B','b_'+p))>=4 and count(lookup(records,6,'B','b_'+p))-count(orig)>=3 for p in ('stop80','cosine_rate'))
    out['complete']=len(records)==100 and all(completed(r) for r in records)
    out['gate']=bool(out['complete'] and out['H_A'] and out['H_B0'] and out['H_B1'])
    return out
