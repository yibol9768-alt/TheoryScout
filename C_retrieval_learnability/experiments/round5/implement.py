"""Mechanical adaptation of the copied round4 engine; retained for provenance."""
from pathlib import Path
p=Path(__file__).with_name('experiment.py')
s=p.read_text().replace('Round 4','Round 5').replace("('OMP_NUM_THREADS','3')","('OMP_NUM_THREADS','1')")
s=s.replace('def data(d,seed):\n    pos=np.array(list(combinations(range(16),3))); A=np.zeros((560,16),bool)\n    A[np.arange(560)[:,None],pos]=True', 'def data(d,seed,n=16,k=3):\n    pos=np.array(list(combinations(range(n),k))); A=np.zeros((len(pos),n),bool)\n    A[np.arange(len(pos))[:,None],pos]=True')
s=s.replace('size=(560,d)','size=(len(A),d)').replace('size=(16,d)','size=(n,d)')
s=s.replace('ordered=np.partition(base, (12,13), axis=1)\n    middle=-(ordered[:,12]+ordered[:,13])/2','cut=base.shape[1]-int(A[0].sum())\n    ordered=np.partition(base, (cut-1,cut), axis=1)\n    middle=-(ordered[:,cut-1]+ordered[:,cut])/2')
s=s.replace('    assert np.all(A.sum(1)==3)','    assert np.all(A.sum(1)==3)\n    if base.shape[1]!=16: return root_bias_reference(base,A,warm)')
s=s.replace('k=3; cutoff=np.partition(S,13,axis=1)[:,13]','k=int(A[0].sum()); cut=S.shape[1]-k; cutoff=np.partition(S,cut,axis=1)[:,cut]')
s=s[:s.index('def train(')]
s+='''def beta_path(schedule,x):
    if schedule=='original': q=x
    elif schedule=='stop80': q=min(x,.8)
    elif schedule=='cosine_rate': q=x+np.sin(np.pi*x)/np.pi
    else: raise ValueError(schedule)
    return float(16.*256.**q)

def train(config,steps,deadline):
    start=time.time(); cpu=time.process_time()
    d,seed=config['d'],config['seed']; mode=config['mode']; schedule=config['schedule']
    A,p=data(d,seed,config.get('n',16),3)
    fingerprint=hashlib.sha256(p['U'].tobytes()+p['V'].tobytes()).hexdigest()
    beta=16.; b=None
    if config['init']=='star':
        b,root_info=root_bias(beta*(p['U']@p['V'].T),A)
    else: b=np.zeros(len(A)); root_info=None
    if mode=='b': p['b']=b.copy()
    else: p['theta']=-b/beta
    if schedule=='train': p['a']=np.array([np.log(beta)])
    rates,cal=calibrate(p,A,mode,beta,b)
    if config['rate']=='legacy': rates['b']=rates['U']
    # Reconstruct the real first update after the chosen rate is applied.
    q={k:v.copy() for k,v in p.items()}; _,g=obj(p,A,mode,beta,b)
    Adam(q,rates).update(q,g)
    nextbeta=float(np.exp(q['a'][0])) if schedule=='train' else beta_path(schedule,1/steps)
    _,sh0=effective(p,mode,beta,b); _,sh_coord=effective(q,mode,beta,b); _,sh_full=effective(q,mode,nextbeta,b)
    cal['first_coordinate_move_rms']=float(np.sqrt(np.mean((sh_coord-sh0)**2)))
    cal['first_full_threshold_move_rms']=float(np.sqrt(np.mean((sh_full-sh0)**2)))
    cal['first_beta']=nextbeta
    if config['rate']=='legacy': cal['bias_reference_effective_move']=None
    opt=Adam(p,rates); history=[]; status='completed'; checkpoint={}
    for t in range(steps+1):
        beta=float(np.exp(p['a'][0])) if schedule=='train' else beta_path(schedule,t/steps)
        if not 1e-8<=beta<=1e4 or not all(np.isfinite(v).all() for v in p.values()):
            status='numerical_or_temperature_failure'; break
        if t%1000==0 and time.time()>deadline: status='budget_truncated'; break
        if t==int(.8*steps) and schedule!='train':
            checkpoint={f'at80_{key}':v.copy() for key,v in p.items()}
            checkpoint['at80_beta']=np.array([beta])
        if t%1000==0 or t==steps:
            met=metrics(p,A,mode,beta,b,arrays=(t==0))
            history.append(dict(step=t,loss=obj(p,A,mode,beta,b)[0],**met))
        if t==steps: break
        _,g=obj(p,A,mode,beta,b,need_loss=False)
        if not all(np.isfinite(v).all() for v in g.values()): status='nonfinite_gradient'; break
        opt.update(p,g,.5*(1+np.cos(np.pi*t/steps)))
    hit={}; sustained={}
    for key in ('all_rank','all_sign','robust_sign'):
        yes=[x['step'] for x in history if x[key]]; hit[key]=min(yes) if yes else None
        sustained[key]=None
        if history and history[-1][key] and status=='completed':
            lastfail=max([x['step'] for x in history if not x[key]],default=-1)
            sustained[key]=min(x['step'] for x in history if x['step']>lastfail)
    final=metrics(p,A,mode,beta,b,True)
    late_loss=bool(any(h['all_sign'] and .8*steps<=h['step']<steps for h in history) and not final['all_sign'])
    result=dict(config=config,status=status,steps=t,requested=steps,fingerprint=fingerprint,rates=rates,
                calibration=cal,initial=history[0],initial_root=root_info,final=final,
                first=hit,sustained=sustained,late_loss=late_loss,
                ever_loss=bool(hit['all_sign'] is not None and not final['all_sign']),history=history,
                seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return result,{**p,'beta':np.array([beta]),**checkpoint}
'''
p.write_text(s)
