import experiment as c
from run import grid,identifier

def main():
    start=c.time.time(); errors={}
    for n in (16,20):
      for mode in ('b','theta'):
        A,p=c.data(7,903,n); beta=16.; b,info=c.root_bias(beta*(p['U']@p['V'].T),A)
        ref,ri=c.root_bias_reference(beta*(p['U']@p['V'].T),A)
        assert info['residual']<=1e-9 and ri['residual']<=1e-9
        assert abs(c.obj(p,A,'solve',beta,b)[0]-c.obj(p,A,'solve',beta,ref)[0])<1e-10
        # Deliberately nonoptimal biases test nonzero bias gradients and temperature chain rule.
        if mode=='b': p['b']=b+.37
        else: p['theta']=-b/beta+.07
        p['a']=c.np.array([c.np.log(beta)])
        _,g=c.obj(p,A,mode,beta)
        worst=0.
        for key,ix in [('U',(3,1)),('V',(2,0)),('a',(0,)),(mode,(3,))]:
            vals=[]
            for sign in (1,-1):
                q={k:v.copy() for k,v in p.items()}; q[key][ix]+=sign*1e-6
                vals.append(c.obj(q,A,mode,float(c.np.exp(q['a'][0])))[0])
            worst=max(worst,abs((vals[0]-vals[1])/2e-6-g[key][ix]))
        errors[f'n{n}_{mode}']=worst
    assert max(errors.values())<1e-7
    A,p=c.data(7,903); p['b']=c.np.zeros(len(A)); p['a']=c.np.array([c.np.log(16.)])
    q={k:v.copy() for k,v in p.items()}; rates={k:.001 for k in p}
    native=c.Adam(p,rates); reference=c.AdamReference(q,rates)
    for t in range(20):
        _,g=c.obj(p,A,'b',float(c.np.exp(p['a'][0])))
        native.update(p,g,.7); reference.update(q,g,.7)
    adam_error=max(float(c.np.max(abs(p[k]-q[k]))) for k in p)
    assert adam_error<1e-12
    assert c.beta_path('original',0)==16 and abs(c.beta_path('original',1)-4096)<1e-10
    assert abs(c.beta_path('cosine_rate',1)-4096)<1e-10
    assert c.beta_path('stop80',.8)==c.beta_path('stop80',1)
    assert all(c.beta_path('stop80',x)==c.beta_path('original',x) for x in c.np.linspace(0,.8,81))
    records=[]; initial={}; timings={}
    configs=[{**x,'seed':903} for x in grid() if x['seed']==30]
    configs += [{**x,'seed':903} for x in grid(True) if x['seed']==40]
    for config in configs:
        r,w=c.train(config,300,c.time.time()+600)
        assert r['status']=='completed'
        assert all(abs(r['calibration']['moves'][k]-.001)<1e-10 for k in ('U','V'))
        if config['rate']=='matched': assert abs(r['calibration']['bias_reference_effective_move']-.001)<1e-12
        if config['init']=='star': assert r['initial_root']['residual']<=1e-9
        key=(config['n'],config['d'],config['init'])
        shift=c.np.array(r['initial']['shift'])
        if key in initial: assert c.np.max(abs(shift-initial[key]))<1e-12
        initial[key]=shift
        name=identifier(config); folder=c.OUT/'smoke'; folder.mkdir(exist_ok=True)
        c.np.savez_compressed(folder/f'{name}.npz',**w)
        with c.np.load(folder/f'{name}.npz') as z:
            A,_=c.data(config['d'],903,config['n'])
            p={k:z[k] for k in ('U','V','b','theta','a') if k in z}
            met=c.metrics(p,A,config['mode'],float(z['beta'][0]),None,True)
            assert c.np.max(abs(c.np.array(met['zeta'])-c.np.array(r['final']['zeta'])))<1e-12
            assert max(c.np.max(abs(c.np.linalg.norm(z[k],axis=1)-1)) for k in ('U','V'))<1e-10
        c.save(folder/f'{name}.json',r)
        timings.setdefault(config['n'],[]).append(r['seconds']/300)
        records.append(dict(id=name,seconds=r['seconds']))
    estimate=1.5*c.T*(100*max(timings[16])+24*max(timings[20]))
    result=dict(adam_reference_error=adam_error,gradient_errors=errors,runs=len(records),records=records,estimated_serial_seconds=estimate,budget_feasible=bool(estimate<=6600),wall_seconds=c.time.time()-start)
    c.save(c.OUT/'smoke.json',result); print(c.json.dumps(result),flush=True)
if __name__=='__main__': main()
