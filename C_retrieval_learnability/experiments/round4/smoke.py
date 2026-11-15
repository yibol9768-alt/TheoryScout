import experiment as c
from experiment import np,time,json
from run import grid,identifier

def main():
    start=time.time(); errors={}
    for mode in c.MODES:
        A,p=c.data(7,901); beta=16.; b,_=c.root_bias(beta*(p['U']@p['V'].T),A)
        if mode in ('b','b_legacy'): p['b']=b.copy()
        if mode=='theta': p['theta']=-b/beta
        p['a']=np.array([np.log(beta)])
        _,g=c.obj(p,A,mode,beta,b)
        worst=0.
        for key,ix in [('U',(3,1)),('V',(2,0)),('a',(0,))]+([('theta',(3,))] if mode=='theta' else [('b',(3,))] if mode in ('b','b_legacy') else []):
            vals=[]
            for sign in (1,-1):
                q={k:v.copy() for k,v in p.items()}; q[key][ix]+=sign*1e-6
                qb=float(np.exp(q['a'][0])); bias=b
                if mode=='solve': bias,_=c.root_bias(qb*(q['U']@q['V'].T),A,b)
                vals.append(c.obj(q,A,mode,qb,bias)[0])
            worst=max(worst,abs((vals[0]-vals[1])/2e-6-g[key][ix]))
        errors[mode]=worst
    assert max(errors.values())<1e-7
    records=[]; timings={m:[] for m in c.MODES}; initial={}
    for d,s,m,p in grid((901,902)):
        r,w=c.train(d,s,m,p,200,time.time()+600)
        assert r['status']=='completed'
        assert r['solver']['max_residual']<=1e-9
        if m in ('b','theta'): assert abs(r['calibration']['bias_reference_effective_move']-.001)<1e-12
        key=(d,s,p)
        if m in ('b','theta','solve'):
            sh=np.array(r['initial']['shift'])
            if key in initial: assert np.max(abs(sh-initial[key]))<1e-12
            initial[key]=sh
        name=identifier(d,s,m,p); target=c.OUT/'smoke'/f'{name}.npz'
        target.parent.mkdir(exist_ok=True); np.savez_compressed(target,**w)
        with np.load(target) as z:
            A,_=c.data(d,s)
            met=c.metrics({k:z[k] for k in z.files if k not in ('beta','solved_b')},A,m,float(z['beta'][0]),z.get('solved_b'),True)
            assert np.max(abs(np.array(met['zeta'])-np.array(r['final']['zeta'])))<1e-12
            assert np.array_equal(z['U'],w['U'])
        c.save(c.OUT/'smoke'/f'{name}.json',r)
        timings[m].append(r['seconds']/200)
        records.append(dict(id=name,seconds=r['seconds']))
    # conservative single process cost; includes per-run setup overhead and factor1.5.
    estimate=1.5*sum(max(v)*c.T*20 for v in timings.values())
    result=dict(gradient_errors=errors,runs=len(records),timings=timings,estimated_serial_seconds=estimate,
                budget_feasible=estimate<=9000,wall_seconds=time.time()-start,records=records)
    c.save(c.OUT/'smoke.json',result); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
