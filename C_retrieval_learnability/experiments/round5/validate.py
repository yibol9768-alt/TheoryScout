"""Independent terminal replay and matching checks; frozen before formal training."""
import experiment as c
from run import grid,identifier,read_records
from assess import assess

def validate():
    manifest=c.json.loads((c.OUT/'manifest.json').read_text()); badhash=[p for p,h in manifest['hashes'].items() if c.digest(c.ROOT/p)!=h]
    errors=[]; max_error=0.; norm_error=0.; prefix_error=0.; prefix_pairs=0; initial={}; fingerprint={}
    records=[]
    for holdout in (False,True):
        folder=c.OUT/('holdout' if holdout else 'main')
        if holdout and not folder.exists(): continue
        for config in grid(holdout):
            name=identifier(config); f=folder/'raw'/f'{name}.json'
            if not f.exists(): errors.append(name+':missing'); continue
            r=c.json.loads(f.read_text()); records.append(r)
            A,p0=c.data(config['d'],config['seed'],config['n'])
            assert r['fingerprint']==c.hashlib.sha256(p0['U'].tobytes()+p0['V'].tobytes()).hexdigest()
            pair=(config['n'],config['d'],config['seed'],config['init']); shift=c.np.array(r['initial']['shift'])
            if pair in initial: assert c.np.max(abs(shift-initial[pair]))<1e-12
            initial[pair]=shift
            if config['rate']=='matched': assert abs(r['calibration']['bias_reference_effective_move']-.001)<1e-12
            assert max(abs(r['calibration']['moves'][k]-.001) for k in ('U','V'))<1e-10
            if config['init']=='star': assert r['initial_root']['residual']<=1e-9
            with c.np.load(folder/'weights'/f'{name}.npz') as z:
                S=z['U']@z['V'].T; beta=float(z['beta'][0]); shift=-z['theta'] if config['mode']=='theta' else z['b']/beta
                low=c.np.where(A,S,c.np.inf).min(1); high=c.np.where(~A,S,-c.np.inf).max(1)
                gamma=low-high; zeta=c.np.minimum(low+shift,-high-shift)
                for key,v in [('gamma',gamma),('zeta',zeta),('shift',shift)]:
                    max_error=max(max_error,float(c.np.max(abs(v-c.np.array(r['final'][key])))))
                assert r['final']['all_sign']==bool(zeta.min()>1e-10)
                assert r['final']['robust_sign']==bool(zeta.min()>=1e-4)
                norm_error=max(norm_error,max(float(c.np.max(abs(c.np.linalg.norm(z[k],axis=1)-1))) for k in ('U','V')))
                if config['stage']=='B' and config['schedule']=='stop80':
                    other={**config,'variant':config['mode']+'_original','schedule':'original'}
                    with c.np.load(folder/'weights'/f'{identifier(other)}.npz') as o:
                        for key in ('at80_U','at80_V','at80_beta','at80_'+config['mode']):
                            prefix_error=max(prefix_error,float(c.np.max(abs(z[key]-o[key]))))
                    orig=c.json.loads((folder/'raw'/f'{identifier(other)}.json').read_text())
                    for h,q in zip(r['history'],orig['history']):
                        if h['step']<=80000:
                            for key in ('beta','loss','min_gamma','min_zeta','sign_fraction'): assert h[key]==q[key]
                    prefix_pairs+=1
            if r['status']!='completed' or r['steps']!=c.T: errors.append(name+':incomplete')
    gate=assess([r for r in records if r['config']['n']==16]); saved=c.json.loads((c.OUT/'gate.json').read_text()); assert gate==saved
    assert (c.OUT/'holdout').exists()==bool(gate['gate'])
    execution=c.json.loads((c.OUT/'execution.json').read_text())
    result=dict(runs=len(records),hash_errors=badhash,trajectory_errors=errors,max_terminal_replay_error=max_error,max_norm_error=norm_error,max_stop80_prefix_error=prefix_error,stop80_prefix_pairs=prefix_pairs,
        prereg_sha256=c.digest(c.ROOT/'PREREG_ROUND5.md'),cpu_seconds=sum(r['cpu_seconds'] for r in records),max_rss_kb=max(r['max_rss_kb'] for r in records),wall_seconds=execution['wall_seconds'],budget_pass=execution['wall_seconds']<=7200 and sum(r['cpu_seconds'] for r in records)<=7200,
        strict_vs_tolerance_count_equal=all(r['final']['all_sign']==r['final']['all_sign_strict'] and r['final']['all_rank']==r['final']['all_rank_strict'] for r in records))
    result['pass']=bool(not badhash and not errors and max_error<=1e-12 and norm_error<=1e-10 and prefix_error<=1e-12 and result['budget_pass'])
    c.save(c.OUT/'validation.json',result); print(c.json.dumps(result),flush=True)
    assert result['pass']
if __name__=='__main__': validate()
