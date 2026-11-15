import experiment as c
from experiment import np,time,json,sys
import subprocess

def identifier(d,s,m,p): return f'd{d}_s{s}_{m}_{p}'
def grid(seeds): return [(d,s,m,p) for d in (6,7) for s in seeds for m in c.MODES for p in ('train','fixed')]

def fixed_geometry():
    results=[]; arrays={}
    for d in (6,7):
      for seed in range(15,20):
       for mode in ('nce','sigmoid_row','sigmoid_zero'):
        source=c.ROOT/f'experiments/round3/weights/d{d}_s{seed}_{mode}.npz'
        with np.load(source) as z: p={k:z[k].copy() for k in z.files}
        A,_=c.data(d,seed); S=p['U']@p['V'].T
        low=np.where(A,S,np.inf).min(1); high=np.where(~A,S,-np.inf).max(1)
        gamma=low-high; mid=(low+high)/2
        for label,beta in [('native',float(np.exp(p['a'][0]))),('16',16.),('256',256.),('4096',4096.)]:
            original=p.get('b',np.zeros(560)).copy()
            star,info=c.root_bias(beta*S,A)
            shifts=[original/beta,-mid,star/beta]
            losses=[]; signs=[]; zetas=[]
            for shift in shifts:
                Z=beta*(S+shift[:,None]); Y=2.*A-1
                losses.append(float(np.logaddexp(0,-Y*Z).mean()))
                z=np.minimum(low+shift,-high-shift); zetas.append(z); signs.append(z>0)
            error=float(np.max(np.abs(zetas[1]-gamma/2)))
            ok=info['residual']<=1e-9 and losses[2]<=min(losses[:2])+1e-10 and error<=1e-12 and np.array_equal(signs[1],gamma>0) and not np.any(signs[2] & (gamma<=0))
            key=f'd{d}_s{seed}_{mode}_{label}'
            arrays[key+'_gamma']=gamma; arrays[key+'_mid_zeta']=zetas[1]; arrays[key+'_star_zeta']=zetas[2]; arrays[key+'_star_b']=star
            results.append(dict(id=key,d=d,seed=seed,mode=mode,beta_label=label,beta=beta,
                                solver=info,loss_original=losses[0],loss_mid=losses[1],loss_star=losses[2],
                                rank_rows=int((gamma>0).sum()),original_sign_rows=int(signs[0].sum()),
                                mid_sign_rows=int(signs[1].sum()),star_sign_rows=int(signs[2].sum()),
                                sorted_but_star_wrong=int(((gamma>0)&~signs[2]).sum()),
                                mid_identity_error=error,pass_H1=bool(ok)))
    c.save(c.OUT/'fixed_geometry.json',results); np.savez_compressed(c.OUT/'fixed_geometry.npz',**arrays)

def worker(index,deadline):
    for i,config in enumerate(grid(range(20,25))):
        if i%2!=index: continue
        if time.time()>deadline: return
        d,s,m,p=config; name=identifier(*config)
        r,w=c.train(d,s,m,p,c.T,deadline)
        np.savez_compressed(c.OUT/'weights'/f'{name}.npz',**w); c.save(c.OUT/'raw'/f'{name}.json',r)
        print(json.dumps(dict(id=name,status=r['status'],seconds=r['seconds'],all_sign=r['final']['all_sign'])),flush=True)

def main():
    if len(sys.argv)>1: worker(int(sys.argv[1]),float(sys.argv[2])); return
    manifest=json.loads((c.OUT/'manifest.json').read_text())
    assert all(c.digest(c.ROOT/p)==h for p,h in manifest['hashes'].items())
    assert json.loads((c.OUT/'smoke.json').read_text())['budget_feasible']
    start=time.time(); deadline=start+9000
    fixed_geometry()
    for folder in ('raw','weights'): (c.OUT/folder).mkdir(exist_ok=True)
    logs=[(c.OUT/f'worker{i}.log').open('w') for i in range(2)]
    ps=[subprocess.Popen([sys.executable,str(c.OUT/'run.py'),str(i),str(deadline)],stdout=logs[i],stderr=subprocess.STDOUT) for i in range(2)]
    while any(p.poll() is None for p in ps):
        if time.time()>deadline+30:
            for p in ps:
                if p.poll() is None: p.terminate()
            break
        time.sleep(1)
    codes=[p.wait() for p in ps]
    for log in logs: log.close()
    c.save(c.OUT/'execution.json',dict(start_epoch=start,end_epoch=time.time(),wall_seconds=time.time()-start,returncodes=codes,deadline=deadline))
    print(json.dumps(dict(returncodes=codes,wall_seconds=time.time()-start)),flush=True)
if __name__=='__main__': main()
