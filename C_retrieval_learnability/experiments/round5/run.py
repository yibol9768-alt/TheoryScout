import experiment as c
from assess import assess
import subprocess,os

def grid(holdout=False):
    ds=(7,) if holdout else (6,7); seeds=(40,41,42) if holdout else (30,31,32,33,34); n=20 if holdout else 16
    rows=[]
    for d in ds:
      for seed in seeds:
        for init in ('zero','star'):
          for rate in ('legacy','matched'):
            rows.append(dict(d=d,seed=seed,n=n,stage='A',variant=('Z' if init=='zero' else 'S')+('L' if rate=='legacy' else 'M'),mode='b',init=init,rate=rate,schedule='train'))
        for mode in ('b','theta'):
          for path in (('original','stop80') if holdout else ('original','stop80','cosine_rate')):
            rows.append(dict(d=d,seed=seed,n=n,stage='B',variant=mode+'_'+path,mode=mode,init='star',rate='matched',schedule=path))
    return rows

def identifier(config): return f"n{config['n']}_d{config['d']}_s{config['seed']}_{config['stage']}_{config['variant']}"
def read_records(folder): return [c.json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]

def worker(index,deadline,holdout):
    folder=c.OUT/('holdout' if holdout else 'main')
    for i,config in enumerate(grid(holdout)):
        if i%2!=index: continue
        if c.time.time()>deadline: return
        r,w=c.train(config,c.T,deadline); name=identifier(config)
        c.np.savez_compressed(folder/'weights'/f'{name}.npz',**w); c.save(folder/'raw'/f'{name}.json',r)
        print(c.json.dumps(dict(id=name,status=r['status'],seconds=r['seconds'],all_sign=r['final']['all_sign'],min_zeta=r['final']['min_zeta'])),flush=True)

def launch(deadline,holdout):
    folder=c.OUT/('holdout' if holdout else 'main')
    for part in ('raw','weights'): (folder/part).mkdir(parents=True,exist_ok=True)
    logs=[(folder/f'worker{i}.log').open('w') for i in range(2)]
    ps=[subprocess.Popen([c.sys.executable,str(c.OUT/'run.py'),'worker',str(i),str(deadline),str(int(holdout))],stdout=logs[i],stderr=subprocess.STDOUT) for i in range(2)]
    while any(p.poll() is None for p in ps):
        if c.time.time()>deadline+5:
            for p in ps:
                if p.poll() is None: p.terminate()
            break
        c.time.sleep(1)
    codes=[p.wait() for p in ps]
    for log in logs: log.close()
    return codes

def main():
    if len(c.sys.argv)>1:
        worker(int(c.sys.argv[2]),float(c.sys.argv[3]),bool(int(c.sys.argv[4]))); return
    manifest=c.json.loads((c.OUT/'manifest.json').read_text())
    assert all(c.digest(c.ROOT/p)==h for p,h in manifest['hashes'].items())
    assert c.json.loads((c.OUT/'smoke.json').read_text())['budget_feasible']
    assert not (c.OUT/'main').exists(), 'Do not overwrite formal outputs'
    start=c.time.time(); deadline=start+7200
    codes=launch(min(deadline,start+5400),False)
    records=read_records(c.OUT/'main'/'raw'); decision=assess(records); c.save(c.OUT/'gate.json',decision)
    holdcodes=None
    if decision['gate'] and c.time.time()<deadline: holdcodes=launch(deadline,True)
    c.save(c.OUT/'execution.json',dict(start_epoch=start,end_epoch=c.time.time(),wall_seconds=c.time.time()-start,deadline=deadline,main_returncodes=codes,holdout_returncodes=holdcodes,holdout_gate=decision['gate'],main_runs=len(records),cpu_seconds=sum(r['cpu_seconds'] for r in records)+sum(r['cpu_seconds'] for r in read_records(c.OUT/'holdout'/'raw'))))
    print((c.OUT/'execution.json').read_text(),flush=True)
if __name__=='__main__': main()
