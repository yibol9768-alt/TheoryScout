import experiment as c
from experiment import np,time,json
import subprocess


def worker(phase,index,deadline,T):
    if phase=='replay':
        for seed in range(10,15):
            if (seed-10)%2!=index: continue
            result,p=c.train(6,seed,'nce',84000,deadline)
            old=json.loads((c.ROOT/f'experiments/round2/raw/n16_k3_d6_s{seed}_nce.json').read_text())
            error=max(float(np.max(abs(np.array(result['final'][key])-old['final'][key]))) for key in ('gamma','zeta'))
            assert result['status']=='completed' and error<1e-10
            A,_,_,_=c.init(6,seed,'nce')
            q,check=c.lift(p,A)
            result['old_replay_max_abs_error']=error; result['deterministic_check']=check
            c.save(c.OUT/'replay'/f's{seed}.json',result)
            np.savez_compressed(c.OUT/'replay'/f's{seed}.npz',**p)
            print(json.dumps(dict(phase=phase,seed=seed,error=error,seconds=result['elapsed_seconds'])),flush=True)
    else:
        for idx,(d,seed) in enumerate((d,s) for d in (6,7,8) for s in range(15,20)):
            if idx%2!=index: continue
            nce_p=None
            for mode in c.MODES:
                identifier=f'd{d}_s{seed}_{mode}'
                if time.time()>deadline: return
                override=None; check=None
                if mode=='sigmoid_lift':
                    A,_,_,_=c.init(d,seed,'nce')
                    if nce_p is None or not c.metrics(nce_p,A)['all_rank']:
                        c.save(c.OUT/'raw'/f'{identifier}.json',dict(config=dict(d=d,seed=seed,mode=mode),status='nce_not_separable_control_unavailable'))
                        continue
                    override,check=c.lift(nce_p,A)
                result,p=c.train(d,seed,mode,T,deadline,override)
                if mode=='nce': nce_p=copy_p={k:v.copy() for k,v in p.items()}
                if check is not None: result['lift_check']=check
                np.savez_compressed(c.OUT/'weights'/f'{identifier}.npz',**p)
                c.save(c.OUT/'raw'/f'{identifier}.json',result)
                print(json.dumps(dict(id=identifier,status=result['status'],seconds=result['elapsed_seconds'])),flush=True)


def main():
    decision=json.loads((c.OUT/'budget_decision.json').read_text()); T=decision['steps']
    if len(c.sys.argv)>1:
        worker(c.sys.argv[1],int(c.sys.argv[2]),float(c.sys.argv[3]),T); return
    manifest=json.loads((c.OUT/'manifest.json').read_text())
    assert all(c.digest(c.ROOT/path)==sha for path,sha in manifest['hashes'].items())
    start=time.time(); deadline=min(start+6000,decision['preparation_start_epoch']+6900)
    for name in ('replay','raw','weights'): (c.OUT/name).mkdir(exist_ok=True)
    codes={}
    for phase in ('replay','main'):
        logs=[(c.OUT/f'{phase}_worker{i}.log').open('w') for i in range(2)]
        ps=[subprocess.Popen([c.sys.executable,str(c.OUT/'run.py'),phase,str(i),str(deadline)],stdout=logs[i],stderr=subprocess.STDOUT) for i in range(2)]
        while any(p.poll() is None for p in ps):
            if time.time()>deadline+20:
                for p in ps:
                    if p.poll() is None: p.terminate()
                break
            time.sleep(1)
        codes[phase]=[p.wait() for p in ps]
        for log in logs: log.close()
        if any(codes[phase]): break
        if phase=='replay': assert len(list((c.OUT/'replay').glob('*.json')))==5
    c.save(c.OUT/'execution.json',dict(start_epoch=start,end_epoch=time.time(),wall_seconds=time.time()-start,deadline_epoch=deadline,returncodes=codes,memory=c.r2.memory()))
    print(json.dumps(codes),flush=True)

if __name__=='__main__': main()
