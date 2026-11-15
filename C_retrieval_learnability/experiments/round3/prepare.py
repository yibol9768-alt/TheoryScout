import experiment as c
from experiment import np,copy,time,json

def main():
    start=time.time()
    errors={}
    for mode in c.MODES[:3]:
        A,pos,neg,p=c.init(4,913,mode)
        _,g=c.obj(p,A,pos,neg,mode)
        worst=0.
        for key,index in [('U',(3,1)),('V',(2,0)),('a',(0,))]+([('b',(3,))] if 'b' in p else []):
            plus,minus=copy.deepcopy(p),copy.deepcopy(p)
            plus[key][index]+=1e-6; minus[key][index]-=1e-6
            fd=(c.obj(plus,A,pos,neg,mode)[0]-c.obj(minus,A,pos,neg,mode)[0])/2e-6
            worst=max(worst,abs(fd-g[key][index]))
        errors[mode]=worst
    assert max(errors.values())<1e-7
    timing=[]
    for d in (6,7,8):
        for mode in c.MODES:
            A,pos,neg,p=c.init(d+(mode=='sigmoid_lift'),15,mode)
            _,g=c.obj(p,A,pos,neg,mode)
            rates,moves=c.calibrate(p,g); opt=c.r2.Adam(p,rates)
            begin=time.perf_counter()
            for t in range(1000):
                _,g=c.obj(p,A,pos,neg,mode)
                opt.update(p,g,.5*(1+np.cos(np.pi*t/100000)))
            timing.append(dict(d=d,mode=mode,seconds=time.perf_counter()-begin))
    enew=5*100*1.5*sum(x['seconds'] for x in timing)
    ereplay=5*84*1.5*timing[0]['seconds']
    T=100000 if enew+ereplay<=6000 else max(1000,min(100000,1000*int(100000*max(0,6000-ereplay)/enew/1000)))
    c.save(c.OUT/'validation.json',dict(gradient_errors=errors,memory=c.r2.memory()))
    c.save(c.OUT/'budget_decision.json',dict(timing=timing,estimate_new=enew,estimate_replay=ereplay,steps=T,preparation_start_epoch=start))
    with (c.ROOT/'PREREG_ROUND3.md').open('a') as f:
        f.write(f'\n## 计时后冻结\n\n12段1000步共{sum(x["seconds"] for x in timing):.6f}秒；保守新训练估计{enew:.3f}秒、重放{ereplay:.3f}秒；按规则统一冻结新训练T={T}。只依据耗时，未查看计时轨迹指标。\n')
    files=[c.ROOT/'PREREG_ROUND3.md',c.OUT/'experiment.py',c.OUT/'prepare.py',c.OUT/'run.py',c.OUT/'summarize.py',c.OUT/'budget_decision.json',c.OUT/'validation.json',c.ROOT/'experiments/round2/core.py',c.ROOT/'experiments/week1_common.py']
    files += sorted((c.ROOT/'experiments/round2/raw').glob('n16_k3_d6_s*_nce.json'))
    c.save(c.OUT/'manifest.json',dict(hashes={str(p.relative_to(c.ROOT)):c.digest(p) for p in files},steps=T,interpreter=c.sys.executable,env={k:c.os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},numpy_version=np.__version__,memory=c.r2.memory()))
    print(json.dumps(dict(steps=T,estimate_new=enew,estimate_replay=ereplay,errors=errors)),flush=True)

if __name__=='__main__': main()
