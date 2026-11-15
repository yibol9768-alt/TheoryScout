"""Independent saved-weights audit and preregistered paired assessment."""
import experiment as c
from experiment import np,json,time


def main():
    rows=[json.loads(p.read_text()) for p in sorted((c.OUT/'raw').glob('*.json'))]
    completed=[r for r in rows if r['status']=='completed']
    lookup={(r['config']['d'],r['config']['seed'],r['config']['mode']):r for r in completed}
    replay=[json.loads(p.read_text()) for p in sorted((c.OUT/'replay').glob('*.json'))]
    errors=[]
    for r in completed:
        cfg=r['config']; tag=f'd{cfg["d"]}_s{cfg["seed"]}_{cfg["mode"]}'
        p=dict(np.load(c.OUT/'weights'/f'{tag}.npz'))
        A,_,_,_=c.init(cfg['d'],cfg['seed'],cfg['mode'])
        met=c.metrics(p,A,True)
        err=max(float(np.max(abs(np.array(met[key])-r['final'][key]))) for key in ('gamma','zeta','effective_zeta'))
        assert err<1e-12
        assert not met['all_sign'] or met['all_rank']
        if cfg['mode']=='sigmoid_zero' and cfg['d']==6: assert not met['all_sign']
        errors.append(err)
    tables=[]
    for d in (6,7,8):
        for mode in c.MODES:
            rr=[lookup[(d,s,mode)] for s in range(15,20) if (d,s,mode) in lookup]
            if not rr: continue
            avg=lambda key:float(np.mean([r['final'][key] for r in rr]))
            tables.append(dict(d=d,mode=mode,count=len(rr),rank_full=sum(r['final']['all_rank'] for r in rr),
                               sign_full=sum(r['final']['all_sign'] for r in rr),robust_full=sum(r['final']['robust_all_sign'] for r in rr),
                               success=avg('success_fraction'),recall=avg('recall_at_k'),min_gap=avg('min_gap'),sign_margin=avg('effective_sign_margin'),beta=avg('beta'),
                               first_sign=[r['first_all_sign_step'] for r in rr]))
    complete_grid=len(completed)==60
    get=lambda d,s,m:lookup[(d,s,m)]
    paired=[]
    if complete_grid:
        for s in range(15,20):
            r,z,n=(get(6,s,m)['final'] for m in ('sigmoid_row','sigmoid_zero','nce'))
            paired.append(dict(seed=s,row_minus_zero=r['success_fraction']-z['success_fraction'],row_minus_nce=r['success_fraction']-n['success_fraction']))
    p2=complete_grid and all(sum(get(6,s,m)['final']['all_rank'] for s in range(15,20))>=4 for m in ('nce','sigmoid_row')) and sum(get(6,s,'sigmoid_row')['final']['all_sign'] for s in range(15,20))>=4 and np.mean([x['row_minus_zero'] for x in paired])>=.10 and sum(x['row_minus_zero']>0 for x in paired)>=4 and np.mean([abs(x['row_minus_nce']) for x in paired])<=.01
    p3_slow=False; p3_obstruct=False
    if complete_grid:
        full={d:sum(get(d,s,'sigmoid_zero')['final']['all_sign'] for s in range(15,20)) for d in (7,8)}
        t7=[get(7,s,'sigmoid_zero')['first_all_sign_step'] for s in range(15,20)]
        t8=[get(8,s,'sigmoid_zero')['first_all_sign_step'] for s in range(15,20)]
        # Exact medians only when both central order statistics are observed.
        if full[7]>=4 and full[8]>=4:
            med7=sorted(t for t in t7 if t is not None)[2]; med8=sorted(t for t in t8 if t is not None)[2]
            slower=sum(a is None and b is not None or a is not None and b is not None and a>b for a,b in zip(t7,t8))
            p3_slow=med7>=2*med8 and slower>=4
        p3_obstruct=full[7]<=1 and full[8]>=4
    p4=complete_grid and all(sum(get(d,s,'sigmoid_lift')['final']['all_sign'] for s in range(15,20))>=4 for d in (6,7)) and all(get(d,s,'sigmoid_lift')['initial']['all_sign'] for d in (6,7) for s in range(15,20))
    hashes=json.loads((c.OUT/'manifest.json').read_text())['hashes']
    assert all(c.digest(c.ROOT/path)==sha for path,sha in hashes.items())
    assessment=dict(completed=len(completed),raw_count=len(rows),P1=len(replay)==5,P2=bool(p2),P3_slow=bool(p3_slow),P3_obstruction=bool(p3_obstruct),P4=bool(p4),paired_d6=paired,table=tables,weights_recompute_max_error=max(errors,default=None),hashes_unchanged=True,memory=c.r2.memory())
    c.save(c.OUT/'assessment.json',assessment)
    print(json.dumps(assessment,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
