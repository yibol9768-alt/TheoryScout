import experiment as c
from experiment import np,json
from run import grid,identifier

def adjudicate(records,fixed):
    def group(d,m,p):
        return [records.get(identifier(d,s,m,p)) for s in range(20,25)]
    def count(d,m,p):
        return sum(r is not None and r['status']=='completed' and r['final']['all_sign'] for r in group(d,m,p))
    def duration(r,key='all_sign'):
        return c.T+1000 if r is None or r['status']!='completed' or r['sustained'][key] is None else r['sustained'][key]
    cnt={f'd{d}_{m}_{p}':int(count(d,m,p)) for d in (6,7) for m in c.MODES for p in ('train','fixed')}
    b=count(7,'b','train'); th=count(7,'theta','train'); so=count(7,'solve','train')
    tb=[duration(r) for r in group(7,'b','train')]; tt=[duration(r) for r in group(7,'theta','train')]
    faster=sum(x<y for x,y in zip(tt,tb)); ratio=float(np.median(tt)/np.median(tb))
    zt=count(7,'zero','train'); lt=count(7,'b_legacy','train'); zf=count(7,'zero','fixed'); lf=count(7,'b_legacy','fixed')
    def success(r): return r is not None and r['status']=='completed' and r['final']['all_sign']
    repaired=sum(success(f) and not success(t) for t,f in zip(group(7,'b_legacy','train'),group(7,'b_legacy','fixed')))
    d6diff=[(f['final']['rank_fraction']-t['final']['rank_fraction']) if t and f and t['status']==f['status']=='completed' else -1. for t,f in zip(group(6,'b_legacy','train'),group(6,'b_legacy','fixed'))]
    h1=len(fixed)==120 and all(r['pass_H1'] for r in fixed)
    h1b=sum(r['sorted_but_star_wrong'] for r in fixed if r['beta_label']=='16')
    return dict(counts=cnt,decisions=dict(H1=h1,H1b=h1b>=1,
        H2=b<=1 and max(th,so)>=4 and max(th,so)-b>=3 and max(count(6,'theta','train'),count(6,'solve','train'))>=3,
        H2s=sum(duration(r)<=c.T for r in group(7,'theta','train'))>=4 and ratio<=.75 and faster>=4,
        H3=zt-lt>=3 and lf>=4 and zf>=4 and abs(zf-lf)<=1 and repaired>=3,
        H3d6=float(np.mean(d6diff))>=.05 and sum(x>0 for x in d6diff)>=4,
        H4=zt>=4 and lt<=1 and zt-lt>=3),
        details=dict(H1b_rows=h1b,H2s_b_times=tb,H2s_theta_times=tt,H2s_ratio=ratio,H2s_faster=faster,
                     H3_repaired= repaired,H3d6_differences=d6diff,H3d6_mean=float(np.mean(d6diff))))

def main():
    manifest=json.loads((c.OUT/'manifest.json').read_text())
    hash_ok=all(c.digest(c.ROOT/p)==h for p,h in manifest['hashes'].items())
    records={}; validation=[]
    for config in grid(range(20,25)):
        name=identifier(*config); path=c.OUT/'raw'/f'{name}.json'
        if not path.exists(): continue
        r=json.loads(path.read_text()); records[name]=r
        d,s,m,p=config; A,_=c.data(d,s)
        with np.load(c.OUT/'weights'/f'{name}.npz') as z:
            weights={k:z[k] for k in z.files if k not in ('beta','solved_b')}
            met=c.metrics(weights,A,m,float(z['beta'][0]),z.get('solved_b'),True)
            normerr=max(float(np.max(abs(np.linalg.norm(z[k],axis=1)-1))) for k in ('U','V'))
        err=max(float(np.max(abs(np.array(met[k])-np.array(r['final'][k])))) for k in ('gamma','zeta','shift'))
        validation.append(dict(id=name,metric_error=err,norm_error=normerr,solver_ok=r['solver']['max_residual']<=1e-9))
    matching=[]
    for d in (6,7):
      for s in range(20,25):
        rr=[records.get(identifier(d,s,m,p)) for m in c.MODES for p in ('train','fixed')]
        fingerprints={r['fingerprint'] for r in rr if r}
        shifts=[np.array(records[identifier(d,s,m,p)]['initial']['shift']) for m in ('b','theta','solve') for p in ('train','fixed') if identifier(d,s,m,p) in records]
        matching.append(dict(d=d,seed=s,fingerprints=len(fingerprints),max_shift_error=max((float(np.max(abs(x-shifts[0]))) for x in shifts),default=0)))
    fixed=json.loads((c.OUT/'fixed_geometry.json').read_text())
    result=adjudicate(records,fixed)
    result.update(total_present=len(records),completed=sum(r['status']=='completed' for r in records.values()),hash_ok=hash_ok,
                  validation=validation,matching=matching)
    result['valid']=hash_ok and len(records)==100 and all(r['metric_error']<1e-12 and r['norm_error']<1e-12 and r['solver_ok'] for r in validation) and all(r['fingerprints']==1 and r['max_shift_error']<1e-12 for r in matching)
    c.save(c.OUT/'assessment.json',result)
    rows=['|d|参数化|beta|全排序|全符号|平均排序比例|平均最小符号间隔|beta均值|持续符号步数中位数(删失101000)|墙钟秒合计|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    per=['|d|seed|参数化|beta路径|状态|排序比例|全符号|最小符号间隔|beta|首次符号|持续符号|持续1e-4|秒|','|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for d in (6,7):
      for m in c.MODES:
       for p in ('train','fixed'):
        rr=[records[identifier(d,s,m,p)] for s in range(20,25) if identifier(d,s,m,p) in records]
        if not rr: continue
        vals=[r['final'] for r in rr]
        times=[r['sustained']['all_sign'] if r['status']=='completed' and r['sustained']['all_sign'] is not None else 101000 for r in rr]
        rows.append(f"|{d}|{m}|{p}|{sum(v['all_rank'] for v in vals)}/5|{sum(r['status']=='completed' and r['final']['all_sign'] for r in rr)}/5|{np.mean([v['rank_fraction'] for v in vals]):.6f}|{np.mean([v['min_zeta'] for v in vals]):.6g}|{np.mean([v['beta'] for v in vals]):.3f}|{np.median(times):.0f}|{sum(r['seconds'] for r in rr):.1f}|")
        for r in rr:
            v=r['final']; per.append(f"|{d}|{r['config']['seed']}|{m}|{p}|{r['status']}|{v['rank_fraction']:.6f}|{v['all_sign']}|{v['min_zeta']:.6g}|{v['beta']:.3f}|{r['first']['all_sign']}|{r['sustained']['all_sign']}|{r['sustained']['robust_sign']}|{r['seconds']:.2f}|")
    (c.OUT/'tables.md').write_text('\n'.join(rows)+'\n\n'+ '\n'.join(per)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('validation','matching','counts')},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
