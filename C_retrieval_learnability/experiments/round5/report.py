"""Post-training aggregation and figures. Never trains or selects a configuration."""
import experiment as c
from run import read_records,identifier
from assess import assess,lookup,count,success,robust,flips
import csv,os
for key in ('MPLCONFIGDIR','XDG_CACHE_HOME','TMPDIR'):
    path=c.OUT/key.lower(); path.mkdir(exist_ok=True); os.environ[key]=str(path)

def main():
    main=read_records(c.OUT/'main'/'raw'); held=read_records(c.OUT/'holdout'/'raw'); records=main+held
    assessment=assess(main)
    if held:
        groups={v:lookup(held,7,'A',v) for v in ('ZL','ZM','SL','SM')}; cnt={v:count(g) for v,g in groups.items()}
        category=assessment['A_d7']['category']
        expected={'initialization':{'SL':True,'ZM':False},'learning_rate':{'SL':False,'ZM':True},'each_sufficient':{'SL':True,'ZM':True},'joint_required':{'SL':False,'ZM':False}}[category]
        ca=cnt['ZL']<=1 and cnt['SM']>=2 and all((cnt[k]>=2 if val else cnt[k]<=1) for k,val in expected.items())
        b={v:lookup(held,7,'B',v) for v in ('b_original','b_stop80','theta_original','theta_stop80')}
        cb=count(b['b_original'],lambda r:r['late_loss'])>=2 and count(b['b_original'])<=1 and count(b['b_stop80'])>=2 and count(b['b_stop80'],robust)>=2 and count(b['b_stop80'],lambda r:r['late_loss'])<=1 and count(b['theta_original'])>=2 and count(b['theta_stop80'])>=2
        assessment['holdout']=dict(executed=True,runs=len(held),A_counts=cnt,H_C_A=bool(ca),H_C_B=bool(cb),B={k:dict(final=count(v),robust=count(v,robust),late_loss=count(v,lambda r:r['late_loss'])) for k,v in b.items()})
    else: assessment['holdout']=dict(executed=False,reason='preregistered gate false',H_C_A=None,H_C_B=None)
    c.save(c.OUT/'assessment.json',assessment)
    summary=[]
    for n in sorted(set(r['config']['n'] for r in records)):
      for d in sorted(set(r['config']['d'] for r in records if r['config']['n']==n)):
       for stage in ('A','B'):
        variants=sorted(set(r['config']['variant'] for r in records if r['config']['n']==n and r['config']['d']==d and r['config']['stage']==stage))
        for variant in variants:
            rs=sorted([r for r in records if r['config']['n']==n and r['config']['d']==d and r['config']['stage']==stage and r['config']['variant']==variant],key=lambda r:r['config']['seed'])
            med=lambda key:float(c.np.median([r['sustained'][key] if r['sustained'][key] is not None else 101000 for r in rs]))
            summary.append(dict(n=n,d=d,stage=stage,variant=variant,seeds=[r['config']['seed'] for r in rs],runs=len(rs),rank=count(rs,lambda r:r['final']['all_rank']),sign=count(rs),robust=count(rs,robust),late_loss=count(rs,lambda r:r['late_loss']),ever_loss=count(rs,lambda r:r['ever_loss']),rank_fraction=float(c.np.mean([r['final']['rank_fraction'] for r in rs])),mean_min_zeta=float(c.np.mean([r['final']['min_zeta'] for r in rs])),min_zeta=[r['final']['min_zeta'] for r in rs],mean_beta=float(c.np.mean([r['final']['beta'] for r in rs])),sustained_sign=med('all_sign'),sustained_robust=med('robust_sign'),seconds=sum(r['seconds'] for r in rs),cpu_seconds=sum(r['cpu_seconds'] for r in rs)))
    c.save(c.OUT/'summary.json',summary)
    lines=['# 完整逐种子结果','', '终态优先；None/删失表示未持续命中；所有间隔均为几何单位。','', '|n|d|seed|实验|分支|状态|排序|符号|鲁棒|末段丢失|min gamma|min zeta|beta|首次符号|持续符号|持续鲁棒|墙钟秒|CPU秒|','|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    csvrows=[]
    for r in records:
        q=r['config']; f=r['final']; name=identifier(q)
        row=[q['n'],q['d'],q['seed'],q['stage'],q['variant'],r['status'],int(f['all_rank']),int(success(r)),int(robust(r)),int(r['late_loss']),f['min_gamma'],f['min_zeta'],f['beta'],r['first']['all_sign'],r['sustained']['all_sign'],r['sustained']['robust_sign'],r['seconds'],r['cpu_seconds']]
        lines.append('|'+ '|'.join(str(x) for x in row)+'|'); csvrows.append(row)
    (c.OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    with (c.OUT/'tables.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['n','d','seed','stage','variant','status','all_rank','all_sign','robust_sign','late_loss','min_gamma','min_zeta','beta','first_sign','sustained_sign','sustained_robust','wall_seconds','cpu_seconds']); w.writerows(csvrows)
    summ=['|n|d|实验|分支|全排序|全符号|鲁棒|末段丢失|平均min zeta|持续符号中位步|持续鲁棒中位步|','|---|---|---|---|---|---|---|---|---|---|---|']
    for s in summary:
        fmt=lambda k:f"{s[k]}/{s['runs']}"
        tm=lambda k:'删失' if s[k]>100000 else str(int(s[k]))
        summ.append(f"|{s['n']}|{s['d']}|{s['stage']}|{s['variant']}|{fmt('rank')}|{fmt('sign')}|{fmt('robust')}|{fmt('late_loss')}|{s['mean_min_zeta']:.7g}|{tm('sustained_sign')}|{tm('sustained_robust')}|")
    (c.OUT/'summary_tables.md').write_text('\n'.join(summ)+'\n')
    # Supplementary diagnostics from already frozen terminal weights, no retraining.
    diagnostic=[]
    for r in records:
        q=r['config']; f=r['final']; gamma=c.np.array(f['gamma']); zeta=c.np.array(f['zeta'])
        diagnostic.append(dict(id=identifier(q),rank_wrong=int((gamma<=1e-10).sum()),sorted_threshold_wrong=int(((gamma>1e-10)&(zeta<=1e-10)).sum()),midpoint_all_sign=bool(gamma.min()/2>1e-10),last_success=max([h['step'] for h in r['history'] if h['all_sign']],default=None),bias_lr=r['rates'].get('b'),query_lr=r['rates']['U'],bias_query_ratio=r['rates'].get('b',0)/r['rates']['U']))
    c.save(c.OUT/'diagnostics.json',diagnostic)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'original':'#c43c39','stop80':'#167a51','cosine_rate':'#245bac'}
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for i,d in enumerate((6,7)):
      for j,mode in enumerate(('b','theta')):
        ax=axes[i,j]
        for path,color in colors.items():
            rs=lookup(main,d,'B',mode+'_'+path)
            hs=c.np.array([[h['min_zeta'] for h in r['history']] for r in rs]); ts=c.np.array([h['step'] for h in rs[0]['history']])
            keep=ts>=70000; hs=hs[:,keep]; ts=ts[keep]
            ax.plot(ts,c.np.median(hs,axis=0),label=path,color=color)
            ax.fill_between(ts,hs.min(0),hs.max(0),alpha=.12,color=color)
        ax.axhline(0,color='black',lw=.7); ax.axhline(1e-4,color='gray',ls=':',lw=.8)
        ax.axvline(80000,color='gray',ls='--',lw=.8); ax.set_title(f'd={d}, {mode}')
        ax.set_yscale('symlog',linthresh=1e-4); ax.set_xlim(70000,100000); ax.set_ylabel('Minimum geometric sign margin')
        if i==1: ax.set_xlabel('Step')
    axes[0,0].legend(fontsize=8); fig.suptitle('Late trajectory: median and seed range; dotted line = robust margin 1e-4')
    fig.tight_layout(); fig.savefig(c.OUT/'late_margins.png',dpi=170); fig.savefig(c.OUT/'late_margins.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,d in zip(axes,(6,7)):
        for i,v in enumerate(('ZL','ZM','SL','SM')):
            rs=lookup(main,d,'A',v); z=[r['final']['min_zeta'] for r in rs]
            ax.scatter(c.np.arange(5)*.04+i-.08,z,s=30,color='#245bac')
            ax.text(i,1.02,f'{count(rs)}/5',transform=ax.get_xaxis_transform(),ha='center')
        ax.axhline(0,color='black',lw=.7); ax.axhline(1e-4,color='gray',ls=':',lw=.8)
        ax.set_yscale('symlog',linthresh=1e-4); ax.set_xticks(range(4),('ZL','ZM','SL','SM')); ax.set_title(f'd={d}',pad=23)
    axes[0].set_ylabel('Terminal minimum geometric sign margin')
    fig.tight_layout(); fig.savefig(c.OUT/'factorial.png',dpi=170); fig.savefig(c.OUT/'factorial.pdf'); plt.close(fig)
    print(c.json.dumps(dict(assessment=assessment,summary=summary)),flush=True)
if __name__=='__main__': main()
