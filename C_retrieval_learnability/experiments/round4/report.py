"""Descriptive reporting only; frozen decisions live in assess.py."""
import os
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'mplconfig')
import experiment as c
from experiment import np,json
from run import identifier

def main():
    assessment=json.loads((c.OUT/'assessment.json').read_text())
    fixed=json.loads((c.OUT/'fixed_geometry.json').read_text())
    records={}
    for path in sorted((c.OUT/'raw').glob('*.json')): records[path.stem]=json.loads(path.read_text())
    rows=['|d|第三轮分支|beta|排序行均值|原偏置符号行均值|中点符号行均值|逻辑最优符号行均值|已排对但逻辑最优错的行数合计|逻辑损失最优均值|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    fixedsummary=[]
    for d in (6,7):
      for mode in ('nce','sigmoid_row','sigmoid_zero'):
       for label in ('native','16','256','4096'):
        rr=[r for r in fixed if r['d']==d and r['mode']==mode and r['beta_label']==label]
        means={k:float(np.mean([r[k] for r in rr])) for k in ('rank_rows','original_sign_rows','mid_sign_rows','star_sign_rows','loss_original','loss_mid','loss_star')}
        mismatch=sum(r['sorted_but_star_wrong'] for r in rr)
        fixedsummary.append(dict(d=d,mode=mode,label=label,**means,mismatch=mismatch))
        rows.append(f"|{d}|{mode}|{label}|{means['rank_rows']:.1f}|{means['original_sign_rows']:.1f}|{means['mid_sign_rows']:.1f}|{means['star_sign_rows']:.1f}|{mismatch}|{means['loss_star']:.6g}|")
    (c.OUT/'fixed_geometry_tables.md').write_text('\n'.join(rows)+'\n')
    paired=[]
    for d in (6,7):
      for seed in range(20,25):
       for mode in c.MODES:
        train=records.get(identifier(d,seed,mode,'train')); fixedr=records.get(identifier(d,seed,mode,'fixed'))
        if train and fixedr:
            paired.append(dict(d=d,seed=seed,mode=mode,
                 fixed_minus_train_rank=fixedr['final']['rank_fraction']-train['final']['rank_fraction'],
                 fixed_minus_train_sign=int(fixedr['final']['all_sign'])-int(train['final']['all_sign']),
                 train_beta=train['final']['beta'],fixed_beta=fixedr['final']['beta']))
    costs=[]
    for mode in c.MODES:
        rr=[r for r in records.values() if r['config']['mode']==mode]
        costs.append(dict(mode=mode,wall_sum=sum(r['seconds'] for r in rr),cpu_sum=sum(r['cpu_seconds'] for r in rr),
                          solver_calls=sum(r['solver']['calls'] for r in rr),solver_row_iterations=sum(r['solver']['iterations'] for r in rr),solver_fallbacks=sum(r['solver']['fallbacks'] for r in rr)))
    diagnostics=dict(fixed_summary=fixedsummary,paired_temperature=paired,costs=costs,
        max_rss_kb=max(r['max_rss_kb'] for r in records.values()),
        cpu_total=sum(r['cpu_seconds'] for r in records.values()),wall_run_sum=sum(r['seconds'] for r in records.values()),
        max_solver_residual=max(r['solver']['max_residual'] for r in records.values()),
        strict_vs_robust_count_disagreement=sum(r['final']['all_sign']!=r['final']['all_sign_strict'] for r in records.values()),
        max_UV_move_error=max(abs(r['calibration']['moves'][k]-.001) for r in records.values() for k in ('U','V')),
        max_initial_bias_shift_move=max(r['calibration']['initial_actual_bias_effective_move'] for r in records.values() if r['config']['mode'] in ('b','theta')),
        max_root_reference_residual=max(x['residual'] for x in json.loads((c.OUT/'validation.json').read_text())['root_reference']))
    c.save(c.OUT/'diagnostics.json',diagnostics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zero='#555555',b='#2365cc',theta='#db7f13',solve='#13885b',b_legacy='#aa3344')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for i,d in enumerate((6,7)):
      for j,schedule in enumerate(('train','fixed')):
        ax=axes[i,j]
        for mode in c.MODES:
            rr=[records[identifier(d,s,mode,schedule)] for s in range(20,25) if identifier(d,s,mode,schedule) in records]
            if len(rr)!=5: continue
            steps=np.array([h['step'] for h in rr[0]['history']])
            if not all(np.array_equal(steps,[h['step'] for h in r['history']]) for r in rr): continue
            values=np.array([[h['sign_fraction'] for h in r['history']] for r in rr])
            ax.plot(steps/1000,values.mean(0),label=mode,color=colors[mode])
            ax.fill_between(steps/1000,values.min(0),values.max(0),color=colors[mode],alpha=.08)
        ax.set_title(f'd={d}, beta={schedule}'); ax.set_ylim(0,1.01); ax.grid(alpha=.2)
        if j==0: ax.set_ylabel('fraction of sign-separated rows')
        if i==1: ax.set_xlabel('steps / 1000')
    axes[0,0].legend(ncol=3,fontsize=8); fig.tight_layout(); fig.savefig(c.OUT/'trajectories.png',dpi=160); plt.close(fig)
    # Show the actual learned temperature paths rather than inferred causal mechanisms.
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for d,ax in zip((6,7),axes):
        for mode in c.MODES:
            rr=[records[identifier(d,s,mode,'train')] for s in range(20,25)]
            steps=np.array([h['step'] for h in rr[0]['history']]); vals=np.array([[h['beta'] for h in r['history']] for r in rr])
            ax.plot(steps/1000,np.median(vals,axis=0),label=mode,color=colors[mode])
        ax.plot(steps/1000,16*256**(steps/c.T),'k--',label='fixed path'); ax.set_yscale('log'); ax.set_title(f'd={d}'); ax.set_xlabel('steps / 1000'); ax.set_ylabel('beta'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(c.OUT/'beta_paths.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
    for i,d in enumerate((6,7)):
      for j,schedule in enumerate(('train','fixed')):
        ax=axes[i,j]
        for mi,mode in enumerate(c.MODES):
            for si,s in enumerate(range(20,25)):
                r=records[identifier(d,s,mode,schedule)]
                value=r['sustained']['all_sign']
                censored=r['status']!='completed' or value is None
                ax.scatter((101000 if censored else value)/1000,mi+(si-2)*.09,
                           marker='>' if censored else 'o',color=colors[mode],s=25,alpha=.8)
        ax.axvline(100,ls='--',color='gray',lw=.8); ax.set_xlim(-2,104)
        ax.set_title(f'd={d}, beta={schedule}'); ax.set_yticks(range(len(c.MODES)),c.MODES)
        ax.grid(axis='x',alpha=.2)
        if i==1: ax.set_xlabel('sustained separation step / 1000 (> = censored)')
    fig.tight_layout(); fig.savefig(c.OUT/'separation_times.png',dpi=160); plt.close(fig)
    print(json.dumps({k:v for k,v in diagnostics.items() if k not in ('fixed_summary','paired_temperature')},ensure_ascii=False))
if __name__=='__main__': main()
