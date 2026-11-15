import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    assert os.environ.get(k)=='2'
import json,time,resource
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

start=time.monotonic();out=Path(__file__).resolve().parent
rows=json.loads((out/'FT_summary.json').read_text())
contrasts=[]
import csv
with (out/'FT_contrasts.csv').open() as f:contrasts=list(csv.DictReader(f))
methods=['GD','Sign','Adam_e0.1','Adam_e1','AdamW_e0.1','Muon']
colors=dict(zip(methods,plt.get_cmap('tab10').colors[:6]))
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
for ax,T in zip(axes[:2],[1,4]):
    for m in methods:
        selected={r['mode']:r for r in rows if r.get('d')==128 and r.get('case')=='independent' and r.get('T')==T and r.get('method')==m}
        for mode,marker in [('fixed','o'),('matched','s')]:
            r=selected[mode]
            ax.errorbar(1-r['relative_loss']['mean'],r['old']['msr']['mean'],
                xerr=1.96*r['relative_loss']['se'],yerr=1.96*r['old']['msr']['se'],
                color=colors[m],marker=marker,ms=5,capsize=2)
        a,b=selected['fixed'],selected['matched']
        ax.plot([1-a['relative_loss']['mean'],1-b['relative_loss']['mean']],
            [a['old']['msr']['mean'],b['old']['msr']['mean']],color=colors[m],lw=.6,alpha=.6)
    ax.axvline(.25,color='gray',lw=.7,ls=':')
    ax.set(title=f'Independent task, T={T}',xlabel='New reducible loss learned',ylabel='Old full-training MSR')
    ax.grid(alpha=.15)
ax=axes[2]
cases=['independent','samekey_random','aligned','conflict','correlated']
for j,m in enumerate(methods[1:]):
    rr=[next(r for r in contrasts if int(r['d'])==128 and r['case']==case and int(r['T'])==4 and r['mode']=='matched' and r['method']==m and r['task']=='old' and r['metric']=='msr') for case in cases]
    xx=[i+(j-2)*.11 for i in range(5)]
    yy=[float(r['mean']) for r in rr]
    ee=[[float(r['mean'])-float(r['lo']) for r in rr],[float(r['hi'])-float(r['mean']) for r in rr]]
    ax.errorbar(xx,yy,yerr=ee,fmt='o',color=colors[m],ms=3,capsize=2)
ax.axhline(0,color='black',lw=.7)
ax.set_xticks(range(5),['indep.','same key','aligned','conflict','corr.'],rotation=30)
ax.set(title='Matched loss, T=4, d=128',ylabel='Old MSR difference from GD')
ax.grid(alpha=.15)
handles=[Line2D([],[],marker='o',color=colors[m],label=m,lw=0) for m in methods]
fig.legend(handles=handles,loc='outside lower center',ncol=6,fontsize=8)
fig.suptitle('Actual Gaussian-task fine-tuning; seed-level 95% intervals (descriptive)',fontsize=11)
for suffix in ['png','pdf']:
    path=out/f'FT_learning_forgetting.{suffix}'
    assert not path.exists()
    fig.savefig(path,dpi=180)
meta=dict(seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
with (out/'FT_plot_meta.json').open('x') as f:json.dump(meta,f,indent=2)
