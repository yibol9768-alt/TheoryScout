"""Export static scientific figures from frozen summaries; no training."""
import csv
import json
import os
from pathlib import Path
import time
import numpy as np
from run import OUT,frozen,dump,utc
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

START=time.monotonic(); frozen()
s=json.loads((OUT/'summary.json').read_text())
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
colors={32:'tab:blue',64:'tab:orange',128:'tab:green'}
fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for n in [32,64,128]:
    for mode in [0,1]:
        cs=[c for c in s['cells'] if c['problem']==2 and c['group']=='learned_adam_cold' and c['N']==n and c['mode']==mode]
        rr=np.array([c['ratio'] for c in cs]); ts=np.array([np.nan if c['tau'] is None else c['tau'] for c in cs])
        ax[0,mode].plot(rr,ts,'o-',label=f'N={n}',color=colors[n])
        for r,t in zip(rr,ts):
            if np.isnan(t): ax[0,mode].scatter([r],[.94-.06*[32,64,128].index(n)],marker='^',color=colors[n],transform=ax[0,mode].get_xaxis_transform())
        if mode==0 and np.isfinite(ts[0]): ax[0,mode].plot(rr,ts[0]*(rr/rr[0])**-.5,':',color=colors[n],alpha=.5)
        ax[0,mode].set(xscale='log',yscale='log',xlabel='D/N' if mode==0 else 'M/N',ylabel='tau_data' if mode==0 else 'tau_epoch',title='B primary: stream' if mode==0 else 'B primary: replay')
        ax[0,mode].legend()
        ax[0,mode].set_xlim(rr.min()/1.12,rr.max()*1.12)
for item in s['spectral']:
    if item['ratio']==256:
        current=np.array(item['current_modes']); initial=np.array(item['initial_modes'])
        ax[1,0].plot(item['eig'],current/initial,'o-',ms=2,color=colors[item['N']],label=f"N={item['N']}")
ax[1,0].axhline(.5,color='k',ls=':'); ax[1,0].set(xscale='log',yscale='log',xlabel='visible eigenvalue',ylabel='mode risk / initial mode risk',title='B primary: unlearned modes at r=256'); ax[1,0].legend()
for group in ['fixed_adam_cold','fixed_proj_cold','fixed_adam_warm','fixed_proj_warm']:
    cs=[c for c in s['cells'] if c['problem']==1 and c['group']==group]
    ax[1,1].plot([c['ratio'] for c in cs],[c['risk'] for c in cs],'o-',label=group.replace('fixed_',''))
ax[1,1].set(xscale='log',yscale='log',xlabel='D/N',ylabel='best full-grid excess risk',title='A: radial and radius interventions'); ax[1,1].legend()
fig.suptitle('V2 frozen-grid diagnostics (triangles denote infinite timescales)')
for ext in ['png','svg']: fig.savefig(OUT/f'diagnostics.{ext}',dpi=170)
plt.close(fig)
with (OUT/'curves.csv').open() as f: curves=list(csv.DictReader(f))
fig,ax=plt.subplots(3,7,figsize=(21,9),layout='constrained')
styles={'learned_adam_cold':'-','learned_proj_cold':'--','learned_adam_warm':':','learned_proj_warm':'-.'}
for i,n in enumerate([32,64,128]):
    for j,(mode,r) in enumerate([(0,r) for r in [16,64,256,1024]]+[(1,r) for r in [16,64,256]]):
        for group,ls in styles.items():
            rows=[x for x in curves if int(x['problem'])==2 and int(x['N'])==n and int(x['mode'])==mode and int(x['ratio'])==r and x['group']==group]
            x=np.array([float(x['value']) for x in rows]); y=np.array([float(x['mean']) for x in rows])
            ax[i,j].plot(x,y,ls,label=group.replace('learned_',''))
        ax[i,j].set(xscale='symlog',yscale='log',title=f"N={n}, {'stream' if mode==0 else 'replay'}, r={r}")
        ax[i,j].set_xscale('symlog',linthresh=1e-6)
        if i==2: ax[i,j].set_xlabel('delta (including zero)')
        if j==0: ax[i,j].set_ylabel('excess risk')
ax[0,0].legend(fontsize=6); fig.suptitle('B: complete decay grids, learned amplitude (oracle and h grids in curves.csv)')
for ext in ['png','svg']: fig.savefig(OUT/f'risk_curves.{ext}',dpi=150)
plt.close(fig)
dump(OUT/'plot_runtime.json',dict(wall_seconds=time.monotonic()-START,utc=utc()))
print('Saved diagnostics and complete-grid risk figures.')
