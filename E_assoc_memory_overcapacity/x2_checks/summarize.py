import os
os.environ['MPLCONFIGDIR']=str(__import__('pathlib').Path(os.environ.get('X2_OUTPUT',str(__import__('pathlib').Path(__file__).resolve().parent)))/'mplconfig')
import json,math,time,hashlib,sys,resource
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtr,ndtri
from checks import ft_independent,gd,ft1,rank_coeff

ROOT=Path(__file__).resolve().parent.parent
H=Path(os.environ.get('X2_OUTPUT',str(Path(__file__).resolve().parent)))
start=time.monotonic()
read=lambda p:json.loads((H/p).read_text())
ss=read('scatter_summary.json'); ww=read('verified/w4_reanalysis.json')
cf=ft_independent(4,1-.75**(1/8))
raw=[json.loads(s) for s in (ROOT/'experiments/w4/W4_raw.jsonl').read_text().splitlines()]
lines=[]
def emit(s): lines.append(s); print(s)
emit('FT4 (bias; independent scatter integrated ALL-query moments)')
for d in [64,128,256,512]:
 for mode in ['fixed','matched']:
    r=next(r for r in ss if r['d']==d and r['method']=='FT4' and r['mode']==mode)
    beta=cf['msr1'] if mode=='fixed' else cf['matched_msr1']
    emit(f"{d} {mode}: {r['msr']-cf['mu']:+.9f} +/- {r['msr_se']:.9f}; predicted {beta/d:+.9f}; residual_SE={(r['msr']-cf['mu']-beta/d)/r['msr_se']:+.3f}")
emit('W4 M GD observed retrieval (seed SE), limiting and corrected prediction')
gp=gd(1,1.)
retr=[]
for d in [64,128,256]:
 rr=[r for r in raw if r['kind']=='M' and r['d']==d and r['method']=='GD']
 for key,p0,p1 in [('mean_rank',gp['mean_rank_0'],gp['mean_rank_1']),('top10',gp['recall_0'],gp['recall_1'])]:
    a=np.array([r[key] for r in rr]); se=a.std(ddof=1)/math.sqrt(len(a)); m=a.mean()
    rec=dict(d=d,key=key,observed=float(m),se=float(se),limit=p0,corrected=p0+p1/d)
    retr.append(rec);emit(str(rec))
emit('FT1 alpha=.14 rx=.6 ry=.7 coefficients and exact effective coefficients')
fc=[]
for k in [0,1]:
 c=ft1(k,.14,.6,.7)
 emit(str({s:c[s] for s in ['k','mu','m1','correct_var1','wrong_var1','mean_rank_1','recall_1']}))
 for r in read('verified/ft1_exact.json'):
    if r['rx']==.6 and r['k']==k: emit(str(r));fc.append(r)
with (H/'report_tables.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
with (H/'w4_retrieval.json').open('x') as f:json.dump(retr,f,indent=2)

fig,axs=plt.subplots(1,3,figsize=(14,4))
colors={'fixed':'#245aa8','matched':'#ce7320'}
for mode in ['fixed','matched']:
 r=[s for s in ss if s['method']=='FT4' and s['mode']==mode]
 x=np.array([s['d'] for s in r]); y=np.array([s['msr']-cf['mu'] for s in r]); se=np.array([s['msr_se'] for s in r])
 axs[0].errorbar(x,y,yerr=1.96*se,fmt='o',color=colors[mode],label=mode+' new exact moments')
 r=[s for s in ww if s['mode']==mode]
 axs[0].errorbar([s['d'] for s in r],[s['bias'] for s in r],yerr=[1.96*s['se'] for s in r],fmt='x',color=colors[mode],alpha=.6,label=mode+' W4')
 grid=np.linspace(64,512,100);beta=cf['msr1'] if mode=='fixed' else cf['matched_msr1']
 axs[0].plot(grid,beta/grid,color=colors[mode])
axs[0].axhline(0,color='gray',lw=.5);axs[0].set(title='FT4 old MSR bias',xlabel='d',ylabel='observed minus limit');axs[0].legend(fontsize=7)
nl=read('verified/nonlinear_exact.json')
for e,method in [(0.,'Sign1'),(.1,'Adam1_e.1')]:
 r=[s for s in nl if s['e']==e]
 axs[1].plot([s['d'] for s in r],[s['sample_variance']-1 for s in r],'-',label=method+' exact integral')
 r=[s for s in ss if s['method']==method]
 axs[1].errorbar([s['d'] for s in r],[s['sample_variance']-1 for s in r],yerr=[1.96*s['sample_variance_se'] for s in r],fmt='o',ms=3)
axs[1].set(title='One-step nonlinear correct variance',xlabel='d',ylabel='variance minus 1');axs[1].legend(fontsize=7)
r=read('rank_reduction.json')
axs[2].plot([s['d'] for s in r],[s['rank_beta'] for s in r],'o-',label='d (mean rank - limit)')
axs[2].axhline(gp['mean_rank_1'],ls='--',label='derived coefficient')
naive=math.exp(-.25)/math.sqrt(4*math.pi)*1.5
axs[2].axhline(naive,ls=':',label='Gaussian moments only')
axs[2].set(title='GD1 rank correction',xlabel='d');axs[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(H/'finite_d.png',dpi=160);fig.savefig(H/'finite_d.pdf');plt.close(fig)
cg=Path('/sys/fs/cgroup')/next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')).lstrip('/')
with (H/'summary_metadata.json').open('x') as f:json.dump(dict(seconds=time.monotonic()-start,python=sys.executable,rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
 controls={s:(cg/s).read_text().strip() for s in ['memory.max','memory.swap.max','memory.peak']},process_threads=len(list(Path('/proc/self/task').iterdir()))),f,indent=2)
