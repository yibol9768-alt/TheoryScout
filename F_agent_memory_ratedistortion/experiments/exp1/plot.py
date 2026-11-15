"""Export scientific endpoint plots; fixed-key seed lines, not population CIs."""
import os
from pathlib import Path
root=Path(__file__).resolve().parent
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']=str(root/'.mplconfig')
import resource,json
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
records=[json.loads(x) for x in (root/'formal/results.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True,layout='constrained')
for ax,panel in zip(axes,('E1','E2_tau03','E2_tau1')):
    layer='E1' if panel=='E1' else 'E2';dd=[d for d in records if d['layer']==layer and d['G']==32 and d['dist']==0 and d['epsilon']==4/15 and d['nu_s']==0 and d['nu_r']==0]
    if panel=='E1':dd=[d for d in dd if d['channel'][1:]==[1.,1.]]
    else:dd=[d for d in dd if d['channel'][0]==(.3 if panel=='E2_tau03' else 1.)]
    values={}
    for seed in (0,1,2):
        seeddata=sorted([d for d in dd if d['seed']==seed],key=lambda d:d['channel'][0 if layer=='E1' else 1]);xx=[];yy=[]
        for d in seeddata:
            x=d['channel'][0 if layer=='E1' else 1]; rr={r['policy']:r for r in d['rows'] if r['policy'].startswith('full_')}; obs=d['paired']['full_summary-full_raw']['mean'];pred=rr['full_summary']['predicted_accuracy']-rr['full_raw']['predicted_accuracy']
            values.setdefault(x,[]).append((obs,pred));xx.append(x);yy.append(obs)
        if xx:ax.plot(xx,yy,color='#2878b5',alpha=.3,linewidth=1)
    xs=sorted(values);means=np.array([np.mean(values[x],axis=0) for x in xs])
    if len(xs):
        ax.plot(xs,means[:,0],'-o',color='#2878b5',label='Observed (3-seed mean)')
        ax.plot(xs,means[:,1],'--s',color='#c45e25',label='Independent calibration prediction')
    if layer=='E1':
        exact=[.5*(lam+1)*(.6/(lam+32)-1/(lam+128)) for lam in xs];ax.plot(xs,exact,':',color='black',label='Known generator formula');ax.set_xscale('log');ax.set_xlabel('Signal lambda');ax.set_title('E1: controlled Gumbel channel')
    else:ax.set_xlabel('Query noise sigma');ax.set_title('E2: frozen geometry, tau='+('.3' if panel=='E2_tau03' else '1.0'))
    ax.axhline(0,color='gray',linewidth=.8);ax.grid(alpha=.15);ax.set_xticks(xs);ax.set_xticklabels([str(x) for x in xs])
axes[0].set_ylabel('Accuracy(summary) - accuracy(raw)');axes[0].legend(fontsize=7,loc='best')
fig.suptitle('G=32, k=4, beta=0.6; thin lines are fixed-key seed realizations',fontsize=11)
fig.savefig(root/'endpoint_accuracy.svg');fig.savefig(root/'endpoint_accuracy.png',dpi=220);print('exported endpoint_accuracy.svg/png')
