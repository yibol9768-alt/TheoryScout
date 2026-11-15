"""Independent AD0 summaries, scalar quadrature, and load sweep. New outputs only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import resource
import time
import numpy as np
from scipy.special import ndtr

OUT=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CPU,(900,920))
wall,cpu=time.monotonic(),time.process_time()

def emit(name,obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

main=json.loads((OUT/'AD0_main_results.json').read_text())
print('MAIN representative rows',flush=True)
for r in main['summary']:
    if r['regime']=='critical' and r['alg']=='GD' and (r['eta']==.5 or r['T']=='infinity') and r['T'] in ('1','2','4','8','128','infinity'):
        print({k:r[k] for k in ('d','n','eta','T','top1','top1_se','top5','cosine_Z')},flush=True)
print('EARLY STOP paired:',flush=True)
for d in (64,128,256):
    rs=[r for r in main['paired'] if r['d']==d and r['regime']=='critical' and r['eta']==.5]
    best=max(rs,key=lambda r:r['top1_minus_infinity'])
    print('d',d,'best .5',best,flush=True)

def scalar(alpha,T=None,parity=None):
    edges=np.arange(0,24+alpha,alpha)
    a=q=0.
    for lo,hi in zip(edges[:-1],edges[1:]):
        g=(lo+hi)/2
        if T is None:
            h=alpha*(2*np.floor((g/alpha-parity)/2+.5)+parity)
        else:
            h=0.
            for _ in range(T):
                h+=alpha*np.sign(g-h)
        phi=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
        a+=2*h*(phi(lo)-phi(hi))
        q+=2*h*h*(ndtr(hi)-ndtr(lo))
    return {'alpha':alpha,'T':T,'parity':parity,'a':float(a),'q':float(q),'c':float(a/np.sqrt(q))}

scalar_rows=[scalar(alpha,T=T) for alpha in (.6,1.5) for T in (1,2,3,4,8,9,32,33,64,65)]
scalar_rows +=[scalar(alpha,parity=p) for alpha in (.6,1.5) for p in (0,1)]
emit('AD0_scalar.json',scalar_rows)
print('SCALAR limits',[r for r in scalar_rows if r['T'] is None],flush=True)

# Deterministic trace-efficiency audit on a stable spectrum, including oscillatory eta.
eig=np.array([.9,1.1])
traces=[]
for eta in (.5,1.,1.5):
    for T in (1,2,3,4,8,32,128):
        p=(1-(1-eta*eig)**T)/eig
        m1=np.mean(eig)
        eff=float(np.mean(eig*p)/np.sqrt(m1*np.mean(eig*p*p)))
        slope=float(T*eta*(1-eta)**(T-1)/(1-(1-eta)**T)-1)
        traces.append({'eta':eta,'T':T,'weighted_efficiency':eff,'log_slope':slope})
        if eff>1+1e-14:
            raise AssertionError('Cauchy-Schwarz failure')
emit('AD0_filter_audit.json',traces)

rng=np.random.default_rng(20261107003)
steps=[1,2,3,4,8,32]
loads=[.2,.4,.6,.8,1.0]
reps=24
design={'seed':20261107003,'loads_nlogn_d2':loads,'d':[64,128,256],
        'reps':reps,'queries':64,'eta':.5,'steps':steps+['infinity'],
        'method':'explicit full-array ranks; all endpoints paired on each dataset'}
emit('AD0_load_design.json',design)
rows=[]
with (OUT/'AD0_load_rows.jsonl').open('x',encoding='utf-8') as stream:
    for d in design['d']:
        for load in loads:
            candidates=np.arange(d+2,2*d*d)
            n=int(candidates[np.argmin(np.abs(candidates*np.log(candidates)/d**2-load))])
            for rep in range(reps):
                X=rng.standard_normal((d,n))/np.sqrt(d)
                Y=rng.standard_normal((d,n))/np.sqrt(d)
                S=d/n*(X@X.T); Z=d/np.sqrt(n)*(Y@X.T)
                eig,V=np.linalg.eigh(S); ZV=Z@V
                ids=rng.choice(n,size=64,replace=False)
                VX=V.T@X[:,ids]
                for T in steps+['infinity']:
                    p=1/eig if T=='infinity' else (1-(1-.5*eig)**T)/eig
                    pred=(ZV*p)@VX
                    scores=Y.T@pred
                    correct=scores[ids,np.arange(64)]
                    ranks=1+np.sum(scores>correct[None,:],axis=0)
                    row={'d':d,'load':load,'n':n,'actual_load':float(n*np.log(n)/d**2),
                         'rep':rep,'T':T,'top1':float(np.mean(ranks<=1)),
                         'top5':float(np.mean(ranks<=5))}
                    rows.append(row);stream.write(json.dumps(row)+'\n')
            stream.flush()
        print('load sweep completed',d,flush=True)
summary=[]
for d in design['d']:
    for load in loads:
        base={r['rep']:r for r in rows if r['d']==d and r['load']==load and r['T']=='infinity'}
        for T in steps+['infinity']:
            rs=[r for r in rows if r['d']==d and r['load']==load and r['T']==T]
            vals=np.array([r['top1'] for r in rs]);dif=np.array([r['top1']-base[r['rep']]['top1'] for r in rs])
            summary.append({'d':d,'load':load,'n':rs[0]['n'],'T':T,
                            'top1':float(vals.mean()),'se':float(vals.std(ddof=1)/np.sqrt(reps)),
                            'minus_infinity':float(dif.mean()),'paired_se':float(dif.std(ddof=1)/np.sqrt(reps))})
result={'design':design,'summary':summary,'cpu_seconds':time.process_time()-cpu,
        'wall_seconds':time.monotonic()-wall,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'Threads':[x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')]}
emit('AD0_load_results.json',result)
for d in design['d']:
    ss=[r for r in summary if r['d']==d and r['T']!='infinity']
    best=max(ss,key=lambda r:r['minus_infinity'])
    print('load max apparent early benefit',best,flush=True)
print({k:result[k] for k in ('cpu_seconds','wall_seconds','max_rss_kib','Threads')},flush=True)

# Publication-style standalone figure; a missing optional matplotlib does not alter results.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(12,6.5))
    for col,d in enumerate(design['d']):
        ax=axs[0,col]
        for eta in (.5,1.):
            rs=[r for r in main['summary'] if r['d']==d and r['regime']=='critical' and r['alg']=='GD' and r['eta']==eta]
            rs.sort(key=lambda r:int(r['T']))
            ax.errorbar([int(r['T']) for r in rs],[r['top1'] for r in rs],
                        yerr=[r['top1_se'] for r in rs],label=f'GD eta={eta}',marker='.',capsize=2)
        end=next(r for r in main['summary'] if r['d']==d and r['regime']=='critical' and r['T']=='infinity')
        ax.axhline(end['top1'],color='black',ls='--',label='least squares')
        ax.set_xscale('log',base=2);ax.set_xlabel('T');ax.set_ylabel('top-1 recall')
        ax.set_title(f'd={d}, critical-window load');ax.legend(fontsize=7)
        ax=axs[1,col]
        for T in (1,2,4,32,'infinity'):
            rs=[r for r in summary if r['d']==d and r['T']==T]
            ax.errorbar([r['load'] for r in rs],[r['top1'] for r in rs],yerr=[r['se'] for r in rs],
                        marker='.',label=f'T={T}',capsize=2)
        ax.set_xlabel('n log(n) / d^2');ax.set_ylabel('top-1 recall');ax.legend(fontsize=7)
    fig.suptitle('Full training arrays; dataset-level SE; stable GD curves')
    fig.tight_layout()
    for name in ('AD0_gd_curves.svg','AD0_gd_curves.png'):
        path=OUT/name
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path,dpi=150)
    plt.close(fig)
except ImportError as exc:
    emit('AD0_plot_unavailable.json',{'reason':str(exc)})
