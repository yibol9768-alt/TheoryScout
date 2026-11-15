import sys, os, json, time, argparse, subprocess
from pathlib import Path
sys.dont_write_bytecode=True
from W4_core import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(out/'mplconfig'); os.environ['XDG_CACHE_HOME']=str(out/'cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=time.monotonic(); ri=resource_info(); d=12 if args.smoke else 32
    x,y,s,z=mdata(d,0,9); ids=rng(9,d,0,4).choice(d*d,min(128,d*d),replace=False)
    mr=[]
    for name,u,q,c,hp in cases(z,s):
        diag=diag_scores(u,x,y,q); r=retrieve(u,x,y,ids,q,c,diag)
        # Independent dense implementation of rank and diagonal, small n only.
        dense=y.T@u@x/math.sqrt(q); truth=np.diag(dense).copy()
        dense[np.arange(d*d),np.arange(d*d)]=-np.inf
        rank=np.sum(dense[:,ids]>truth[ids],axis=0)/(d*d-1)
        assert np.array_equal(rank,np.array(r['ranks']))
        assert np.max(np.abs(diag-truth))<1e-11
        mr.append(dict(method=name,c=c,full_variance=float(diag.var(ddof=1)),
                       reduction_mse=float(np.mean((u-hp)**2)/q),**r))
    er=ext_point(d,4)+ext_point(d,16); fr=ft(d,0,9)
    ec=brentq(lambda e:efficiency(e)[2]-8/(3*math.pi),.001,.2)
    cert='smoke skips certificate; full replay runs the interval certificate'
    if not args.smoke:
        cert=subprocess.check_output([sys.executable,'-B',str(HERE/'W4_certificate.py'),'certificate'],text=True)
        assert 'CERTIFIED: 0.050156130219 < e_* < 0.050156130220' in cert
    (out/'certificate.txt').write_text(cert)
    fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    names=[r['method'] for r in mr]
    axs[0,0].plot(names,[r['mean_rank'] for r in mr],'o',label='fresh simulation')
    axs[0,0].plot(names,[ndtr(-r['c']/math.sqrt(2)) for r in mr],'x',label='limit')
    axs[0,0].set_title('F1: mean rank'); axs[0,0].legend()
    axs[0,1].bar(names,[r['reduction_mse'] for r in mr]); axs[0,1].set_title('F2: actual/proxy MSE')
    rr=next(r for r in er if r['gamma']==4 and r['tau']==0 and r['method']=='Sign')
    axs[0,2].plot([0,1],[rr['variance'],1+4/(math.pi*4)],'o'); axs[0,2].set(xticks=[0,1],xticklabels=['sample','limit'],title='F3: Sign wrong variance, gamma4')
    a,q,c=efficiency(.1)
    aa=next(r for r in er if r['gamma']==16 and r['tau']==0 and r['method']=='Adam_e0.1')
    mm=next(r for r in er if r['gamma']==16 and r['tau']==0 and r['method']=='Muon')
    axs[1,0].plot([0,1],[c,8/(3*math.pi)],'o'); axs[1,0].set(xticks=[0,1],xticklabels=['c(e=.1)','8/(3pi)'],title=f'F4: e*={ec:.8f}; ratio={aa["msr"]/mm["msr"]:.3f}')
    for method in ['Adam_e0.1','Muon']:
        rr=[r for r in er if r['gamma']==16 and r['method']==method]
        axs[1,1].plot([r['tau'] for r in rr],[r['msr_scaled'] for r in rr],'o-',label=method)
    axs[1,1].set_title('F5: parameter noise'); axs[1,1].legend()
    axs[1,2].plot([r['mode'] for r in fr],[r['old']['msr'] for r in fr],'o')
    axs[1,2].axhline(fr[0]['prediction'],ls='--'); axs[1,2].set_title('F6: old MSR, independent GD4')
    fig.savefig(out/'representatives.png',dpi=150); fig.savefig(out/'representatives.pdf'); plt.close(fig)
    result=dict(d=d,seed=0,M=mr,EXT=er,FT=fr,ecrit=ec,checks=dict(dense_rank=True,dense_correct_score=True,GD_spectral=True,interval_certificate=not args.smoke),
                resources=ri,elapsed_seconds=time.monotonic()-start,resources_after=resource_info())
    (out/'raw.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(seconds=result['elapsed_seconds'],checks=result['checks'])),flush=True)

if __name__=='__main__': main()
