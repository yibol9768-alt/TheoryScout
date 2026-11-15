"""Frozen EXP3 decisions, independent sufficient-statistic audit, static figures."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import datetime
import json
import math
import resource
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig')
os.environ['XDG_CACHE_HOME']=str(HERE/'cache')
sys.path.insert(0,str(HERE))
import numpy as np
import model
from run import verify_frozen


def aggregate(rows,part='full',control=False):
    stats=[r['metrics'][part]['control'] if control else r['metrics'][part] for r in rows]
    def total(k): return sum(s[k] for s in stats)
    den=total('prediction_energy');noise=total('mean_noise');signal=total('signal')
    observed=sum((np.array(s['noise_output']) for s in stats),np.zeros((4,4)))
    predicted=sum((np.array(r['predicted_noise_output']) for r in rows),np.zeros((4,4)))
    return dict(error=math.sqrt(max(0.,total('prediction_error'))/den),
        gain=total('gain_numerator')/den,snr=math.sqrt(den/noise),
        resolved=bool(signal>0 and den/noise>=100),signal=signal,
        noise_risk_ratio=total('noise_risk')/sum(r['predicted_noise_risk'] for r in rows),
        noise_output_error=float(np.linalg.norm(observed-predicted)/np.linalg.norm(predicted)),
        totalrisk_ratio=total('total_risk')/sum(r['predicted_totalrisk'] for r in rows),
        prediction_energy=den,prediction_error=total('prediction_error'),mean_noise=noise,
        scalar_error=math.sqrt(max(0.,total('scalar_error'))/den))


def decide(values,predicate,precision=False):
    bad=[i for i,x in enumerate(values) if not predicate(x)]
    confirmed=[i for i in bad if not precision or values[i]['resolved']]
    unresolved=[i for i,x in enumerate(values) if precision and not x['resolved']]
    return dict(status='失败' if confirmed else ('无法判断' if unresolved else '通过'),
        violations=len(bad),confirmed_violations=len(confirmed),unresolved=len(unresolved),total=len(values))


def main():
    started=time.monotonic()
    verify_frozen()
    scope=sys.argv[1]
    props=subprocess.check_output(['systemctl','--user','show',scope,'-p','MemoryMax','-p','MemorySwapMax'],text=True)
    assert 'MemoryMax=2147483648' in props and 'MemorySwapMax=0' in props
    (HERE/'analysis_scope_properties.txt').write_text(props)
    folder=HERE/'formal'
    def read(name): return json.loads((folder/(name+'.json')).read_text())
    manifest=read('manifest');hold=read('holdout');training=read('training')
    sman=json.loads((HERE/'smoke/manifest.json').read_text())
    smoke=json.loads((HERE/'smoke/smoke.json').read_text())
    assert manifest['status']==sman['status']=='complete'
    assert len(hold)==384 and len(training)==16
    assert len({(r['seed'],r['n'],r['alpha'],r['ai'],r['b']) for r in hold})==384
    tests=[s for r in training for s in r['tests']]
    assert len(tests)==64
    assert {r['seed'] for r in hold}=={6101,6102,6103,6104}
    assert {r['seed'] for r in training}=={6201,6202}
    meanerr=cov_error=prediction_error=0.
    for row in hold+tests:
        with np.load(folder/row['file']) as z:
            for prefix,blocks in (('','q_blockmeans'),('u_','u_blockmeans')):
                meanerr=max(meanerr,float(np.max(np.abs(z[prefix+'mean_full']-z[blocks].mean(axis=0)))))
                meanerr=max(meanerr,float(np.max(np.abs(z[prefix+'mean_full']-(z[prefix+'mean_first']+z[prefix+'mean_second'])/2))))
            noise=float(np.einsum('n,nii->',z['lam'],z['rowcov_full']))
            assert abs(noise-row['metrics']['full']['noise_risk'])<1e-11
            p=dict(lam=z['lam'],cov=z['Sigma0'])
            m=model.bridge(p,z['E'],row['b'])
            prediction_error=max(prediction_error,float(np.max(np.abs(m['pred']-z['prediction']))))
            err=model.norm2(z['mean_full']-m['pred'],z['lam'])-noise/row['batches']
            assert abs(err-row['metrics']['full']['prediction_error'])<1e-11
            if 'full_covariance' in z:
                cov_error=max(cov_error,abs(float(np.trace(z['full_covariance']))-float(np.trace(z['rowcov_full'].sum(axis=0)))))
    assert max(meanerr,cov_error,prediction_error)<1e-11
    groups=[]
    for alpha in (1.5,3.):
        for n in (16,32,64,128):
            for ai,amp in enumerate((1.,.25,.0625)):
                rows=[r for r in hold if (r['alpha'],r['n'],r['ai'])==(alpha,n,ai)]
                assert len(rows)==16
                gram_error=math.sqrt(max(0.,sum(r['gram_error'] for r in rows))/sum(r['gram_energy'] for r in rows))
                groups.append(dict(alpha=alpha,n=n,amplitude=amp,
                    parts={part:aggregate(rows,part) for part in ('full','first','second')},
                    control=aggregate(rows,control=True),gram_error=gram_error))
    online=[]
    for alpha in (1.5,3.):
        for n in (16,64):
            for b in (4,64):
                for t in model.TIMES:
                    rows=[r for r in tests if (r['alpha'],r['n'],r['b'],r['T'])==(alpha,n,b,t)]
                    assert len(rows)==2
                    stat=aggregate(rows)
                    online.append(dict(alpha=alpha,n=n,b=b,T=t,metrics=stat,
                        measured_increment=sum(r['measured_increment'] for r in rows)/2,
                        predicted_increment=sum(r['predicted_increment'] for r in rows)/2))
    allparts=[g['parts'][part] for g in groups for part in ('full','first','second')]
    full=[g['parts']['full'] for g in groups]
    controls=[g['control'] for g in groups]
    overall=aggregate(hold)
    improvement=1-overall['error']/max(overall['scalar_error'],1e-300)
    p0=all(g['gram_error']<=.10 for g in groups) and all(c['error']<=.10 and .8<=c['noise_risk_ratio']<=1.2 for c in controls)
    gates=dict(P0=dict(status='通过' if p0 else '失败'),
        P1=decide(allparts,lambda x:x['error']<=.25,True),
        P2=decide(allparts,lambda x:abs(x['gain']-1)<=.20,True),
        P3=dict(status='通过' if improvement>=.25 else '失败',improvement=improvement),
        P4=decide(full,lambda x:.7<=x['noise_risk_ratio']<=1.3 and x['noise_output_error']<=.30),
        P5_drift=decide([x['metrics'] for x in online],lambda x:x['error']<=.25 and abs(x['gain']-1)<=.20,True),
        P5_noise=decide([x['metrics'] for x in online],lambda x:.7<=x['noise_risk_ratio']<=1.3 and x['noise_output_error']<=.30))
    all_reachable=True;max_stability=0.
    for r in training:
        p=model.problem(r['seed'],r['n'],r['alpha'])
        radius=np.linalg.norm(p['teacher'])+np.sqrt(4)*r['eta']*r['steps']
        upper=(1+1/r['b'])*radius**2+np.sum(p['lam'])/r['b']*(radius**2+np.linalg.norm(p['cov'],2))
        a_star=p['lam'][-1]/np.sqrt(upper)
        max_stability=max(max_stability,r['eta']*a_star)
        r['risk_bound_constants']=dict(Mplus=upper,a_star=a_star)
        for pt in r['rows']:
            all_reachable &= pt['distance']<=pt['reachable_radius']+1e-10
            assert pt['D']==r['b']*pt['T'] and pt['nD']==r['n']*pt['D']
    gates['P6']=dict(status='通过' if all_reachable and max_stability<=1 else '失败',
        all_steps_asserted=True,all_saved_states_reachable=bool(all_reachable),max_eta_a_star=max_stability)
    costs={}
    for name,rows in (('holdout',hold),('online_training',training),('online_diagnostics',tests)):
        costs[name]={k:sum(r[k] for r in rows) for k in ('samples','sample_seconds','gradient_seconds','svd_seconds','bridge_seconds','elapsed_seconds')}
    prior=sman['elapsed_seconds']+manifest['elapsed_seconds']
    result=dict(groups=groups,online_groups=online,overall=overall,gates=gates,costs=costs,
        audit=dict(mean_reconstruction_max=meanerr,covariance_trace_max=cov_error,prediction_reconstruction_max=prediction_error),
        gram=dict(bad_fraction_min=min(r['gram_bad_fraction'] for r in hold+tests),
            bad_fraction_max=max(r['gram_bad_fraction'] for r in hold+tests),
            bad_fraction_median=float(np.median([r['gram_bad_fraction'] for r in hold+tests])),
            states_with_bad_fraction_le_point1=sum(r['gram_bad_fraction']<=.1 for r in hold+tests),
            states=len(hold+tests),
            empirical_delta2_min=min(r['empirical_delta2'] for r in hold+tests),
            empirical_delta2_max=max(r['empirical_delta2'] for r in hold+tests)),
        smoke={k:v for k,v in smoke.items() if k not in ('local','training')},
        prior_compute_seconds=prior,peak_rss_bytes=max(sman['peak_rss_bytes'],manifest['peak_rss_bytes']),
        training_bound_constants=[dict(seed=r['seed'],n=r['n'],alpha=r['alpha'],b=r['b'],**r['risk_bound_constants']) for r in training])
    lines=['# EXP3 冻结规则的完整分组表','',
        '| α | n | a | 部分 | 漂移误差 | 增益 | SNR | 噪声迹比 | 输出噪声误差 |',
        '|---|---|---|---|---|---|---|---|---|']
    for g in groups:
        for part,s in g['parts'].items():
            lines.append(f"| {g['alpha']} | {g['n']} | {g['amplitude']} | {part} | {s['error']:.6f} | {s['gain']:.6f} | {s['snr']:.3f} | {s['noise_risk_ratio']:.6f} | {s['noise_output_error']:.6f} |")
    lines+=['','## 真实路径独立 batch 诊断','',
        '| α | n | b | t | 漂移误差 | 增益 | SNR | 噪声迹比 | 输出噪声误差 | 实测 ΔV | 预测 ΔV |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for g in online:
        s=g['metrics']
        lines.append(f"| {g['alpha']} | {g['n']} | {g['b']} | {g['T']} | {s['error']:.6f} | {s['gain']:.6f} | {s['snr']:.3f} | {s['noise_risk_ratio']:.6f} | {s['noise_output_error']:.6f} | {g['measured_increment']:.8f} | {g['predicted_increment']:.8f} |")
    (HERE/'TABLES.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.5))
    x=np.arange(len(groups))
    axs[0].plot(x,[s['error'] for s in full],'o-',label='matrix')
    axs[0].plot(x,[s['scalar_error'] for s in full],'x--',label='scalar ablation')
    axs[0].axhline(.25,c='k',ls=':');axs[0].set_ylabel('relative drift error');axs[0].legend()
    axs[1].plot(x,[s['noise_risk_ratio'] for s in full],'o-');axs[1].axhspan(.7,1.3,alpha=.15);axs[1].set_ylabel('risk noise ratio Q / GB')
    axs[2].plot(x,[s['noise_output_error'] for s in full],'o-');axs[2].axhline(.30,c='k',ls=':');axs[2].set_ylabel('output noise error')
    for ax in axs: ax.set_xlabel('fixed (alpha,n,amplitude) group');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(HERE/'bridge_holdout.png',dpi=170);fig.savefig(HERE/'bridge_holdout.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for alpha in (1.5,3.):
        for b in (4,64):
            rows=[r for r in training if r['alpha']==alpha and r['b']==b]
            risk=[np.mean([r['rows'][j]['risk'] for r in rows]) for j in range(4)]
            axs[0].loglog(np.array(model.TIMES)+1,risk,'o-',label=f'alpha={alpha},b={b}')
    axs[0].set_xlabel('T+1');axs[0].set_ylabel('mean risk (n,seed pooled)');axs[0].legend(fontsize=7)
    axs[1].scatter([g['predicted_increment'] for g in online],[g['measured_increment'] for g in online],s=15)
    lim=[min(g['predicted_increment'] for g in online),max(g['predicted_increment'] for g in online)]
    axs[1].plot(lim,lim,'k:');axs[1].set_xlabel('predicted fresh-batch risk increment');axs[1].set_ylabel('measured increment')
    fig.tight_layout();fig.savefig(HERE/'online_risk.png',dpi=170);fig.savefig(HERE/'online_risk.pdf');plt.close(fig)
    result['analysis_elapsed_seconds']=time.monotonic()-started
    result['total_compute_seconds']=prior+result['analysis_elapsed_seconds']
    result['analysis_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result['analysis_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    gates['P7']=dict(status='通过' if result['total_compute_seconds']<=3600 else '失败',
        local_units=len(hold),online_paths=len(training),online_snapshot_units=len(tests),
        threads=1,scope_properties=props,total_compute_seconds=result['total_compute_seconds'])
    (HERE/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(gates=gates,overall=overall,gram=result['gram'],total_compute_seconds=result['total_compute_seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
