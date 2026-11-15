"""Read frozen formal data, audit, apply predeclared thresholds, export figures."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import hashlib
import json
import math
import resource
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


def aggregate(records, part='full', control=False):
    rows=[r['metrics'][part]['control'] if control else r['metrics'][part] for r in records]
    def total(key):
        return sum(s[key] for s in rows)
    den=total('prediction_energy')
    signal=total('signal')
    noise=total('mean_noise')
    result=dict(relative_error=math.sqrt(max(0.,total('prediction_error'))/den),
                gain_ratio=total('gain_numerator')/den,
                snr=math.sqrt(den/noise),signal=signal,
                prediction_energy=den,mean_noise=noise,
                fixed_relative_error=math.sqrt(max(0.,total('fixed_error'))/total('fixed_energy')))
    for name in ('shape','orth','coupling'):
        result[name+'_error']=math.sqrt(max(0.,total(name+'_error'))/max(signal,1e-300))
    measured_risk=total('noise_risk')
    predicted_risk=sum(s['predicted_noise_risk'] for s in rows) if control else sum(r['predicted_noise_risk'] for r in records)
    result['noise_risk_ratio']=measured_risk/predicted_risk
    if not control:
        observed=sum((np.array(s['noise_output']) for s in rows),np.zeros((4,4)))
        predicted=sum((np.array(r['predicted_noise_output']) for r in records),np.zeros((4,4)))
        result['noise_output_error']=float(np.linalg.norm(observed-predicted)/np.linalg.norm(predicted))
    result['resolved']=bool(signal>0 and result['snr']>=10)
    return result


def status(checks, resolved=True):
    return '失败' if not all(checks) else ('通过' if resolved else '无法判断')


def main():
    started=time.monotonic()
    verify_frozen()
    folder=HERE/'formal'
    def read(name):
        return json.loads((folder/(name+'.json')).read_text())
    manifest=read('manifest')
    assert manifest['status']=='complete'
    hold=read('holdout');cal=read('calibration')
    training=read('training');tc=read('training_calibration');locked=read('locked_model')
    smoke=json.loads((HERE/'smoke'/'smoke.json').read_text())
    sman=json.loads((HERE/'smoke'/'manifest.json').read_text())
    assert sman['status']=='complete'
    assert len(hold)==384 and len(cal)==16 and len(training)==96 and len(tc['records'])==40
    keys=lambda rows:{(r['seed'],r['n'],r['alpha'],r.get('ai',-1),r['b']) for r in rows}
    assert len(keys(hold))==384 and len(keys(cal))==16 and len(keys(training))==96
    assert {r['seed'] for r in hold}=={2001,2002,2003,2004}
    assert not ({r['seed'] for r in hold}&{r['seed'] for r in cal})
    # Independent checks from saved first/second moments and block means.
    maximum_mean_error=maximum_noise_error=maximum_cov_error=0.
    for row in hold+cal:
        with np.load(folder/row['file']) as z:
            maximum_mean_error=max(maximum_mean_error,float(np.max(np.abs(z['mean_full']-(z['mean_first']+z['mean_second'])/2))))
            maximum_mean_error=max(maximum_mean_error,float(np.max(np.abs(z['mean_full']-z['q_blockmeans'].mean(axis=0)))))
            rc=z['rowcov_full']
            noise=float(np.einsum('n,nii->',z['lam'],rc))
            maximum_noise_error=max(maximum_noise_error,abs(noise-row['metrics']['full']['noise_risk']))
            if 'full_covariance' in z:
                maximum_cov_error=max(maximum_cov_error,abs(float(np.trace(z['full_covariance']))-float(np.trace(rc.sum(axis=0)))))
            # Recompute locked predictions and full-sample de-noised errors independently.
            if row['seed']>=2000:
                p=model.problem(row['seed'],row['n'],row['alpha'])
                m=model.moments(p,z['E'],row['b'],locked['state']['nu'])
                pred=locked['state']['c']*m['gamma']*m['A']
                err=model.norm2(z['mean_full']-pred,z['lam'])-noise/row['batches']
                assert abs(err-row['metrics']['full']['prediction_error'])<1e-11
    assert maximum_mean_error<1e-11 and maximum_noise_error<1e-11 and maximum_cov_error<1e-11
    for r in training+tc['records']:
        assert all(pt['D']==r['b']*pt['T'] and pt['nD']==r['n']*pt['D'] for pt in r['rows'])
        assert r['rows'][-1]['T']==r['steps']
    groups=[]
    for alpha in (1.5,3.):
        for n in (16,32,64,128):
            for ai,amp in enumerate((1.,.25,.0625)):
                rows=[r for r in hold if (r['alpha'],r['n'],r['ai'])==(alpha,n,ai)]
                assert len(rows)==16
                groups.append(dict(alpha=alpha,n=n,amplitude=amp,
                    parts={part:aggregate(rows,part) for part in ('full','first','second')},
                    control=aggregate(rows,control=True)))
    allparts=[g['parts'][part] for g in groups for part in ('full','first','second')]
    full=[g['parts']['full'] for g in groups]
    controls=[g['control'] for g in groups]
    overall=aggregate(hold)
    improvement=1-overall['relative_error']/overall['fixed_relative_error']
    gamma_groups=[]
    for alpha in (1.5,3.):
        for b in (4,16,64,256):
            rows=[r for r in training if (r['alpha'],r['b'])==(alpha,b)]
            points=[p for r in rows for p in r['rows']]
            gs=[p['gamma'] for p in points]
            gamma_groups.append(dict(alpha=alpha,b=b,min=min(gs),max=max(gs),ratio=max(gs)/min(gs),
                max_proxy_stability=max(p['proxy_stability'] for p in points)))
    dimension_slopes=[]
    for alpha in (1.5,3.):
        for b in (4,16,64,256):
            for t in model.TIMES:
                pairs=[(r['n'],p['gamma']) for r in training if (r['alpha'],r['b'])==(alpha,b)
                       for p in r['rows'] if p['T']==t]
                slope=float(np.polyfit(np.log([x[0] for x in pairs]),np.log([x[1] for x in pairs]),1)[0])
                dimension_slopes.append(dict(alpha=alpha,b=b,T=t,slope=slope))
    time_slopes=[]
    for r in training:
        slope=float(np.polyfit(np.log([p['T']+1 for p in r['rows']]),np.log([p['gamma'] for p in r['rows']]),1)[0])
        time_slopes.append(dict(seed=r['seed'],alpha=r['alpha'],n=r['n'],b=r['b'],slope=slope))
    trajectory_points=[p for r in training for p in r['rows']]
    family_fraction=sum(p['family']['in_relaxed_family'] for p in trajectory_points)/len(trajectory_points)
    costs={}
    for name,rows in (('local_calibration',cal),('local_holdout',hold),('lr_calibration',tc['records']),('online',training)):
        costs[name]=dict(configurations=len(rows),samples=sum(r['samples'] for r in rows),
            elapsed_seconds=sum(r['elapsed_seconds'] for r in rows),
            **{key:sum(r.get(key,0.) for r in rows) for key in
               ('sample_seconds','gradient_seconds','svd_seconds','gamma_seconds')})
    frozen_match=manifest['frozen']==sman['frozen']
    limits_ok=all(m['scope_properties'].find('MemoryMax=2147483648')>=0 and
                  m['scope_properties'].find('MemorySwapMax=0')>=0 and
                  all(int(v)<=2 for v in m['threads'].values()) for m in (manifest,sman))
    gates={
      'P0':dict(status=status([smoke['rotation_max_error']<1e-11,smoke['rank_deficient_pass'],
             smoke['analytic_trace_identity_pass']]+[s['relative_error']<=.10 and .8<=s['noise_risk_ratio']<=1.2 for s in controls]),
             max_control_error=max(s['relative_error'] for s in controls),
             control_noise_range=[min(s['noise_risk_ratio'] for s in controls),max(s['noise_risk_ratio'] for s in controls)]),
      'P1':dict(status=status([s['relative_error']<=.25 for s in allparts],all(s['resolved'] for s in allparts)),
                failures=sum(s['relative_error']>.25 for s in allparts),max=max(s['relative_error'] for s in allparts)),
      'P2':dict(status=status([abs(s['gain_ratio']-1)<=.20 for s in allparts],all(s['resolved'] for s in allparts)),
                failures=sum(abs(s['gain_ratio']-1)>.20 for s in allparts),
                range=[min(s['gain_ratio'] for s in allparts),max(s['gain_ratio'] for s in allparts)]),
      'P3':dict(status=status([s['shape_error']<=.15 for s in allparts],all(s['resolved'] for s in allparts)),
                failures=sum(s['shape_error']>.15 for s in allparts),max=max(s['shape_error'] for s in allparts)),
      'P4':dict(status=status([s['orth_error']<=.15 and s['coupling_error']<=.15 for s in allparts],all(s['resolved'] for s in allparts)),
                failures=sum(s['orth_error']>.15 or s['coupling_error']>.15 for s in allparts),
                max_orth=max(s['orth_error'] for s in allparts),max_coupling=max(s['coupling_error'] for s in allparts)),
      'P5':dict(status=status([improvement>=.25]),relative_improvement=improvement,
                state_error=overall['relative_error'],fixed_error=overall['fixed_relative_error']),
      'P6':dict(status=status([.7<=s['noise_risk_ratio']<=1.3 and s['noise_output_error']<=.30 for s in full]),
                failures=sum(not (.7<=s['noise_risk_ratio']<=1.3 and s['noise_output_error']<=.30) for s in full),
                risk_ratio_range=[min(s['noise_risk_ratio'] for s in full),max(s['noise_risk_ratio'] for s in full)],
                max_output_error=max(s['noise_output_error'] for s in full)),
      'P7':dict(status=status([g['ratio']<=4 and g['max_proxy_stability']<=.5 for g in gamma_groups]),
                failures=sum(g['ratio']>4 or g['max_proxy_stability']>.5 for g in gamma_groups),
                max_gain_range=max(g['ratio'] for g in gamma_groups),
                max_proxy_stability=max(g['max_proxy_stability'] for g in gamma_groups)),
      'P8':dict(status=status([limits_ok,frozen_match,manifest['elapsed_seconds']+sman['elapsed_seconds']+120<=3600]),
                complete_counts=[len(cal),len(hold),len(tc['records']),len(training)],
                resource_limits_verified=limits_ok,conservative_total_seconds=manifest['elapsed_seconds']+sman['elapsed_seconds']+120)
    }
    out=dict(audit=dict(passed=True,max_mean_reconstruction_error=maximum_mean_error,
        max_noise_reconstruction_error=maximum_noise_error,max_covariance_trace_error=maximum_cov_error,
        frozen_match=frozen_match),gates=gates,locked=locked,groups=groups,overall=overall,
        unresolved_groups=sum(not g['parts']['full']['resolved'] for g in groups),
        half_unresolved_groups=sum(not s['resolved'] for s in allparts),
        gamma_groups=gamma_groups,dimension_slopes=dimension_slopes,time_slopes=time_slopes,
        family_fraction=family_fraction,family_points=len(trajectory_points),costs=costs,
        locked_learning_rates=tc['locked'],formal_manifest=manifest,smoke_manifest=sman)
    (HERE/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# EXP2 冻结规则数值表','',
           '| α | n | a | 漂移误差 | 增益比 | 最佳标量后形状误差 | 行正交 | 输出耦合 | 风险噪声比 | 输出噪声误差 | SNR |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for g in groups:
        s=g['parts']['full']
        vals=[g['alpha'],g['n'],g['amplitude']]+[s[k] for k in ('relative_error','gain_ratio','shape_error','orth_error','coupling_error','noise_risk_ratio','noise_output_error','snr')]
        lines.append('| '+' | '.join(f'{v:.6g}' for v in vals)+' |')
    lines.extend(['','全部前半/后半指标、每单元统计、时钟斜率及资源审计见 analysis.json 与 formal/。'])
    (HERE/'TABLES.md').write_text('\n'.join(lines)+'\n')
    # Standalone scientific exports; no new data, fitted windows or thresholds.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    colors={1.5:'tab:blue',3.:'tab:orange'}
    for alpha in (1.5,3.):
        subset=[g for g in groups if g['alpha']==alpha]
        axes[0].scatter([g['parts']['full']['gain_ratio'] for g in subset],
                        [g['parts']['full']['shape_error'] for g in subset],label=f'alpha={alpha}',color=colors[alpha])
        for amp,mark in ((1.,'o'),(.25,'s'),(.0625,'^')):
            subset=[g for g in groups if g['alpha']==alpha and g['amplitude']==amp]
            axes[1].plot([g['n'] for g in subset],[g['parts']['full']['relative_error'] for g in subset],
                         marker=mark,color=colors[alpha],label=f'{alpha}, a={amp}')
    axes[0].axvspan(.8,1.2,color='green',alpha=.1)
    axes[0].axhline(.15,color='black',ls='--')
    axes[0].set(xlabel='Predicted / observed projection gain (ideal 1)',ylabel='Shape error after best scalar')
    # Actual ratio is observed/predicted, consistent with tables.
    axes[0].set_xlabel('Observed / predicted projection gain (ideal 1)')
    axes[0].legend()
    axes[1].axhline(.25,color='black',ls='--')
    axes[1].set(xlabel='Model dimension n',ylabel='Held-out full drift relative error',xscale='log')
    axes[1].legend(fontsize=7)
    for g in gamma_groups:
        axes[2].scatter(g['b'],g['ratio'],color=colors[g['alpha']],marker='o' if g['alpha']==1.5 else 's')
    axes[2].axhline(4,color='black',ls='--')
    axes[2].set(xlabel='Batch b',ylabel='Online max gamma / min gamma',xscale='log',yscale='log')
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(HERE/('bridge_holdout.'+suffix),dpi=180)
    plt.close(fig)
    out['analysis_elapsed_seconds']=time.monotonic()-started
    out['analysis_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    (HERE/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(audit=out['audit'],gates=gates,locked=locked,
        unresolved_groups=out['unresolved_groups'],family_fraction=family_fraction,
        costs=costs,analysis_elapsed_seconds=out['analysis_elapsed_seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
