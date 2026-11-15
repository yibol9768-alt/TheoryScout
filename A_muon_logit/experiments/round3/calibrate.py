"""Auxiliary calibration only: fixed-design Gaussian, NOT full-state bootstrap."""
import argparse
import json
import time
import numpy as np
from scipy.signal import lfilter
from util import (HERE, save, sha, rng, fit_beta, predict, max_acf, holm,
                  metadata, verify_prereg, check_time)


def simulate(g,m,bound,factor,generator,repeats,rho=0.,batch=32):
    out=np.empty(repeats)
    for lo in range(0,repeats,batch):
        n=min(batch,repeats-lo)
        noise=generator.standard_normal((n,450,len(bound)))@factor.T
        if rho:
            # Stationary initial row, thereafter AR(1) with covariance Sigma.
            scale=np.sqrt(1-rho*rho)
            noise[:,0,:]/=scale
            noise=lfilter([scale],[1,-rho],noise,axis=1)
        y=m[None,:,:]+noise
        beta=fit_beta(g[:150],y[:,:150,:],bound)
        residual=y-predict(g[None,:,:],beta[:,None,:],bound)
        out[lo:lo+n]=max_acf(residual)
    return out


def permutation_test(residual,generator,repeats=1999,batch=32):
    out=np.empty(repeats)
    for lo in range(0,repeats,batch):
        n=min(batch,repeats-lo)
        permutations=np.argsort(generator.random((n,450)),axis=1)
        out[lo:lo+n]=max_acf(residual[permutations])
    return out


def analyze_one(path,out,repeats):
    start=time.monotonic(); runid=path.stem
    raw=np.load(path);fit=np.load(path.with_name(runid+'_fit.npz'))
    feedback=json.loads((out/'feedback'/f'{runid}.json').read_text())
    index=feedback['index'];bound=raw['bound'];g=raw['g'];m=fit['m'];residual=fit['innovation']
    covariance=np.cov(residual[:150],rowvar=False,ddof=1)
    ridge=1e-12*float(np.max(np.diag(covariance)))
    covariance+=np.eye(len(bound))*ridge
    eigenvalues,vectors=np.linalg.eigh(covariance)
    factor=vectors*np.sqrt(np.maximum(eigenvalues,0))[None,:]
    samples={}
    for stage,label,rho in ((1,'calibration',0.),(2,'validation',0.),(3,'rho_0.2',.2),(4,'rho_0.4',.4)):
        samples[label]=simulate(g,m,bound,factor,rng(index,stage),repeats,rho)
        print(f'{runid} {label} n={repeats} elapsed={time.monotonic()-start:.1f}s',flush=True)
    critical=float(np.quantile(samples['calibration'],.95,method='higher'))
    fwer=float(np.mean(samples['validation']>critical))
    observed=float(max_acf(residual))
    perm=permutation_test(residual,rng(index,5))
    samples['permutation']=perm
    np.savez_compressed(out/'calibration'/f'{runid}_statistics.npz',**samples)
    report=dict(run_id=runid,index=index,channels=len(bound),windows=4,lags=20,repeats_each=repeats,
                observed_T=observed,critical_95pct=critical,independent_FWER=fwer,
                calibration_gate=bool(.04<=fwer<=.06),
                FWER_MC_SE=float(np.sqrt(fwer*(1-fwer)/repeats)),
                power={str(rho):float(np.mean(samples[label]>critical)) for rho,label in ((.2,'rho_0.2'),(.4,'rho_0.4'))},
                power_gate={str(rho):bool(np.mean(samples[label]>critical)>=.8) for rho,label in ((.2,'rho_0.2'),(.4,'rho_0.4'))},
                approximate_Gaussian_p=float((1+np.sum(samples['calibration']>=observed))/(repeats+1)),
                exchangeability_sensitivity_p=float((1+np.sum(perm>=observed))/2000),
                covariance=covariance,ridge=ridge,
                residual_variance_max_min_ratio=feedback['residual_variance_max_min_ratio'],
                exact_refit='same constrained beta fit, fixed g, all registered channels; same four windows and ACF denominator',
                primary_P3_status='不可检验：本附项不模拟完整状态，不是 §7.3 主反馈 bootstrap',
                caveats=['Plug-in covariance and beta, no confidence-set worst case',
                         'Gaussian homoscedastic fixed design is an auxiliary null, not a fitted full stochastic training model',
                         'Row permutation p only valid under exchangeability; heteroscedasticity invalidates that interpretation',
                         'Failure to reject does not establish independence or the martingale difference condition'],
                source_array_sha256=sha(path),source_fit_sha256=sha(path.with_name(runid+'_fit.npz')),
                resources=metadata(start),script_sha256=sha(__file__))
    save(out/'calibration'/f'{runid}.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify_prereg();start=time.monotonic();out=HERE/('smoke' if args.smoke else 'full')
    (out/'calibration').mkdir(exist_ok=True)
    paths=[p for p in sorted((out/'arrays').glob('*.npz')) if not p.stem.endswith('_fit')]
    reports=[]
    for path in paths:
        reports.append(analyze_one(path,out,200 if args.smoke else 20000))
        save(out/'calibration_partial.json',dict(completed=[r['run_id'] for r in reports],resources=metadata(start)))
        check_time(start)
    for label in ('approximate_Gaussian_p','exchangeability_sensitivity_p'):
        adjusted=holm([r[label] for r in reports])
        for r,p in zip(reports,adjusted):
            r[label+'_Holm']=float(p)
    for r in reports:
        save(out/'calibration'/f'{r["run_id"]}.json',r)
    summary=dict(runs=reports,run_count=len(reports),expected_full_runs=18,
                 primary_P3_status='不可检验：无闭合反馈状态和噪声生成律',
                 full_size_auxiliary=not args.smoke,
                 all_auxiliary_FWER_validated=all(r['calibration_gate'] for r in reports),
                 auxiliary_Holm_rejections={label:sum(r[label+'_Holm']<=.05 for r in reports) for label in ('approximate_Gaussian_p','exchangeability_sensitivity_p')},
                 resources=metadata(start))
    save(out/'calibration_summary.json',summary)


if __name__=='__main__':
    main()
