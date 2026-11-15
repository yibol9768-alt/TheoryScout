"""Frozen round-three statistics; time windows, candidates and gates never adapt."""
import argparse
import json
import time
import numpy as np
from scipy.stats import rankdata
from util import (HERE, DATA, save, sha, rng, blocks, fit_beta, predict,
                  correlation, verify_prereg, metadata)


def improvement(a,b):
    return float(1-a/b) if b>0 else None


def rho_regression(state, response, eta, c):
    design=np.column_stack([np.ones(len(state)),state-state.mean(0)])
    slope=np.linalg.lstsq(design,response,rcond=None)[0][1:].T
    return float(np.max(np.abs(np.linalg.eigvals(c*np.eye(2)+eta*slope))))


def contraction(state, pred, xi, eta, c, generator):
    train_idx=blocks(150,generator,500); hold_idx=blocks(300,generator,500)
    pr=np.asarray([rho_regression(state[:150][ix],pred[:150][ix],eta,c) for ix in train_idx])
    ob=np.asarray([rho_regression(state[150:][ix],xi[150:][ix],eta,c) for ix in hold_idx])
    pci=np.quantile(pr,[.025,.975]);oci=np.quantile(ob,[.025,.975])
    return dict(predicted=rho_regression(state[:150],pred[:150],eta,c),
                observed=rho_regression(state[150:],xi[150:],eta,c),predicted_CI95=pci,observed_CI95=oci,
                gate=bool(pci[0]>=0 and pci[1]<1 and max(pci[0],oci[0])<=min(pci[1],oci[1])),
                limitation='Projected local regression, not an autonomous multi-key state law')


def analyze_feedback(path, replay):
    arr=np.load(path); index=replay['index']; gen=rng(index,0)
    xi,g,bound=arr['xi'],arr['g'],arr['bound']
    channels=xi.shape[-1]; nc=channels//2
    ab=arr['ab'][:-1].reshape(450,nc,2)
    beta=fit_beta(g[:150],xi[:150],bound)
    m=predict(g,beta,bound)
    base=np.broadcast_to(xi[:150].mean(0),xi.shape)
    innovation=xi-m
    # Independent direct ACF implementation is audited in validate.py.
    from util import max_acf
    wm=xi[150:].reshape(3,100,channels).mean(1)
    pm=m[150:].reshape(3,100,channels).mean(1)
    bm=base[150:].reshape(3,100,channels).mean(1)
    persistent=np.asarray([xi[50:150].mean(0),xi[150:250].mean(0),xi[250:350].mean(0)])
    errf=(wm-pm)**2;errb=(wm-bm)**2;errp=(wm-persistent)**2
    early=xi[:150].reshape(3,50,channels)
    earlymeans=early.mean(1)
    maxi=earlymeans.argmax(0);mini=earlymeans.argmin(0)
    sampled=np.asarray([e[blocks(50,gen,500)].mean(1) for e in early])
    bootdiff=sampled[maxi,:,np.arange(channels)]-sampled[mini,:,np.arange(channels)]
    halfwidth=(np.quantile(bootdiff,.975,axis=1)-np.quantile(bootdiff,.025,axis=1))/2
    spread=earlymeans.max(0)-earlymeans.min(0)
    enough=(spread>=halfwidth).reshape(nc,2).any(1)
    obscoh=[];predcoh=[]
    for lo in (50,150,250,350):
        obscoh.append(np.linalg.norm((xi[lo:lo+100]/bound).mean(0).reshape(nc,2),axis=1))
        predcoh.append(np.linalg.norm((m[lo:lo+100]/bound).mean(0).reshape(nc,2),axis=1))
    obscoh=np.asarray(obscoh);predcoh=np.asarray(predcoh)
    earlyblock=(xi[50:150]/bound).reshape(10,10,channels).mean(1)
    boots=earlyblock[gen.integers(10,size=(500,10))].mean(1)
    earlyci=np.quantile(boots,[.025,.975],axis=0)
    signal=((earlyci[0]>0)|(earlyci[1]<0)).reshape(nc,2).any(1)
    candidates=[]; contraction_generator=rng(index,6)
    positive={(r['head'],r['candidate']):r['logit']>0 for r in replay['checkpoints'] if r['step']==150}
    for i in range(nc):
        sl=slice(2*i,2*i+2)
        obsdrop=float(1-obscoh[-1,i]/obscoh[0,i]) if obscoh[0,i]>0 else None
        preddrop=float(1-predcoh[-1,i]/predcoh[0,i]) if predcoh[0,i]>0 else None
        sync={}
        for name,window in [('fit',slice(0,150)),('holdout',slice(150,450))]:
            coeff=ab[window,i]; rms=np.sqrt(np.mean(coeff**2,axis=0))
            diff=float(np.sqrt(np.mean((coeff[:,0]/rms[0]-coeff[:,1]/rms[1])**2)))
            cor=correlation(innovation[window,2*i],innovation[window,2*i+1])
            sync[name]=dict(normalized_trajectory_RMS_difference=diff,innovation_QK_correlation=cor,
                            RMS_QK=rms,RMS_scale_ratio=float(rms[0]/rms[1]),
                            same_sign_fraction=float(np.mean(coeff[:,0]*coeff[:,1]>0)),
                            gate=bool(diff<=.2 and cor is not None and cor>=.8))
        con=contraction(ab[:,i],m[:,sl],xi[:,sl],replay['eta'],1-replay['alpha'],contraction_generator)
        candidates.append(dict(head=i//4,candidate=i%4,positive_at_freeze=positive[i//4,i%4],enough_variation=bool(enough[i]),
                               fit_window_mean_spread=spread[sl],fit_window_difference_CI_halfwidth=halfwidth[sl],
                               window_MSE_feedback=errf[:,sl].mean(),window_MSE_constant=errb[:,sl].mean(),
                               improvement=improvement(errf[:,sl].mean(),errb[:,sl].mean()),
                               step_MSE_feedback=np.mean(innovation[150:,sl]**2),
                               step_MSE_constant=np.mean((xi-base)[150:,sl]**2),
                               early_signal=bool(signal[i]),predicted_coherence_drop=preddrop,observed_coherence_drop=obsdrop,
                               transition_gate=bool(signal[i] and preddrop is not None and preddrop>=.3 and obsdrop is not None and obsdrop>0),
                               observed_coherence=obscoh[:,i],predicted_coherence=predcoh[:,i],
                               synchronization=sync,contraction=con))
    spatial=json.loads((DATA/'v2_offline/diagnostics.json').read_text())['runs']
    spatialrow=next(r for r in spatial if r['run_id']==path.stem)
    selectedheads=[h['spatial_gate'] for h in spatialrow['heads']]
    def summarize(mask):
        mask=np.asarray(mask,dtype=bool)
        if not mask.any():
            return dict(candidates=0,identifiable=False)
        channelmask=np.repeat(mask,2)
        mse_f=float(errf[:,channelmask].mean());mse_b=float(errb[:,channelmask].mean())
        # Resample complete 100-step evaluation windows, never individual steps.
        ids=gen.integers(3,size=(500,3))
        bf=errf[:,channelmask].mean(1)[ids].mean(1);bb=errb[:,channelmask].mean(1)[ids].mean(1)
        boot=1-bf/np.maximum(bb,1e-300)
        transitions=[r['transition_gate'] for r,keep in zip(candidates,mask) if keep]
        valid=[r for r,keep in zip(candidates,mask) if keep and r['early_signal'] and r['predicted_coherence_drop'] is not None]
        drop=float(np.median([r['predicted_coherence_drop'] for r in valid])) if valid else None
        odrop=float(np.median([r['observed_coherence_drop'] for r in valid])) if valid else None
        rate=improvement(mse_f,mse_b)
        return dict(candidates=int(mask.sum()),identifiable=True,window_MSE_feedback=mse_f,window_MSE_constant=mse_b,
                    window_MSE_persistence=float(errp[:,channelmask].mean()),improvement=rate,
                    improvement_CI95_approx=np.quantile(boot,[.025,.975]),
                    mean_gate=bool(rate is not None and rate>=.2),
                    transition_identifiable_candidates=len(valid),transition_passing_candidates=sum(transitions),
                    median_predicted_drop=drop,median_observed_drop=odrop,
                    transition_gate=bool(drop is not None and drop>=.3 and odrop is not None and odrop>0))
    residual_variance=innovation[50:450].reshape(4,100,channels).var(1,ddof=1)
    ratios=residual_variance.max(0)/np.maximum(residual_variance.min(0),1e-300)
    sigma=np.divide(bound,beta,out=np.full_like(beta,np.inf),where=beta>0)
    noise_sd=np.sqrt(np.mean(arr['noise_sd'][:150]**2,axis=0))
    report=dict(run_id=path.stem,index=index,seed=replay['seed'],condition=replay['condition'],eta=replay['eta'],
                beta=beta,sigma=sigma,
                fitted_block_noise_RMS_SD=noise_sd,sigma_over_block_noise_RMS_SD=sigma/noise_sd,
                bound=bound,max_actual_normalized_amplitude=float(np.max(np.abs(xi/bound))),
                all_candidates=summarize(np.ones(nc,dtype=bool)),
                varying_candidates=summarize(enough),
                varying_spatial_candidates=summarize(enough & np.repeat(selectedheads,4)[:nc]),
                candidates=candidates,observed_max_acf=float(max_acf(innovation)),
                residual_window_variances=residual_variance,residual_variance_max_min_ratio=ratios,
                closure_identified=False,
                closure_reason='g uses current full 8-key weights/activations; no tested autonomous (a,b) law or stochastic full-state law',
                primary_P3_status='not_testable_without_closed_state_noise_law',
                source_array_sha256=sha(path))
    return report,dict(m=m,beta=beta,innovation=innovation,constant=base)


def summarize_seeds(feedback):
    rows=[]
    for condition in ('isotropic','anisotropic'):
        for eta in (.01,.03,.1):
            runs=[r for r in feedback if r['condition']==condition and r['eta']==eta]
            row=dict(condition=condition,eta=eta,seeds=[r['seed'] for r in runs],groups={})
            for group in ('all_candidates','varying_candidates','varying_spatial_candidates'):
                vals=[r[group] for r in runs]
                passes=sum(v.get('mean_gate',False) for v in vals)
                trans=sum(v.get('transition_gate',False) for v in vals)
                row['groups'][group]=dict(improvements=[v.get('improvement') for v in vals],
                                         mean_pass_seeds=passes,mean_gate=passes>=2,
                                         transition_pass_seeds=trans,transition_gate=trans>=2)
            rows.append(row)
    return rows


def spearman(x,y):
    return correlation(rankdata(x),rankdata(y)) if len(x)>1 else None


def optimizer_summary(replays):
    main=[r for r in replays if r['primary']]
    rows=[]
    for condition in ('isotropic','anisotropic'):
        for eta in (.01,.03,.1):
            runs=[r for r in main if r['condition']==condition and r['eta']==eta]
            if not runs:
                continue
            checkpoints=[dict(seed=r['seed'],**cp) for r in runs for cp in r['checkpoints']]
            row=dict(condition=condition,eta=eta,total_candidates=len(checkpoints),weak_masked_candidates=sum(c['weak_masked'] for c in checkpoints),modes={})
            for mode in ('original_k0','matched_fro'):
                vals={}
                for other in ('signsgd','adamw'):
                    byseed=[]
                    for run in runs:
                        cp=run['checkpoints']
                        select=[c for c in cp if c['weak_masked'] and c['comparisons'][mode]['ratio_'+other] is not None]
                        allvalid=[c for c in cp if c['comparisons'][mode]['ratio_'+other] is not None]
                        byseed.append(dict(seed=run['seed'],weak_masked_count=len(select),
                                           median_ratio_masked=float(np.median([c['comparisons'][mode]['ratio_'+other] for c in select])) if select else None,
                                           median_ratio_all=float(np.median([c['comparisons'][mode]['ratio_'+other] for c in allvalid])) if allvalid else None,
                                           mask_predictor_spearman_all=spearman([c['masking_score'] for c in allvalid],[c['comparisons'][mode]['ratio_'+other] for c in allvalid])))
                    selected=[c['comparisons'][mode]['ratio_'+other] for c in checkpoints if c['weak_masked'] and c['comparisons'][mode]['ratio_'+other] is not None]
                    med=float(np.median(selected)) if selected else None
                    same=sum(r['median_ratio_masked'] is not None and r['median_ratio_masked']>1 for r in byseed)
                    pred=sum(r['mask_predictor_spearman_all'] is not None and r['mask_predictor_spearman_all']>0 for r in byseed)
                    vals[other]=dict(median_ratio_masked=med,median_ratio_all=float(np.median([c['comparisons'][mode]['ratio_'+other] for c in checkpoints])),
                                     same_direction_seeds=same,predictor_positive_seeds=pred,by_seed=byseed,
                                     gate=bool(med is not None and med>=1.5 and same>=2 and pred>=2))
                row['modes'][mode]=vals
            row['gate']=all(v['gate'] for v in row['modes']['matched_fro'].values())
            rows.append(row)
    # Summary across all primary checkpoints is descriptive, not seed replication.
    allcp=[c for r in main for c in r['checkpoints']]
    totals={}
    for mode in ('original_k0','matched_fro'):
        totals[mode]={}
        for other in ('signsgd','adamw'):
            vals=[c['comparisons'][mode]['ratio_'+other] for c in allcp]
            finite=[v for v in vals if v is not None]
            totals[mode][other]=dict(median_all=float(np.median(finite)),q25_q75=np.quantile(finite,[.25,.75]),zero_denominators=len(vals)-len(finite))
    return dict(strata=rows,all_primary=totals,total_primary_candidates=len(allcp),weak_masked=sum(c['weak_masked'] for c in allcp),
                primary_sources=len(main),control_sources=len(replays)-len(main),
                gate=bool(rows and any(r['gate'] for r in rows)),
                limitation='Local counterfactual only, not a full-trajectory causal ordering')


def closest(run,target):
    points=run['dense_CE']
    best=min(points,key=lambda r:(abs(r['ce']-target),r['step']))
    bracket=None
    for a,b in zip(points[:-1],points[1:]):
        if a['ce']>=target and b['ce']<=target:
            bracket=dict(lower_step=a,upper_step=b,logit_endpoint_range=[min(a['Mabs'],b['Mabs']),max(a['Mabs'],b['Mabs'])],dense=bool(b['step']-a['step']==1))
            break
    return dict(closest=best,error=abs(best['ce']-target),eligible=abs(best['ce']-target)<=.005,first_downcrossing=bracket)


def ce_summary(replays):
    lookup={(r['condition'],r['eta'],r['seed'],r['optimizer']):r for r in replays}
    rows=[]
    for main in (r for r in replays if r['primary']):
        for other in ('muon09','adamw','signsgd'):
            control=lookup.get((main['condition'],main['eta'],main['seed'],other))
            if control is None:
                continue
            for target in (1.,.8):
                a,b=closest(main,target),closest(control,target)
                difference=abs(a['closest']['ce']-b['closest']['ce'])
                rows.append(dict(condition=main['condition'],eta=main['eta'],seed=main['seed'],comparator=other,target=target,
                                 muon=a,other=b,CE_difference=difference,
                                 matched=a['eligible'] and b['eligible'] and difference<=.01))
    counts=[]
    for other in ('muon09','adamw','signsgd'):
        for target in (1.,.8):
            sub=[r for r in rows if r['comparator']==other and r['target']==target]
            good=[r for r in sub if r['matched']]
            ratios=[r['muon']['closest']['Mabs']/r['other']['closest']['Mabs'] for r in good]
            counts.append(dict(comparator=other,target=target,total_pairs=len(sub),matched_pairs=len(good),missing_pairs=len(sub)-len(good),
                               median_Mabs_ratio_muon_other=float(np.median(ratios)) if ratios else None))
    return dict(counts=counts,pairs=rows)


def spatial_supplement():
    old=json.loads((DATA/'v2_offline/diagnostics.json').read_text())
    rows=[]
    for run in old['runs']:
        d=json.loads((DATA/'runs'/f'{run["run_id"]}.json').read_text())
        for h in range(4):
            points=[p for p in d['probe'] if p['step']>=160]
            values=np.asarray([p['head_Mabs'][h] for p in points]);times=np.asarray([p['step'] for p in points])
            j=int(np.argmax(values));peak=values[j]
            late=float(np.median(values[times>=501]))
            after=values[j+1:]<=.9*peak
            sustained=bool(np.any(after[:-1]&after[1:]))
            falling=bool(1-late/peak>=.1 and sustained)
            rows.append(dict(run_id=run['run_id'],seed=run['seed'],condition=run['condition'],eta=run['eta'],head=h,
                             spatial_gate=run['heads'][h]['spatial_gate'],peak_step=int(times[j]),peak=peak,
                             late_median=late,drop_fraction=1-late/peak,sustained_two_points=sustained,falling=falling,
                             ever_above_8=bool(values.max()>8)))
    return dict(rows=rows,total=len(rows),spatial_pass=sum(r['spatial_gate'] for r in rows),
                falling_heads=sum(r['falling'] for r in rows),falling_spatial_pass=sum(r['falling'] and r['spatial_gate'] for r in rows),
                crossing8_heads=sum(r['ever_above_8'] for r in rows),crossing8_spatial_pass=sum(r['ever_above_8'] and r['spatial_gate'] for r in rows))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify_prereg();start=time.monotonic();out=HERE/('smoke' if args.smoke else 'full')
    replays=[json.loads(p.read_text()) for p in sorted((out/'replays').glob('*.json'))]
    feedback=[]
    for replay in (r for r in replays if r['primary']):
        report,arrays=analyze_feedback(out/'arrays'/f'{replay["run_id"]}.npz',replay)
        feedback.append(report)
        save(out/'feedback'/f'{replay["run_id"]}.json',report)
        np.savez_compressed(out/'arrays'/f'{replay["run_id"]}_fit.npz',**arrays)
        print(replay['run_id'],report['varying_candidates'],flush=True)
    report=dict(feedback=summarize_seeds(feedback),optimizer=optimizer_summary(replays),CE=ce_summary(replays),
                spatial=spatial_supplement(),
                P3_main='不可检验：没有闭合状态/噪声生成律；附加校准另存，不能补票',
                P5='跳过：尚无闭合反馈，且必须先通过 P2/P3/P4',resources=metadata(start),script_sha256=sha(__file__))
    save(out/'statistics.json',report)


if __name__=='__main__':
    main()
