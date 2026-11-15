"""Read-only analysis of the complete frozen V2 inventory."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from run import ROOT,OUT,DELTAS,HS,B,digest,frozen,dump,utc

START=time.monotonic()


def finite(x):
    return float(x) if np.isfinite(x) else None


def interval(values):
    values=np.asarray(values,float); ok=np.isfinite(values)
    return dict(valid_fraction=float(ok.mean()),ci95=np.quantile(values[ok],[.025,.975]).tolist()
        if ok.mean()>=.9 else None)


def line(r,t):
    t=np.array([np.inf if x is None else x for x in t])
    if not np.isfinite(t).all(): return None
    x=np.log(r); y=np.log(t); slope,intercept=np.polyfit(x,y,1)
    sst=np.sum((y-y.mean())**2)
    return dict(slope=float(slope),r2=float(1-np.sum((y-intercept-slope*x)**2)/sst) if sst else 1.,
        endpoint_ratio=float(t[-1]/t[0]),max_min=float(t.max()/t.min()))


def window(initial,current,eig):
    good=current>=.5*initial; segments=[]; start=None
    for j,ok in enumerate(np.r_[good,False]):
        if ok and start is None: start=j
        if not ok and start is not None:
            width=float(eig[start]/eig[j-1]); length=j-start
            segments.append(dict(first=start+1,last=j,modes=length,width=width,passed=bool(length>=4 and width>=10)))
            start=None
    return dict(passed=any(s['passed'] for s in segments),segments=segments,
        max_width=max([s['width'] for s in segments if s['modes']>=4],default=0.))


def bridge(rows,problem,group):
    rr=[r for r in rows if r['problem']==problem and r['group']==group]
    invalid=[r for r in rr if r['D']<=0 or r['V']<=0 or not np.isfinite(r['D']+r['V'])]
    valid=[r for r in rr if r not in invalid]
    for r in valid: r['F']=r['V']/(2*r['D'])
    def istrain(r): return r['grid_index']%2==1 and (problem==1 or r['N'] in [32,128])
    train=[r for r in valid if istrain(r)]; test=[r for r in valid if not istrain(r)]
    def design(rows,noise=False,with_n=True):
        arr=[[1.,np.log(r['D'] if noise else r['delta'])]+(
            [np.log(r['N']/64)] if with_n and problem==2 else [])+[np.log(r['ratio']/64)] for r in rows]
        return np.array(arr)
    def fit_predict(noise=False):
        if len(train)<5 or not test: return None
        coeff=np.linalg.lstsq(design(train,noise),np.log([r['F'] if noise else r['D'] for r in train]),rcond=None)[0]
        pred=np.exp(design(test,noise)@coeff); obs=np.array([r['F'] if noise else r['D'] for r in test])
        errors=abs(pred/obs-1)
        y=np.log([r['F'] if noise else r['D'] for r in train]); yp=design(train,noise)@coeff
        return dict(coefficients=coeff.tolist(),max_relative_error=float(errors.max()),
            median_relative_error=float(np.median(errors)),p90_relative_error=float(np.quantile(errors,.9)),
            training_r2=float(1-np.sum((y-yp)**2)/np.sum((y-y.mean())**2)),
            holdout_predictions=[dict(N=r['N'],ratio=r['ratio'],grid_index=r['grid_index'],
                observed=float(o),predicted=float(p),relative_error=float(e)) for r,o,p,e in zip(test,obs,pred,errors)])
    exponents=[]
    for n in ([64] if problem==1 else [32,64,128]):
        tr=[r for r in valid if r['N']==n and r['grid_index']%2==1]
        if len(tr)>=5:
            z=np.linalg.lstsq(design(tr,False,False),np.log([r['D'] for r in tr]),rcond=None)[0][1]
            nu=np.linalg.lstsq(design(tr,True,False),np.log([r['F'] for r in tr]),rcond=None)[0][1]
            exponents.append(dict(N=n,zeta=float(z),nu=float(nu)))
    drift=fit_predict(False); noise=fit_predict(True)
    zr=float(np.ptp([x['zeta'] for x in exponents])) if exponents else None
    nr=float(np.ptp([x['nu'] for x in exponents])) if exponents else None
    passed=bool(not invalid and drift and noise and drift['max_relative_error']<=.15 and
        noise['max_relative_error']<=.15 and len(exponents)==(1 if problem==1 else 3) and
        all(.4<=x['zeta']<=.6 and .85<=x['nu']<=1.15 for x in exponents) and zr<.1 and nr<.1)
    return dict(problem=problem,group=group,passed=passed,rows=len(rr),invalid_count=len(invalid),
        invalid_cells=[{k:r[k] for k in ['N','ratio','grid_index','D','V']} for r in invalid],
        train_rows=len(train),test_rows=len(test),drift=drift,noise=noise,exponents=exponents,
        zeta_range=zr,nu_range=nr,diagnostic_fit_drops_invalid=bool(invalid))


def main():
    prereg=frozen(); meta=json.loads((OUT/'formal/metadata.json').read_text())
    assert meta['complete'] and meta['code_sha256']==digest(OUT/'run.py')
    expected={f'p1_n64_s{s}_m0_r0.npz' for s in range(2101,2109)}
    expected|={f'p2_n{n}_s{s}_m0_r0.npz' for n in [32,64,128] for s in range(3101,3109)}
    expected|={f'p2_n{n}_s{s}_m1_r{r}.npz' for n in [32,64,128] for r in [16,64,256] for s in range(3101,3109)}
    assert {t['file'] for t in meta['tasks']}==expected and len(meta['calibration'])==32
    assert all(t['min_norm']>1e-12 for t in meta['tasks']+meta['calibration'])
    rng=np.random.default_rng(20261107); bs=rng.integers(0,8,(1000,8))
    cells=[]; raw={}; boot={}; curves=[]; prows=[]; spectral=[]; models=[]; endpoint_count=0
    for problem,n in [(1,64)]+[(2,n) for n in [32,64,128]]:
        mo=np.load(OUT/'formal'/f'model_p{problem}_n{n}.npz')
        models.append(dict(problem=problem,N=n,s2=float(mo['s2']),oracle_amplitude=float(mo['a0']),
            eig_max=float(mo['eig'][0]),eig_min=float(mo['eig'][-1]),condition=float(mo['eig'][0]/mo['eig'][-1])))
        seeds=range(2101,2109) if problem==1 else range(3101,3109)
        for mode,ratio in [(0,0)]+([(1,r) for r in [16,64,256]] if problem==2 else []):
            files=[np.load(OUT/'formal'/f'p{problem}_n{n}_s{s}_m{mode}_r{ratio}.npz') for s in seeds]
            risk=np.array([f['risks'] for f in files]); endpoint_count+=risk.size
            assert np.isfinite(risk).all()
            assert all(np.isfinite(f['monitor']).all() for f in files)
            groups=files[0]['groups']; unique=list(dict.fromkeys(groups))
            checks=([64,256,1024] if problem==1 else [16,64,256,1024]) if not mode else [ratio]
            for group in unique:
                idx=np.flatnonzero(groups==group); angle=group.endswith('angle'); grid=HS if angle else DELTAS
                for j,r in enumerate(checks):
                    vals=risk[:,j,idx]; mean=vals.mean(0); se=vals.std(0,ddof=1)/np.sqrt(8)
                    best=int(mean.argmin()); near=np.flatnonzero(mean<=1.01*mean[best])
                    bad=[0,len(grid)-1] if angle else [0,1,len(grid)-1]
                    identifiable=best not in bad and not np.isin(near,bad).any() and grid[near[-1]]/grid[near[0]]<=2.5
                    key=f'{problem}/{group}/{n}/{mode}/{r}'
                    raw[key]=vals; bb=vals[bs].mean(1).argmin(-1); boot[key]=bb
                    tau=1/(grid[best]*r*n/B) if not angle and best else np.inf
                    bt=np.full(1000,np.nan)
                    if not angle:
                        mask=bb>0; bt[mask]=1/(grid[bb[mask]]*r*n/B)
                    paired=(vals[:,0]-vals[:,best])/vals[:,0]
                    d=np.array([f['diagnostics'][j,idx[best]] for f in files])
                    c=dict(key=key,problem=problem,group=group,N=n,mode=mode,ratio=r,
                        best_index=best,best_value=float(grid[best]),tau=None if angle else finite(tau),
                        risk=float(mean[best]),se=float(se[best]),baseline=float(mean[0]),
                        improvement=None if angle else float(1-mean[best]/mean[0]),
                        interior=bool(best not in bad),identifiable=bool(identifiable),
                        near_indices=near.tolist(),near_width=float(grid[near[-1]]/grid[near[0]]) if grid[near[0]]>0 else None,
                        seed_best_indices=vals.argmin(1).tolist(),seed_risks_at_best=vals[:,best].tolist(),
                        tau_bootstrap=None if angle else interval(bt),
                        paired_gain_ci95=None if angle else interval(paired[bs].mean(1)),
                        norm_mean=float(d[:,0].mean()),amplitude_mean=float(d[:,1].mean()),
                        radial_raw_mean=float(d[:,5].mean()),radial_applied_mean=float(d[:,6].mean()),angle_mean=float(d[:,7].mean()))
                    cells.append(c)
                    for k,value in enumerate(grid):
                        curves.append(dict(problem=problem,group=group,N=n,mode=mode,ratio=r,
                            index=k,value=float(value),mean=float(mean[k]),se=float(se[k]),
                            **{f'seed_{s}':float(vals[ii,k]) for ii,s in enumerate(seeds)}))
                    if problem==2 and mode==0 and group=='learned_adam_cold' and r in [64,256,1024]:
                        init=np.mean([f['initial_modes'][idx[best]] for f in files],axis=0)
                        current=np.mean([f['modes'][j,idx[best]] for f in files],axis=0)
                        spectral.append(dict(N=n,ratio=r,selected_delta=float(grid[best]),
                            initial_modes=init.tolist(),current_modes=current.tolist(),eig=mo['eig'].tolist(),
                            **window(init,current,mo['eig'])))
                if not mode and not angle:
                    pr=np.array([f['probes'][:,idx] for f in files])
                    assert pr.shape==(8,3,42,8)
                    assert np.isfinite(pr[:,:,:,[0,1,3,4,5,6,7]]).all()
                    avg=pr.mean(axis=0)
                    for j,r in enumerate([64,256,1024]):
                        for k in range(1,42):
                            # Preserve nonpositive signal; no selective filtering for gates.
                            prows.append(dict(problem=problem,group=group,N=n,ratio=r,grid_index=k,
                                delta=float(DELTAS[k]),D=float(avg[j,k,0]),V=float(avg[j,k,1]),
                                D_mc_se=float(np.sqrt(np.sum(pr[:,j,k,3]**2))/8),
                                mean_angle=float(avg[j,k,4]),mean_radial_raw=float(avg[j,k,5]),
                                mean_radial_applied=float(avg[j,k,6])))
            for f in files: f.close()
    assert endpoint_count==69480
    fits=[]; gates=[]
    for group in dict.fromkeys(c['group'] for c in cells if c['problem']==2):
        cc=[c for c in cells if c['problem']==2 and c['group']==group]
        if group.endswith('angle'):
            gates.append(dict(group=group,angle=True,stream_identifiable=sum(c['identifiable'] for c in cc if c['mode']==0),
                replay_identifiable=sum(c['identifiable'] for c in cc if c['mode']==1)))
            continue
        gf=[]
        for mode in [0,1]:
            for n in [32,64,128]:
                cs=[c for c in cc if c['mode']==mode and c['N']==n]; rr=np.array([c['ratio'] for c in cs])
                fit=line(rr,[c['tau'] for c in cs]); slopes=[]
                for q in range(1000):
                    kk=np.array([boot[c['key']][q] for c in cs])
                    slopes.append(np.nan if np.any(kk==0) else np.polyfit(np.log(rr),np.log(1/(DELTAS[kk]*rr*n/B)),1)[0])
                gf.append(dict(group=group,N=n,mode=mode,fit=fit,slope_bootstrap=interval(slopes)))
        fits+=gf; sc=[c for c in cc if c['mode']==0]; rc=[c for c in cc if c['mode']==1]
        g1=all(c['identifiable'] for c in sc) and sum(c['improvement']>=.02 for c in sc)>=9
        sf=gf[:3]; rf=gf[3:]; numeric=all(f['fit'] is not None for f in sf)
        spread=float(np.ptp([f['fit']['slope'] for f in sf])) if numeric else None
        g2num=bool(numeric and spread<.1 and all(-.65<=f['fit']['slope']<=-.35 and
            f['fit']['r2']>=.95 and .067<=f['fit']['endpoint_ratio']<=.168 for f in sf))
        rt=[c['tau'] for c in rc]; globalratio=max(rt)/min(rt) if all(x is not None for x in rt) else None
        g3=bool(all(c['identifiable'] for c in rc) and globalratio is not None and globalratio<=2 and
            all(f['fit'] is not None and f['fit']['max_min']<=1.5 and abs(f['fit']['slope'])<=.1 for f in rf))
        gates.append(dict(group=group,angle=False,G1=bool(g1),G2_evaluable=bool(g1),G2_numeric_only=g2num,
            G2=bool(g1 and g2num),G3=g3,stream_interior=sum(c['interior'] for c in sc),
            stream_identifiable=sum(c['identifiable'] for c in sc),stream_gain_2pct=sum(c['improvement']>=.02 for c in sc),
            replay_interior=sum(c['interior'] for c in rc),replay_identifiable=sum(c['identifiable'] for c in rc),
            stream_slope_range=spread,replay_global_ratio=globalratio))
    aa=[c for c in cells if c['problem']==1]; causal=[]
    for typ in ['adam','proj']:
        cold=next(c for c in aa if c['group']==f'fixed_{typ}_cold' and c['ratio']==64)
        warm=next(c for c in aa if c['group']==f'fixed_{typ}_warm' and c['ratio']==64)
        causal.append(dict(update=typ,cold_gain=cold['improvement'],warm_gain=warm['improvement'],
            warm_removes_advantage=bool(cold['improvement']>=.02 and warm['improvement']<.02)))
    ac=next(c for c in aa if c['group']=='fixed_adam_cold' and c['ratio']==64)
    pc=next(c for c in aa if c['group']=='fixed_proj_cold' and c['ratio']==64)
    av=raw[ac['key']][:,ac['best_index']]; pv=raw[pc['key']][:,ac['best_index']]
    gain=(av-pv)/av; ci=interval(gain[bs].mean(1))
    radial_gain=float(1-pv.mean()/av.mean())
    radial_effect=dict(at_delta=ac['best_value'],relative_gain=radial_gain,paired_ci95=ci,
        passed=bool(radial_gain>=.02 and ci['ci95'][0]>0))
    bridges=[]
    for p in [1,2]:
        for group in dict.fromkeys(r['group'] for r in prows if r['problem']==p):
            bridges.append(bridge(prows,p,group))
    primary=next(g for g in gates if g['group']=='learned_adam_cold')
    primary_bridge=next(b for b in bridges if b['problem']==2 and b['group']=='learned_adam_cold')
    spectrum_ok=all(s['passed'] for s in spectral)
    summary=dict(prereg_sha256=prereg,analysis_utc=utc(),analysis_code_sha256=digest(__file__),
        G0=True,endpoint_count=endpoint_count,formal_wall_seconds=meta['wall_seconds'],peak_rss_MiB=meta['peak_rss_MiB'],
        min_norm=min(t['min_norm'] for t in meta['tasks']+meta['calibration']),models=models,
        primary=primary,spectral_passed=sum(s['passed'] for s in spectral),spectral_all_passed=spectrum_ok,
        primary_bridge_passed=primary_bridge['passed'],continue_full_theory=bool(primary['G1'] and primary['G2'] and primary['G3'] and spectrum_ok and primary_bridge['passed']),
        A_initialization=causal,A_radial_intervention=radial_effect,A_groups=[dict(group=g,
            identifiable=sum(c['identifiable'] for c in aa if c['group']==g)) for g in dict.fromkeys(c['group'] for c in aa)],
        gates=gates,fits=fits,cells=cells,bridges=bridges,spectral=spectral,
        costs=dict(calibration_samples=sum(t['samples'] for t in meta['calibration']),
            calibration_sample_parameter_products=sum(t['sample_parameter_products'] for t in meta['calibration']),
            formal_stream_samples=sum(t['samples'] for t in meta['tasks'] if not t['mode']),
            formal_replay_seen=sum(t['samples'] for t in meta['tasks'] if t['mode']),
            formal_replay_unique=sum(t['unique_samples'] for t in meta['tasks'] if t['mode']),
            formal_sample_parameter_products=sum(t['sample_parameter_products'] for t in meta['tasks']),
            probe_samples=sum(t['probe_samples'] for t in meta['tasks']),
            probe_sample_parameter_products=sum(t['probe_sample_parameter_products'] for t in meta['tasks'])),
        analysis_wall_seconds=time.monotonic()-START)
    dump(OUT/'summary.json',summary)
    # Seed names differ between A/B; use the union to retain every raw value.
    fields=list(dict.fromkeys(k for row in curves for k in row))
    with (OUT/'curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(curves)
    with (OUT/'probes.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in prows for k in r)); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(prows)
    lines=['# 全部225格：δ与h分开报告','','|问题|臂|N|制度|r|δ或h最优|τ|风险±SE|改善|可识别|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---|']
    for c in cells:
        tau='—' if c['group'].endswith('angle') else ('∞' if c['tau'] is None else f"{c['tau']:.5g}")
        gain='—' if c['improvement'] is None else f"{c['improvement']:.2%}"
        lines.append(f"|{c['problem']}|{c['group']}|{c['N']}|{'stream' if c['mode']==0 else 'replay'}|{c['ratio']}|{c['best_value']:.6g}|{tau}|{c['risk']:.6g}±{c['se']:.2g}|{gain}|{c['identifiable']}|")
    lines+=['','## B各臂严格门槛','','|臂|单遍识别/12|改善≥2%/12|多遍识别/9|G1|G2|G3|','|---|---:|---:|---:|---|---|---|']
    for g in gates:
        lines.append(f"|{g['group']}|{g['stream_identifiable']}|{g.get('stream_gain_2pct','—')}|{g['replay_identifiable']}|{g.get('G1','—')}|{g.get('G2','—')}|{g.get('G3','—')}|")
    lines+=['','## 漂移/噪声留出预测','','|问题|臂|无效格|漂移最大相对误差|噪声最大相对误差|通过|','|---|---|---:|---:|---:|---|']
    for b in bridges:
        dd=b['drift']; nn=b['noise']
        lines.append(f"|{b['problem']}|{b['group']}|{b['invalid_count']}|{dd['max_relative_error'] if dd else '—'}|{nn['max_relative_error'] if nn else '—'}|{b['passed']}|")
    lines+=['','summary.json含全部拟合、bootstrap有效率、近优集合、每种子最优及谱窗口。',
        'curves.csv含原始八种子风险。paired_gain_ci95为原选点条件的区间，存在选择偏差。',
        '桥接若有非正漂移，正值子集拟合仅为诊断；严格通过标志仍为false。']
    (OUT/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cells','spectral','bridges','fits','gates']},ensure_ascii=False,indent=2))
    dump(OUT/'analysis_runtime.json',dict(wall_seconds=time.monotonic()-START,utc=utc()))


if __name__=='__main__': main()
