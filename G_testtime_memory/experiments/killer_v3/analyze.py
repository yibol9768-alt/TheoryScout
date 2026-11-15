"""Frozen full-grid gates; conditional/oracle/forecast roles stay distinct."""
import csv
import json
from collections import defaultdict
import numpy as np
from design import *
from common import OUT,begin,finish,mean_se,prior_elapsed

def write_csv(name,rows):
    with (OUT/name).open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def key(r): return (r['family'],r['d'],r['load'],r['alpha'],r['method'],r['readout'])
def mk(r): return (r['family'],r['d'],r['load'],r['alpha'],r['method'])
def primary(r): return r['family']=='bounded' and r['alpha']>=1 and r['readout']!='full'

def read_seed_file(name,keyfun,expected,groups):
    out=defaultdict(dict); count=0
    for line in (OUT/name).read_text().splitlines():
        r=json.loads(line); k=keyfun(r)
        assert r['seed'] not in out[k]
        out[k][r['seed']]=r; count+=1
    assert count==expected and len(out)==groups,(name,count,len(out))
    for k,rr in out.items(): assert set(rr)==set(TEST_SEEDS),(name,k)
    return out,count

def main():
    state=begin('analysis')
    assert json.loads((OUT/'formal_meta.json').read_text())['complete']
    forecast={key(r):r for r in json.loads((OUT/'forecasts.json').read_text())}
    uncond={mk(r):r for r in json.loads((OUT/'unconditional_moments.json').read_text())}
    assert len(forecast)==1728 and len(uncond)==432
    raw,count=read_seed_file('raw.jsonl',key,27648,1728)
    moments,mcount=read_seed_file('moments.jsonl',mk,6912,432)
    oracle,ocount=read_seed_file('oracle.jsonl',key,384,24)
    rows=[]; lookup={}
    for k,pred in forecast.items():
        rr=raw[k]
        av,se=mean_se([rr[s]['top1'] for s in TEST_SEEDS])
        cp=mean_se([rr[s]['predicted'] for s in TEST_SEEDS])[0]
        residual,rse=mean_se([rr[s]['top1']-rr[s]['predicted'] for s in TEST_SEEDS])
        row={z:pred[z] for z in ('family','d','load','alpha','method','readout','p')}
        row.update({'mean':av,'se':se,'predicted':cp,'abs_error':abs(av-cp),
            'paired_residual_se':rse,'forecast':pred['predicted'],
            'forecast_error':abs(av-pred['predicted']),'forecast_integration_se':pred['integration_se']})
        for b in range(4): row[f'age{b}']=mean_se([rr[s]['age_top1'][b] for s in TEST_SEEDS])[0]
        rows.append(row); lookup[k]=row
    write_csv('method_summary.csv',rows)
    cells=[]
    for family in FAMILIES:
        for d in DIMS:
            for load in LOADS:
                for a in ALPHAS:
                    for r in READOUTS:
                        ds=[lookup[family,d,load,a,m,r] for m in METHODS[1:]]
                        # Frozen selector uses only pre-data forecast, not the
                        # conditional predictions or observations of validation.
                        j=max(range(len(BETAS)),key=lambda j:ds[j]['forecast'])
                        h=lookup[family,d,load,a,'hebb',r]; chosen=ds[j]
                        hk=(family,d,load,a,'hebb',r); dk=(family,d,load,a,chosen['method'],r)
                        gain,se=mean_se([raw[dk][s]['top1']-raw[hk][s]['top1'] for s in TEST_SEEDS])
                        best=max(x['mean'] for x in ds); jb=max(range(len(ds)),key=lambda j:ds[j]['mean'])
                        near=[b for b,row in zip(BETAS,ds) if best-row['mean']<=.01]
                        factor=min(max(BETAS[j]/b,b/BETAS[j]) for b in near)
                        pg=chosen['predicted']-h['predicted']; fg=chosen['forecast']-h['forecast']
                        cells.append({'family':family,'d':d,'load':load,'alpha':a,'readout':r,
                            'beta_pred':BETAS[j],'beta_best':BETAS[jb],
                            'best_boundary':jb in (0,len(BETAS)-1),
                            'hebb':h['mean'],'delta':chosen['mean'],'gain':gain,'gain_se':se,
                            'lower_3se':gain-3*se,'P1_cell_pass':gain>=.05 and gain-3*se>0,
                            'predicted_gain':pg,'gain_error':abs(gain-pg),
                            'forecast_gain':fg,'forecast_gain_error':abs(gain-fg),
                            'regret':best-chosen['mean'],'near_factor':factor})
    write_csv('cell_summary.csv',cells)
    momentrows=[]
    for k,rr in moments.items():
        pred=uncond[k]
        for b in range(4):
            row={z:pred[z] for z in ('family','d','load','alpha','method')}; row['age_bin']=b
            for metric in ('m','Q'):
                av,se=mean_se([rr[s][metric+'_bins'][b] for s in TEST_SEEDS])
                cp=mean_se([rr[s][metric+'_cond_bins'][b] for s in TEST_SEEDS])[0]
                up=pred[metric+'_bins'][b]
                res,rse=mean_se([rr[s][metric+'_bins'][b]-rr[s][metric+'_cond_bins'][b] for s in TEST_SEEDS])
                row.update({metric+'_mean':av,metric+'_se':se,metric+'_conditional':cp,
                    metric+'_unconditional':up,metric+'_conditional_relative_error':abs(av-cp)/abs(cp),
                    metric+'_unconditional_relative_error':abs(av-up)/abs(up),metric+'_paired_residual_se':rse})
            for field in ('m_variance_pred_bins','m_squared_residual_bins','Q_squared_residual_bins',
                          'rho_sd_bins','max_wrong_scaled_bins'):
                row[field]=mean_se([rr[s][field][b] for s in TEST_SEEDS])[0]
            momentrows.append(row)
    write_csv('moment_summary.csv',momentrows)
    orows=[]
    for k,rr in oracle.items():
        base=lookup[k]; row={z:base[z] for z in ('family','d','load','alpha','method','readout','p','mean','se','predicted','abs_error')}
        for field in ('oracle_w','oracle_rho','coupling_bound_mean','coupling_bound_max','coupling_nonvacuous_fraction'):
            row[field]=mean_se([rr[s][field] for s in TEST_SEEDS])[0]
        for field in ('oracle_w','oracle_rho'):
            res,rse=mean_se([rr[s]['observed']-rr[s][field] for s in TEST_SEEDS])
            row[field+'_error']=abs(res); row[field+'_paired_residual_se']=rse
        samples=sum(rr[s]['MC_samples'] for s in TEST_SEEDS)
        row['MC_samples']=samples; row['MC_99pct_simultaneous_error']=float(np.sqrt(np.log(2*24/.01)/(2*samples)))
        orows.append(row)
    write_csv('oracle_summary.csv',orows)
    mainrows=[r for r in rows if primary(r)]; maincells=[r for r in cells if primary(r)]
    mainmom=[r for r in momentrows if r['family']=='bounded' and r['alpha']>=1]
    assert len(mainrows)==432 and len(maincells)==54 and len(mainmom)==576
    coverage=[]
    for r in READOUTS[:-1]:
        sub=[x for x in maincells if x['readout']==r and x['d']>=128]
        assert len(sub)==12
        coverage.append({'readout':r,'passing':sum(x['P1_cell_pass'] for x in sub),'total':12,
            'mean_gain':float(np.mean([x['gain'] for x in sub]))})
    near_rate=float(np.mean([r['near_factor']<=2 for r in maincells]))
    age=[{'d':d,'gap':lookup['bounded',d,8,0.,'delta_0.8','full']['age3']-
          lookup['bounded',d,8,0.,'delta_0.8','full']['age0']} for d in DIMS]
    gates={'P1':all(r['passing']>=9 and r['mean_gain']>=.08 for r in coverage),
        'P2_conditional':all(r['abs_error']<=.03 for r in mainrows),
        'P2_forecast':all(r['forecast_error']<=.03 for r in mainrows),
        'P3_conditional':all(r['gain_error']<=.03 for r in maincells),
        'P3_forecast':all(r['forecast_gain_error']<=.03 for r in maincells),
        'P4':all(r['regret']<=.03 for r in maincells) and near_rate>=.8,
        'P5':all(r['gap']>=.15 for r in age),
        'P7_oracle':all(r['oracle_w_error']<=.03 for r in orows)}
    for metric in ('m','Q'):
        for kind in ('conditional','unconditional'):
            gates[f'P6{metric}_{kind}']=all(r[f'{metric}_{kind}_relative_error']<=.05 for r in mainmom)
    numeric=json.loads((OUT/'numerical_audit.json').read_text())
    gates['N0']=max(r['quadrature_max'] for r in numeric)<=.002 and max(r['interpolation_max'] for r in numeric)<=.0005
    stats={}
    for name,group,field,cutoff in (
        ('P2_conditional',mainrows,'abs_error',.03),('P2_forecast',mainrows,'forecast_error',.03),
        ('P3_conditional',maincells,'gain_error',.03),('P3_forecast',maincells,'forecast_gain_error',.03),
        ('P4_regret',maincells,'regret',.03),('P7_oracle_w',orows,'oracle_w_error',.03),
        ('oracle_rho',orows,'oracle_rho_error',.03)):
        stats[name]={'total':len(group),'fail':sum(r[field]>cutoff for r in group),
            'max_error':max(r[field] for r in group),'mean_error':float(np.mean([r[field] for r in group])),
            'worst':sorted(group,key=lambda r:r[field],reverse=True)[:5]}
    for metric in ('m','Q'):
        for kind in ('conditional','unconditional'):
            field=f'{metric}_{kind}_relative_error'; name=f'P6{metric}_{kind}'
            stats[name]={'total':576,'fail':sum(r[field]>.05 for r in mainmom),
                'max_error':max(r[field] for r in mainmom),
                'worst':sorted(mainmom,key=lambda r:r[field],reverse=True)[:5]}
    strata=[]
    for family in FAMILIES:
        for r in READOUTS:
            sub=[x for x in rows if x['family']==family and x['readout']==r and x['alpha']>=1]
            strata.append({'family':family,'readout':r,'total':len(sub),
                'conditional_fail':sum(x['abs_error']>.03 for x in sub),
                'conditional_max':max(x['abs_error'] for x in sub),
                'forecast_fail':sum(x['forecast_error']>.03 for x in sub),
                'forecast_max':max(x['forecast_error'] for x in sub)})
    mstrata=[]
    for family in FAMILIES:
        for d in DIMS:
            for method in METHODS:
                sub=[x for x in momentrows if x['family']==family and x['d']==d and x['method']==method and x['alpha']>=1]
                row={'family':family,'d':d,'method':method,'total':len(sub)}
                for metric in ('m','Q'):
                    for kind in ('conditional','unconditional'):
                        field=f'{metric}_{kind}_relative_error'
                        row[field+'_fail']=sum(x[field]>.05 for x in sub)
                        row[field+'_max']=max(x[field] for x in sub)
                mstrata.append(row)
    core=[k for k in gates if k not in ('P5','P7_oracle')]
    summary={'gates':gates,'killer_pass':all(gates[k] for k in core),
        'H1':gates['P1'],'H2':all(gates[k] for k in ('P2_conditional','P2_forecast','P3_conditional','P3_forecast','P4')),
        'coverage':coverage,'stats':stats,'near_fraction':near_rate,
        'boundary_best_count':sum(r['best_boundary'] for r in maincells),'age_gaps':age,
        'strata':strata,'moment_strata':mstrata,
        'oracle_all_bounds_vacuous':all(r['coupling_nonvacuous_fraction']==0 for r in orows),
        'oracle_MC_max_bound':max(r['MC_99pct_simultaneous_error'] for r in orows),
        'raw_records':count,'moment_records':mcount,'oracle_records':ocount,
        'max_quadrature_difference':max(r['quadrature_max'] for r in numeric),
        'max_interpolation_difference':max(r['interpolation_max'] for r in numeric),
        'limitations':'No d^4 variance tensor on main grid; oracle uses true U; no proven nonvacuous non-oracle recall bound.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    finish(state,cumulative_before_analysis=prior_elapsed('analysis'))
    print(json.dumps({'gates':gates,'coverage':coverage,'stats':{k:{a:b for a,b in v.items() if a!='worst'} for k,v in stats.items()},
                      'near_fraction':near_rate,'age_gaps':age},indent=2))

if __name__=='__main__': main()
