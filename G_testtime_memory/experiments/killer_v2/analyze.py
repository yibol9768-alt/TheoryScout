"""One complete-grid, seed-level analysis; thresholds fixed in preregistration."""
import csv
import json
from collections import defaultdict
import numpy as np
from design import *
from common import OUT, begin, finish, mean_se, prior_elapsed

def csv_write(name, rows):
    with (OUT/name).open('x') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def key(r):
    return (r['family'],r['d'],r['load'],r['alpha'],r['method'],r['readout'])

def moment_key(r):
    return (r['family'],r['d'],r['load'],r['alpha'],r['method'])

def primary(r):
    return r['family']=='bounded' and r['alpha']>=1 and r['readout']!='full'

def main():
    state = begin('analysis')
    assert json.loads((OUT/'formal_meta.json').read_text())['complete']
    predictions = {key(r):r for r in json.loads((OUT/'predictions.json').read_text())}
    assert len(predictions)==1728
    raw = defaultdict(dict)
    count = 0
    with (OUT/'raw.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            k = (r['split'],)+key(r)
            assert r['seed'] not in raw[k]
            raw[k][r['seed']] = r
            count += 1
    assert count == 32256 and len(raw)==2304
    for k,values in raw.items():
        assert set(values)==set(CAL_SEEDS if k[0]=='calibration' else TEST_SEEDS)
        if k[0]=='calibration': assert k[2]==64
    rows,lookup = [],{}
    for k,pred in predictions.items():
        rr = raw[('validation',)+k]
        av,se = mean_se([rr[s]['top1'] for s in TEST_SEEDS])
        row = {z:pred[z] for z in ('family','d','load','alpha','method','readout','p')}
        row.update({'mean':av,'se':se,'predicted':pred['predicted'],
                    'abs_error':abs(av-pred['predicted']), 'quadrature_delta':pred['quadrature_delta']})
        for b in range(4): row[f'age{b}'] = mean_se([rr[s]['age_top1'][b] for s in TEST_SEEDS])[0]
        for metric in ('top10','mean_rank'): row[metric] = mean_se([rr[s][metric] for s in TEST_SEEDS])[0]
        rows.append(row); lookup[k]=row
    csv_write('method_summary.csv',rows)
    cells = []
    for family in FAMILIES:
        for d in DIMS:
            for load in LOADS:
                for a in ALPHAS:
                    for readout in READOUTS:
                        ds = [lookup[family,d,load,a,m,readout] for m in METHODS[1:]]
                        j = max(range(len(ds)),key=lambda j:ds[j]['predicted'])
                        h = lookup[family,d,load,a,'hebb',readout]
                        chosen = ds[j]
                        dk = (family,d,load,a,chosen['method'],readout)
                        hk = (family,d,load,a,'hebb',readout)
                        gain,se = mean_se([raw[('validation',)+dk][s]['top1']-
                                           raw[('validation',)+hk][s]['top1'] for s in TEST_SEEDS])
                        best = max(r['mean'] for r in ds)
                        jb = max(range(len(ds)),key=lambda j:ds[j]['mean'])
                        near = [b for b,r in zip(BETAS,ds) if best-r['mean']<=.01]
                        ratio = min(max(BETAS[j]/b,b/BETAS[j]) for b in near)
                        pred_gain = chosen['predicted']-h['predicted']
                        cells.append({'family':family,'d':d,'load':load,'alpha':a,'readout':readout,
                                      'beta_pred':BETAS[j],'beta_best':BETAS[jb],
                                      'best_boundary':jb in (0,len(BETAS)-1),
                                      'hebb':h['mean'],'delta':chosen['mean'],'gain':gain,'gain_se':se,
                                      'lower_3se':gain-3*se,'P1_cell_pass':gain>=.05 and gain-3*se>0,
                                      'predicted_gain':pred_gain,'gain_error':abs(gain-pred_gain),
                                      'regret':best-chosen['mean'],'near_factor':ratio,
                                      'argmax_factor':max(BETAS[j]/BETAS[jb],BETAS[jb]/BETAS[j])})
    csv_write('cell_summary.csv',cells)
    moment_raw = defaultdict(dict)
    moment_count = 0
    with (OUT/'moments.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            k = (r['split'],)+moment_key(r)
            assert r['seed'] not in moment_raw[k]
            moment_raw[k][r['seed']] = r
            moment_count += 1
    assert moment_count==8064 and len(moment_raw)==576
    for k,values in moment_raw.items():
        assert set(values)==set(CAL_SEEDS if k[0]=='calibration' else TEST_SEEDS)
    moment_rows = []
    for split,seeds in (('calibration',CAL_SEEDS),('validation',TEST_SEEDS)):
        for k,rr in moment_raw.items():
            if k[0]!=split: continue
            pk = k[1:]+('tau4',)
            pred = predictions[pk]
            for b in range(4):
                row = {'split':split,**{z:pred[z] for z in ('family','d','load','alpha','method')},'age_bin':b}
                for metric in ('m','Q'):
                    av,se = mean_se([rr[s][metric+'_bins'][b] for s in seeds])
                    p = pred[metric+'_bins'][b]
                    row.update({metric+'_mean':av,metric+'_se':se,metric+'_predicted':p,
                                metric+'_absolute_error':abs(av-p),
                                metric+'_relative_error':abs(av-p)/abs(p)})
                row['max_wrong_scaled_mean'] = mean_se([rr[s]['max_wrong_scaled_bins'][b] for s in seeds])[0]
                row['no_feedback_Q_predicted'] = (pred['no_feedback_Q_bins'][b] if pred['method']!='hebb' else None)
                row['no_feedback_Q_relative_error'] = (abs(row['Q_mean']-row['no_feedback_Q_predicted'])/
                                                       abs(row['no_feedback_Q_predicted'])
                                                       if pred['method']!='hebb' else None)
                moment_rows.append(row)
    csv_write('moment_summary.csv',moment_rows)
    maincells = [r for r in cells if primary(r)]
    absrows = [r for r in rows if primary(r)]
    mainmom = [r for r in moment_rows if r['split']=='validation' and r['family']=='bounded' and r['alpha']>=1]
    assert len(maincells)==54 and len(absrows)==432 and len(mainmom)==576
    coverage = []
    for readout in READOUTS[:-1]:
        rs = [r for r in maincells if r['readout']==readout and r['d']>=128]
        assert len(rs)==12
        coverage.append({'readout':readout,'passing':sum(r['P1_cell_pass'] for r in rs),
                         'total':len(rs),'mean_gain':float(np.mean([r['gain'] for r in rs]))})
    near_rate = float(np.mean([r['near_factor']<=2 for r in maincells]))
    age = [{'d':d,'gap':lookup['bounded',d,8,0.,'delta_0.8','full']['age3']-
                        lookup['bounded',d,8,0.,'delta_0.8','full']['age0']} for d in DIMS]
    gates = {'P1':all(r['passing']>=9 and r['mean_gain']>=.08 for r in coverage),
             'P2':all(r['abs_error']<=.03 for r in absrows),
             'P3':all(r['gain_error']<=.03 for r in maincells),
             'P4':all(r['regret']<=.03 for r in maincells) and near_rate>=.8,
             'P5':all(r['gap']>=.15 for r in age),
             'P6m':all(r['m_relative_error']<=.05 for r in mainmom),
             'P6Q':all(r['Q_relative_error']<=.05 for r in mainmom)}
    strata = []
    for family in FAMILIES:
        for readout in READOUTS:
            rr = [r for r in rows if r['family']==family and r['readout']==readout and r['alpha']>=1]
            cc = [r for r in cells if r['family']==family and r['readout']==readout and r['alpha']>=1]
            strata.append({'family':family,'readout':readout,'max_top1_error':max(r['abs_error'] for r in rr),
                           'top1_fail':sum(r['abs_error']>.03 for r in rr),'top1_total':len(rr),
                           'max_gain_error':max(r['gain_error'] for r in cc),
                           'max_regret':max(r['regret'] for r in cc)})
    moment_strata = []
    for family in FAMILIES:
        for d in DIMS:
            for method in METHODS:
                rr = [r for r in moment_rows if r['split']=='validation' and r['family']==family and r['d']==d and r['method']==method and r['alpha']>=1]
                moment_strata.append({'family':family,'d':d,'method':method,'total':len(rr),
                                       'm_fail':sum(r['m_relative_error']>.05 for r in rr),
                                       'Q_fail':sum(r['Q_relative_error']>.05 for r in rr),
                                       'm_max_relative_error':max(r['m_relative_error'] for r in rr),
                                       'Q_max_relative_error':max(r['Q_relative_error'] for r in rr)})
    summary = {'gates':gates,'H1':gates['P1'],'H2':all(gates[k] for k in ('P2','P3','P4')),
               'feedback_pass':gates['P6m'] and gates['P6Q'],
               'killer_pass':all(gates[k] for k in ('P1','P2','P3','P4','P6m','P6Q')),
               'P1_coverage':coverage,'P2_total':len(absrows),
               'P2_fail_count':sum(r['abs_error']>.03 for r in absrows),
               'P2_max_error':max(r['abs_error'] for r in absrows),
               'P3_total':len(maincells),'P3_fail_count':sum(r['gain_error']>.03 for r in maincells),
               'P3_max_error':max(r['gain_error'] for r in maincells),
               'P4_regret_fail_count':sum(r['regret']>.03 for r in maincells),
               'P4_max_regret':max(r['regret'] for r in maincells),
               'P4_near_fraction':near_rate,'P4_best_boundary_count':sum(r['best_boundary'] for r in maincells),
               'P5_age_gaps':age,'moment_total':len(mainmom),
               'P6m_fail_count':sum(r['m_relative_error']>.05 for r in mainmom),
               'P6Q_fail_count':sum(r['Q_relative_error']>.05 for r in mainmom),
               'P6m_max_error':max(r['m_relative_error'] for r in mainmom),
               'P6Q_max_error':max(r['Q_relative_error'] for r in mainmom),
               'strata':strata,'moment_strata':moment_strata,
               'worst_top1':sorted(absrows,key=lambda r:r['abs_error'],reverse=True)[:10],
               'worst_m':sorted(mainmom,key=lambda r:r['m_relative_error'],reverse=True)[:10],
               'worst_Q':sorted(mainmom,key=lambda r:r['Q_relative_error'],reverse=True)[:10],
               'raw_records':count,'moment_records':moment_count,
               'calibration_note':'No parameter fitting; calibration observations do not change predictions or gates.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    finish(state, cumulative_before_analysis=prior_elapsed('analysis'))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('worst_top1','worst_m','worst_Q','strata','moment_strata')},indent=2))

if __name__=='__main__': main()
