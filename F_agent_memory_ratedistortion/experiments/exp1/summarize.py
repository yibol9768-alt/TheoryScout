"""Read frozen EXP1 outputs; never re-fit or change decision thresholds."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import json, csv, math, resource, hashlib
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
ROOT=Path(__file__).resolve().parent
excluded={'padded_summary','cold_archive','route_raw'}

def stats(rows,key='abs_error'):
    if not rows: return {'n':0,'mae':None,'p95':None,'max':None}
    a=np.array([r[key] for r in rows]); return dict(n=len(a),mae=float(a.mean()),p95=float(np.quantile(a,.95)),max=float(a.max()))

def summarize():
    data=[]
    with (ROOT/'formal/results.jsonl').open() as f:
        for line in f:
            try: data.append(json.loads(line))
            except json.JSONDecodeError: break
    output={}; flat=[]; paired=[]; calibration=[]; allstrata=[]
    for layer in ('E1','E2'):
        dd=[d for d in data if d['layer']==layer]; models={}
        for d in dd:
            for row in d['rows']:
                row['active_item_count']=d['G'] if row['policy']=='route_raw' else row['s']+4*row['n']+(3*d['G'] if row['policy']=='padded_summary' else 0)
                row['W_model']=row['W']
                row['W_generator']=((row['s']*d['channel'][1]+4*row['n']*d['channel'][2]+(3*d['G']*d['channel'][2] if row['policy']=='padded_summary' else 0)) if d['layer']=='E1' and row['policy']!='route_raw' else None)
                row['storage_exceeds_budget']=row['stored_bytes']>136*row['budget']
                flat.append({**{k:d[k] for k in ('id','layer','seed','G','dist','epsilon','nu_s','nu_r','v_s_hat','v_r_hat','entropy','estimated_entropy')},'channel':json.dumps(d['channel']),'domain_pass':d['calibration']['homogeneous']['domain_pass'],'extended_domain_pass':d['calibration']['extended']['domain_pass'],**row})
            for name,val in d['paired'].items(): paired.append(dict(id=d['id'],layer=layer,seed=d['seed'],contrast=name,mean=val['mean'],lo=val['ci95'][0],hi=val['ci95'][1]))
            for kind in ('homogeneous','extended'):
                key=(d['seed'],d['G'],tuple(d['channel']),kind)
                if key in models: continue
                m=d['calibration'][kind]; models[key]=m
                calibration.append(dict(layer=layer,seed=d['seed'],G=d['G'],channel=json.dumps(d['channel']),model=kind,**m,fit_trials=sum(x['n'] for x in d['calibration']['fit']),validation_trials=sum(x['n'] for x in d['calibration']['validation']),minimum_validation_cell_n=min(x['n'] for x in d['calibration']['validation'])))
        rows=[r for r in flat if r['layer']==layer and r['policy'] not in excluded]
        domain=[r for r in rows if r['domain_pass']]
        extended=[dict(r,abs_error=abs(r['accuracy']-r['extended_predicted_accuracy'])) for r in rows]
        domain_ex=[r for r in extended if r['extended_domain_pass']]
        dp=[r for r in rows if r['policy']=='dp']; dppass=[r for r in dp if r['domain_pass']]
        endpoint=[]; boundaries=[]
        for d in dd:
            rr={r['policy']:r for r in d['rows'] if r['policy'].startswith('full_')}
            delta=rr['full_summary']['predicted_accuracy']-rr['full_raw']['predicted_accuracy']
            observed=d['paired']['full_summary-full_raw']['mean']
            rec=dict(id=d['id'],seed=d['seed'],G=d['G'],dist=d['dist'],channel=d['channel'],predicted_difference=delta,observed_difference=observed,ci95=d['paired']['full_summary-full_raw']['ci95'],eligible=abs(delta)>=.02,correct=delta*observed>0,domain_pass=d['calibration']['homogeneous']['domain_pass'])
            endpoint.append(rec)
            if d['G']==32 and d['dist']==0 and d['epsilon']==4/15 and d['nu_s']==0 and d['nu_r']==0:
                boundaries.append({**rec,'summary_accuracy':rr['full_summary']['accuracy'],'raw_accuracy':rr['full_raw']['accuracy'],'summary_prediction':rr['full_summary']['predicted_accuracy'],'raw_prediction':rr['full_raw']['predicted_accuracy'],'lambda_hat':d['calibration']['homogeneous']['lambda_hat']})
        eligible=[e for e in endpoint if e['eligible']]; elig_domain=[e for e in eligible if e['domain_pass']]
        def ordering(ee): return dict(n=len(ee),correct=sum(e['correct'] for e in ee),rate=sum(e['correct'] for e in ee)/len(ee) if ee else None)
        byseed={str(s):dict(all=stats([r for r in rows if r['seed']==s]),domain=stats([r for r in domain if r['seed']==s]),dp_regret=stats([r for r in dp if r['seed']==s],'regret')) for s in (0,1,2)}
        tests=[r for r in rows if r['abs_error']>max(.015,4*r['se'])]
        extended_tests=[r for r in extended if r['abs_error']>max(.015,4*r['se'])]
        oracle=[dict(r,abs_error=abs(r['accuracy']-r['oracle_accuracy'])) for r in rows if 'oracle_accuracy' in r]
        # Eq26 uses gain on a fixed source/query set; report accuracy relation residual.
        pad=[]
        for d in dd:
            rr={r['policy']:r for r in d['rows'] if r['policy'] in ('full_raw','padded_summary','full_summary','route_raw','cold_archive')}
            beta=d['v_s_hat']/d['v_r_hat']
            residual=(2*rr['padded_summary']['accuracy']-1)-beta*(2*rr['full_raw']['accuracy']-1)
            pad.append(dict(id=d['id'],seed=d['seed'],residual_gain=residual,beta_hat=beta,summary_difference=d['paired']['full_summary-full_raw']['mean'],padded_difference=d['paired']['padded_summary-full_raw']['mean']))
        # Reader invariance compares the same frozen retrieval condition, not independent Monte Carlo.
        lookup={(d['seed'],d['G'],d['dist'],d['epsilon'],tuple(d['channel']),d['nu_s'],d['nu_r']):d for d in dd}; recallchanges=[]
        for key,d in lookup.items():
            if key[-2:]==(0.,0.): continue
            base=lookup.get((*key[:-2],0.,0.))
            if base:
                rr={r['policy']:r for r in d['rows'] if r['policy'].startswith('full_')}; bb={r['policy']:r for r in base['rows'] if r['policy'].startswith('full_')}
                recallchanges.extend(abs(rr[k]['recall']-bb[k]['recall']) for k in rr)
        output[layer]=dict(configurations=len(dd),unique_calibrations={kind:dict(n=sum(k[-1]==kind for k in models),passes=sum(m['domain_pass'] for k,m in models.items() if k[-1]==kind)) for kind in ('homogeneous','extended')},all=stats(rows),domain=stats(domain),extended_all=stats(extended),extended_domain=stats(domain_ex),dp_regret=stats(dp,'regret'),dp_regret_domain=stats(dppass,'regret'),threshold_exceedances=len(tests),extended_threshold_exceedances=len(extended_tests),oracle=stats(oracle),endpoint_ordering=ordering(eligible),endpoint_ordering_domain=ordering(elig_domain),by_seed=byseed,boundaries=boundaries,padding_residual_gain=stats([{'abs_error':abs(x['residual_gain'])} for x in pad]),reader_recall_comparisons=len(recallchanges),reader_recall_max_difference=max(recallchanges,default=None),cold_archive_identical=all(abs(d['paired']['cold_archive-full_summary']['mean'])==0 for d in dd),permutations=sorted(set(d['permutations_completed'] for d in dd if d['permutations_completed'])),endpoints=endpoint)
        for d in dd:
            rr=[r for r in d['rows'] if r['policy'] not in excluded]
            allstrata.append(dict(layer=layer,id=d['id'],seed=d['seed'],domain_pass=d['calibration']['homogeneous']['domain_pass'],mae=float(np.mean([r['abs_error'] for r in rr])),max_error=max(r['abs_error'] for r in rr),dp_max_regret=max(r['regret'] for r in rr if r['policy']=='dp')))
        output[layer]['conditions_over_05']=[x for x in allstrata if x['layer']==layer and x['domain_pass'] and (x['mae']>.05 or x['dp_max_regret']>.05)]
    def write_csv(name,rows):
        if not rows: return
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (ROOT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    write_csv('metrics.csv',flat); write_csv('paired_seed_differences.csv',paired); write_csv('calibration.csv',calibration); write_csv('condition_summary.csv',allstrata)
    output['e0']=json.loads((ROOT/'checks_v2_results.json').read_text())
    if (ROOT/'formal/metadata.json').exists(): output['resources']=json.loads((ROOT/'formal/metadata.json').read_text())
    if (ROOT/'formal/status.json').exists():
        st=json.loads((ROOT/'formal/status.json').read_text()); output['status']={layer:{s:sum(x['id'].startswith(layer) and x['state']==s for x in st) for s in sorted(set(x['state'] for x in st))} for layer in ('E1','E2')}
    (ROOT/'summary.json').write_text(json.dumps(output,indent=2,ensure_ascii=False))
    for layer in ('E1','E2'):
        print(layer,json.dumps({k:v for k,v in output[layer].items() if k not in ('endpoints','boundaries','conditions_over_05')},ensure_ascii=False))
if __name__=='__main__': summarize()
