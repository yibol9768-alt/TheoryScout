"""Descriptive cell uncertainties and model-null sampling audit; no refitting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import resource,json,csv,math
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
root=Path(__file__).resolve().parent
seen=set(); rows=[]; out=[]
with (root/'formal/results.jsonl').open() as f:
    for line in f:
        try: d=json.loads(line)
        except json.JSONDecodeError: break
        key=(d['layer'],d['seed'],d['G'],tuple(d['channel']))
        if key in seen: continue
        seen.add(key)
        for kind in ('homogeneous','extended'):
            par=d['calibration'][kind]; la=par['lambda_hat']; ws=par['ws_hat']
            for cell in d['calibration']['validation']:
                a=(la+(ws if cell['t']==0 else 1))/(la+cell['ns']*ws+cell['nr']); empirical=cell['hits']/cell['n']
                rows.append(dict(layer=d['layer'],seed=d['seed'],G=d['G'],channel=str(d['channel']),model=kind,**cell,predicted_recall=a,empirical_recall=empirical,absolute_residual=abs(a-empirical),binomial_se=math.sqrt(a*(1-a)/cell['n'])))
        # Null calculation only when E1 generator is known, including correct scale.
        if d['layer']=='E1':
            la,ws,wr=d['channel']; cells=d['calibration']['validation']; nn=np.array([x['n'] for x in cells]); pp=np.array([(la+(ws if x['t']==0 else wr))/(la+x['ns']*ws+x['nr']*wr) for x in cells])
            rr=np.random.default_rng(np.random.SeedSequence([d['seed'],d['G'],int(la),991])); maxima=[]
            for start in range(0,2000,128):
                sample=rr.binomial(nn,pp,size=(min(128,2000-start),len(nn))); maxima.extend(np.max(np.abs(sample/nn-pp),axis=1).tolist())
            out.append(dict(seed=d['seed'],G=d['G'],channel=d['channel'],min_cell_n=int(nn.min()),validation_cells=len(cells),oracle_observed_max_residual=float(max(abs(x['hits']/x['n']-p) for x,p in zip(cells,pp))),null_probability_max_residual_exceeds_003=float(np.mean(np.array(maxima)>.03)),null_median_max_residual=float(np.median(maxima))))
with (root/'calibration_cells.csv').open('w') as f:
    if rows:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(root/'calibration_sampling_diagnostic.json').write_text(json.dumps({'scope':'post hoc descriptive sampling audit; original thresholds unchanged; not a new support test','E1_model_null':out},indent=2))
print(json.dumps({'unique_calibrations':len(seen),'cell_rows':len(rows),'E1_null_failure_probability_range':[min((x['null_probability_max_residual_exceeds_003'] for x in out),default=None),max((x['null_probability_max_residual_exceeds_003'] for x in out),default=None)]}))
