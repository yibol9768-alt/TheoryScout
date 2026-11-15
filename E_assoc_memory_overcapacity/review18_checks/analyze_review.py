"""Summarize independent MC and independently re-read original formal ranks."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from scipy.stats import chi2, norm, t
from independent_mc import ROOT, OUT, coefficients, resources, write_json

start = time.monotonic()
params = coefficients()
datasets = {d:json.loads((OUT/f'mc_d{d}.json').read_text()) for d in (64,128,256,512)}
arrays = {d:np.load(OUT/f'mc_d{d}.npz',allow_pickle=False)['groups'] for d in datasets}
allrows = [row for result in datasets.values() for row in result['results']]

def csv_write(name, rows):
    with (OUT/name).open('x',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

csv_write('mc_summary.csv',[{k:v for k,v in row.items() if k!='raw_moments'} for row in allrows])
fits = []
moments = []
for index, key in enumerate(params):
    r, method, task = key
    cfg = params[key]
    rows = [next(v for v in datasets[d]['results'] if
                 (v['r'],v['method'],v['task'])==key) for d in datasets]
    dvals = np.array([v['d'] for v in rows])
    y = np.array([v['scaled_bias'] for v in rows])
    se = np.array([v['scaled_se'] for v in rows])
    for power in (.5,1.):
        design = np.column_stack([np.ones(4),dvals.astype(float)**(-power)])
        covariance = np.linalg.inv((design.T/se**2)@design)
        estimate = covariance@((design.T/se**2)@y)
        residual = y-design@estimate
        chisq = float(np.sum((residual/se)**2))
        fits.append(dict(r=r,method=method,task=task,power=power,BR=cfg['BR'],
                         intercept=float(estimate[0]),se=float(np.sqrt(covariance[0,0])),
                         lower95=float(estimate[0]-1.96*np.sqrt(covariance[0,0])),
                         upper95=float(estimate[0]+1.96*np.sqrt(covariance[0,0])),
                         slope=float(estimate[1]),chisq=chisq,df=2,
                         descriptive_p=float(chi2.sf(chisq,2))))
    for d,row in zip(dvals,rows):
        raw = np.array(row['raw_moments'])
        meanA,meanB=raw[2:4]
        moments.append(dict(d=int(d),r=r,method=method,task=task,
                            meanA_coefficient=float(d*(meanA-cfg['mu'])),target_m=cfg['m'],
                            meanB_coefficient=float(d*(meanB-1)),target_ell=cfg['ell'],
                            varA_coefficient=float(d*(raw[4]-meanA**2)),
                            target_varA=4+3*cfg['mu']**2,
                            varB_coefficient=float(d*(raw[5]-meanB**2)),target_varB=4.,
                            covAB_coefficient=float(d*(raw[6]-meanA*meanB)),target_covAB=4*cfg['mu'],
                            projection_coefficient=float(d*raw[7]),target_projection=4+cfg['mu']**2,
                            lambda_second_coefficient=float(d*raw[9]),target_lambda_second=4.,
                            lambda_third_scaled=float(d**1.5*raw[10])))
csv_write('fits.csv',fits)
csv_write('moment_checks.csv',moments)

# Independently recover observed differences from stored success and rank arrays.
authored = json.loads((ROOT/'z2b_checks/Z2B_outer_result.json').read_text())
formal = []
author_arithmetic = []
rawrows = list(csv.DictReader((ROOT/'z2b_checks/Z2B_outer_rows.csv').open()))
diagnostics = list(csv.DictReader((ROOT/'y3d/formal/analysis/diagnostics72.csv').open()))
secondary = list(csv.DictReader((ROOT/'y3d/formal/analysis/secondary_old45.csv').open()))
input_hashes = {}
for r, ir in [(0.,1),(-.5,2)]:
    values = []
    gd_values, sign_values = [], []
    queries = []
    for path in sorted((ROOT/'y3d/formal').glob(f'd512_r{ir}_s*.npz')):
        input_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as f:
            gd = np.asarray(f['GD_old_success'],dtype=float)
            sign = np.asarray(f['Sign_old_success'],dtype=float)
            assert np.array_equal(gd.astype(bool),f['GD_old_ranks']<=math.ceil(.1*(512*512//4)))
            assert np.array_equal(sign.astype(bool),f['Sign_old_ranks']<=math.ceil(.1*(512*512//4)))
            values.append(float(np.mean(gd-sign)))
            gd_values.append(float(gd.mean()))
            sign_values.append(float(sign.mean()))
            queries.append(len(gd))
    values=np.array(values)
    n=len(values)
    sd=float(values.std(ddof=1))
    se=sd/math.sqrt(n)
    mean=float(values.mean())
    a=next(v for v in authored['results'] if v['r']==r)
    desc=next(v for v in diagnostics if v['d']=='512' and int(v['ir'])==ir
              and v['task']=='old' and v['label']=='GD-Sign')
    assert abs(mean-float(desc['observed']))<1e-15
    assert abs(se-float(desc['se']))<1e-15
    for method,vs in [('GD',gd_values),('Sign',sign_values)]:
        s=next(v for v in secondary if v['d']=='512' and int(v['ir'])==ir and v['method']==method)
        assert abs(np.mean(vs)-float(s['mean']))<1e-15
    idxgd=list(params).index((r,'GD','old'))
    idxsign=list(params).index((r,'Sign','old'))
    difference=arrays[512][:,idxgd,0]-arrays[512][:,idxsign,1]
    difference_upper=arrays[512][:,idxgd,1]-arrays[512][:,idxsign,0]
    middle=(difference+difference_upper)/2
    mcmean=float(middle.mean())
    mcse=float(middle.std(ddof=1)/math.sqrt(len(middle)))
    logterm=math.log(2/.0025)
    independent_radius=(math.sqrt(2*difference.var(ddof=1)*logterm/len(difference))+
                        14*logterm/(3*(len(difference)-1)))
    independent_radius_upper=(math.sqrt(2*difference_upper.var(ddof=1)*logterm/len(difference_upper))+
                              14*logterm/(3*(len(difference_upper)-1)))
    formal.append(dict(r=r,seeds=n,queries_per_seed=queries[0],GD=float(np.mean(gd_values)),
                       Sign=float(np.mean(sign_values)),difference=mean,se=se,
                       descriptive_lower95=mean-t.ppf(.975,n-1)*se,
                       descriptive_upper95=mean+t.ppf(.975,n-1)*se,
                       author_lower=a['interval_lower'],author_upper=a['interval_upper'],
                       mean_is_covered=a['interval_lower']<=mean<=a['interval_upper'],
                       individual_seeds_covered=int(np.count_nonzero((values>=a['interval_lower'])&(values<=a['interval_upper']))),
                       mc_mean=mcmean,mc_se=mcse,mc_lower99=float(difference.mean()-independent_radius),
                       mc_upper99=float(difference_upper.mean()+independent_radius_upper),
                       author_vs_independent_z=(a['mean_lower']-mcmean)/math.sqrt(mcse**2+a['sd_lower']**2/2048),
                       scaled_difference=512*(mean-a['limit']),
                       BR_difference=params[(r,'GD','old')]['BR']-params[(r,'Sign','old')]['BR']))
    selected=[v for v in rawrows if float(v['r'])==r]
    for side in ('lower','upper'):
        vec=np.array([float(v[side]) for v in selected])
        radius=math.sqrt(2*vec.var(ddof=1)*logterm/len(vec))+14*logterm/(3*(len(vec)-1))
        endpoint=float(vec.mean()+(-radius if side=='lower' else radius))
        reference=a['interval_'+side]
        author_arithmetic.append(dict(r=r,side=side,mean=float(vec.mean()),radius=radius,
                                      endpoint=endpoint,reference=reference,error=endpoint-reference))
csv_write('formal_comparison.csv',formal)
write_json('interval_arithmetic.json',author_arithmetic)

# An exported scientific figure, plotting the residual relative to the claimed
# coefficient. Error bars describe MC error, not the asymptotic remainder.
os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
for ax,(r,task) in zip(axes.flat,[(0.,'old'),(0.,'new'),(-.5,'old'),(-.5,'new')]):
    for method in ('GD','Sign','Adam.1','Adam1'):
        row=[v for v in allrows if (v['r'],v['method'],v['task'])==(r,method,task)]
        ax.errorbar([v['d']**-.5 for v in row],
                    [v['scaled_bias']-v['BR'] for v in row],
                    yerr=[1.96*v['scaled_se'] for v in row],marker='o',capsize=2,label=method)
    ax.axhline(0,color='black',linewidth=.7)
    ax.set_title(f'r={r:g}, {task}')
    ax.grid(alpha=.2)
    ax.set_ylabel('d * (E recall - Rinf) - BR')
    ax.set_xlabel('1 / sqrt(d)')
axes[0,0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'scaled_bias.png',dpi=180)
fig.savefig(OUT/'scaled_bias.pdf')
plt.close(fig)

evidence_files=['Z2B_BR_PROOF.md','Z1_G1_PROOF.md','Z2_RECALL_BRIDGE.md',
                'Y3_FINITE_D_NONLINEAR.md','PROOF_REVIEW13.md','PROOF_REVIEW16.md',
                'y3_checks/Y3_coefficients.csv','z2b_checks/Z2B_residual.py',
                'z2b_checks/Z2B_audit.py','z2b_checks/Z2B_design.json',
                'z2b_checks/Z2B_outer_rows.csv','z2b_checks/Z2B_outer_result.json',
                'y3d/formal/analysis/diagnostics72.csv','y3d/formal/analysis/secondary_old45.csv']
for name in evidence_files:
    input_hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
write_json('input_hashes.json',input_hashes)
write_json('analysis_summary.json',dict(fits=fits,formal=formal,
                                      max_coefficient_discrepancy=max(v['max_reference_error'] for v in allrows),
                                      total_outer=sum(v['outer'] for v in datasets.values()),
                                      total_queries=sum(v['outer']*v['inner'] for v in datasets.values()),
                                      probability_evaluations=sum(v['outer']*v['inner']*16 for v in datasets.values()),
                                      simulation_seconds=sum(v['wall_seconds'] for v in datasets.values()),
                                      simulation_cpu_seconds=sum(v['resources']['cpu_seconds'] for v in datasets.values()),
                                      analysis_seconds=time.monotonic()-start,resources=resources()))

lines=['# Independent Monte Carlo output','',
       'Values below use recall probability, not percentage points. Errors after ± are one Monte Carlo SE.',
       '', '|r|method|task|BR|d64|d128|d256|d512|','|---|---|---|---:|---:|---:|---:|---:|']
for key in params:
    row=[v for v in allrows if (v['r'],v['method'],v['task'])==key]
    lines.append('|'+ '|'.join([str(key[0]),key[1],key[2],f"{row[0]['BR']:.5f}"]+
                              [f"{v['scaled_bias']:.5f} ± {v['scaled_se']:.5f}" for v in row])+'|')
lines+=['','|r|method|task|BR|intercept in 1/sqrt(d), ± 1 SE|intercept in 1/d, ± 1 SE|',
        '|---|---|---|---:|---:|---:|']
for key in params:
    f=[v for v in fits if (v['r'],v['method'],v['task'])==key]
    lines.append('|'+ '|'.join([str(key[0]),key[1],key[2],f"{f[0]['BR']:.5f}"]+
                              [f"{v['intercept']:.5f} ± {v['se']:.5f}" for v in f])+'|')
with (OUT/'numerical_tables.md').open('x') as f:
    f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(formal=formal,max_coefficient_discrepancy=max(v['max_reference_error'] for v in allrows),
                      analysis_seconds=time.monotonic()-start),ensure_ascii=False),flush=True)
