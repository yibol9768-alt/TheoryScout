"""Summarize dataset-seed uncertainty and check closed forms independently."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import t

BASE = Path(__file__).resolve().parent
P = Path(os.environ.get('W9_OUTPUT_DIR', str(BASE))).resolve()
assert P.is_relative_to(BASE), 'Outputs must stay within w9_checks/'
rows = [json.loads(s) for s in (P/'formal_rows.jsonl').read_text().splitlines()]
meta = json.loads((P/'formal_metadata.json').read_text())
pred = json.loads((P/'formal_predictions.json').read_text())
groups = defaultdict(list)
for row in rows:
    groups[(row['case'],row['d'],row['method'])].append(row)
assert len(rows) == 600 and len(meta['completed']) == 120
assert len(groups) == 75 and all(len(v)==8 for v in groups.values())
assert all(np.isfinite(v) for row in rows for v in row.values() if isinstance(v,(int,float)))

def estimate(x):
    x = np.asarray(x)
    mean = float(x.mean())
    half = float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return mean,half

# Check numerical reference integration against exact independent formulas.
rho = .75
exact = {
    ('iso_g','gd1'): 1.,
    ('diag_g','gd1'): 1/np.sqrt(1+rho*rho),
    ('rot_g','gd1'): 1/np.sqrt(1+rho*rho),
    ('iso_g','sign1'): np.sqrt(2/np.pi),
    ('diag_g','sign1'): np.sqrt(2/np.pi)*(np.sqrt(1-rho)+np.sqrt(1+rho))/2,
    ('rot_g','sign1'): np.sqrt(2/np.pi)/np.sqrt(1+2*rho*np.arcsin(rho)/np.pi),
}
checks = []
for (case, method), value in exact.items():
    error = abs(pred[case][method]['C']-value)
    assert error < 3e-5, (case,method,error)
    checks.append(dict(case=case,method=method,exact_C=value,integration_C=pred[case][method]['C'],error=error))
for case in ['iso_g','diag_g','rot_g']:
    b = np.eye(2) if case=='iso_g' else (np.diag([.25,1.75]) if case=='diag_g' else np.array([[1.,.75],[.75,1.]]))
    lam = np.linalg.eigvalsh(b)
    g = 1-(1-.5*lam)**4
    value = np.mean(g)/np.sqrt(np.mean(g*g))
    error = abs(pred[case]['gd4']['C']-value)
    assert error < 3e-5
    checks.append(dict(case=case,method='gd4',exact_C=value,integration_C=pred[case]['gd4']['C'],error=error))
metrics=['recall10','correct_mean','correct_var','wrong_var','cdf_grid_error','rank_mean',
         'frob_error','correct_error2','wrong_error2']
summary=[]
for (case,d,method), rr in sorted(groups.items()):
    record=dict(case=case,d=d,method=method,**pred[case][method])
    for metric in metrics:
        record[metric],record[metric+'_half95']=estimate([r[metric] for r in rr])
    record['pred_mean']=record['C']/np.sqrt(.5)
    record['pred_rank_mean']=float(ndtr(-record['C']/np.sqrt(1.)))
    summary.append(record)
with (P/'summary.csv').open('x') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]))
    w.writeheader(); w.writerows(summary)
paired=[]
for d in [64,128,256]:
    for method,left,right in [
        ('sign1','iso_g','rot_g'),('sign1','diag_g','rot_g'),
        ('gd1','diag_g','rot_g'),('gd4','diag_g','rot_g'),
        ('adam4','diag_g','rot_g'),
        ('sign4','rot_radkeys','rot_g'),
        ('adam4','rot_radkeys','rot_g'),
        ('sign4','rot_radboth','rot_g'),
        ('adam4','rot_radboth','rot_g')]:
        aa={r['seed']:r['recall10'] for r in groups[(left,d,method)]}
        bb={r['seed']:r['recall10'] for r in groups[(right,d,method)]}
        mean,half=estimate([aa[s]-bb[s] for s in sorted(aa)])
        paired.append(dict(d=d,method=method,left=left,right=right,
                           delta=mean,half95=half,pred_delta=pred[left][method]['pred_recall10']-pred[right][method]['pred_recall10']))
(P/'closed_form_checks.json').write_text(json.dumps(checks,indent=2))
(P/'paired_contrasts.json').write_text(json.dumps(paired,indent=2))
lines=[
    '# W9 数值核查',
    '',
    f"正式模拟 {meta['seconds']:.2f} 秒，峰值 RSS {meta['maxrss_kib']/1024:.1f} MiB；120 个数据集、600 个算法端点。",
    'κ=.5；每个数据集 128 个训练查询、全部 n 个训练值候选；区间按 8 个独立数据集种子计算，逐项 t₇ 95% 区间，未多重校正。预测独立于训练数据。',
    '',
    '## d=256：训练检索与分数',
    '',
    '|情形|方法|C|预测召回|观测召回 ±95%半宽|忽略协方差预测|正确均值 / 预测|正确方差|错误方差|CDF网格误差|',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'
]
for r in summary:
    if r['d']==256:
        lines.append(f"|{r['case']}|{r['method']}|{r['C']:.4f}|{r['pred_recall10']:.4f}|{r['recall10']:.4f} ± {r['recall10_half95']:.4f}|{r['naive_recall10']:.4f}|{r['correct_mean']:.3f} / {r['pred_mean']:.3f}|{r['correct_var']:.3f}|{r['wrong_var']:.3f}|{r['cdf_grid_error']:.4f}|")
lines += ['', '## d=256：配对召回差', '',
          '|方法|左 − 右|观测差 ±95%半宽|预测差|','|---|---|---:|---:|']
for r in paired:
    if r['d']==256:
        lines.append(f"|{r['method']}|{r['left']} − {r['right']}|{r['delta']:.4f} ± {r['half95']:.4f}|{r['pred_delta']:.4f}|")
lines += ['', '## 旋转高斯块的轨迹和训练分数迁移', '',
          '|d|方法|Frobenius 均方误差|正确训练分数误差²|错误训练分数误差²|','|---:|---|---:|---:|---:|']
for r in summary:
    if r['case']=='rot_g' and r['method'] in ['gd4','sign4','adam4']:
        lines.append(f"|{r['d']}|{r['method']}|{r['frob_error']:.5g}|{r['correct_error2']:.5g}|{r['wrong_error2']:.5g}|")
lines += ['', '## 解释限制', '',
    'CDF 误差只是在五个固定正态阈值对查询汇总后的诊断量，不是定理要求的逐查询 sup-CDF 误差，也不是统一置信界。',
    'rot_radboth 的 Sign/Adam 是探索结果，未证明其普适性。有限维正确分数方差可有偏差；区间覆盖不等于渐近定理的证实或等价性认证。',
    '所有首步 actual=proxy 的 Frobenius/分数误差都为零；所有 9 项独立闭式核验与 Sobol 预测之差 <3e-5。',
    'summary.csv 包含全部维度、指标及种子区间；formal_rows.jsonl 是未汇总原始结果。']
assert all(r['frob_error']==0 and r['correct_error2']==0 and r['wrong_error2']==0
           for r in rows if r['method'] in ['gd1','sign1'])
(P/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
