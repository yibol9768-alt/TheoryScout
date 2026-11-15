"""Independent, focused precision check of the equal-scalar-c counterexample."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='2'
import json, time, resource
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import t

BASE=Path(__file__).resolve().parent
P=Path(os.environ.get('W9_OUTPUT_DIR',str(BASE))).resolve()
assert P.is_relative_to(BASE), 'Outputs must stay within w9_checks/'
P.mkdir(parents=True,exist_ok=True)
start=time.monotonic()
d=256; n=d*d//2; nq=1024; ns=24; alpha=.6; budget=720
rho=.75
blocks={'iso':np.eye(2),'diag':np.diag([.25,1.75]),'rot':np.array([[1.,rho],[rho,1.]])}
cs={'iso':np.sqrt(2/np.pi),
    'diag':np.sqrt(2/np.pi)*(np.sqrt(1-rho)+np.sqrt(1+rho))/2,
    'rot':np.sqrt(2/np.pi)/np.sqrt(1+2*rho*np.arcsin(rho)/np.pi)}
qs={'iso':alpha**2,'diag':alpha**2,'rot':alpha**2*(1+2*rho*np.arcsin(rho)/np.pi)}
rows=[]
with (P/'focus_rows.jsonl').open('x') as log:
    for seed in range(ns):
        if time.monotonic()-start>budget:
            break
        rng=np.random.default_rng(19000000+seed)
        base=rng.normal(size=(n,d))/np.sqrt(d)
        y=rng.normal(size=(n,d))/np.sqrt(d)
        ids=rng.choice(n,nq,replace=False)
        for case,b in blocks.items():
            x=(base.reshape(n,d//2,2)@np.linalg.cholesky(b).T).reshape(n,d)
            z=(d/np.sqrt(n))*(y.T@x)
            # Exact actual first step; no proxy replacement.
            u=alpha*np.sign(z)
            scores=(y@(u@x[ids].T))/np.sqrt(qs[case])
            correct=scores[ids,np.arange(nq)].copy()
            ranks=1+np.sum(scores>correct[None,:],axis=0)
            # Querywise sup-CDF diagnostic on a fixed 32-query subset.
            ks=[]
            for k in range(32):
                wrong=np.concatenate([scores[:ids[k],k],scores[ids[k]+1:,k]])
                f=ndtr(np.sort(wrong)); nn=len(f)
                ks.append(max(np.max(np.arange(1,nn+1)/nn-f),
                              np.max(f-np.arange(nn)/nn)))
            row=dict(seed=seed,case=case,d=d,n=n,queries=nq,
                     recall=float(np.mean(ranks<=np.ceil(.1*n))),
                     mean_score=float(correct.mean()),var_score=float(correct.var(ddof=1)),
                     query_KS_mean=float(np.mean(ks)),query_KS_max=float(np.max(ks)),
                     C=cs[case],pred=float(ndtr(cs[case]/np.sqrt(.5)-ndtri(.9))))
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps(dict(seed=seed,case=case,recall=row['recall'],elapsed=round(time.monotonic()-start,1))),flush=True)
            del scores,x
        del base,y

def est(a):
    a=np.asarray(a)
    return float(a.mean()),float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
summary=[]
for case in blocks:
    rr=[r for r in rows if r['case']==case]
    assert len(rr)>=2
    m,h=est([r['recall'] for r in rr])
    summary.append(dict(case=case,seeds=len(rr),mean=m,half95=h,pred=rr[0]['pred'],
                        query_KS_mean=float(np.mean([r['query_KS_mean'] for r in rr])),
                        correct_mean=float(np.mean([r['mean_score'] for r in rr])),
                        correct_var=float(np.mean([r['var_score'] for r in rr]))))
contrasts=[]
for a,b in [('iso','rot'),('diag','rot')]:
    aa={r['seed']:r for r in rows if r['case']==a}
    bb={r['seed']:r for r in rows if r['case']==b}
    seeds=sorted(aa.keys()&bb.keys())
    m,h=est([aa[s]['recall']-bb[s]['recall'] for s in seeds])
    contrasts.append(dict(left=a,right=b,delta=m,half95=h,
                          pred=aa[seeds[0]]['pred']-bb[seeds[0]]['pred']))
metadata=dict(d=d,n=n,queries=nq,seeds=len(rows)//3,seconds=time.monotonic()-start,
              maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              reason='After broad run: focused precision for equal marginal c and same spectrum; new independent data seeds, not preregistered.',
              seed_base=19000000,budget_seconds=budget)
(P/'focus_summary.json').write_text(json.dumps(dict(summary=summary,contrasts=contrasts,metadata=metadata),indent=2))
lines=['# W9 定向精度核查','',
       '首轮每数据集 128 个查询不足以分辨较小的基依赖差；追加独立种子、每集 1024 个查询，仅检验 Sign 首步的解析反例。此追加设计不是预注册。',
       '',f"d=256，κ=.5，{metadata['seeds']} 个配对种子；用时 {metadata['seconds']:.1f} 秒，峰值 RSS {metadata['maxrss_kib']/1024:.1f} MiB。",
       '', '|情形|预测召回|实际召回 ±95%半宽|逐查询 KS 平均|正确均值|正确方差|',
       '|---|---:|---:|---:|---:|---:|']
for r in summary:
    lines.append(f"|{r['case']}|{r['pred']:.5f}|{r['mean']:.5f} ± {r['half95']:.5f}|{r['query_KS_mean']:.4f}|{r['correct_mean']:.4f}|{r['correct_var']:.4f}|")
lines+=['','|配对差|预测|观测 ±95%半宽|','|---|---:|---:|']
for r in contrasts:
    lines.append(f"|{r['left']} − {r['right']}|{r['pred']:.5f}|{r['delta']:.5f} ± {r['half95']:.5f}|")
lines+=['','区间按数据集种子计算，逐项 t 区间；每个数据集的查询与全部训练值比较。KS 在每集固定选取的前 32 个查询上逐一计算完整错误候选 sup-CDF 差，然后按查询和数据集平均，不能用其平均值证明一致概率收敛。']
(P/'FOCUS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
