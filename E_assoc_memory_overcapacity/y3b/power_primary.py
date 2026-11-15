"""已有规划协方差上的确定性主终点功效补表；无随机抽样。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,25))
import json
import math
import numpy as np
from scipy.stats import t,nct
from common import OUT,METHODS,PAIRS,DIMS,dump,config,environment,usage
from predict import load
environment();cfg=config();pred,_=load()
covs={(r['d'],r['ir']):r for r in json.loads((OUT/'planning_covariances.json').read_text())}
rows=[]
for task,K in [('old',27),('new',45)]:
    mean=0.;variance=0.
    for d in DIMS:
        for ir in range(3):
            row=covs[min(d,256),ir];C=np.array(row['C2048']);W=np.array(row['W'])
            Q=cfg['query_target'] if (d,ir)==(512,2) else cfg['query_default']
            B=C-W/2048
            # 样本估计的负数据间方差不用于提高功效：PSD投影后再加W/Q。
            eig,vec=np.linalg.eigh(B);C=(vec*np.maximum(eig,0))@vec.T+W/Q
            weights=np.zeros(10);delta=[]
            if task=='old':
                for a,b in PAIRS:
                    p,q=pred[d,ir,a,task],pred[d,ir,b,task]
                    v=(p['recall_first_order']-p['recall_limit'])-(q['recall_first_order']-q['recall_limit'])
                    delta.append(v);weights[METHODS.index(a)]+=v;weights[METHODS.index(b)]-=v
            else:
                for j,m in enumerate(METHODS):
                    p=pred[d,ir,m,task];v=p['recall_first_order']-p['recall_limit']
                    delta.append(v);weights[5+j]=v
            mean+=sum(v*v for v in delta)/K
            variance+=float(4*weights@C@weights)/(K*K)
    nc=mean/math.sqrt(variance/cfg['blocks'])
    crit=t.ppf(1-.05/(2*cfg['confirm_family']),cfg['blocks']-1)
    rows.append(dict(task=task,N=cfg['blocks'],expected_loss_improvement=mean,block_sd=math.sqrt(variance),
                     power=float(nct.sf(crit,cfg['blocks']-1,nc))))
summary=json.loads((OUT/'power_summary.json').read_text())
dump(OUT/'primary_power.json',dict(rows=rows,primary_joint_union_lower=max(0.,sum(r['power'] for r in rows)-1),
     all_three_joint_union_lower=max(0.,sum(r['power'] for r in rows)+summary['chosen']['power']-2),
     no_random_samples=True,usage=usage()))
print(json.dumps(rows))
