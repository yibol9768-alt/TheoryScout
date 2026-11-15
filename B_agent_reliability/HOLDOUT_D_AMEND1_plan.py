"""M1: offline only, no clients/imports from execution code; exclusive new outputs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='2'
import csv, json, hashlib, math
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'HOLDOUT_D_DESIGN.json').read_text())
N=144; B=10000; OUTER=300
BASE=ROOT/'experiments/real/analysis_length'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def training():
    tasks=list(csv.DictReader((BASE/'ANALYSIS_LENGTH_C_tasks.csv').open()))
    steps=list(csv.DictReader((BASE/'ANALYSIS_LENGTH_C_steps.csv').open()))
    by=defaultdict(list)
    for s in steps:
        if int(s['resolved']): by[s['batch'],s['id']].append(s)
    rows=[]; groups=defaultdict(list); batches=defaultdict(list)
    for t in tasks:
        row=[]
        for early in (True,False):
            z=[s for s in by[t['batch'],t['id']] if (int(s['step'])<16)==early]
            e=sum(int(s['error']) for s in z)
            m=sum(s['r2']!='' for s in z)
            row.append([len(z),sum(int(s['fatal']) for s in z),e,m,sum(int(s['r2']) for s in z if s['r2']!='')])
        groups[t['batch'],t['length']].append(len(rows)); batches[t['batch']].append(len(rows));rows.append(row)
    rows=np.array(rows,dtype=float)
    assert rows.sum(0).tolist()==[[1463,34,333,72,38],[1339,14,289,56,42]]
    rng=np.random.default_rng(116990004); v=np.zeros((B,2,5))
    for ids in groups.values():
        v+=rows[np.array(ids)[rng.integers(len(ids),size=(B,len(ids)))]].sum(1)
    return v,{b:rows[ids].sum(0) for b,ids in batches.items()},groups

def kernels(v):
    n,d,e,m,s=np.moveaxis(v,-1,0)
    p=e/n;r1=(e-m)/e;r2=s/m;h=d/n
    assert np.allclose(p*(1-r1)*(1-r2),h)
    return np.stack((p,r1,r2,h),-1)

def simulate(k,rng):
    count=len(k);shape=(count,N);live=np.ones(shape,bool);c=np.zeros(shape,int);out=[]
    for j,L in enumerate((16,48)):
        p,r1,r2,h=(k[:,j,i,None] for i in range(4))
        f=rng.geometric(h,size=shape);ex=np.minimum(f,L)*live;death=(f<=L)&live
        nonfatal=ex-death
        e=rng.binomial(nonfatal,(p-h)/(1-h))
        s=rng.binomial(e,(1-r1)*r2/(r1+(1-r1)*r2))
        out.append(np.stack([ex,death,e+death,s+death,s],-1).astype(float))
        c+=nonfatal;live &= ~death
    return np.stack(out,2),c

def indicators(t,c):
    cost=t[:,:,:,[0,2,3]].sum((1,2,3));complete=cost<=9000
    main=np.array(D['comparison']['weights_log_D_over_A'])[np.searchsorted([8,16,32,64],c,side='right')].sum(1)>.541
    cal=np.ones(len(t),bool);success=[]
    for L,lo,hi in ((8,107,130),(16,85,113),(32,70,97),(64,46,74)):
        x=(c>=L).sum(1);success.append(x);cal &= (x>=lo)&(x<=hi)
    total=t.sum(1);h=total[:,:,1]/total[:,:,0];r=total[:,:,4]/np.maximum(total[:,:,3],1)
    elig=(total[:,:,0]>0).all(1)&(total[:,:,3]>=30).all(1)&(total[:,:,1]>=10).all(1)
    hs=(t[:,:,:,1]-h[:,None,:]*t[:,:,:,0])/total[:,None,:,0]
    rs=(t[:,:,:,4]-r[:,None,:]*t[:,:,:,3])/np.maximum(total[:,None,:,3],1)
    se=lambda v:np.sqrt(N/(N-1)*((v[:,:,0]-v[:,:,1])**2).sum(1))
    hp=h[:,0]-h[:,1]>1.959963984540054*se(hs)
    rp=r[:,1]-r[:,0]>1.959963984540054*se(rs)
    return cost,dict(potential_primary=main,potential_calibration=cal,potential_primary_calibration=main&cal,
                     budget_complete=complete,primary=complete&main,calibration=complete&cal,joint=complete&main&cal,
                     potential_count_eligibility=elig,count_eligibility=complete&elig,
                     fatal_normal=complete&elig&hp,r2_normal=complete&elig&rp,
                     mechanisms_normal=complete&elig&hp&rp,all_gates_normal=complete&main&cal&elig&hp&rp)

def actual_bootstrap(t,flags,weights):
    values=defaultdict(list);valid_counts=[]
    for i,row in enumerate(t[:OUTER]):
        v=(weights@row.reshape(N,10)).reshape(B,2,5)
        valid=(v[:,:,0]>0).all(1)&(v[:,:,3]>0).all(1);valid_counts.append(int(valid.sum()))
        eligible=bool(flags['potential_count_eligibility'][i] and valid.sum()>=9900)
        hp=rp=False
        if valid.sum()>=9900:
            z=v[valid];h=z[:,:,1]/z[:,:,0];r=z[:,:,4]/z[:,:,3]
            hp=bool(np.quantile(h[:,0]-h[:,1],.025,method='linear')>0)
            rp=bool(np.quantile(r[:,1]-r[:,0],.025,method='linear')>0)
        complete=flags['budget_complete'][i]
        for name,x in dict(potential_bootstrap_eligible=eligible,bootstrap_eligible=complete&eligible,
                           fatal_bootstrap=complete&eligible&hp,r2_bootstrap=complete&eligible&rp,
                           mechanisms_bootstrap=complete&eligible&hp&rp,
                           all_gates_bootstrap=flags['joint'][i]&eligible&hp&rp).items(): values[name].append(bool(x))
    return dict(outer=OUTER,inner=B,bootstrap_seed_each_experiment=116990005,
                rates={k:float(np.mean(v)) for k,v in values.items()},counts={k:sum(v) for k,v in values.items()},
                valid_min=min(valid_counts),valid_max=max(valid_counts),
                normal_same_outer={k:float(v[:OUTER].mean()) for k,v in flags.items()},
                max_MC_SE=math.sqrt(.25/OUTER),note='实际冻结配对整题bootstrap，有限外层模拟；不是精确认证概率')

def main():
    dest=ROOT/'HOLDOUT_D_AMEND1_RESULTS.json'
    assert not dest.exists()
    v,batches,groups=training(); ks=kernels(v)
    rng=np.random.default_rng(116990005)
    ids=rng.integers(N,size=(B,N))
    weights=np.zeros((B,N),float)
    np.add.at(weights,(np.arange(B)[:,None],ids),1)
    point=np.array([[D['models']['M_D'][s][key] for key in ('p','r1','r2','h')] for s in ('E','T')])
    scenarios={'parameter_resampling':(ks,'训练分层整题参数混合，非后验/非跨批保证'),
               'point_reference':(np.broadcast_to(point,(B,2,4)).copy(),'冻结点参数参考')}
    weak=point.copy();weak[:,3]=[.018,.014];weak[:,2]=1-weak[:,3]/(weak[:,0]*(1-weak[:,1]))
    scenarios['weak_effect_stress']=(np.broadcast_to(weak,(B,2,4)).copy(),'较小分段效应压力测试，无发生权重')
    for b,counts in batches.items():
        scenarios['batch_'+b+'_stress']=(np.broadcast_to(kernels(counts),(B,2,4)).copy(),'仅用旧批次核的漂移压力测试，无发生权重；不是新的拟合对照')
    results={}
    for i,(name,(k,note)) in enumerate(scenarios.items()):
        seed=216990101+i;t,c=simulate(k,np.random.default_rng(seed));cost,flags=indicators(t,c)
        results[name]=dict(note=note,simulations=B,seed=seed,rates={n:float(x.mean()) for n,x in flags.items()},
                           counts={n:int(x.sum()) for n,x in flags.items()},
                           uncapped_demand_mean=float(cost.mean()),uncapped_demand_quantiles={str(q):float(np.quantile(cost,q)) for q in (.5,.95,.99,.999)},
                           uncapped_demand_max=int(cost.max()),exhaustions=int((cost>9000).sum()),
                           actual_return_cap=9000,actual_returns_mean=float(np.minimum(cost,9000).mean()),
                           mean_segment_counts=t.sum(1).mean(0).tolist(),actual_bootstrap=actual_bootstrap(t,flags,weights))
        print(name,json.dumps(results[name]['rates']),flush=True)
    result=dict(no_network=True,no_llm=True,N=N,successful_cap=9000,attempt_cap=12000,
                decision='保持N=144，接受较低规划成功率与触限缺失风险，绝不突破硬限/放宽门槛',
                group_sizes={str(k):len(x) for k,x in groups.items()},batch_counts={k:x.tolist() for k,x in batches.items()},
                joint_kernel_columns=['p','r1','r2','h'],kernel_quantiles=np.quantile(ks,[.025,.5,.975],axis=0).tolist(),
                parameter_seed=116990004,simulations_per_scenario=B,max_MC_SE=math.sqrt(.25/B),
                scope='一次参数抽样由未来144题共用；模拟停止/修复及潜在未截断需求，超过9000的实验所有确认门槛均失败。无传输失败，12000尝试尾部不从本概率模型推算。',
                inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'HOLDOUT_D_DESIGN.json',ROOT/'HOLDOUT_D_PREREG.md',ROOT/'HOLDOUT_D_offline.py',BASE/'ANALYSIS_LENGTH_C_tasks.csv',BASE/'ANALYSIS_LENGTH_C_steps.csv']},scenarios=results)
    with dest.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__': main()
