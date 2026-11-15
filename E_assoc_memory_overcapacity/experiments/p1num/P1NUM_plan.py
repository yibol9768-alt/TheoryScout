import csv,json,math,statistics
from pathlib import Path
import numpy as np
from scipy.stats import norm,t
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/p1num'
def read(p):return list(csv.DictReader((ROOT/p).open()))
def stat(x):return float(np.mean(x)),float(np.std(x,ddof=1))
def required(sd,bias,margin,alpha):
 for n in range(8,10001):
  se=sd/math.sqrt(n);z=t.ppf(1-alpha/2,n-1)
  power=norm.cdf((margin-z*se-bias)/se)-norm.cdf((-margin+z*se-bias)/se)
  if power>=.8:return n,float(power)
 return 10000,0.
def main():
 unresolved=[r for r in read('experiments/exp2/old_76_retest.csv') if r['noise_class']=='未分辨'];assert len(unresolved)==37
 selected=[]
 for k in (.5,1.,2.):
  for family in ('Adam','Sign'):
   r=min((r for r in unresolved if float(r['kappa'])==k and r['name'].startswith(family)),key=lambda r:(r['name'],r['metric']))
   selected.append({key:r[key] for key in ('kappa','name','metric')})
 decomp=read('experiments/exp2/P3_seed_decomposition.csv');power=[]
 for r in selected:
  old=[s for s in decomp if int(s['d'])==256 and float(s['kappa'])==float(r['kappa']) and s['name']==r['name'] and s['metric']==r['metric']]
  D=np.array([float(s['D']) for s in old]);v=np.array([float(s['v']) for s in old]);n0=512
  # Conservative: max empirical seed variance, estimated query variance; inflate 2x. No benefit from more queries assumed for P3.
  sd=math.sqrt(2*max(float(D.var(ddof=1)),float(v.mean())))
  n,p=required(sd,0.,.005,.05/6)
  power.append(dict(kind='P3',**r,d=256,M=512,N=n,old_N=len(D),old_mean=float(D.mean()),planning_sd=sd,power_at_zero_bias=p,margin=.005))
 # P2 paired old data; read only required rows streaming.
 rows={}
 for line in (ROOT/'experiments/exp2/formal_rows.jsonl').open():
  s=json.loads(line)
  if s['d']==512 and s['name'] in ('Sign1','Sign2_a1.5_actual'):rows[s['kappa'],s['seed'],s['name']]=s
 for k,metric in ((1.,'top0.01'),(2.,'top0.1')):
  Ds=[];vs=[]
  n=round(k*512**2)
  for seed in range(24):
   a=np.array(rows[k,seed,'Sign2_a1.5_actual']['ranks']);b=np.array(rows[k,seed,'Sign1']['ranks'])
   delta=(a<=math.ceil(float(metric[3:])*n)).astype(float)-(b<=math.ceil(float(metric[3:])*n)).astype(float)
   Ds.append(delta.mean());vs.append((1-512/n)*delta.var(ddof=1)/512)
  var=np.var(Ds,ddof=1);v=np.mean(vs);tau=max(0.,var-v)
  # Choose M2048: retain full positive tau, inflate BOTH components 2x; no d extrapolation.
  M=2048;sd=math.sqrt(2*(tau+v*512/M*(1-M/n)/(1-512/n)))
  nn,pp=required(sd,.003,.01,.05/4) # preserve original 4-context multiplicity
  power.append(dict(kind='P2',kappa=k,name='Sign2_a1.5_actual',metric=metric,d=512,M=M,N=nn,old_N=24,old_mean=float(np.mean(Ds)),old_seed_variance=float(var),old_query_variance=float(v),old_tau2=float(tau),planning_sd=sd,design_bias=.003,power_at_design_bias=pp,margin=.01))
 # EXT old ratio variance worst across all sampled e and two d; 4x variance for tighter grid/interpolation unknown.
 groups={}
 for line in (ROOT/'experiments/ext/EXT_formal_rows.jsonl').open():
  r=json.loads(line)
  if r['tau']==0:groups[r['d'],r['gamma'],r['seed'],r['method']]=r['msr_scaled']
 for gamma in (16,64,256):
  sd=max(np.std([groups[d,gamma,s,e]/groups[d,gamma,s,'polar']-1 for s in range(16)],ddof=1) for d in (128,256) for e in ('0.0','0.01','0.1','1.0'))*2
  # 6 gamma/d cells x 12 e; epsilon contrasts are separately corrected
  alpha=.05/72;half=.002
  nn=next(n for n in range(16,10001) if (t.ppf(1-alpha/2,n-1)+norm.ppf(.9))*sd/math.sqrt(n)<=half)
  power.append(dict(kind='A',gamma=gamma,N=nn,planning_sd=float(sd),seed_level_ratio_halfwidth_target=half,halfwidth_success_normal_planning=.8))
 result=dict(selection_rule='Among the 37 unresolved old76 entries: for each kappa=.5,1,2 and family Adam/Sign select lexicographically smallest (name,metric); d256 only. No outcome magnitude used to rank.',unresolved37=unresolved,selected6=selected,power=power)
 with (OUT/'P1NUM_power.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
 print(json.dumps(dict(selected=selected,power=power),indent=2))
if __name__=='__main__':main()
