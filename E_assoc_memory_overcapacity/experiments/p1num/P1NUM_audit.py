"""Independent replay of existing P1NUM formal seeds; no additional seed or outcome selection."""
import csv,hashlib,json,math,time
from collections import defaultdict
import numpy as np
from scipy.stats import t
from P1NUM_run import ROOT,OUT,put,metadata,frozen

def main():
 st=time.monotonic();meta=metadata();frozen();rows=[json.loads(s) for s in (OUT/'P1NUM_formal_rows.jsonl').read_text().splitlines()]
 # A: full dense n*n, different formula and path, first registered seed and smallest cell.
 d=128;g=16;n=d*g;ent=[20261106,910631,1,d,n,0,11]
 X=np.random.default_rng(np.random.SeedSequence(ent+[0])).standard_normal((d,n));Y=np.random.default_rng(np.random.SeedSequence(ent+[1])).standard_normal((d,n))
 Xc=X-X.mean(axis=1,keepdims=True);Yc=Y-Y.mean(axis=1,keepdims=True);Z=Yc@Xc.T/math.sqrt(n)
 a,s,bt=np.linalg.svd(Z,full_matrices=False);polar=math.sqrt(d)*a@bt;errors=[];a_checks=[]
 aa=[r for r in rows if r['kind']=='A' and r['d']==d and r['gamma']==g and r['seed']==0]
 for r in aa:
  U=polar if r['label']=='polar' else np.sign(Z) if r['e']==0 else Z/(np.abs(Z)+r['e'])
  scores=Y.T@U@X/d;diag=np.diag(scores).copy();scores[np.arange(n),np.arange(n)]=0
  wrong=scores.sum(axis=0)/(n-1);var=(scores*scores).sum(axis=0)/(n-1)-wrong**2
  mean=float(np.mean(diag-wrong));variance=float(var.mean());err=max(abs(mean-r['mean']),abs(variance-r['variance']))
  assert err<3e-11,(r['label'],err);errors.append(err);a_checks.append(r['label'])
 del X,Y,Xc,Yc,Z,scores,U,polar
 # M: first registered d256/k.5 seed; dense X,Y, original physical W-unit Adam.
 d=256;k=.5;n=round(k*d*d);ent=[20261106,910631,2,d,n,0,11]
 rx=np.random.default_rng(np.random.SeedSequence(ent+[0]));ry=np.random.default_rng(np.random.SeedSequence(ent+[1]));rq=np.random.default_rng(np.random.SeedSequence(ent+[2]))
 ids=rq.choice(n,512,replace=False)
 X=np.concatenate([rx.standard_normal((d,min(8192,n-lo)))/math.sqrt(d) for lo in range(0,n,8192)],axis=1)
 Y=np.concatenate([ry.standard_normal((d,min(8192,n-lo)))/math.sqrt(d) for lo in range(0,n,8192)],axis=1)
 S=d/n*(X@X.T);Z=d/math.sqrt(n)*(Y@X.T);scale=math.sqrt(n)
 a=.2;W1=a/scale*np.sign(Z);grad=W1@S-Z/scale;Ws=W1-a/scale*np.sign(grad);Ps=2*a/scale*np.sign(Z)*(np.abs(Z)>a)
 W=np.zeros((d,d));mm=W.copy();vv=W.copy();P=W.copy();mp=W.copy();vp=W.copy()
 for it in range(1,17):
  grad=W@S-Z/scale;mm=.9*mm+.1*grad;vv=.999*vv+.001*grad*grad
  W-=.2/scale*(mm/(1-.9**it))/(np.sqrt(vv/(1-.999**it))+.1/scale)
  grad=P-Z/scale;mp=.9*mp+.1*grad;vp=.999*vp+.001*grad*grad
  P-=.2/scale*(mp/(1-.9**it))/(np.sqrt(vp/(1-.999**it))+.1/scale)
 checks=[]
 for r in rows:
  if r['kind']!='P3' or r['kappa']!=k or r['seed']!=0:continue
  assert np.array_equal(ids,r['query_indices'])
  for label,mat in [('actual',Ws if r['name'].startswith('Sign') else W),('baseline',Ps if r['name'].startswith('Sign') else P)]:
   scores=Y.T@(mat@X[:,ids]);truth=np.einsum('ij,ij->j',Y[:,ids],mat@X[:,ids]);scores[ids,np.arange(512)]=-np.inf
   rank=1+(scores>truth[None,:]).sum(axis=0);assert np.array_equal(rank,r['ranks_'+label]),(r['name'],label)
   checks.append(dict(name=r['name'],side=label,ranks=512))
 del X,Y,scores
 # Preregistered sensitivity to the OLD 240-item correction, keeping same N/data.
 design=json.loads((OUT/'P1NUM_design.json').read_text());groups=defaultdict(list)
 for r in rows:
  if r['kind']=='P3':groups[r['kappa'],r['name'],r['metric']].append(r)
 sensitivity=[]
 for p in design['power']:
  if p['kind']!='P3':continue
  rr=groups[float(p['kappa']),p['name'],p['metric']];n=len(rr)
  if n<2:continue
  vals=np.array([r['D'] for r in rr]);avals=np.array([r['A'] for r in rr]);mean=float(vals.mean());se=float(vals.std(ddof=1)/math.sqrt(n));A=float(avals.mean());ase=float(avals.std(ddof=1)/math.sqrt(n))
  tc=float(t.ppf(1-.05/(2*240),n-1));ta=float(t.ppf(1-.05/240,n-1));eq=abs(mean)+tc*se<=.005;upper=A+ta*ase
  sensitivity.append(dict(kappa=p['kappa'],name=p['name'],metric=p['metric'],N=n,signed_bound_old240=abs(mean)+tc*se,A_upper_old240=upper,old240_noise_compatible=eq and upper<=.005**2 and n==p['N']))
 with (OUT/'P1NUM_P3_old240_sensitivity.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(sensitivity[0]));writer.writeheader();writer.writerows(sensitivity)
 # Exact full expected record membership, including seeds and grid labels.
 ds=design['design'];actual=set();planned=set()
 for r in rows:
  key=(r['kind'],r['d'],r.get('gamma',r.get('kappa')),r['seed'],r.get('label',r.get('name')),r.get('metric',''))
  assert key not in actual,key;actual.add(key)
 for p in design['power']:
  if p['kind']=='A':
   for d0 in ds['dims']:
    for seed in range(p['N']):
     for label in ['polar']+[str(e) for e in ds['e_grid']]+['physical_epsilon']:planned.add(('A',d0,p['gamma'],seed,label,''))
  else:
   for seed in range(p['N']):planned.add((p['kind'],p['d'],float(p['kappa']),seed,p['name'],p['metric']))
 assert actual<=planned
 frozen();put(OUT/'P1NUM_independent_audit.json',dict(dense_A_methods=len(a_checks),dense_A_score_pairs=n if False else 2048**2,max_dense_A_error=max(errors),dense_M_rank_checks=checks,rank_checks_total=sum(r['ranks'] for r in checks),expected_records=len(planned),actual_records=len(actual),missing_records=len(planned-actual),complete=actual==planned,frozen_inputs_unchanged=True,seconds=time.monotonic()-st,resource_start=meta,resource_end=metadata()))
 print(json.dumps(dict(A_max_error=max(errors),M_ranks=sum(r['ranks'] for r in checks),expected=len(planned),actual=len(actual),seconds=time.monotonic()-st)))
if __name__=='__main__':main()
