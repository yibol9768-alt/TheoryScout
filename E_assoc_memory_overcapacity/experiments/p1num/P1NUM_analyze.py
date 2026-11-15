"""Predata-fixed seed-level gates and raw data audit."""
import csv,json,math,time
from collections import defaultdict
import numpy as np
from scipy.stats import t,norm
from P1NUM_run import ROOT,OUT,put,frozen,metadata,tasks

def stat(x):
 a=np.array(x,dtype=float);return float(a.mean()),float(a.std(ddof=1)/math.sqrt(len(a)))
def save(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 start=time.monotonic();frozen();res=metadata();design=json.loads((OUT/'P1NUM_design.json').read_text());fc=json.loads((OUT/'P1NUM_forecasts.json').read_text())
 rows=[json.loads(s) for s in (OUT/'P1NUM_formal_rows.jsonl').read_text().splitlines()];a=defaultdict(list);m=defaultdict(list)
 for r in rows:
  if r['kind']=='A':a[r['d'],r['gamma'],r['label']].append(r)
  else:
   assert len(r['query_indices'])==512 and len(set(r['query_indices']))==512
   aa=np.array(r['ranks_actual']);bb=np.array(r['ranks_baseline']);assert np.all((aa>=1)&(aa<=r['n'])) and np.all((bb>=1)&(bb<=r['n']))
   q=float(r['metric'][3:]);delta=(aa<=math.ceil(q*r['n'])).astype(float)-(bb<=math.ceil(q*r['n'])).astype(float)
   D=float(delta.mean());v=float((1-512/r['n'])*delta.var(ddof=1)/512)
   assert abs(D-r['D'])<1e-14 and abs(v-r['v'])<1e-14 and abs(D*D-v-r['A'])<1e-14
   m[r['kind'],r['kappa'],r['name'],r['metric']].append(r)
 ag=[];pg=[];bounds=[];contrasts=[];mg=[]
 for f in fc['cells']:
  rr=a[f['d'],f['gamma'],f['label']];n=len(rr);planned=next(r['N'] for r in design['power'] if r['kind']=='A' and r['gamma']==f['gamma']);complete=n==planned
  assert len({r['seed'] for r in rr})==n
  if n<2:continue
  mean,se=stat([r['paired_ratio'] for r in rr]);tc=float(t.ppf(1-.05/(2*78),n-1));half=tc*se;lower=mean-half;upper=mean+half
  decision='above' if lower>0 else 'below' if upper<0 else 'unresolved'
  ag.append(dict(d=f['d'],gamma=f['gamma'],label=f['label'],e=f['e'],epsilon=f['epsilon'],N=n,planned_N=planned,complete=complete,mean=mean,SE=se,critical_t=tc,lower=lower,upper=upper,precision_target=.005,precision_pass=half<=.005 and complete,ratio_center=f['ratio_center'],center_error=mean-f['ratio_center'],center_equivalence_bound=abs(mean-f['ratio_center'])+half,center_pass=abs(mean-f['ratio_center'])+half<=.01 and complete,envelope_direction=f['envelope_direction'],envelope_direction_confirmed=(decision==f['envelope_direction']) if f['envelope_direction']!='unresolved' else None,decision=decision))
 for d in design['design']['dims']:
  for g in design['design']['gammas']:
   rr=a[d,g,'polar'];n=len(rr);mean,se=stat([r['msr_scaled'] for r in rr]);tc=float(t.ppf(1-.05/(2*6),n-1));f=next(f for f in fc['cells'] if f['d']==d and f['gamma']==g)
   pg.append(dict(d=d,gamma=g,N=n,mean=mean,SE=se,lower=mean-tc*se,upper=mean+tc*se,L=f['muon_L'],U=f['muon_U'],finite_d_tolerance=.005,envelope_pass=mean-tc*se>=f['muon_L']-.005 and mean+tc*se<=f['muon_U']+.005))
   grid=[r for r in ag if r['d']==d and r['gamma']==g and r['label']!='physical_epsilon'];below=[r['e'] for r in grid if r['decision']=='below'];above=[r['e'] for r in grid if r['decision']=='above']
   lo=max(below) if below else None;hi=min(above) if above else None;root=next(r for r in fc['roots'] if r['gamma']==g)
   # Existence/uniqueness of the empirical mean crossing is NOT asserted.
   bounds.append(dict(d=d,gamma=g,last_confirmed_below=lo,first_confirmed_above=hi,bracket_width=hi-lo if hi is not None and lo is not None else None,bracket_valid=hi is not None and lo is not None and lo<hi,center_root=root['center_root'],center_inside=(lo<=root['center_root']<=hi) if lo is not None and hi is not None else None,envelope_root_lower=root['envelope_root_lower'],envelope_root_upper=root['envelope_root_upper'],unresolved_grid=','.join(str(r['e']) for r in grid if r['decision']=='unresolved')))
   phys={r['seed']:r for r in a[d,g,'physical_epsilon']};ctrl={r['seed']:r for r in a[d,g,'0.06']};ids=sorted(phys.keys()&ctrl.keys());diff=[phys[s]['paired_ratio']-ctrl[s]['paired_ratio'] for s in ids]
   mean,se=stat(diff);tc=float(t.ppf(1-.05/(2*6),len(ids)-1));pred=next(f['ratio_center'] for f in fc['cells'] if f['d']==d and f['gamma']==g and f['label']=='physical_epsilon')-next(f['ratio_center'] for f in fc['cells'] if f['d']==d and f['gamma']==g and f['label']=='0.06')
   contrasts.append(dict(d=d,gamma=g,N=len(ids),physical_epsilon=design['design']['epsilon_physical'],physical_e=phys[ids[0]]['e'],fixed_e=.06,ratio_contrast=mean,SE=se,lower=mean-tc*se,upper=mean+tc*se,prediction=pred,equivalence_bound=abs(mean-pred)+tc*se,pass_1pp=abs(mean-pred)+tc*se<=.01))
 for p in design['power']:
  if p['kind']=='A':continue
  kind=p['kind'];k=float(p['kappa']);rr=m[kind,k,p['name'],p['metric']];n=len(rr);complete=n==p['N'];assert len({r['seed'] for r in rr})==n
  if n<2:continue
  mean,se=stat([r['D'] for r in rr]);alpha=.05/(4 if kind=='P2' else 6);tc=float(t.ppf(1-alpha/2,n-1));taa=float(t.ppf(1-alpha,n-1));target=0.
  if kind=='P2':
   c1=math.sqrt(2/math.pi);phi=math.exp(-1.5**2/2)/math.sqrt(2*math.pi);cdf=norm.cdf(-1.5);c2=math.sqrt(2)*phi/math.sqrt(cdf);z=norm.ppf(float(p['metric'][3:]));target=float(norm.cdf(c2/math.sqrt(k)+z)-norm.cdf(c1/math.sqrt(k)+z))
  bound=abs(mean-target)+tc*se;eq=bound<=p['margin'] and complete
  row=dict(kind=kind,d=p['d'],kappa=k,name=p['name'],metric=p['metric'],N=n,planned_N=p['N'],complete=complete,mean=mean,SE=se,target=target,critical_t=tc,lower=mean-tc*se,upper=mean+tc*se,equivalence_bound=bound,margin=p['margin'],signed_equivalence_pass=eq,planned_power=p.get('power_at_design_bias',p.get('power_at_zero_bias')),achieved_MDE80=(tc+norm.ppf(.8))*se)
  if kind=='P2':
   direction=(-mean-taa*se)>0;row.update(direction_confirmed=direction,combined_pass=direction and eq,status='相容到1pp' if direction and eq else '未分辨')
  else:
   A,Ase=stat([r['A'] for r in rr]);lower=A-taa*Ase;upper=A+taa*Ase
   row.update(A=A,A_SE=Ase,A_lower=lower,A_upper=upper,mean_query_v=float(np.mean([r['v'] for r in rr])),noise_compatible=eq and upper<=.005**2,status='查询噪声相容_0.5pp' if eq and upper<=.005**2 else '显著有限维残留' if lower>0 or abs(mean)>tc*se else '未分辨')
  mg.append(row)
 save('P1NUM_A_gates.csv',ag);save('P1NUM_muon_gates.csv',pg);save('P1NUM_boundary.csv',bounds);save('P1NUM_epsilon_contrast.csv',contrasts);save('P1NUM_M_gates.csv',mg)
 selected={(float(r['kappa']),r['name'],r['metric']) for r in design['selected6']};disclosure=[]
 for r in design['unresolved37']:
  disclosure.append(dict(kappa=r['kappa'],name=r['name'],metric=r['metric'],selected=(float(r['kappa']),r['name'],r['metric']) in selected,reason='每负载每家族字典序首项' if (float(r['kappa']),r['name'],r['metric']) in selected else '资源受限：未选；本轮不更新判决'))
 save('P1NUM_old37_disclosure.csv',disclosure)
 events=[json.loads(s) for s in (OUT/'P1NUM_formal_events.jsonl').read_text().splitlines()];expected=tasks(design);observed=[(r['kind'],r['d'],r['k'],r['seed']) for r in events]
 assert observed==[(kind,d,k,seed) for kind,d,k,seed,_ in expected][:len(observed)]
 assert len(rows)==sum(r['records']-(events[i-1]['records'] if i else 0) for i,r in enumerate(events))
 summary=dict(A_gates=len(ag),A_precision_pass=sum(r['precision_pass'] for r in ag),A_center_pass=sum(r['center_pass'] for r in ag),A_envelope_called=sum(r['envelope_direction']!='unresolved' for r in ag),A_envelope_confirmed=sum(r['envelope_direction_confirmed'] is True for r in ag),muon_envelope_pass=sum(r['envelope_pass'] for r in pg),epsilon_contrast_pass=sum(r['pass_1pp'] for r in contrasts),boundaries=bounds,M=mg,records=len(rows),planned_configurations=len(expected),completed_configurations=len(events),raw_rank_audit=True,seconds=time.monotonic()-start,resource_start=res,resource_end=metadata())
 frozen();put(OUT/'P1NUM_summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
