"""Independent numerical checks and parameter-uncertainty sensitivity; no network."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import json,math,hashlib,csv,collections,ast,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;O=ROOT/'experiments/real/analysis_length'
models=json.loads((O/'ANALYSIS_LENGTH_C_models.json').read_text())
boot=json.loads((O/'ANALYSIS_LENGTH_C_bootstrap_parameters.json').read_text())
T=list(csv.DictReader((O/'ANALYSIS_LENGTH_C_tasks.csv').open()));S=list(csv.DictReader((O/'ANALYSIS_LENGTH_C_steps.csv').open()))
held=[t for t in T if t['batch']!='calib1'];rng=np.random.default_rng(640815)
groups=collections.defaultdict(list)
for i,t in enumerate(held):groups[(t['batch'],t['length'])].append(i)
n=np.array([int(t['n']) for t in held]);death=np.array([int(t['d']) for t in held],bool)
for i,t in enumerate(held):
 ss=[s for s in S if s['id']==t['id'] and s['batch']==t['batch']]
 if int(t['censor']) and ss and not int(ss[-1]['resolved']):n[i]-=1

def curve(model,param):
 L=np.arange(129)
 if model=='A':return (1-param['h'])**L
 def beta(b):
  if b['boundary']:return (1-b['mean'])**L
  return np.r_[1.,np.exp(np.cumsum(np.log1p(-b['alpha']/(b['k']+np.arange(128)))))]
 if model=='B_fatal':return beta(param)
 cut=param['cut'];early=(1-param['h0'])**np.minimum(cut,L)
 if model[0]=='C':return early*(1-param['hplus'])**np.maximum(0,L-cut)
 return early*beta(param['tail'])[np.maximum(0,L-cut)]
def vec_ll(cur):
 pr=np.where(death,cur[np.maximum(n-1,0)]-cur[n],cur[n])
 return np.where(pr>0,np.log(np.maximum(pr,np.finfo(float).tiny)),-np.inf)
diffs={k:[] for k in ('B_fatal','C1','C8','D1','D8')}
for fitted in boot:
 baseline=vec_ll(curve('A',fitted['A']));idx=[]
 for ids in groups.values():idx.extend(rng.choice(ids,len(ids),replace=True))
 for k in diffs:diffs[k].append(float(np.sum((vec_ll(curve(k,fitted[k]))-baseline)[idx])))
D=json.loads((O/'ANALYSIS_LENGTH_C_score_differences.json').read_text())
for k,v in diffs.items():
 quant=np.quantile(v,[.025,.975],method='inverted_cdf')
 D[k]['cluster95_calibration_refit_and_heldout']=[float(x) if np.isfinite(x) else None for x in quant]
 D[k]['refit_negative_infinity_replicates']=int(sum(x==-np.inf for x in v))
 D[k]['null_lower_endpoint_means_negative_infinity']=bool(quant[0]==-np.inf)
 D[k]['refit_interval_note']='Zero estimated first-step hazard can assign zero probability to heldout failure: retain -infinity, no smoothing or probability floor.'
(O/'ANALYSIS_LENGTH_C_score_differences.json').write_text(json.dumps(D,indent=2)+'\n')
# Verify models' stored survival, raw-input immutability, SVG parse, probability sums.
checks=0
for k in ('A','B_fatal','C1','C8','D1','D8'):
 cur=curve(k,models[k]['parameters']);assert cur[0]==1 and np.all(np.diff(cur)<=1e-12) and np.min(cur)>=0
 for L,p in models[k]['predictions'].items():assert abs(cur[int(L)]-p)<1e-10;checks+=1
 calc=vec_ll(cur).sum();assert abs(calc-models[k]['scores']['holdout_bc']['trajectory_ll'])<1e-8;checks+=1
p=models['A']['parameters'];assert abs(p['h']-19/1279)<1e-12;assert abs(p['p']*(1-p['r1'])*(1-p['r2'])-p['h'])<1e-12
# Independent exact beta survival via lgamma at the MLE.
b=models['B_fatal']['parameters']
for L in (8,16,32,64,128):
 exact=math.exp(math.lgamma(b['beta']+L)-math.lgamma(b['k']+L)-math.lgamma(b['beta'])+math.lgamma(b['k']))
 assert abs(exact-models['B_fatal']['predictions'][str(L)])<1e-10;checks+=1
inputs=json.loads((O/'ANALYSIS_LENGTH_C_inputs_sha256.json').read_text())
for p,h in inputs.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h;checks+=1
ET.parse(O/'ANALYSIS_LENGTH_C_survival.svg');checks+=1
future=json.loads((O/'ANALYSIS_LENGTH_C_HOLDOUT_D_DRAFT.json').read_text())
for key in ('cells_A','cells_new'):assert abs(sum(future['comparison'][key])-1)<1e-12;checks+=1
assert future['status']=='DRAFT_ONLY_NOT_RUN';assert future['comparison']['critical']==math.log(40);checks+=2
# The prediction sequence is a draft, but check its proposed seed range against actual and scheduled seeds now.
newseeds=set(range(*[future['proposed_seed_range'][0],future['proposed_seed_range'][1]+1]));oldseeds=set()
for name in ('calib1','holdout_b','holdout_c'):
 j=json.loads((ROOT/'experiments/real'/f'{name}_real'/'summary.json').read_text())
 for row in j['results']+j.get('schedule',[]):
  if 'seed' in row:oldseeds.add(row['seed'])
assert not(newseeds&oldseeds);checks+=1
# Static check disallows service APIs and importing runners in delivered source files.
for p in ROOT.glob('ANALYSIS_LENGTH_C_*.py'):
 tree=ast.parse(p.read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):imports=[x.name for x in node.names]
  elif isinstance(node,ast.ImportFrom):imports=[node.module or '']
  else:continue
  assert all(not x.startswith(('requests','http','urllib','socket','openai','holdout','reliability_core','subprocess')) for x in imports)
 checks+=1
verification=json.loads((O/'ANALYSIS_LENGTH_C_verification.json').read_text());verification.update(independent_numerical_checks=checks,
 calibration_and_heldout_refit_replicates=len(boot),proposed_D_seeds_disjoint_checked=True,
 draft_future_sampling='Only hypothetical multinomial outcome counts simulated for design power; no D tasks/prompts/model responses generated')
(O/'ANALYSIS_LENGTH_C_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print('VALIDATED',checks,'checks')
print('REFIT_SCORE_DIFFERENCES',json.dumps(D))
