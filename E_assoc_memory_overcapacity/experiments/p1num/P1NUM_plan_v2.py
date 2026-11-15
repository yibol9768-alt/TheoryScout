import json,math
from pathlib import Path
from scipy.stats import norm,t
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/p1num'
p=json.loads((OUT/'P1NUM_power.json').read_text())
for r in p['power']:
 if r['kind']=='P2':
  r['previous_predata_M']=r['M'];r['previous_predata_N']=r['N'];r['M']=512
  sd=math.sqrt(2*r['old_seed_variance']);r['planning_sd']=sd
  def power(n,bias):
   se=sd/math.sqrt(n);z=t.ppf(1-.05/8,n-1)
   return float(norm.cdf((.01-z*se-bias)/se)-norm.cdf((-.01+z*se-bias)/se)) if z*se<.01 else 0.
  required=next(n for n in range(8,10001) if power(n,.003)>=.8)
  r['required_N80']=required;r['N']=min(required,64 if r['kappa']==1 else 112)
  r['power_at_design_bias']=power(r['N'],.003);r['power_at_zero_bias']=power(r['N'],0.)
  r['planned_halfwidth']=float(t.ppf(1-.05/8,r['N']-1)*sd/math.sqrt(r['N']))
  r['budget_cap_applied']=r['N']<required
 if r['kind']=='A':
  sd=r['planning_sd'];alpha=.05/78 # 6 cells * (12 grid +1 physical epsilon)
  r['seed_level_ratio_halfwidth_target']=.005
  r['N']=next(n for n in range(24,10001) if (t.ppf(1-alpha/2,n-1)+norm.ppf(.9))*sd/math.sqrt(n)<=.005)
  r['alpha_family']=alpha
p['design']=dict(gammas=[16,64,256],dims=[128,256],e_grid=[0,.03,.04,.05,.055,.06,.065,.07,.08,.1,.14,.24],epsilon_physical=.06/math.sqrt(128*64),fixed_e_control=.06,seed_domain=910631,formal_tag=11,smoke_tag=91,p2_caps=[64,112],formal_seconds_limit=3450)
with (OUT/'P1NUM_design.json').open('x') as f:json.dump(p,f,indent=2,ensure_ascii=False)
print(json.dumps(p['power'],indent=2))
