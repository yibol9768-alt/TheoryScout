"""Read only existing and newly derived results; prepare exploratory D draft, no LLM."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
import math,json,csv,collections,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/real/analysis_length'
rng=np.random.default_rng(640812)
T=list(csv.DictReader((OUT/'ANALYSIS_LENGTH_C_tasks.csv').open()))
S=list(csv.DictReader((OUT/'ANALYSIS_LENGTH_C_steps.csv').open()))
A=list(csv.DictReader((OUT/'ANALYSIS_LENGTH_C_actions.csv').open()))
for rows in (T,S,A):
    for r in rows:
        for k in list(r):
            if k not in ('batch','id','status','error_fields'):
                try:r[k]=int(r[k])
                except ValueError:r[k]=None if r[k] in ('','None') else r[k]
def fisher(tab):
 a,b=tab[0];c,d=tab[1];n=a+b+c+d;r=a+b;col=a+c
 def pr(x):return math.comb(col,x)*math.comb(n-col,r-x)/math.comb(n,r)
 obs=pr(a);return min(1.,sum(pr(x) for x in range(max(0,r-(n-col)),min(r,col)+1) if pr(x)<=obs*(1+1e-10)))
def cluster_position(tasks,steps,B=20000):
 # Stratify by batch and planned length; no extrapolation of missing tails.
 rows=[]
 for t in tasks:
  ss=[s for s in steps if s['id']==t['id'] and s['batch']==t['batch']];row=[]
  for early in (True,False):
   z=[s for s in ss if (s['step']<16)==early and s['resolved']]
   row.extend([len(z),sum(s['fatal'] for s in z),sum(s['error'] for s in z),sum(s['r1']==1 for s in z),
               sum(s['r1'] is not None for s in z),sum(s['r2']==1 for s in z),sum(s['r2'] is not None for s in z)])
  rows.append(row)
 rows=np.array(rows,float);groups=collections.defaultdict(list)
 for i,t in enumerate(tasks):groups[(t['batch'],t['length'])].append(i)
 total=np.zeros((B,14))
 for ids in groups.values():total+=rows[np.array(ids)[rng.integers(0,len(ids),(B,len(ids)))]].sum(1)
 def diff(k,n):return total[:,k]/np.maximum(total[:,n],1)-total[:,k+7]/np.maximum(total[:,n+7],1)
 result={}
 for label,k,n in [('h',1,0),('p',2,0),('r1',3,4),('r2',5,6)]:
  good=(total[:,n]>0)&(total[:,n+7]>0);v=diff(k,n)[good]
  result[label]=dict(early_minus_late_ci95=np.quantile(v,[.025,.975]).tolist(),valid=int(sum(good)))
 return result
POS={}
for b in ('calib1','holdout_b','holdout_c','all'):
 ts=[t for t in T if b=='all' or t['batch']==b];ss=[s for s in S if b=='all' or s['batch']==b]
 POS[b]=cluster_position(ts,ss)
 if b=='holdout_c':POS['holdout_c_long_only']=cluster_position([t for t in ts if t['length']==64],[s for s in ss if s['length']==64])

# Detection after explicit feedback. Invalid JSON/schema can force both detect=0 and failure.
DEP=collections.defaultdict(collections.Counter)
for b in ('calib1','holdout_b','holdout_c'):
 folder=ROOT/'experiments/real'/f'{b}_real';summary=json.loads((folder/'summary.json').read_text())
 for t in summary['results']:
  for line in (folder/(t['id']+'.jsonl')).read_text().splitlines():
   a=json.loads(line)
   if a['event']!='action' or not a['review']:continue
   parsed=a['action'];rv=a['review'];detect=bool(parsed and parsed['detect'])
   DEP[f'{b}_r{rv}'][f'parsed{int(parsed is not None)}_detect{int(detect)}_correct{int(a["correct"])}']+=1
   DEP[f'all_r{rv}'][f'parsed{int(parsed is not None)}_detect{int(detect)}_correct{int(a["correct"])}']+=1

# Exploratory train-on-all future model, deliberately separate from CALIB-only holdout evaluation.
# Terminal hazard estimated on COMPLETELY RESOLVED steps; one B incomplete repair omitted.
resolved=[s for s in S if s['resolved']]
early=[s for s in resolved if s['step']<16];late=[s for s in resolved if s['step']>=16]
hE=sum(s['fatal'] for s in early)/len(early);hT=sum(s['fatal'] for s in late)/len(late)
Ls=[8,16,32,64,128];N=400
surv=lambda L:(1-hE)**min(L,16)*(1-hT)**max(0,L-16)
# Stratified, whole-task bootstrap for the all-data exploratory prediction intervals.
bootN=10000;aggregate=[]
for t in T:
 ss=[s for s in resolved if s['batch']==t['batch'] and s['id']==t['id']]
 aggregate.append([sum(s['step']<16 for s in ss),sum(s['fatal'] and s['step']<16 for s in ss),
                   sum(s['step']>=16 for s in ss),sum(s['fatal'] and s['step']>=16 for s in ss)])
aggregate=np.array(aggregate,float);groups=collections.defaultdict(list)
for i,t in enumerate(T):groups[(t['batch'],t['length'])].append(i)
v=np.zeros((bootN,4))
for ids in groups.values():v+=aggregate[np.array(ids)[rng.integers(0,len(ids),(bootN,len(ids)))]].sum(1)
hes=v[:,1]/v[:,0];hts=v[:,3]/v[:,2]
def binom_pmf(n,p):
 if p<=0:return np.r_[1.,np.zeros(n)]
 if p>=1:return np.r_[np.zeros(n),1.]
 j=np.arange(n+1);lp=np.array([math.lgamma(n+1)-math.lgamma(int(k)+1)-math.lgamma(n-int(k)+1) for k in j])+j*math.log(p)+(n-j)*math.log1p(-p)
 return np.exp(lp)
def binom_interval(n,p,alpha=.01):
 c=np.cumsum(binom_pmf(n,p));return [int(np.searchsorted(c,alpha/2)),int(np.searchsorted(c,1-alpha/2))]
def mix_interval(n,ps,alpha=.01):
 # Stable log computation, including all calibration uncertainty, without a giant matrix.
 j=np.arange(n+1);lc=np.array([math.lgamma(n+1)-math.lgamma(int(k)+1)-math.lgamma(n-int(k)+1) for k in j]);mass=np.zeros(n+1)
 for p in ps:mass+=np.exp(lc+j*np.log(max(p,1e-300))+(n-j)*np.log(max(1-p,1e-300)))
 mass/=len(ps);c=np.cumsum(mass);return [int(np.searchsorted(c,alpha/2)),int(np.searchsorted(c,1-alpha/2))]
PRED={}
for L in Ls:
 p=surv(L);ps=(1-hes)**min(L,16)*(1-hts)**max(0,L-16)
 PRED[str(L)]=dict(S=p,calibration95=np.quantile(ps,[.025,.975]).tolist(),fixed_binomial_n400_99=binom_interval(N,p),
                   bootstrap_mixture_n400_99=mix_interval(N,ps))
# N=400 max-length trajectories provide nested checkpoint outcomes. A fixed simple-vs-simple
# likelihood ratio uses only those 6 multinomial cells, avoiding independence claims for prefixes.
# Bins: fail before 8, 8..15, 16..31, 32..63, 64..127, survive128.
hA=19/1279
sA=np.array([1]+[(1-hA)**L for L in Ls]);sD=np.array([1]+[surv(L) for L in Ls])
pA=np.r_[sA[:-1]-sA[1:],sA[-1]];pD=np.r_[sD[:-1]-sD[1:],sD[-1]]
w=np.log(pD/pA)
SIM_N=100000;nullstat=rng.multinomial(N,pA,SIM_N)@w
# Quantile 'higher', strict > gives conservative rejection at <=.025 Monte Carlo reference.
crit=float(np.quantile(nullstat,.975,method='higher'))
altstat=rng.multinomial(N,pD,SIM_N)@w
mc_critical=crit
# The actual proposal requires a likelihood ratio >40. Under fixed A, E_A[LR]=1,
# so Markov's inequality bounds type I error by 1/40 without Monte Carlo tuning.
crit=math.log(40)
power=float(np.mean(altstat>crit));type1=float(np.mean(nullstat>crit))
# Sample-size sensitivity computed from exact same multinomial mechanism.
POW={}
for nn in (100,200,300,400):
 n0=rng.multinomial(nn,pA,50000)@w;c=math.log(40)
 a1=rng.multinomial(nn,pD,50000)@w
 POW[str(nn)]=dict(critical=c,type1=float(np.mean(n0>c)),power=float(np.mean(a1>c)))
D=dict(status='DRAFT_ONLY_NOT_RUN',fit_data='CALIB1 + HOLDOUT_B observed resolved steps + HOLDOUT_C; EXPLORATORY, no holdout-validation claim',
       model='piecewise terminal hazard with cut16 (zero-based steps0..15 early)',early=dict(d=sum(s['fatal'] for s in early),n=len(early),h=hE),
       tail=dict(d=sum(s['fatal'] for s in late),n=len(late),h=hT),N=N,length_design='400 independent max-length128 tasks; nested prefix endpoints8/16/32/64/128; length absent from prompts',
       proposed_seed_range=[101280001,101280400],predictions=PRED,conditional_S64_given16=(1-hT)**48,
       comparison=dict(A_h=hA,cells_A=pA.tolist(),cells_new=pD.tolist(),weights_log_new_over_A=w.tolist(),
                       statistic='sum six bin counts * weights',critical=crit,strict_reject='T > critical',alpha=.025,
                       simulations=SIM_N,seed=640812,type1_bound=.025,type1_MC=type1,power_MC=power,sample_size_sensitivity=POW,
                       MC_975_null_quantile_reference_only=mc_critical),
       requirements='No calls now. Actual execution needs separate user authorization and frozen engineering prereg. No replacement of model failures or exclusion of smoke. Retain censoring.',
       uncertainty_note='Pooled batch/length stratified cluster bootstrap; transport shift and causal uncertainty remain. 99% marginal count intervals have simultaneous >=95% union-bound coverage if fixed model correct; nested endpoints not independent.')
(OUT/'ANALYSIS_LENGTH_C_HOLDOUT_D_DRAFT.json').write_text(json.dumps(D,indent=2,ensure_ascii=False)+'\n')

# Describe cohort position differences without treating them as confirmation.
TESTS={}
for b in ('calib1','holdout_b','holdout_c','all'):
 ss=[s for s in resolved if b=='all' or s['batch']==b]
 ee=[s for s in ss if s['step']<16];ll=[s for s in ss if s['step']>=16]
 tab=[[sum(s['fatal'] for s in ee),len(ee)-sum(s['fatal'] for s in ee)],[sum(s['fatal'] for s in ll),len(ll)-sum(s['fatal'] for s in ll)]]
 TESTS[b]=dict(table_fatal_by_position=tab,fisher_action_IID_reference=fisher(tab),clusterCI=POS[b])
TESTS['C_prefix16_short_vs_long']=dict(table=[[10,10],[13,7]],fisher=fisher([[10,10],[13,7]]))
# schema-conditioned detection table
(OUT/'ANALYSIS_LENGTH_C_supplement.json').write_text(json.dumps(dict(position_tests=TESTS,position_cluster_intervals=POS,detection_stratified={k:dict(v) for k,v in DEP.items()}),indent=2)+'\n')
# Input preservation verification.
inputs=json.loads((OUT/'ANALYSIS_LENGTH_C_inputs_sha256.json').read_text());changed=[]
for p,h in inputs.items():
 if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:changed.append(p)
assert not changed
(OUT/'ANALYSIS_LENGTH_C_verification.json').write_text(json.dumps(dict(input_files_checked=len(inputs),changed=changed,raw_actions=len(A),raw_steps=len(S),raw_tasks=len(T),
       scope_memory_max='2G',scope_swap_max=0,threads_max=2,network_calls=0,llm_calls=0),indent=2)+'\n')
print('D',json.dumps(D,ensure_ascii=False),flush=True)
print('SUPPLEMENT',json.dumps(dict(tests=TESTS,det={k:dict(v) for k,v in DEP.items()})),flush=True)
# Frozen local kernels of the exploratory piecewise proposal, excluding one unresolved B episode.
D['local_kernels']={}
for label,ss in [('early',early),('tail',late)]:
 e=sum(s['error'] for s in ss);r1n=sum(s['r1'] is not None for s in ss);r1ok=sum(s['r1']==1 for s in ss)
 r2n=sum(s['r2'] is not None for s in ss);r2ok=sum(s['r2']==1 for s in ss)
 p=e/len(ss);r1=r1ok/r1n;r2=r2ok/r2n
 assert abs(p*(1-r1)*(1-r2)-(hE if label=='early' else hT))<1e-12
 D['local_kernels'][label]=dict(p=p,r1=r1,r2=r2,error=[e,len(ss)],repair1=[r1ok,r1n],repair2=[r2ok,r2n])
(OUT/'ANALYSIS_LENGTH_C_HOLDOUT_D_DRAFT.json').write_text(json.dumps(D,indent=2,ensure_ascii=False)+'\n')

# Paired task bootstrap on heldout likelihood differences with fitted parameters fixed.
models=json.loads((OUT/'ANALYSIS_LENGTH_C_models.json').read_text())
def ssurv(name,L):
 par=models[name]['parameters']
 def bsur(b,nn):
  if b['boundary']:return (1-b['mean'])**nn
  return math.exp(sum(math.log1p(-b['alpha']/(b['k']+j)) for j in range(nn)))
 if name=='A':return (1-par['h'])**L
 if name=='B_fatal':return bsur(par,L)
 if name=='B_error':return (1-par['p']*par['f'])**L # observed MLE is homogeneous boundary
 surv=(1-par['h0'])**min(par['cut'],L);nn=max(0,L-par['cut'])
 return surv*((1-par['hplus'])**nn if name[0]=='C' else bsur(par['tail'],nn))
held=[t for t in T if t['batch']!='calib1'];groups=collections.defaultdict(list)
for i,t in enumerate(held):groups[(t['batch'],t['length'])].append(i)
DELTA={}
for name in models:
 vals=[]
 for t in held:
  n=t['n'];local=[s for s in S if s['id']==t['id'] and s['batch']==t['batch']]
  if t['censor'] and local and not local[-1]['resolved']:n-=1
  def prob(m):return ssurv(m,n-1)-ssurv(m,n) if t['d'] else ssurv(m,n)
  vals.append(math.log(max(prob(name),1e-300))-math.log(max(prob('A'),1e-300)))
 vals=np.array(vals);tot=np.zeros(20000)
 for ids in groups.values():tot+=vals[np.array(ids)[rng.integers(0,len(ids),(20000,len(ids)))]].sum(1)
 DELTA[name]=dict(delta_LL_vs_A=float(vals.sum()),cluster95_fixed_fit=np.quantile(tot,[.025,.975]).tolist())
(OUT/'ANALYSIS_LENGTH_C_score_differences.json').write_text(json.dumps(DELTA,indent=2)+'\n')

# Standalone code-native SVG: no image model, plotting service, or network.
kmdata=json.loads((OUT/'ANALYSIS_LENGTH_C_survival.json').read_text())['KM']
W,H=1050,440;left,top,pw,ph=72,40,680,330
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
 '<rect width="100%" height="100%" fill="white"/>','<g font-family="sans-serif" font-size="13">',
 '<text x="72" y="22" font-size="16">Terminal survival: CALIB-only forecasts vs observed prefixes</text>']
def pt(L,s):return (left+L/128*pw,top+(1-s)*ph)
for y in [0,.25,.5,.75,1]:
 xx,yy=pt(0,y);svg.append(f'<path d="M {left} {yy} H {left+pw}" stroke="#ddd"/><text x="32" y="{yy+5}">{y:.2f}</text>')
for L in [0,8,16,32,64,96,128]:
 xx,yy=pt(L,0);svg.append(f'<text x="{xx-8}" y="{yy+23}">{L}</text>')
curves=[('A CALIB homogeneous','#333',lambda L:ssurv('A',L),'6,4'),('B CALIB task frailty','#9467bd',lambda L:ssurv('B_fatal',L),'3,3'),
        ('Exploratory pooled piecewise16','#e68a00',surv,'8,3')]
# Avoid name collision: surv is scalar in prior helper scope only; global remains callable lambda.
for label,color,fn,dash in curves:
 pts=' '.join(f'{x:.2f},{y:.2f}' for x,y in [pt(L,fn(L)) for L in range(129)])
 svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
legend=[(x[0],x[1]) for x in curves]
for key,label,color in [('calib1_all','CALIB observed (L32)','#278349'),('holdout_c_16','C short (L16)','#d43a3a'),('holdout_c_64','C long prefixes (L64)','#147eb5')]:
 points=[pt(0,1)];last=1
 for row in kmdata[key]:
  L=row['step_1based'];points.extend([pt(L,last),pt(L,row['KM'])]);last=row['KM']
 pts=' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
 svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>');legend.append((label,color))
for i,(label,color) in enumerate(legend):
 y=top+25+i*35;svg.extend([f'<path d="M 780 {y} h 24" stroke="{color}" stroke-width="3"/>',f'<text x="812" y="{y+5}">{label}</text>'])
svg+=['<text x="370" y="420">Number of completed steps / target length</text>','<text x="780" y="300">Post-hoc exploration;</text>','<text x="780" y="320">pooled curve is not heldout.</text>','</g></svg>']
(OUT/'ANALYSIS_LENGTH_C_survival.svg').write_text('\n'.join(svg))
print('FINAL_D_POWER',D['comparison'],flush=True)
print('SCORE_DELTA',DELTA,flush=True)
