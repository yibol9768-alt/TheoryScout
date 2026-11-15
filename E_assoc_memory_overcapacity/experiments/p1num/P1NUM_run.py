"""P1NUM: immutable predata implementation. CPU, local inputs, exclusive outputs."""
import argparse,ctypes,hashlib,json,math,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/p1num'
def put(path,data):
 with path.open('x') as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def frozen():
 for p,h in json.loads((OUT/'P1NUM_frozen.json').read_text()).items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/'P1NUM_PREREG.md')==(ROOT/'P1NUM_PREREG.sha256').read_text().split()[0]
def metadata():
 # Only operating-system runtime metadata outside project, never other projects.
 cg=next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
 root=Path('/sys/fs/cgroup')/cg.lstrip('/')
 controls={n:(root/n).read_text().strip() for n in ('memory.max','memory.swap.max','memory.peak','memory.events')}
 assert controls['memory.max']=='2147483648' and controls['memory.swap.max']=='0'
 pools=[]
 for path in sorted({x.split()[-1] for x in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in x and '/' in x}):
  lib=ctypes.CDLL(path)
  for s in ('scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads','openblas_get_num_threads64_'):
   try:fn=getattr(lib,s)
   except AttributeError:continue
   fn.restype=ctypes.c_int;pools.append(dict(library=path,threads=fn()));break
  else:raise RuntimeError(path)
 assert pools and all(x['threads']<=2 for x in pools)
 assert all(os.environ.get(n)=='2' for n in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'))
 return dict(controls=controls,threadpools=pools,rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,python=sys.executable,device='CPU')
def moments(e):
 if e==0:return dict(a=math.sqrt(2/math.pi),q=1.,c=math.sqrt(2/math.pi))
 density=lambda z:math.sqrt(2/math.pi)*math.exp(-z*z/2)
 a=quad(lambda z:density(z)*z*z/(z+e),0,np.inf,epsabs=2e-13,epsrel=2e-13)[0]
 q=quad(lambda z:density(z)*z*z/(z+e)**2,0,np.inf,epsabs=2e-13,epsrel=2e-13)[0]
 return dict(a=a,q=q,c=a/math.sqrt(q))
def F(c,g):return c/math.sqrt(1+2*c*c/g)
def pred(g,e):
 b=8/(3*math.pi);c=moments(e)['c'];delta=52/g**1.5
 L=b*(1-.1/g-1/g**2)/math.sqrt(1+delta);U=b*(1-.1/g+1/g**2)/math.sqrt(1-delta)
 return dict(c=c,variance=1+2*c*c/g,msr=F(c,g),ratio_center=F(c,g)/(b*(1-.1/g))-1,muon_L=L,muon_U=U,envelope_direction='below' if F(c,g)<L else 'above' if F(c,g)>U else 'unresolved')
def prepare():
 start=time.monotonic();m=metadata();design=json.loads((OUT/'P1NUM_design.json').read_text());ds=design['design'];fc=[];roots=[]
 for g in ds['gammas']:
  p=pred(g,.06)
  def invert(y):
   if y<=F(math.sqrt(2/math.pi),g):return 0.
   if y>=F(1.,g):return None
   return brentq(lambda e:F(moments(e)['c'],g)-y,1e-8,100,xtol=1e-13)
  roots.append(dict(gamma=g,center_root=invert(8/(3*math.pi)*(1-.1/g)),envelope_root_lower=invert(p['muon_L']),envelope_root_upper=invert(p['muon_U'])))
  for d in ds['dims']:
   for label,e in [(str(e),e) for e in ds['e_grid']]+[('physical_epsilon',ds['epsilon_physical']*math.sqrt(d*g))]:
    fc.append(dict(d=d,gamma=g,label=label,e=e,epsilon=e/math.sqrt(d*g),**pred(g,e)))
 put(OUT/'P1NUM_forecasts.json',dict(cells=fc,roots=roots,seconds=time.monotonic()-start,resource=m))
 print(json.dumps(roots))
def randoms(d,n,seed,kind,tag):
 ent=[20261106,910631,kind,d,n,seed,tag]
 return ent,[np.random.default_rng(np.random.SeedSequence(ent+[j])) for j in range(3)]
class Budget:
 def __init__(self,deadline):self.deadline=deadline;self.max_op=0.
 def check(self):
  if time.time()+max(3.,2*self.max_op)>self.deadline:raise TimeoutError('registered wall-clock reserve')
  if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024>1700:raise TimeoutError('registered RSS reserve')
 def dot(self,a,b):
  self.check();st=time.monotonic();r=a@b;self.max_op=max(self.max_op,time.monotonic()-st);return r

def aevaluate(u,x,y,gx,gy,budget):
 d,n=x.shape;sy=y.sum(axis=1);diag=np.empty(n);wrong=np.empty(n)
 for lo in range(0,n,4096):
  hi=min(n,lo+4096);p=budget.dot(u,x[:,lo:hi]);diag[lo:hi]=np.sum(p*y[:,lo:hi],axis=0)/d
  wrong[lo:hi]=(sy@p/d-diag[lo:hi])/(n-1)
 total=float(np.sum((budget.dot(budget.dot(u.T,gy),u))*gx.T)/(d*d))
 mean=float(np.mean(diag-wrong));var=float((total-diag@diag)/(n*(n-1))-wrong@wrong/n)
 return dict(mean=mean,variance=var,msr_scaled=mean/math.sqrt(var)/math.sqrt(d/(n/d))),diag,wrong

def arun(d,g,seed,tag,ds,budget,smoke=False):
 n=d*g;ent,(rx,ry,_)=randoms(d,n,seed,1,tag)
 x=rx.standard_normal((d,n));y=ry.standard_normal((d,n));z=(budget.dot(y,x.T)-np.outer(y.sum(axis=1),x.sum(axis=1))/n)/math.sqrt(n)
 gx=budget.dot(x,x.T);gy=budget.dot(y,y.T);a,_,bt=np.linalg.svd(z,full_matrices=False)
 polar=math.sqrt(d)*budget.dot(a,bt);base,_,_=aevaluate(polar,x,y,gx,gy,budget)
 rows=[dict(kind='A',d=d,gamma=g,seed=seed,entropy=ent,label='polar',e=None,**base)]
 es=[(str(e),e) for e in ds['e_grid']]+[('physical_epsilon',ds['epsilon_physical']*math.sqrt(n))]
 for label,e in es:
  u=np.sign(z) if e==0 else z/(np.abs(z)+e) # MSR exactly scale invariant; no estimated norm or c fit.
  r,diag,wrong=aevaluate(u,x,y,gx,gy,budget)
  if smoke:
   scores=budget.dot(budget.dot(y.T,u),x)/d;ix=np.arange(n);diag0=scores[ix,ix].copy();scores[ix,ix]=0
   w=scores.sum(axis=0)/(n-1);v=(scores*scores).sum(axis=0)/(n-1)-w*w
   assert np.allclose(diag,diag0,atol=2e-11) and np.allclose(w,wrong,atol=2e-11)
   assert abs(r['variance']-v.mean())<2e-11
   assert abs(r['mean']-np.sum(u*z)*math.sqrt(n)/(d*(n-1)))<2e-11
  rows.append(dict(kind='A',d=d,gamma=g,seed=seed,entropy=ent,label=label,e=e,epsilon=e/math.sqrt(n),paired_ratio=r['msr_scaled']/base['msr_scaled']-1,**r))
 return rows

def adam(z,s):
 u=np.zeros_like(z);m=np.zeros_like(z);v=np.zeros_like(z)
 for tt in range(1,17):
  r=u-z if s is None else u@s-z;m=.9*m+.1*r;v=.999*v+.001*r*r
  u-=.2*(m/(1-.9**tt))/(np.sqrt(v/(1-.999**tt))+.1)
 return u

def mrun(d,k,seed,tag,targets,budget,smoke=False):
 n=round(k*d*d);M=min(512,n);ent,(rx,ry,rq)=randoms(d,n,seed,2,tag)
 ids=rq.choice(n,M,replace=False);xq=np.empty((d,M));yq=np.empty_like(xq);s=np.zeros((d,d));z=np.zeros_like(s);xx=[];yy=[]
 for lo in range(0,n,8192):
  width=min(8192,n-lo);x=rx.standard_normal((d,width))/math.sqrt(d);y=ry.standard_normal((d,width))/math.sqrt(d)
  s+=budget.dot(x,x.T)*d/n;z+=budget.dot(y,x.T)*d/math.sqrt(n)
  mask=(ids>=lo)&(ids<lo+width);xq[:,mask]=x[:,ids[mask]-lo];yq[:,mask]=y[:,ids[mask]-lo]
  if smoke:xx.append(x);yy.append(y)
 names=sorted({t['name'] for t in targets});us={}
 for name in names:
  if name.startswith('Sign'):
   a=1.5 if '_a1.5_' in name else .2;u1=a*np.sign(z)
   us[name]=u1+a*np.sign(z-budget.dot(u1,s))
   us['Sign1' if d==512 else name.replace('actual','proxy')]=u1 if d==512 else u1+a*np.sign(z-u1)
  else:us[name]=adam(z,s);us[name.replace('actual','proxy')]=adam(z,None)
 names=sorted(us);vectors=np.concatenate([budget.dot(us[name],xq) for name in names],axis=1)
 truth=np.stack([np.sum(vectors[:,j*M:(j+1)*M]*yq,axis=0) for j in range(len(names))]);flat=truth.ravel();greater=np.zeros(len(flat),np.int64);ties=np.zeros(len(flat),np.int64);allids=np.tile(ids,len(names))
 _,(_,ry,_)=randoms(d,n,seed,2,tag)
 for lo in range(0,n,8192):
  width=min(8192,n-lo);y=ry.standard_normal((d,width))/math.sqrt(d)
  for start in range(0,width,1024):
   end=min(start+1024,width);score=budget.dot(vectors.T,y[:,start:end]);low=lo+start;high=lo+end
   mask=(allids>=low)&(allids<high);ir=np.flatnonzero(mask);score[ir,allids[mask]-low]=-np.inf
   greater+=np.sum(score>flat[:,None],axis=1);ties+=np.sum(score==flat[:,None],axis=1)
 assert not np.any(ties),'Nontrivial ties: report implementation failure rather than silently choose a rank'
 ranks=(greater+1).reshape(len(names),M);rankdict={name:ranks[j] for j,name in enumerate(names)}
 if smoke:
  X=np.concatenate(xx,axis=1);Y=np.concatenate(yy,axis=1)
  for j,name in enumerate(names):
   score=(us[name]@X[:,ids]).T@Y;score[np.arange(M),ids]=-np.inf
   assert np.array_equal(1+np.sum(score>truth[j,:,None],axis=1),ranks[j])
  W=np.zeros_like(z);mm=np.zeros_like(z);vv=np.zeros_like(z)
  for tt in range(1,17):
   grad=W@s-z/math.sqrt(n);mm=.9*mm+.1*grad;vv=.999*vv+.001*grad*grad
   W-=.2/math.sqrt(n)*(mm/(1-.9**tt))/(np.sqrt(vv/(1-.999**tt))+.1/math.sqrt(n))
  assert np.allclose(W*math.sqrt(n),adam(z,s),atol=3e-12)
 rows=[]
 for target in targets:
  name=target['name'];base='Sign1' if d==512 else name.replace('actual','proxy');metric=target['metric'];q=float(metric[3:])
  a=rankdict[name];b=rankdict[base];delta=(a<=math.ceil(q*n)).astype(float)-(b<=math.ceil(q*n)).astype(float)
  D=float(delta.mean());v=float((1-M/n)*delta.var(ddof=1)/M)
  rows.append(dict(kind='P2' if d==512 else 'P3',d=d,kappa=k,n=n,seed=seed,entropy=ent,name=name,baseline=base,metric=metric,M=M,query_indices=ids.tolist(),ranks_actual=a.tolist(),ranks_baseline=b.tolist(),D=D,v=v,A=D*D-v,nontrivial_ties=int(ties.sum())))
 return rows

def tasks(design):
 ds=design['design'];power=design['power'];result=[]
 # A first; P3 next; then P2 in seed-interleaved order. Never select on outcomes.
 for seed in range(max(r['N'] for r in power if r['kind']=='A')):
  for r in power:
   if r['kind']=='A' and seed<r['N']:
    result.extend(('A',d,r['gamma'],seed,[]) for d in ds['dims'])
 for seed in range(max(r['N'] for r in power if r['kind']=='P3')):
  for k in (.5,1.,2.):
   rows=[r for r in power if r['kind']=='P3' and float(r['kappa'])==k and seed<r['N']]
   if rows:result.append(('M',256,k,seed,rows))
 for seed in range(max(r['N'] for r in power if r['kind']=='P2')):
  for r in power:
   if r['kind']=='P2' and seed<r['N']:result.append(('M',512,r['kappa'],seed,[r]))
 return result

def run(mode):
 frozen();design=json.loads((OUT/'P1NUM_design.json').read_text());ds=design['design'];start=time.time();res=metadata()
 if mode=='smoke':put(OUT/'P1NUM_budget.json',dict(smoke_start=start,formal_limit_seconds=3450))
 else:assert json.loads((OUT/'P1NUM_smoke_metadata.json').read_text())['success']
 budget=Budget(start+(120 if mode=='smoke' else 3450))
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('hard stage deadline')));signal.alarm(120 if mode=='smoke' else 3450)
 jobs=[('A',12,16,0,[]),('M',16,1.,0,[dict(name='Sign2_a0.2_actual',metric='top0.01'),dict(name='Adam_b0.9_e0.1_T16_actual',metric='top0.5')]),('M',512,1.,100000,[r for r in design['power'] if r['kind']=='P2' and r['kappa']==1.])] if mode=='smoke' else tasks(design)
 done=0;count=0;error=None
 try:
  with (OUT/f'P1NUM_{mode}_rows.jsonl').open('x',buffering=1) as f,(OUT/f'P1NUM_{mode}_events.jsonl').open('x',buffering=1) as ev:
   for kind,d,k,seed,targets0 in jobs:
    budget.check();st=time.monotonic();tag=91 if mode=='smoke' else 11
    rows=arun(d,int(k),seed,tag,ds,budget,mode=='smoke') if kind=='A' else mrun(d,k,seed,tag,targets0,budget,mode=='smoke' and d<512)
    for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    done+=1;count+=len(rows);event=dict(kind=kind,d=d,k=k,seed=seed,seconds=time.monotonic()-st,elapsed=time.time()-start,completed=done,planned=len(jobs),records=count)
    ev.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
 except Exception as ex:error=repr(ex);print(error,flush=True)
 finally:
  signal.alarm(0);frozen();put(OUT/f'P1NUM_{mode}_metadata.json',dict(success=done==len(jobs) and error is None,completed=done,planned=len(jobs),records=count,error=error,seconds=time.time()-start,max_matmul_seconds=budget.max_op,resource_start=res,resource_end=metadata()))
 if error:raise RuntimeError(error)
if __name__=='__main__':
 arg=argparse.ArgumentParser();arg.add_argument('mode',choices=['prepare','smoke','formal']);mode=arg.parse_args().mode
 if mode=='prepare':prepare()
 else:run(mode)
