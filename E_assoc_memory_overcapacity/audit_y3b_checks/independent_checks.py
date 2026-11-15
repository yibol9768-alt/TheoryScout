"""独立审计：小维审计域数据，旧数据只用于规划复核，不启动正式入口。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, (90, 100))
import ast, csv, hashlib, itertools, json, math, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import t, nct
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'audit_y3b_checks'
sys.path.insert(0,str(ROOT/'y3b'))
import common
env=common.environment()
cfg=json.loads((ROOT/'y3b/config.json').read_text())
def save(name,obj):
    with (OUT/name).open('x') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
summary={}
manifest=[]
for line in (ROOT/'Y3B_FROZEN.sha256').read_text().splitlines():
    h,rel=line.split('  ',1)
    assert (ROOT/rel).resolve().is_relative_to(ROOT)
    assert sha(ROOT/rel)==h, rel
    manifest.append(rel)
assert len(set(manifest))==len(manifest)
inputs=json.loads((ROOT/'y3b/input_hashes.json').read_text())
assert all(sha(ROOT/p)==h for p,h in inputs.items())
assert sha(ROOT/'y3b/predictions.csv')==sha(ROOT/'y3_checks/Y3_frozen_predictions.csv')
summary['freeze']={'files':len(manifest),'old_inputs':len(inputs),'all_match':True,
                   'formal_exists':(ROOT/'y3b/formal').exists()}
import predict
opened=[]; watching=False
def hook(event,args):
    if watching and event=='open': opened.append(str(args[0]))
sys.addaudithook(hook)
watching=True
pred,params=predict.load()
watching=False
assert set(opened)=={str(ROOT/'y3b/predictions.csv'),str(ROOT/'y3b/parameters.json')}
pop=json.loads((ROOT/'experiments/x1/X1_population.json').read_text())['rows']
assert all(p['amplitude']==next(v for v in pop if v['ir']==ir and v['method']==m)['amplitude']
           for (ir,m),p in params.items())
summary['prediction_reads']=opened

# 独立从192份原始成功指示量计算配对方差，不使用规划协方差文件。
zs=[]; ws=[]
for s in range(192):
    with np.load(ROOT/f'experiments/x1/X1_formal_d256_r2_s{s}.npz',allow_pickle=False) as f:
        z=f['GD_old_success'].astype(float)-f['Sign_old_success'].astype(float)
        assert len(z)==2048
        zs.append(float(z.mean()))
        ws.append(float(np.sum((z-z.mean())**2)/(len(z)-1)))
v2048=float(np.sum((np.array(zs)-np.mean(zs))**2)/191)
w=float(np.mean(ws)); remainder=v2048-w/2048
p0=pred[512,2,'GD','old']['recall_limit']-pred[512,2,'Sign','old']['recall_limit']
p1=pred[512,2,'GD','old']['recall_first_order']-pred[512,2,'Sign','old']['recall_first_order']
delta=p1-p0; vq=max(0.,remainder)+w/8192
nc=abs(delta)*math.sqrt(32)/(2*math.sqrt(vq)); critical=float(t.ppf(1-.05/6,31))
power=float(nct.sf(critical,31,nc)); power125=float(nct.sf(critical,31,nc/math.sqrt(1.25)))
audit_rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261106,68732590,932,0])))
mc=200000
normal=audit_rng.standard_normal(mc); denomin=np.sqrt(audit_rng.chisquare(31,mc)/31)
mc_power=float(np.mean((normal+nc)/denomin>critical))
inflation_limit=brentq(lambda f:nct.sf(critical,31,nc/math.sqrt(f))-.8,1.,3.)
summary['power']={'p0':p0,'p1':p1,'delta':delta,'variance2048':v2048,'within_variance':w,
    'raw_remainder':remainder,'V8192':vq,'loss_mean':delta**2,'loss_sd':2*abs(delta)*math.sqrt(vq),
    'noncentrality':nc,'critical':critical,'power':power,'power_variance125':power125,
    'independent_ratio_monte_carlo':mc_power,'mc_N':mc,
    'variance_inflation_at_80pct':inflation_limit,
    'warning':'条件于修正预测为真、d256方差可外推；模拟仅统计量，不生成正式MR数据'}

# 从所有可见旧记录和静态字面种子重建旧域。额外保守范围独立列出。
old=set(); parse_errors=[]
def add(v):
    if isinstance(v,(tuple,list)) and v and all(type(x) is int and x>=0 for x in v): old.add(tuple(v))
def walk(v):
    if isinstance(v,dict):
        for k,x in v.items():
            if 'entropy' in k.lower():
                add(x)
                if isinstance(x,list):
                    for j in range(6): add(x+[j])
            walk(x)
    elif isinstance(v,list):
        for x in v:
            if isinstance(x,(dict,list)): walk(x)
for p in ROOT.rglob('*'):
    if not p.is_file() or p.is_symlink() or OUT in p.parents or ROOT/'y3b' in p.parents: continue
    if p.suffix in ['.json','.jsonl'] and p.stat().st_size<25000000:
        txt=p.read_text()
        if 'entropy' in txt:
            try:
                if p.suffix=='.json': walk(json.loads(txt))
                else:
                    for line in txt.splitlines():
                        if 'entropy' in line: walk(json.loads(line))
            except Exception as e: parse_errors.append([str(p.relative_to(ROOT)),repr(e)])
    if p.suffix=='.py':
        try: tree=ast.parse(p.read_text())
        except SyntaxError: continue
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['SeedSequence','default_rng','RandomState'] and node.args:
                try: val=ast.literal_eval(node.args[0])
                except (ValueError,TypeError): continue
                add([val] if type(val) is int else val)
for ph,d,ir,s,k in itertools.product([11,22,33],[16,128,256,512],range(3),range(192),range(4)):
    add([20261106,913752,ph,d,ir,s,k])
for d,ir,s,k in itertools.product([128,256,512],range(3),range(64),range(4)):
    add([20261106,30917401,71,d,ir,s,k])
domains=[
    ((20261106,440004,k,d,s,j) for k,d,s,j in itertools.product([1,2,9],[16,32,64,128,256,512],range(32),range(6))),
    ((350611,10,d,s) for d,s in itertools.product([16,32,64,128],range(4))),
    ((350611,20,d,g,s) for d,g,s in itertools.product([32,64,128,256],[16,64],range(5))),
    ((730619,c,d,s) for c,d,s in itertools.product(range(6),[32,64,128],range(4))),
    ((20261106,880008,d,s) for d,s in itertools.product([64,128,256],range(64))),
    ((880082,d,s) for d,s in itertools.product([32,64,96],range(16))),
    ((900000+10000*d+s,) for d,s in itertools.product([64,128,256],range(8))),
    ((19000000+s,) for s in range(24)),
    ((20261106,220099,d,s) for d,s in itertools.product([64,128,256,512],range(96))),
    ((202611063,g,s) for g,s in itertools.product([16,64],range(3))),
    ((611063027,d,c,s) for d,c,s in itertools.product([64,128],range(6),range(16))),
    itertools.chain([(611063104,)],((611064018,d,c,s) for d,c,s in itertools.product([64,128],[1,2],range(12)))),
    ((8117903,d,s,c) for d,s,c in itertools.product([16,32,64,128,256,512,1024],range(64),range(4))),
    ((100000*d+1000*k+s,) for d,k,s in itertools.product([32,64,128],[1,2,4],range(5))),
    ((2026110600+100000*d+1000*g+s,) for d,g,s in itertools.product([16,32,64,128,256,512],[2,4],range(64))),
    ((120000+d*10+s,) for d,s in itertools.product([256,512],range(3))),
    ((s,) for s in range(10000))]
for domain in domains:
    for e in domain: add(e)
streams=json.loads((ROOT/'y3b/streams.json').read_text())
base=[(d,ir) for ir in range(3) for d in [128,256,512]]
expected=[]
for s in range(32):
    for d,ir in base[s%9:]+base[:s%9]:
        for k in range(4):
            expected.append(dict(d=d,ir=ir,seed=s,stream=k,entropy=[20261106,87410329,83,d,ir,s,k]))
assert streams==expected
def state(e):return np.random.SeedSequence(e).generate_state(8).tobytes()
new={state(row['entropy']) for row in streams}
assert len(new)==1152
collisions=[list(e) for e in old if state(e) in new]
assert not collisions and not parse_errors
save('independent_old_entropies.json',[list(e) for e in sorted(old)])
summary['seeds']={'planned':len(new),'old_domain_count':len(old),'collisions':collisions,'parse_errors':parse_errors}

# 实际受审函数只替换rng为审计域；d=16，不使用任何正式熵元组。
import run
audit_entropies=[]
def audit_generator(d,ir,s,k):
    ent=[20261106,68732590,932,d,ir,s,k]; audit_entropies.append(ent)
    assert state(ent) not in new and tuple(ent) not in old
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(ent)))
run.rng=audit_generator
maxima={'data':0.,'optimizer_U':0.,'D':0.,'V':0.}; cases=[]
for ir,r in enumerate([.5,0.,-.5]):
    d=16;n=64;Q=7
    def arr(k):return audit_generator(d,ir,0,k).standard_normal((d,n),dtype=np.float64)/math.sqrt(d)
    x=arr(0);y0=arr(1);noise=arr(2);y1=math.sqrt(1-r*r)*noise+r*y0
    ids=audit_generator(d,ir,0,3).choice(n,Q,replace=False,shuffle=True)
    actual=run.dataset(d,ir,0,Q)
    # 从原始损失梯度、W坐标和偏差校正的Adam重新实现，不复制Z/H更新式。
    W0=d/n*(y0@x.T)
    gradient=d/n*((W0@x-y1)@x.T)
    for a,b in zip(actual[:4],[x,y0,y1,ids]):
        maxima['data']=max(maxima['data'],float(np.max(np.abs(a-b))))
    assert np.array_equal(ids,actual[3])
    cellcfg=dict(cfg,query_default=Q,query_chunk=3,candidate_chunk=11)
    actual_cell=run.cell(d,ir,0,cellcfg,params)
    arrays={'x':x,'y0':y0,'y1':y1,'ids':ids}
    for m in common.METHODS:
        p=params[ir,m];alpha=p['amplitude']
        if m=='GD': W=W0-alpha*gradient
        elif m=='Sign': W=W0-alpha/math.sqrt(n)*np.sign(gradient)
        else:
            e={'Adam2':2.,'Adam1':1.,'Adam.1':.1}[m]
            beta1,beta2=.9,.999
            mt=(1-beta1)*gradient;vt=(1-beta2)*gradient**2
            W=W0-alpha/math.sqrt(n)*(mt/(1-beta1))/(np.sqrt(vt/(1-beta2))+e*math.sqrt(2*(1-r))/math.sqrt(n))
        U=math.sqrt(n)*W
        actual_U=actual[4]+alpha*run.direction(actual[5],m,r)
        maxima['optimizer_U']=max(maxima['optimizer_U'],float(np.max(abs(U-actual_U))))
        for task,y in [('old',y0),('new',y1)]:
            scores=y.T@(U@x[:,ids]); ranks=[];ds=[];vs=[]
            for j,i in enumerate(ids):
                order=sorted(range(n),key=lambda k:(-scores[k,j],k))
                ranks.append(order.index(int(i))+1)
                ds.append(scores[i,j]/math.sqrt(p['q']))
                vs.append(sum(scores[k,j]**2 for k in range(n) if k!=i)/((n-1)*p['q']))
            pre=m+'_'+task+'_'
            assert np.array_equal(actual_cell[pre+'ranks'],ranks)
            assert np.array_equal(actual_cell[pre+'success'],np.array(ranks)<=math.ceil(.1*n))
            for key,reference in [('D',ds),('V',vs)]:
                err=float(np.max(abs(actual_cell[pre+key]-reference)))
                maxima[key]=max(maxima[key],err);assert err<1e-11
            cases.append(dict(d=d,ir=ir,r=r,method=m,task=task,rank_match=True))
            arrays[pre+'ranks']=np.array(ranks);arrays[pre+'D']=np.array(ds);arrays[pre+'V']=np.array(vs)
    with (OUT/f'independent_d16_r{ir}.npz').open('xb') as f:np.savez(f,**arrays)
save('model_cases.json',cases)
summary['model']={'cases':len(cases),'max_errors':maxima,'audit_entropy_prefix':[20261106,68732590,932],
    'formal_random_streams_used':0,'actual_runner_functions_tested':['dataset','direction','cell','ranks']}

import analyze
checks=[]
for family in [3,72,45]:
    values=[-.01,.002,.004,.008,.012]
    result=analyze.interval(values,family)
    mean=sum(values)/len(values);se=math.sqrt(sum((v-mean)**2 for v in values)/20)
    assert abs(result['mean']-mean)<1e-16 and abs(result['se']-se)<1e-16
    assert abs(t.cdf(result['critical'],4)-(1-.05/(2*family)))<1e-11
    checks.append('interval_family_'+str(family))
assert not analyze.equivalent(dict(lower=-.005,upper=.004))
assert not analyze.equivalent(dict(lower=-.004,upper=.005))
assert analyze.interval([.1],3)['lower'] is None
summary['statistics']={'checks':checks,'equivalence_band':cfg['equivalence_band'],'boundary_rejected':True}
u=resource.getrusage(resource.RUSAGE_SELF)
summary['resources']={'environment':env,'cpu_seconds':u.ru_utime+u.ru_stime,'peak_rss_mib':u.ru_maxrss/1024}
save('independent_summary.json',summary)
print(json.dumps(summary,ensure_ascii=False))
