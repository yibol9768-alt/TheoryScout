"""确定性合成测试：不生成MR数组、不构造随机Generator、不抽查询。"""
import resource
import os
import csv
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from scipy.stats import t
import run  # 先导入：其RLIMIT仅下调；下面分析模块再下调到50 CPU秒
import analyze
import common
from common import ROOT, OUT, METHODS, TASKS, DIMS, dump, sha, environment, usage
from predict import load

environment()
name=sys.argv[2] if len(sys.argv)==3 and sys.argv[1]=='--name' else 'synthetic'
assert name.startswith('synthetic') and name.replace('_','').isalnum()
TEST=ROOT/'audit_y3c_checks'/'synthetic_replay';TEST.mkdir()
checks=[]
def check(label, condition):
    assert condition,label
    checks.append(label)

# 禁止本进程任何新训练/查询抽样；下面只能用显式数组及整数运算。
def forbidden(*args,**kwargs): raise AssertionError('合成测试禁止随机抽样')
run.rng=forbidden
np.random.default_rng=forbidden
np.random.Generator=forbidden
np.random.PCG64=forbidden

v=[.001,.002,.004,.003]
for family in [2,3,72]:
    s=analyze.interval(v,family)
    se=math.sqrt(sum((x-.0025)**2 for x in v)/(3*4))
    check(f'family{family}_SE',abs(s['se']-se)<1e-17)
    check(f'family{family}_tail',abs(t.cdf(s['critical'],3)-(1-.05/(2*family)))<2e-12)
    check(f'family{family}_bounds',abs(s['lower']-(.0025-s['critical']*se))<1e-17)
    raw=2*t.sf(abs(.0025/se),3)
    check(f'family{family}_p',abs(s['p_bonferroni']-min(1.,raw*family))<1e-15)
check('n0_unavailable',analyze.interval([],72)['lower'] is None)
check('n1_unavailable',analyze.interval([.1],72)['se'] is None)
check('zero_variance_positive',analyze.interval([.01]*24,3)['lower']>0)
check('zero_variance_zero',analyze.interval([0.]*24,3)['p_two_sided']==1.)
check('equivalence_inside',analyze.equivalent(dict(lower=-.00499,upper=.00499)))
check('equivalence_boundary_fails',not analyze.equivalent(dict(lower=-.005,upper=.004)))
check('equivalence_outside',not analyze.equivalent(dict(lower=-.006,upper=.001)))
check('equivalence_unavailable',not analyze.equivalent(dict(lower=None,upper=None)))
for y,p0,p1 in [(.25,.3,.24),(.2,.1,.3),(.51,.53,.52)]:
    check('loss_identity_'+str(y),abs(analyze.loss(y,p0,p1)-((y-p0)**2-(y-p1)**2))<2e-17)

pred,_=load(); cfg=common.config()
def records(count,where='corrected'):
    rows={}
    for s in range(count):
        for d in DIMS:
            for ir in range(3):
                rr={}
                for m in METHODS:
                    for task in TASKS:
                        p=pred[d,ir,m,task]
                        # 同一seed的所有格共享确定性±1e-5扰动：仅为测试区组聚合。
                        recall=p['recall_first_order' if where=='corrected' else 'recall_limit'] + (1 if s%2 else -1)*1e-5
                        rr[m,task]=dict(recall=recall,D=p['mu']+p['m']/d,V=1+p['w']/d,
                            var_D=1.,var_V=4/d,cov_DV=4*p['mu']/d)
                rows[d,ir,s]=rr
    return rows

for label,n,truth in [('pass32',32,'corrected'),('fail32',32,'uncorrected'),('insufficient23',23,'corrected')]:
    dest=TEST/label;dest.mkdir()
    result=analyze.analysis(records(n,truth),pred,cfg,dest)
    check(label+'_main',result['primary_both_passed']==(truth=='corrected' and n>=24))
    check(label+'_d512',result['d512_passed']==(truth=='corrected' and n==cfg['blocks']))
    with (dest/'diagnostics72.csv').open() as f: rr=list(csv.DictReader(f))
    check(label+'_72_cells',len(rr)==72)
    with (dest/'secondary_old45.csv').open() as f: rr=list(csv.DictReader(f))
    check(label+'_adam1_reported',sum(r['method']=='Adam1' for r in rr)==9)
    dump(dest/'result.json',result)

# 不完整区组：9*24-1份完成数据全部进入逐格表；主分析只能23区组。
rr=records(24);rr.pop((128,0,23))
dest=TEST/'incomplete';dest.mkdir();res=analyze.analysis(rr,pred,cfg,dest)
check('incomplete_block_filter',res['complete_blocks']==23 and not res['primary_both_passed'])
check('ungrouped_retained',res['complete_cells']==215 and res['ungrouped_complete_cells']==8)
with (dest/'all_completed.csv').open() as f: allrows=list(csv.DictReader(f))
check('all_completed_in_table',len(allrows)==2150)
dump(dest/'result.json',res)
dest=TEST/'empty';dest.mkdir();res=analyze.analysis({},pred,cfg,dest)
check('empty_no_confirmation',not res['primary_both_passed'] and not res['d512_passed'])
with (dest/'all_completed.csv').open() as f: check('empty_no_fake_rows',len(list(csv.DictReader(f)))==0)

# 确定性小矩阵检索；含平局、自候选排除、不同块大小；不抽任何随机数组。
x=np.eye(4,dtype=float);u=np.eye(4,dtype=float)
y=np.array([[1,1,1,1],[0,2,2,0],[2,0,1,2],[1,1,0,1]],dtype=float)
ids=np.array([3,1,0,2],dtype=np.int64)
rank,ok,D,V=run.ranks(u,x,y,ids,2.,query_chunk=2,candidate_chunk=2)
score=y.T@(u@x[:,ids]);expected=[]
for j,idx in enumerate(ids):
    order=np.lexsort((np.arange(4),-score[:,j]))
    expected.append(int(np.flatnonzero(order==idx)[0])+1)
check('ranks_ties_self_exclusion',np.array_equal(rank,expected))
check('D_normalization',np.allclose(D,np.diag(y)[ids]/math.sqrt(2)))
check('V_uncentered_wrong_only',np.allclose(V,[(sum(score[:,j]**2)-score[idx,j]**2)/6 for j,idx in enumerate(ids)]))
run.soft_stop(signal.SIGXCPU,None)
try: run.checkpoint()
except run.StopRun: checks.append('cpu_soft_checkpoint')
else: raise AssertionError('cpu_soft_checkpoint')
run.STOP=None
run.soft_stop(signal.SIGALRM,None)
try: run.checkpoint()
except run.StopRun: checks.append('wall_soft_checkpoint')
else: raise AssertionError('wall_soft_checkpoint')
run.STOP=None
path=TEST/'atomic.npz';run.commit(path,dict(a=np.array([1,2,3])))
check('atomic_commit',path.is_file() and not path.with_suffix('.partial').exists())
try: run.commit(path,dict(a=np.array([9])))
except FileExistsError: checks.append('atomic_no_overwrite')
else: raise AssertionError('atomic_no_overwrite')
check('atomic_original_preserved',np.array_equal(np.load(path)['a'],[1,2,3]))

# 正式NPZ装载器：十个方法/任务必须齐全，rank与success须一致。
loader=TEST/'loader';loader.mkdir()
Q=cfg['query_default'];aa={'ids':np.arange(Q,dtype=np.int64),
    'metadata':np.array(json.dumps(dict(d=128,ir=0,seed=0,manifest_sha256='test',schema=1)))}
for m in METHODS:
    for task in TASKS:
        for suffix,value in [('ranks',np.ones(Q,dtype=np.int64)),('success',np.ones(Q,dtype=bool)),
                             ('D',np.ones(Q)),('V',np.ones(Q))]:
            aa[m+'_'+task+'_'+suffix]=value
run.commit(loader/'d128_r0_s0.npz',aa)
key,loaded=analyze.read_cell(loader/'d128_r0_s0.npz','test',cfg)
check('loader_complete_ten',key==(128,0,0) and len(loaded)==10)
broken=TEST/'broken_loader';broken.mkdir();bad=dict(aa);bad.pop('Adam1_new_V')
run.commit(broken/'d128_r0_s0.npz',bad)
try: analyze.read_cell(broken/'d128_r0_s0.npz','test',cfg)
except AssertionError: checks.append('loader_missing_Adam1_rejected')
else: raise AssertionError('loader_missing_Adam1_rejected')

# 隔离夹具中的清单校验/篡改拒绝，绝不写主项目旧文件或正式清单。
fixture=TEST/'freeze_fixture';(fixture/'y3c').mkdir(parents=True)
required=['Y3C_PREREG.md','Y3C_READY.md','y3c/config.json','y3c/environment.json',
          'y3c/predictions.csv','y3c/streams.json','y3c/run.py','y3c/analyze.py',
          'y3c/common.py','y3c/launch.py','y3c/predict.py','y3c/parameters.json',
          'y3c/integrity.py','y3c/publication.py']
for rel in required:
    with (fixture/rel).open('x') as f: f.write('{}' if rel.endswith('environment.json') else 'fixture')
with (fixture/'Y3C_FROZEN.sha256').open('x') as f:
    for rel in required: f.write(sha(fixture/rel)+'  '+rel+'\n')
oldroot,oldout,oldenv=common.ROOT,common.OUT,common.environment
common.ROOT=fixture;common.OUT=fixture/'y3c';common.environment=lambda:{}
try:
    check('manifest_valid',common.verify()==sha(fixture/'Y3C_FROZEN.sha256'))
    with (fixture/'y3c/predictions.csv').open('a') as f:f.write('tampered')
    try: common.verify()
    except AssertionError: checks.append('manifest_tamper_rejected')
    else: raise AssertionError('manifest_tamper_rejected')
finally:
    common.ROOT,common.OUT,common.environment=oldroot,oldout,oldenv

# 缩短限额的真实OS核验：软CPU信号处理后写状态，硬CPU忽略软信号后被杀。
for label,code in [
 ('soft',"import resource,signal,sys; resource.setrlimit(resource.RLIMIT_CPU,(1,2)); signal.signal(signal.SIGXCPU,lambda a,b:sys.exit(23)); exec('while True: pass')"),
 ('hard',"import resource,signal; resource.setrlimit(resource.RLIMIT_CPU,(1,2)); signal.signal(signal.SIGXCPU,signal.SIG_IGN); exec('while True: pass')")]:
    p=subprocess.run([sys.executable,'-B','-c',code],timeout=8,capture_output=True)
    check('RLIMIT_CPU_'+label,p.returncode==(23 if label=='soft' else -9))
child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(5)'])
try: child.wait(timeout=.05)
except subprocess.TimeoutExpired:
    child.kill();child.wait();checks.append('wall_hard_supervisor_kill')
else: raise AssertionError('wall_hard_supervisor_kill')

# 静态隔离检查：预测、正式运行与分析不包含任何X1数据路径或旧程序导入。
for filename in ['predict.py','run.py','analyze.py']:
    txt=(OUT/filename).read_text()
    check(filename+'_no_X1_path','experiments/x1' not in txt and 'X1_formal' not in txt and 'import prepare' not in txt)
source=(OUT/'run.py').read_text()
check('verify_before_sampling',source.index('mh = verify()') < source.index('arrays = cell('))

children=resource.getrusage(resource.RUSAGE_CHILDREN)
dump(ROOT/'audit_y3c_checks'/'synthetic_results.json',dict(passed=True,checks=checks,no_formal_sampling=True,
    no_random_training_or_queries=True,usage=usage(),cpu_seconds_children=children.ru_utime+children.ru_stime))
print(json.dumps(dict(passed=True,checks=len(checks),usage=usage()),ensure_ascii=False))
