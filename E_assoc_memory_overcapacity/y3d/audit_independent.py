"""独立故障注入：只复用确定性旧审计NPZ，无Generator/正式运行器。"""
import ast, contextlib, copy, csv, difflib, hashlib, json, os
from pathlib import Path
import resource, shutil, signal, subprocess, sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'y3d'/'audit_rerun'
sys.path.insert(0,str(ROOT/'y3d'))
import common, integrity, publication
common.environment()
import numpy as np
cfg=common.config();mh=common.sha(ROOT/'Y3C_FROZEN.sha256')
TEST=OUT/'independent_faults';TEST.mkdir()
results=[]
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def forbidden(*a,**k):raise RuntimeError('审计禁止随机流')
np.random.default_rng=forbidden;np.random.Generator=forbidden;np.random.PCG64=forbidden
U=dict(cpu_seconds=1.,peak_rss_mib=100.,monotonic=1.)
plan=list(common.jobs(cfg['blocks']))
okrun=dict(program='run.py',returncode=0,wall_hard_killed=False)
okanalyze=dict(program='analyze.py',returncode=0,wall_hard_killed=False)
def status(rc=0):
    return dict(stages=[copy.deepcopy(okrun),dict(okanalyze,returncode=rc)],
                wall_seconds=100.,cpu_seconds_children=3.,supervisor_usage=U.copy())
BASE=TEST/'control'/'formal';BASE.mkdir(parents=True)
source=ROOT/'audit_y3b_checks/full_plan_run/target_fixture'
# 仅把确定性合成NPZ的metadata清单绑定改为真实Y3C清单；所有40个测量数组原样复制。
for d,ir,s in plan:
    name=f'd{d}_r{ir}_s{s}.npz'
    with np.load(source/name,allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    meta=json.loads(str(a['metadata']));meta['manifest_sha256']=mh
    a['metadata']=np.array(json.dumps(meta))
    with (BASE/name).open('xb') as f:np.savez_compressed(f,**a)
save(BASE/'launch.json',dict(schema=1,manifest_sha256=mh,cpu_hard_split=[21340,50,10],
    preparation_cpu_reserve=200,wall_hard_total=12600,usage=U))
save(BASE/'start.json',dict(schema=1,manifest_sha256=mh,streams_sha256=common.sha(ROOT/'y3d/streams.json'),
    program_sha256=common.sha(ROOT/'y3d/run.py'),cfg=cfg,usage=U,no_draw_before_this_record=True))
events=[]
for k in plan:
    name=f'd{k[0]}_r{k[1]}_s{k[2]}.npz'
    events.extend([dict(schema=1,state='started',cell=list(k),usage=U),
        dict(schema=1,state='complete',cell=list(k),sha256=common.sha(BASE/name),usage=U)])
events.append(dict(schema=1,state='stopped',reason='planned_complete',error=None,incomplete_cell=None,usage=U))
with (BASE/'events.jsonl').open('x') as f:
    for e in events:f.write(json.dumps(e)+'\n')
save(BASE/'runner_status.json',dict(schema=1,reason='planned_complete',error=None,completed=[list(k) for k in plan],
    incomplete_cell=None,usage=U,wall_seconds=100.))
save(BASE/'run_exit.json',okrun)
WORKER=r'''
import os,sys,signal
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import analyze
data=Path(sys.argv[2]);mh=sys.argv[3];mode=sys.argv[4]
analyze.OUT=data.parent;analyze.verify=lambda:mh
sys.argv=['analyze.py','--formal-via-launcher'];os.environ['Y3D_LAUNCHER']='1'
analyze.main()
if mode=='kill_after_pending':os.kill(os.getpid(),signal.SIGKILL)
if mode=='clobber_report_after_pending':
    p=data/'analysis/all_completed.csv'
    with p.open('rb') as f:header=f.readline()
    p.write_bytes(header)
'''
def analyze(data,mode='normal'):
    with (data.parent/'actual_analyzer.log').open('x') as f:
        child=subprocess.run([sys.executable,'-B','-c',WORKER,str(ROOT/'y3d'),str(data),mh,mode],
                             stdout=f,stderr=subprocess.STDOUT,timeout=30)
    return child.returncode
def clone(label,analysis=False):
    data=TEST/label/'formal';data.mkdir(parents=True)
    for p in BASE.iterdir():
        if p.is_file() and (p.suffix=='.npz' or p.name in publication.INPUTS):shutil.copyfile(p,data/p.name)
    if analysis:
        shutil.copytree(BASE/'analysis',data/'analysis',ignore=shutil.ignore_patterns('FINAL_ACCEPTED*'))
    return data
def verify(data):
    try:publication.verify_accepted(data,mh,cfg)
    except Exception as e:return {'verified':False,'error':repr(e)}
    return {'verified':True}
def commit(data,st=None):return publication.finalize(data,mh,cfg,st or status())
assert analyze(BASE)==0
assert commit(BASE) and verify(BASE)['verified']
pending=json.loads((BASE/'analysis/PENDING.json').read_text())
assert pending['candidate_decisions']['primary_both_passed'] and pending['candidate_decisions']['d512_passed']
results.append(dict(case='true_y3c_binding_control',accepted=True,verified=True,complete_cells=pending['complete_cells']))

# 新故障1：监管器在完整凭据临时文件fsync后、原子link前真实SIGKILL。
data=clone('supervisor_kill_before_publish',True)
supervisor_worker=r'''
import os,sys,signal,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import common,publication
data=Path(sys.argv[2]);mh=sys.argv[3];status=json.loads(sys.argv[4])
original=publication.os.link
def kill_before_link(src,dst,*a,**kw):
    if Path(dst).name=='FINAL_ACCEPTED.json':os.kill(os.getpid(),signal.SIGKILL)
    return original(src,dst,*a,**kw)
publication.os.link=kill_before_link
publication.finalize(data,mh,common.config(),status)
'''
with (data.parent/'actual_supervisor.log').open('x') as f:
    child=subprocess.run([sys.executable,'-B','-c',supervisor_worker,str(ROOT/'y3d'),str(data),mh,json.dumps(status())],
                         stdout=f,stderr=subprocess.STDOUT,timeout=10)
r=dict(case=data.parent.name,returncode=child.returncode,
       pending_credential=(data/'analysis/FINAL_ACCEPTED.pending').exists(),
       final_exists=(data/'analysis/FINAL_ACCEPTED.json').exists(),**verify(data))
assert child.returncode==-9 and r['pending_credential'] and not r['final_exists'] and not r['verified'];results.append(r)

# 新故障2：FINAL_ACCEPTED提交后篡改NPZ，且保留可装载、合法形状。
data=clone('npz_mutation_after_accept',True);assert commit(data)
p=data/'d128_r0_s0.npz'
with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
a['GD_old_D']=a['GD_old_D'].copy();a['GD_old_D'][0]+=.125
with p.open('wb') as f:np.savez_compressed(f,**a)
r=dict(case=data.parent.name,final_exists=True,**verify(data));assert not r['verified'];results.append(r)

# 新故障3：已发布凭据后改表（真实分析完成和监管提交后）。
data=clone('report_mutation_after_accept',True);assert commit(data)
with (data/'analysis/endpoints.csv').open('ab') as f:f.write(b'tampered\n')
r=dict(case=data.parent.name,final_exists=True,**verify(data));assert not r['verified'];results.append(r)

# 新故障4：分析main写完PENDING后、真实子进程正常返回前清空数据表。
data=clone('report_mutation_before_accept');rc=analyze(data,'clobber_report_after_pending')
assert rc==0
accepted=commit(data,status(rc));r=dict(case=data.parent.name,analyzer_returncode=rc,accepted=accepted,**verify(data))
with (data/'analysis/all_completed.csv').open() as f:r['reported_data_rows']=len(list(csv.DictReader(f)))
r['candidate_complete_cells']=json.loads((data/'analysis/PENDING.json').read_text())['complete_cells']
r['decisions']=json.loads((data/'analysis/FINAL_ACCEPTED.json').read_text())['primary_both_passed'] if accepted else None
assert not accepted and not r['verified'] and r['reported_data_rows']==0;results.append(r)

# 新故障5：仅候选PENDING的统计标志在提交前损坏，原NPZ/表/退出不改。
data=clone('decision_mutation_before_accept',True)
p=data/'analysis/PENDING.json';x=json.loads(p.read_text());before=x['candidate_decisions']['full72_equivalent']
x['candidate_decisions']['full72_equivalent']=not before;p.write_text(json.dumps(x))
accepted=commit(data);r=dict(case=data.parent.name,original_full72=before,mutated_full72=not before,accepted=accepted,**verify(data))
assert not accepted and not r['verified'];results.append(r)

# 新故障6：监管重查后、发布前NPZ改变；二次采信必须拒绝。
data=clone('npz_change_during_finalize',True);calls=[]
def probe():
    calls.append(1)
    if len(calls)==1:
        with (data/'d128_r0_s0.npz').open('ab') as f:f.write(b'late-change')
    return dict(wall_seconds=100.,supervisor_usage=U)
accepted=publication.finalize(data,mh,cfg,status(),budget_probe=probe)
r=dict(case=data.parent.name,accepted_path_created=accepted,**verify(data));assert not r['verified'];results.append(r)

save(TEST/'results.json',dict(results=results,manifest_sha256=mh,no_formal_sampling=True,
    new_random_training_or_queries=False,usage=common.usage(),
    child_cpu_seconds=sum((getattr(resource.getrusage(resource.RUSAGE_CHILDREN),k) for k in ['ru_utime','ru_stime']))))
print(json.dumps(results,ensure_ascii=False))
