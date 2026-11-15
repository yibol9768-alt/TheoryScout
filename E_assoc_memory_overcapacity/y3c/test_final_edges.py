"""对最终版模块重核全部账本反例，并检查额外schema和发布边界。"""
import ast,contextlib,copy,csv,json,os,resource,shutil,subprocess,sys
from pathlib import Path
import run
import analyze
import common,integrity,publication
from common import ROOT,OUT,sha,dump,usage
resource.setrlimit(resource.RLIMIT_CPU,(49,50))
common.environment()
TEST=OUT/'final_edges';TEST.mkdir()
PRIOR=OUT/'protocol_v1'
checks=[]
def check(n,c):
    assert c,n
    checks.append(n)
cfg=common.config();mh=sha(ROOT/'Y3B_FROZEN.sha256')
def forbidden(*a,**k):raise AssertionError('no random sampling')
run.rng=forbidden;run.cell=forbidden
for name in ['default_rng','Generator','PCG64']:setattr(analyze.np.random,name,forbidden)
def inspect(data):
    hashes={p.name:sha(p) for p in data.glob('*.npz')}
    keys={tuple(map(int,p.stem.replace('d','').replace('r','').replace('s','').split('_'))) for p in data.glob('*.npz')}
    return integrity.inspect(data,mh,cfg,keys,hashes)
prior=json.loads((PRIOR/'FAILED.json').read_text());check('prior_failure_preserved',prior['passed'] is False)
for name,functions in [('analyze.py',['interval','equivalent','loss','analysis','read_cell']),
                        ('run.py',['soft_stop','checkpoint','rng','dataset','direction','ranks','cell','commit']),
                        ('common.py',['entropy','jobs'])]:
    def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    before=nodes(ROOT/'y3b'/name);after=nodes(OUT/name)
    for fn in functions:check('unchanged_'+name+'_'+fn,before[fn]==after[fn])
for name in ['config.json','parameters.json','predictions.csv','streams.json','seed_audit.json',
             'rng_inventory.json','power_summary.json','primary_power.json','planning_covariances.json','power_grid.csv','environment.json']:
    check('byte_identical_'+name,sha(ROOT/'y3b'/name)==sha(OUT/name))
check('prereg_byte_identical',sha(ROOT/'Y3B_PREREG.md')==sha(ROOT/'Y3C_PREREG.md'))
# 每份原审计/单故障夹具使用最终版模块再次校验。
for case in sorted(PRIOR.iterdir()):
    if case.is_dir() and (case.name.startswith('audit_') or (case/'integrity_test.json').exists()):
        check('final_integrity_'+case.name,not inspect(case/'formal')['valid'])
        if case.name.startswith('audit_'):
            pending=json.loads((case/'formal/analysis/PENDING.json').read_text())
            with (case/'formal/analysis/all_completed.csv').open() as f:rows=list(csv.DictReader(f))
            check('audit_all_readable_disclosed_'+case.name,len(rows)==10*len(list((case/'formal').glob('*.npz'))) and not pending['confirmation_valid'])
for name in ['control_normal','soft_stop_partial','soft_stop_empty']:
    check('final_valid_'+name,inspect(PRIOR/name/'formal')['valid'])
    check('final_credential_'+name,publication.verify_accepted(PRIOR/name/'formal',mh,cfg)['confirmation_valid'])
def clone(name,include_analysis=False):
    data=TEST/name/'formal';data.mkdir(parents=True)
    base=PRIOR/'control_normal/formal'
    for p in base.iterdir():
        if p.is_file() and (p.suffix=='.npz' or p.name in publication.INPUTS):shutil.copyfile(p,data/p.name)
    if include_analysis:shutil.copytree(base/'analysis',data/'analysis',ignore=shutil.ignore_patterns('FINAL_ACCEPTED.json'))
    return data
for name,file,field,value in [
 ('runner_reason_list','runner_status.json','reason',[]),
 ('runner_schema_bool','runner_status.json','schema',True),
 ('start_schema_bool','start.json','schema',True),
 ('launch_schema_bool','launch.json','schema',True),
 ('completed_boolean_cell','runner_status.json','completed',[[128,False,0]]),
]:
    data=clone(name);p=data/file;obj=json.loads(p.read_text());obj[field]=value;p.write_text(json.dumps(obj))
    check(name,not inspect(data)['valid'])
    if name=='runner_reason_list':
        analyze.OUT=data.parent;analyze.verify=lambda:mh
        sys.argv=['analyze.py','--formal-via-launcher'];os.environ['Y3C_LAUNCHER']='1'
        with (data.parent/'stdout').open('x') as f,contextlib.redirect_stdout(f):analyze.main()
        pending=json.loads((data/'analysis/PENDING.json').read_text())
        with (data/'analysis/all_completed.csv').open() as f:rows=list(csv.DictReader(f))
        check('malformed_status_still_discloses_288',len(rows)==2880 and not pending['candidate_confirmation_valid'])
# 一份NPZ已损坏，其他287份依然披露。
data=clone('corrupt_npz');(data/'d128_r0_s0.npz').write_bytes(b'broken')
analyze.OUT=data.parent;analyze.verify=lambda:mh
sys.argv=['analyze.py','--formal-via-launcher'];os.environ['Y3C_LAUNCHER']='1'
with (data.parent/'stdout').open('x') as f,contextlib.redirect_stdout(f):analyze.main()
pending=json.loads((data/'analysis/PENDING.json').read_text())
with (data/'analysis/all_completed.csv').open() as f:rows=list(csv.DictReader(f))
check('corrupt_npz_other_287_disclosed',len(rows)==2870 and len(pending['invalid_files'])==1 and not pending['candidate_confirmation_valid'])
# 最后一次预算探针也必须通过，不能仅在哈希检查前采一次值。
data=clone('late_budget_exceeded',True)
st=dict(stages=[dict(program='run.py',returncode=0,wall_hard_killed=False),
                dict(program='analyze.py',returncode=0,wall_hard_killed=False)],
    wall_seconds=1.,cpu_seconds_children=3.,supervisor_usage=dict(cpu_seconds=1.,peak_rss_mib=100.,monotonic=2.))
probes=[]
def probe():
    probes.append(1)
    return dict(wall_seconds=1. if len(probes)==1 else 12601.,supervisor_usage=st['supervisor_usage'])
check('late_budget_no_publication',not publication.finalize(data,mh,cfg,st,budget_probe=probe))
check('late_budget_no_credential',not (data/'analysis/FINAL_ACCEPTED.json').exists() and (data/'publication_aborted.json').exists())
# 修改候选数据哈希也必须由监管器拒绝。
data=clone('changed_npz_after_analysis',True)
(data/'d128_r0_s0.npz').write_bytes(b'changed')
check('changed_npz_rejected_by_supervisor',not publication.finalize(data,mh,cfg,st))
# 独立复制每份状态，逐项验证，避免初次测试的共享字典污染。
for label,update in [
 ('analysis_nonzero',lambda s:s['stages'][1].update(returncode=7)),
 ('analysis_hard_flag',lambda s:s['stages'][1].update(wall_hard_killed=True)),
 ('analysis_negative_return',lambda s:s['stages'][1].update(returncode=-9)),
 ('supervisor_over_budget',lambda s:s['supervisor_usage'].update(cpu_seconds=10)),
 ('total_cpu_over_budget',lambda s:s.update(cpu_seconds_children=21400)),
 ('total_wall_over_budget',lambda s:s.update(wall_seconds=12601)),
 ('exit_evidence_disagrees',lambda s:s['stages'][0].update(returncode=1)),
]:
    data=clone(label,True);state=copy.deepcopy(st);update(state)
    check(label,not publication.finalize(data,mh,cfg,state) and not (data/'analysis/FINAL_ACCEPTED.json').exists())
worker='''
import os,sys,signal
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import analyze
analyze.OUT=Path(sys.argv[2]);mh=sys.argv[3];mode=sys.argv[4]
analyze.verify=lambda:mh
sys.argv=['analyze.py','--formal-via-launcher'];os.environ['Y3C_LAUNCHER']='1'
if mode=='before_pending':
    original=analyze.dump
    def dump(path,obj):
        if path.name=='PENDING.json':os.kill(os.getpid(),signal.SIGKILL)
        original(path,obj)
    analyze.dump=dump
analyze.main()
if mode=='after_pending':os.kill(os.getpid(),signal.SIGKILL)
if mode=='nonzero':raise SystemExit(7)
'''
injections=[]
for mode,expected in [('after_pending',-9),('before_pending',-9),('nonzero',7)]:
    data=clone('actual_analysis_'+mode)
    with (data.parent/'child.stdout').open('x') as f:
        child=subprocess.run([sys.executable,'-B','-c',worker,str(OUT),str(data.parent),mh,mode],stdout=f,stderr=subprocess.STDOUT,timeout=30)
    check('real_analysis_'+mode+'_exit',child.returncode==expected)
    state=copy.deepcopy(st);state['stages'][1]['returncode']=child.returncode
    if mode!='before_pending':
        pending=json.loads((data/'analysis/PENDING.json').read_text())
        check('real_analysis_'+mode+'_valid_candidate_never_confirms',pending['candidate_confirmation_valid'] and not pending['confirmation_valid'])
    check('real_analysis_'+mode+'_no_final_credential',not publication.finalize(data,mh,cfg,state) and not (data/'analysis/FINAL_ACCEPTED.json').exists())
    injections.append(dict(case=mode,returncode=child.returncode,accepted=False))
child=subprocess.run([sys.executable,'-B','-c','import os,signal;os.kill(os.getpid(),signal.SIGKILL)'],timeout=5)
check('real_runner_sigkill_exit',child.returncode==-9)
data=clone('real_runner_sigkill',True)
state=copy.deepcopy(st);state['stages'][0]['returncode']=child.returncode
(data/'run_exit.json').write_text(json.dumps(state['stages'][0]))
check('real_runner_sigkill_candidate_rejected',not inspect(data)['valid'])
check('real_runner_sigkill_no_credential',not publication.finalize(data,mh,cfg,state))
# 报告/候选缺失、候选分析CPU超额及提交后证据变化。
for label,mutate in [
 ('missing_report',lambda d:(d/'analysis/endpoints.csv').unlink()),
 ('missing_pending',lambda d:(d/'analysis/PENDING.json').unlink()),
]:
    data=clone(label,True);mutate(data);check(label,not publication.finalize(data,mh,cfg,copy.deepcopy(st)))
data=clone('analysis_cpu_over_budget',True);p=data/'analysis/PENDING.json';obj=json.loads(p.read_text());obj['usage']['cpu_seconds']=51.;p.write_text(json.dumps(obj))
check('analysis_cpu_over_budget',not publication.finalize(data,mh,cfg,copy.deepcopy(st)))
base=PRIOR/'control_normal/formal';p=base/'analysis/endpoints.csv';original_bytes=p.read_bytes();p.write_bytes(original_bytes+b'tampered\n')
try:publication.verify_accepted(base,mh,cfg)
except AssertionError:checks.append('post_commit_tamper_rejected')
else:raise AssertionError('post_commit_tamper_rejected')
finally:p.write_bytes(original_bytes)
check('positive_credential_restored',publication.verify_accepted(base,mh,cfg)['confirmation_valid'])
# 原59项在冻结必需模块扩展后再跑一次；与原测试标签逐项一致。
original=json.loads((ROOT/'y3b/synthetic_v4_tests.json').read_text())
last=json.loads((OUT/'synthetic_final_tests.json').read_text())
check('final_original_59',last['passed'] and last['checks']==original['checks'] and len(last['checks'])==59)
check('never_created_formal_directory',not (OUT/'formal').exists() and not (ROOT/'y3b/formal').exists())
children=resource.getrusage(resource.RUSAGE_CHILDREN)
dump(TEST/'results.json',dict(passed=True,checks=checks,injections=injections,usage=usage(),no_formal_sampling=True,
    no_random_training_or_queries=True,source_sha256={p.name:sha(p) for p in OUT.glob('*.py')},
    cpu_seconds_children=children.ru_utime+children.ru_stime))
print(json.dumps(dict(passed=True,checks=len(checks),usage=usage())))
