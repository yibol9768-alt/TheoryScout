"""R1/R2确定性反向测试；复用审计NPZ，禁止任何MR抽样。"""
import ast
import contextlib
import copy
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
# 父测试进程会累计运行很多次分析；正式子进程仍执行原45/50限额。
setrlimit = resource.setrlimit
def import_limit(which, limits):
    if which != resource.RLIMIT_CPU: setrlimit(which,limits)
resource.setrlimit = import_limit
try:
    import run
    import analyze
finally:
    resource.setrlimit = setrlimit
import common
import integrity
import publication
from common import ROOT, OUT, dump, sha, usage
# 测试父进程累计执行多次分析；不改变真实分析子进程45/50秒限额。
resource.setrlimit(resource.RLIMIT_CPU,(70,75))

common.environment()
TEST=ROOT/'audit_y3c_checks'/'protocol_replay'
TEST.mkdir()
checks=[];outcomes=[]
def check(label,condition):
    assert condition,label
    checks.append(label)
def forbidden(*a,**kw): raise AssertionError('禁止MR正式流/训练/查询抽样')
run.rng=forbidden;run.cell=forbidden
for k in ['default_rng','Generator','PCG64']: setattr(analyze.np.random,k,forbidden)
cfg=common.config();mh=sha(ROOT/'Y3B_FROZEN.sha256')
plan=list(common.jobs(cfg['blocks']))
source=ROOT/'audit_y3b_checks/full_plan_run/target_fixture'
base_hashes={p.name:sha(p) for p in source.glob('*.npz')}
U=dict(cpu_seconds=1.,peak_rss_mib=100.,monotonic=1.)
run_ok=dict(program='run.py',returncode=0,wall_hard_killed=False)
analysis_ok=dict(program='analyze.py',returncode=0,wall_hard_killed=False)
def fixture(label, count=288, reason='planned_complete', active=None):
    case=TEST/label;data=case/'formal';data.mkdir(parents=True)
    selected=plan[:count]
    for d,ir,s in selected:
        name=f'd{d}_r{ir}_s{s}.npz';shutil.copyfile(source/name,data/name)
    dump(data/'launch.json',dict(schema=1,manifest_sha256=mh,cpu_hard_split=[21340,50,10],
        preparation_cpu_reserve=200,wall_hard_total=12600,usage=U))
    dump(data/'start.json',dict(schema=1,manifest_sha256=mh,streams_sha256=sha(OUT/'streams.json'),
        program_sha256=sha(OUT/'run.py'),cfg=cfg,usage=U,no_draw_before_this_record=True))
    events=[]
    for k in selected:
        name=f'd{k[0]}_r{k[1]}_s{k[2]}.npz'
        events.extend([dict(schema=1,state='started',cell=list(k),usage=U),
                       dict(schema=1,state='complete',cell=list(k),sha256=base_hashes[name],usage=U)])
    if active is not None: events.append(dict(schema=1,state='started',cell=list(active),usage=U))
    events.append(dict(schema=1,state='stopped',reason=reason,error=None,incomplete_cell=active,usage=U))
    write_events(data,events)
    dump(data/'runner_status.json',dict(schema=1,reason=reason,error=None,completed=[list(k) for k in selected],
        incomplete_cell=active,usage=U,wall_seconds=100.))
    dump(data/'run_exit.json',run_ok)
    return data

def write_events(data,events):
    (data/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
def change(data,name,fn):
    p=data/name;o=json.loads(p.read_text());fn(o);p.write_text(json.dumps(o))
def get_events(data):return [json.loads(s) for s in (data/'events.jsonl').read_text().splitlines()]
def audit(data):
    hashes={p.name:sha(p) for p in data.glob('*.npz')}
    keys={tuple(map(int,p.stem.replace('d','').replace('r','').replace('s','').split('_'))) for p in data.glob('*.npz')}
    return integrity.inspect(data,mh,cfg,keys,hashes)
def analyze_case(data):
    analyze.OUT=data.parent;analyze.verify=lambda:mh
    os.environ['Y3C_LAUNCHER']='1';sys.argv=['analyze.py','--formal-via-launcher']
    with (data.parent/'analyze.stdout').open('x') as log,contextlib.redirect_stdout(log):analyze.main()
    result=json.loads((data/'analysis/PENDING.json').read_text())
    check(data.parent.name+'_pending_never_confirms',result['confirmation_valid'] is False and not result['primary_both_passed'] and not result['d512_passed'])
    check(data.parent.name+'_no_final_before_supervisor',not (data/'analysis/FINAL_ACCEPTED.json').exists())
    with (data/'analysis/all_completed.csv').open() as f: n=len(list(csv.DictReader(f)))
    check(data.parent.name+'_readable_disclosed',n==10*len(list(data.glob('*.npz'))))
    return result

def status(run_state=None, analysis_state=None):
    return dict(stages=[copy.deepcopy(run_state or run_ok),copy.deepcopy(analysis_state or analysis_ok)],wall_seconds=101.,
        cpu_seconds_children=3.,supervisor_usage=dict(cpu_seconds=1.,peak_rss_mib=100.,monotonic=2.))

def commit(data,expected,st=None):
    good=publication.finalize(data,mh,cfg,st or status())
    check(data.parent.name+'_publication',good is expected)
    check(data.parent.name+'_credential_presence',(data/'analysis/FINAL_ACCEPTED.json').exists() is expected)
    if good:check(data.parent.name+'_credential_verifies',publication.verify_accepted(data,mh,cfg)['confirmation_valid'])
    outcomes.append(dict(case=data.parent.name,accepted=good,
        rejection=json.loads((data/'launch_status.json').read_text())['publication_errors']))

# 保持原统计与生成实现逐函数AST相同，配置、预测、种子、规划逐字节相同。
for name, functions in [('analyze.py',['interval','equivalent','loss','analysis','read_cell']),
                        ('run.py',['soft_stop','checkpoint','rng','dataset','direction','ranks','cell','commit']),
                        ('common.py',['entropy','jobs'])]:
    def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    before=nodes(ROOT/'y3b'/name);after=nodes(OUT/name)
    for fn in functions:check('unchanged_'+name+'_'+fn,before[fn]==after[fn])
for name in ['config.json','parameters.json','predictions.csv','streams.json','seed_audit.json',
             'rng_inventory.json','power_summary.json','primary_power.json','planning_covariances.json','power_grid.csv','environment.json']:
    check('byte_identical_'+name,sha(ROOT/'y3b'/name)==sha(OUT/name))
check('prereg_byte_identical',sha(ROOT/'Y3B_PREREG.md')==sha(ROOT/'Y3C_PREREG.md'))
original=json.loads((ROOT/'y3b/synthetic_v4_tests.json').read_text())
rerun=json.loads((OUT/'synthetic_y3c_tests.json').read_text())
check('original_59_checks_all_rerun',rerun['passed'] and rerun['checks']==original['checks'] and len(rerun['checks'])==59)

# 原审计的全部13个持久化运行夹具原样重放，不补记录；必须拒绝并完整披露。
raws=sorted((ROOT/'audit_y3b_checks').glob('state_*/formal'))+sorted((ROOT/'audit_y3b_checks/full_plan_run').glob('state_*/formal'))+[ROOT/'audit_y3b_checks/analysis_killed_actual/formal']
check('all_13_original_audit_fixtures_enumerated',len(raws)==13)
for i,src in enumerate(raws):
    data=TEST/f'audit_{i}_{src.parent.name}'/'formal';data.mkdir(parents=True)
    for p in src.iterdir():
        if p.is_file() and (p.suffix=='.npz' or p.name in publication.INPUTS):shutil.copyfile(p,data/p.name)
    result=analyze_case(data)
    check(data.parent.name+'_rejected',not result['candidate_confirmation_valid'] and bool(result['integrity_errors']))
    st=status(json.loads((data/'run_exit.json').read_text()))
    if (src/'launch_status.json').exists():st['stages']=json.loads((src/'launch_status.json').read_text())['stages']
    commit(data,False,st)

# 新schema合法、288格、三确认全部通过的阳性对照。原审计NPZ原样复用。
base=fixture('control_normal');result=analyze_case(base)
check('positive_candidate_valid',result['candidate_confirmation_valid'])
check('positive_all_three_pass',result['candidate_decisions']['primary_both_passed'] and result['candidate_decisions']['d512_passed'])
commit(base,True)
try:publication.finalize(base,mh,cfg,status())
except FileExistsError:checks.append('credential_cannot_overwrite')
else:raise AssertionError('credential_cannot_overwrite')

# R2逐项隔离故障；四个关键反例也走完整分析器，验证披露未丢。
mutations={
 'missing_events':lambda d:(d/'events.jsonl').unlink(),
 'empty_events':lambda d:(d/'events.jsonl').write_text(''),
 'missing_start':lambda d:(d/'start.json').unlink(),
 'missing_launch':lambda d:(d/'launch.json').unlink(),
 'missing_status':lambda d:(d/'runner_status.json').unlink(),
 'missing_exit':lambda d:(d/'run_exit.json').unlink(),
 'start_manifest_mismatch':lambda d:change(d,'start.json',lambda x:x.update(manifest_sha256='wrong')),
 'start_streams_mismatch':lambda d:change(d,'start.json',lambda x:x.update(streams_sha256='0'*64)),
 'start_program_mismatch':lambda d:change(d,'start.json',lambda x:x.update(program_sha256='0'*64)),
 'start_cfg_mismatch':lambda d:change(d,'start.json',lambda x:x.update(cfg={})),
 'runner_error_with_normal_reason':lambda d:change(d,'runner_status.json',lambda x:x.update(error='exception')),
 'runner_completed_mismatch':lambda d:change(d,'runner_status.json',lambda x:x.update(completed=[])),
 'runner_incomplete_mismatch':lambda d:change(d,'runner_status.json',lambda x:x.update(incomplete_cell=[128,0,0])),
 'runner_reason_mismatch':lambda d:change(d,'runner_status.json',lambda x:x.update(reason='wall_soft')),
 'runner_cpu_over_budget':lambda d:change(d,'runner_status.json',lambda x:x['usage'].update(cpu_seconds=21341)),
 'runner_wall_over_budget':lambda d:change(d,'runner_status.json',lambda x:x.update(wall_seconds=12526)),
 'runner_nonzero':lambda d:change(d,'run_exit.json',lambda x:x.update(returncode=7)),
 'runner_sigkill':lambda d:change(d,'run_exit.json',lambda x:x.update(returncode=-9,wall_hard_killed=True)),
 'runner_hard_flag_zero_code':lambda d:change(d,'run_exit.json',lambda x:x.update(wall_hard_killed=True)),
 'runner_bool_returncode':lambda d:change(d,'run_exit.json',lambda x:x.update(returncode=False)),
}
event_mutations={
 'legal_json_illegal_event':lambda e:e.insert(0,{'state':'bogus'}),
 'legal_json_array':lambda e:e.insert(0,[]),
 'event_missing_schema':lambda e:e[0].pop('schema'),
 'event_unknown_field':lambda e:e[0].update(extra=1),
 'event_noninteger_key':lambda e:e[0].update(cell=[128.,0,0]),
 'event_outside_plan':lambda e:e[0].update(cell=[16,0,0]),
 'event_wrong_order':lambda e:e[0].update(cell=[256,0,0]),
 'duplicate_started':lambda e:e.insert(1,copy.deepcopy(e[0])),
 'duplicate_complete':lambda e:e.insert(2,copy.deepcopy(e[1])),
 'missing_started':lambda e:e.pop(0),
 'missing_complete':lambda e:e.pop(1),
 'missing_stopped':lambda e:e.pop(),
 'duplicate_stopped':lambda e:e.append(copy.deepcopy(e[-1])),
 'event_after_stop':lambda e:e.append(copy.deepcopy(e[0])),
 'commit_hash_mismatch':lambda e:e[1].update(sha256='0'*64),
 'malformed_hash':lambda e:e[1].update(sha256='bad'),
 'event_error_inconsistent':lambda e:e[-1].update(error='failed'),
 'event_status_mismatch':lambda e:e[-1].update(reason='cpu_soft'),
 'event_incomplete_mismatch':lambda e:e[-1].update(incomplete_cell=[128,0,0]),
 'event_usage_backwards':lambda e:e[1]['usage'].update(cpu_seconds=0),
}
for label,mutate in event_mutations.items():
    def operation(d,fn=mutate):
        events=get_events(d);fn(events);write_events(d,events)
    mutations[label]=operation
mutations['malformed_json']=lambda d:(d/'events.jsonl').write_text('{bad}\n')
mutations['duplicate_json_key']=lambda d:(d/'events.jsonl').write_text('{"schema":1,"state":"started","state":"stopped"}\n')
mutations['json_nan']=lambda d:(d/'events.jsonl').write_text('{"schema":NaN}\n')
for label,mutate in mutations.items():
    data=fixture(label);mutate(data);result=audit(data)
    check(label+'_integrity_invalid',not result['valid'] and bool(result['errors']))
    dump(data.parent/'integrity_test.json',dict(errors=result['errors']))
    if label in ['missing_events','legal_json_illegal_event','commit_hash_mismatch','event_status_mismatch']:
        pending=analyze_case(data)
        check(label+'_candidate_rejected',not pending['candidate_confirmation_valid'])
        commit(data,False)

# 合法预算软停（含最后一格尚未提交）、空运行；确认资格与统计显著性分开。
for label,count,active in [('soft_stop_partial',215,list(plan[215])),('soft_stop_empty',0,None)]:
    data=fixture(label,count,'cpu_soft',active)
    result=analyze_case(data)
    check(label+'_valid_execution',result['candidate_confirmation_valid'])
    check(label+'_statistics_ineligible',not result['candidate_decisions']['primary_both_passed'] and not result['candidate_decisions']['d512_passed'])
    commit(data,True)

# 监管器独立否决：候选已写好仍不能掩盖分析退出、硬杀和预算问题。
for label,update in [
 ('analysis_nonzero',lambda s:s['stages'][1].update(returncode=7)),
 ('analysis_hard_flag',lambda s:s['stages'][1].update(wall_hard_killed=True)),
 ('analysis_negative_return',lambda s:s['stages'][1].update(returncode=-9)),
 ('supervisor_over_budget',lambda s:s['supervisor_usage'].update(cpu_seconds=10)),
 ('total_cpu_over_budget',lambda s:s.update(cpu_seconds_children=21400)),
 ('total_wall_over_budget',lambda s:s.update(wall_seconds=12601)),
 ('exit_evidence_disagrees',lambda s:s['stages'][0].update(returncode=1)),
]:
    data=fixture(label);shutil.copytree(base/'analysis',data/'analysis',ignore=shutil.ignore_patterns('FINAL_ACCEPTED.json'))
    st=status();update(st);commit(data,False,st)

# 真实子进程：在分析器main写完PENDING后立即SIGKILL/非零退出，或在写PENDING前杀死。
worker='''
import os,sys,signal
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import analyze
analyze.OUT=Path(sys.argv[2]);analyze.verify=lambda:sys.argv_manifest
sys.argv_manifest=sys.argv[3];mode=sys.argv[4]
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
for mode,expected in [('after_pending',-9),('before_pending',-9),('nonzero',7)]:
    data=fixture('actual_analysis_'+mode)
    with (data.parent/'child.stdout').open('x') as log:
        child=subprocess.run([sys.executable,'-B','-c',worker,str(OUT),str(data.parent),mh,mode],stdout=log,stderr=subprocess.STDOUT,timeout=30)
    check('actual_analysis_'+mode+'_returncode',child.returncode==expected)
    st=status(analysis_state=dict(program='analyze.py',returncode=child.returncode,wall_hard_killed=False))
    if mode!='before_pending':
        pending=json.loads((data/'analysis/PENDING.json').read_text())
        check(mode+'_candidate_valid_but_never_confirms',pending['candidate_confirmation_valid'] and not pending['confirmation_valid'])
    commit(data,False,st)

child=subprocess.run([sys.executable,'-B','-c','import os,signal;os.kill(os.getpid(),signal.SIGKILL)'],timeout=5)
check('actual_runner_SIGKILL',child.returncode==-9)
data=fixture('actual_runner_kill_after_status')
change(data,'run_exit.json',lambda s:s.update(returncode=child.returncode))
result=analyze_case(data);check('actual_runner_kill_rejects_candidate',not result['candidate_confirmation_valid'])
commit(data,False,status(json.loads((data/'run_exit.json').read_text())))

# 分析后证据变化、缺报告、缺候选与提交后篡改不能采用。
for label,mutate in [
 ('changed_after_analysis',lambda d:change(d,'runner_status.json',lambda x:x.update(error='late failure'))),
 ('missing_report',lambda d:(d/'analysis/endpoints.csv').unlink()),
 ('missing_pending',lambda d:(d/'analysis/PENDING.json').unlink()),
 ('analysis_cpu_over_budget',lambda d:change(d,'analysis/PENDING.json',lambda x:x['usage'].update(cpu_seconds=51))),
]:
    data=fixture(label);shutil.copytree(base/'analysis',data/'analysis',ignore=shutil.ignore_patterns('FINAL_ACCEPTED.json'))
    mutate(data);commit(data,False)
original_bytes=(base/'analysis/endpoints.csv').read_bytes()
with (base/'analysis/endpoints.csv').open('ab') as f:f.write(b'tampered\n')
try:publication.verify_accepted(base,mh,cfg)
except AssertionError:checks.append('post_commit_tamper_rejected')
else:raise AssertionError('post_commit_tamper_rejected')
# 恢复的是本测试自己创建的文件，阳性凭据保留可复验。
(base/'analysis/endpoints.csv').write_bytes(original_bytes)
check('positive_credential_restored',publication.verify_accepted(base,mh,cfg)['confirmation_valid'])
check('formal_directory_never_created',not (OUT/'formal').exists() and not (ROOT/'y3b/formal').exists())
children=resource.getrusage(resource.RUSAGE_CHILDREN)
dump(TEST/'results.json',dict(passed=True,checks=checks,outcomes=outcomes,no_formal_sampling=True,
    no_random_training_or_queries=True,audit_fixtures=len(raws),usage=usage(),
    cpu_seconds_children=children.ru_utime+children.ru_stime))
print(json.dumps(dict(passed=True,checks=len(checks),outcomes=len(outcomes),usage=usage(),cpu_seconds_children=children.ru_utime+children.ru_stime)))
