"""调用原分析main的隔离状态反例；冻结校验/路径重定向，统计/装载/判决原样。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(40,21600))
import contextlib, io, json, os, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_y3b_checks'
sys.path.insert(0,str(ROOT/'y3b'))
import common, predict, analyze
common.environment()
cfg=common.config();pred,params=predict.load()
mh=hashlib.sha256((ROOT/'Y3B_FROZEN.sha256').read_bytes()).hexdigest()
source=OUT/'target_fixture';source.mkdir()
def dump(p,o):
    with p.open('x') as f:json.dump(o,f,indent=2)
for s in range(32):
    Q=8192
    aa={'ids':np.arange(Q,dtype=np.int64),'metadata':np.array(json.dumps(dict(d=512,ir=2,seed=s,manifest_sha256=mh,schema=1)))}
    for m in common.METHODS:
        for task in common.TASKS:
            count=round(Q*pred[512,2,m,task]['recall_first_order'])
            if m=='GD' and task=='old':count+=s%3-1
            ok=np.arange(Q)<count
            prefix=m+'_'+task+'_'
            aa[prefix+'ranks']=np.where(ok,1,65536).astype(np.int64)
            aa[prefix+'success']=ok
            aa[prefix+'D']=np.full(Q,pred[512,2,m,task]['mu'],dtype=np.float64)
            aa[prefix+'V']=np.ones(Q,dtype=np.float64)
    with (source/f'd512_r2_s{s}.npz').open('xb') as f:np.savez_compressed(f,**aa)

outcomes=[]
for label,exit_state,event_mode in [
    ('control_normal',dict(program='run.py',returncode=0,wall_hard_killed=False),'valid'),
    ('hard_kill',dict(program='run.py',returncode=-9,wall_hard_killed=True),'valid'),
    ('nonzero_exit',dict(program='run.py',returncode=1,wall_hard_killed=False),'valid'),
    ('missing_events',dict(program='run.py',returncode=0,wall_hard_killed=False),'absent'),
    ('commit_hash_mismatch',dict(program='run.py',returncode=0,wall_hard_killed=False),'wrong_hash'),
    ('analysis_bad_return',dict(program='run.py',returncode=0,wall_hard_killed=False),'valid'),
]:
    fixture=OUT/('state_'+label);data=fixture/'formal';data.mkdir(parents=True)
    for p in source.glob('*.npz'):os.link(p,data/p.name)
    dump(data/'run_exit.json',exit_state)
    dump(data/'runner_status.json',dict(reason='cpu_soft',error=None,completed=[[512,2,s] for s in range(32)],
        incomplete_cell=None,usage=dict(cpu_seconds=21101,peak_rss_mib=100,monotonic=0),wall_seconds=100))
    if event_mode!='absent':
        with (data/'events.jsonl').open('x') as f:
            for s in range(32):
                f.write(json.dumps(dict(state='started',cell=[512,2,s]))+'\n')
                h=hashlib.sha256((data/f'd512_r2_s{s}.npz').read_bytes()).hexdigest()
                f.write(json.dumps(dict(state='complete',cell=[512,2,s],sha256=('0'*64 if event_mode=='wrong_hash' else h)))+'\n')
            f.write(json.dumps(dict(state='stopped',reason='cpu_soft',error=None,incomplete_cell=None))+'\n')
    analyze.OUT=fixture;analyze.verify=lambda:mh;analyze.config=lambda:cfg;analyze.load=lambda:(pred,params)
    sys.argv=['analyze.py','--formal-via-launcher'];os.environ['Y3B_LAUNCHER']='1'
    with (fixture/'analysis_stdout.log').open('x') as log,contextlib.redirect_stdout(log):analyze.main()
    result=json.loads((data/'analysis/FINAL.json').read_text())
    if label=='analysis_bad_return':
        # 与真实监管器相同：记录分析退出，却不撤销已存在FINAL。
        dump(data/'launch_status.json',dict(stages=[exit_state,dict(program='analyze.py',returncode=-9,wall_hard_killed=True)],
             analysis_final_exists=True))
    outcomes.append(dict(case=label,runner_exit=exit_state,confirmation_valid=result['confirmation_valid'],
        d512_passed=result['d512_passed'],complete_cells=result['complete_cells'],
        event_parse_errors=result['event_parse_errors'],invalid_files=result['invalid_files']))
    assert result['confirmation_valid'] and result['d512_passed'],label
dump(OUT/'adversarial_outcomes.json',outcomes)
print(json.dumps(outcomes,ensure_ascii=False))
