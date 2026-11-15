import copy, math, time
from datetime import datetime,timezone
from unittest.mock import patch
from common import *
from independent_score import independent_wal,apply_ref,describe_ref,mechanism_ref,integer_step,compact,SEEDS,D
guards();sys.path.insert(0,str(RUNNER))
import score as target
from protocol import Task,MODEL
from evidence import version,smoke_snapshot,CHECKS
out=OUT/'score_checks_r2';out.mkdir()
res=resources();results=[]
# Cache one real read-only freeze verification; every compared WAL sees the same frozen tree.
freeze=target.frozen_check();assert not freeze,freeze
def compare(label,path):
    ref,_=independent_wal(path)
    with patch.object(target,'frozen_check',return_value=freeze):z=target.score(path)
    for k in ('five_cells','terminal','prefix_bounds','resolved_per_task','tasks','T','G1'):assert ref[k]==z[k],(label,k)
    if not z['G0'] or not ref['G1']:
        assert z['G2'] is None and z['G3'] is None and z['G4']['passed'] is None and z['decision']=='信息不足'
    else:
        g2=all(not b['robust_incompatible'] for b in ref['prefix_bounds'].values());g3=ref['T']>.541;g4=mechanism_ref(ref['resolved_per_task'])
        assert z['G2']==g2 and z['G3']==g3
        for k in ('eligible','valid','passed','status'):assert z['G4'][k]==g4[k],(label,k,z['G4'],g4)
        if g4['intervals']:
            assert max(abs(a-b) for aa,bb in zip(g4['intervals'],z['G4']['intervals']) for a,b in zip(aa,bb))<1e-15
        decision=('在此反馈协议下，分段修复风险假说获得一次新批次支持' if g2 and g3 and g4['passed'] else
                  '预测相容且相对A获支持，分段修复机制证据未完成' if g2 and g3 else '未满足全部确认门槛')
        assert z['decision']==decision
    path_out=out/(label+'_comparison.json')
    with path_out.open('x') as f:json.dump(dict(reference=ref,scorer=z),f,ensure_ascii=False,indent=2)
    results.append(dict(label=label,five_cells=z['five_cells'],T=z['T'],G0=z['G0'],G1=z['G1'],G2=z['G2'],G3=z['G3'],G4=z['G4'],decision=z['decision']))
    return z

# Independent comparison against all rerun boundary/fault WALs, including unresolved review.
base=OUT/'all_tests'
for name in ('v2_boundary_7','v2_boundary_8','v2_boundary_15','v2_boundary_16','v2_boundary_31','v2_boundary_32','v2_boundary_63','v2_boundary_64','v2_service_score','v2_budget_score','v2_full_score'):
    compare(name,base/name/'HOLDOUT_D_WAL.jsonl')

# New synthetic WALs: state machine is used only as a trace generator. The reference
# above reconstructs integer arithmetic and resolved counts without that machine.
def generate(name,cells,live=True,repairs=True):
    events=[];start=time.time()-7200;success=0;calls=0;attempt={};ts={s:Task(s) for s in SEEDS}
    def add(kind,**kw):
        e=dict(seq=len(events)+1,prev=events[-1]['digest'] if events else '0'*64,kind=kind,utc=start+len(events)*.01,**kw)
        e['digest']=digest(compact(e).encode());events.append(e);return e
    add('init',mode='live' if live else 'mock',fixture=None,freeze_verified=True,amend2_manifest_sha256=digest((ROOT/'HOLDOUT_D_AMEND2.sha256').read_bytes()),runner_version=version(),protocol_source_sha256=digest((ROOT/'experiments/real/reliability_core.py').read_bytes()))
    add('recovery',prefix_hash=events[-1]['digest'])
    if live:add('launch',live=True,environment=dict(memory_max='2147483648',swap_max='0',stdin='/dev/null',tty=False,sid=987654,original_sid=987653,threads={k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
    desired=[c for n,c in zip(cells,[7,15,31,63,64]) for _ in range(n)]
    assert len(desired)==144
    for idx,seed in enumerate(SEEDS):
        t=ts[seed];fatal_c=desired[idx]
        while not t.terminal:
            if success%100==0:add('quota',ok=True,five_hour=0.,week=0.,successful=success,checked=start+len(events)*.01)
            body=t.body();rid=t.rid;attempt[rid]=attempt.get(rid,0)+1;calls+=1
            add('reserve',call=calls,rid=rid,seed=seed,step=t.step,review=t.review,attempt=attempt[rid],body=base64.b64encode(body).decode(),body_sha256=digest(body))
            p=json.loads(json.loads(body)['messages'][1]['content']);gold=integer_step(p['checkpoint'],p['operation'])
            repair_here=repairs and ((idx<10 and t.step==0) or (fatal_c>=31 and t.step==16))
            wrong=t.step==fatal_c or (repair_here and t.review<2)
            content='AUDIT_SYNTHETIC_INVALID_JSON' if wrong else compact(dict(writes=gold,detect=t.review>0))
            response=dict(model=MODEL,usage=dict(prompt_tokens=100,completion_tokens=50,total_tokens=150),choices=[dict(message=dict(role='assistant',content=content),finish_reason='stop')])
            raw=compact(response).encode();success+=1
            add('response',call=calls,status=200,raw=base64.b64encode(raw).decode(),raw_sha256=digest(raw),reason='AUDIT_SYNTHETIC_NO_MODEL',finished=start+len(events)*.01)
            add('action',call=calls,record=t.apply(response),recovered=False)
        if idx==1 and live:
            snapshot=smoke_snapshot(events,version())
            approval=dict(schema='HOLDOUT_D_AMEND2_HUMAN_REVIEW_V1',approved=True,reviewer='FORGED_SYNTHETIC_AUDIT_INPUT_NOT_AUTHORIZATION',reviewed_at=datetime.fromtimestamp(start+len(events)*.01,timezone.utc).isoformat(),review_method='manual',checked_items={k:True for k in CHECKS},snapshot=snapshot)
            add('smoke_approved',approval=approval)
    assert calls<=9000
    p=out/(name+'_SYNTHETIC_NOT_REAL.jsonl')
    with p.open('x') as f:f.write(''.join(compact(e)+'\n' for e in events))
    return p

import base64
for name,cells,live,repairs in [('all_gates',[25,20,15,24,60],True,True),('G0_mock',[25,20,15,24,60],False,True),
                              ('G2_fail',[144,0,0,0,0],True,False),('G3_fail',[19,20,21,28,56],True,True),
                              ('G4_fail',[25,20,15,24,60],True,False)]:
    compare(name,generate(name,cells,live,repairs))

# Independent C reconstruction: use stored task initial state and raw response, not D Task.
c_details=[];counts=[0]*5;actions=0
for path in sorted((ROOT/'experiments/real/holdout_c_real').glob('sheet_d0_*.jsonl')):
    ev=[json.loads(x) for x in path.read_text().splitlines()];task=next(x['task'] for x in ev if x['event']=='task')
    req={x['call']:x for x in ev if x['event']=='request'};resp={x['call']:x for x in ev if x['event']=='response'};st=None
    for e in ev:
        if e['event']!='action':continue
        body=compact(dict(messages=req[e['call']]['messages'])).encode()
        st,rec=apply_ref(st,body,resp[e['call']]['response'],task['seed'])
        for k,v in rec.items():
            if k!='seed':assert e[k]==v,(path,k)
        actions+=1
    if task['length']==64:
        assert st['terminal'];c=st['c'];counts[0 if c<8 else 1 if c<16 else 2 if c<32 else 3 if c<64 else 4]+=1
    c_details.append(dict(file=path.name,length=task['length'],c=st['c'],terminal64=st['terminal'],resolved=st['counts']))
c=target.c_replay();assert c['actions']==actions and c['files']==len(c_details) and c['full64_cells']==counts
assert c['details']==c_details
save('c_independent.json',dict(resources=res,files=len(c_details),actions=actions,cells64=counts,details=c_details))
assert not target.primary_pass(.541) and target.primary_pass(math.nextafter(.541,math.inf))
save('score_results.json',dict(resources=res,freeze_errors=freeze,comparisons=results,C=dict(files=len(c_details),actions=actions,cells64=counts),
    note='All synthetic live flags/launch/reviewer data are fabricated fixtures solely to exercise G0-G4. They are not D data or approval. Freeze checks are one real check cached for identical read-only inputs.'))
print(json.dumps(dict(comparisons=len(results),C_files=len(c_details),C_actions=actions,C_cells=counts),ensure_ascii=False))
