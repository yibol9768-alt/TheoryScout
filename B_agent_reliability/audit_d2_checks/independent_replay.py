"""从 C 原始轨迹独立解析、重算整数、构建提示，再与 D Task 对比。"""
import bisect
import json
from collections import Counter
from common import ROOT, dump_file, install_guards, resource_info

install_guards()
import sys
sys.path.insert(0,str(ROOT/'HOLDOUT_D_runner'))
from protocol import Task

SYSTEM = ('You execute verifiable integer transactions. Return JSON only with exactly '
          '{"writes":{key:integer,...},"detect":boolean}. No extra fields, duplicate keys, or prose. '
          'All arithmetic uses Python modulo: x%997 is in 0..996 even for negative x. '
          'Assignments are sequential, including aliases: each RHS reads the latest state. '
          'Writes must contain exactly required_writes keys, even unchanged ones. '
          'detect means you believe the previous applied action was wrong; false on the first step. '
          'Do only the current transaction. Do not call tools. '
          'State has r0..r5, derived cells F0,F1,F2. For each instruction in order: '
          'r[a]=(r[a]+m*r[b]-r[c]+d)%997; then r[c]=(r[c]+r[a])%997. '
          'After ALL instructions: F0=(r0+2*r1-r2)%997; F1=(3*F0+r3-r4)%997; F2=(F1+2*r5)%997. '
          'Output all six registers and all three formula cells.')
KEYS = ['r0','r1','r2','r3','r4','r5','F0','F1','F2']
compact = lambda z:json.dumps(z,ensure_ascii=False,separators=(',',':'))

def transition(state, operation):
    s=state.copy()
    for instruction in operation['instructions']:
        a,b,c = [f'r{instruction[key]}' for key in ('a','b','c')]
        s[a]=(s[a]+instruction['m']*s[b]-s[c]+instruction['d'])%997
        s[c]=(s[c]+s[a])%997
    s['F0']=(s['r0']+2*s['r1']-s['r2'])%997
    s['F1']=(3*s['F0']+s['r3']-s['r4'])%997
    s['F2']=(s['F1']+2*s['r5'])%997
    return s

def parse(content):
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj:raise ValueError('duplicate')
            obj[k]=v
        return obj
    try:
        a=json.loads(content,object_pairs_hook=unique)
    except (ValueError,TypeError):return None,'json'
    if not (type(a) is dict and set(a)=={'writes','detect'} and type(a['detect']) is bool
            and type(a['writes']) is dict and set(a['writes'])==set(KEYS)
            and all(type(v) is int and 0<=v<997 for v in a['writes'].values())):
        return None,'schema'
    return a,None

def reference_messages(task, step, review, state, start, last):
    p=dict(step=step,operation=task['operations'][step],required_writes=KEYS,checkpoint=state.copy() if review==0 else start.copy(),table=task['table'])
    messages=[dict(role='system',content=SYSTEM),dict(role='user',content=compact(p))]
    if review:
        feedback=dict(mode='validator_triggered_review',instruction='The applied transaction failed validation. Recompute and correct the required writes. State whether you detect a previous error.',
                      step=step,operation=task['operations'][step],required_writes=KEYS,applied_state=state.copy(),checkpoint=start.copy(),table=task['table'])
        messages.extend([dict(role='assistant',content=last or ''),dict(role='user',content=compact(feedback))])
    return messages

details=[];errors=[];kinds=Counter();review_counts=Counter();full64_cells=[0]*5;full64_missing=0
files=sorted((ROOT/'experiments/real/holdout_c_real').glob('sheet_d0_*.jsonl'))
for p in files:
    events=[json.loads(line) for line in p.read_text().splitlines()]
    saved=next(e['task'] for e in events if e['event']=='task')
    seed=saved['seed'];dt=Task(seed);state=saved['initial'].copy();gold=state.copy()
    oracle_prefix=[gold.copy()]
    for operation in saved['operations']:
        gold=transition(gold,operation);oracle_prefix.append(gold)
    assert oracle_prefix==saved['prefix'],p
    assert dt.t['operations'][:min(64,saved['length'])]==saved['operations'][:64],p
    assert dt.t['prefix'][:min(64,saved['length'])+1]==oracle_prefix[:65],p
    req={e['call']:e for e in events if e['event']=='request'}
    resp={e['call']:e for e in events if e['event']=='response'}
    step=review=0;start=None;last=None;fatal=False;local_counts=[[0]*5 for _ in range(2)];actions=0
    for old in events:
        if old['event']!='action':continue
        if step>=64:break
        call=old['call'];messages=reference_messages(saved,step,review,state,start,last)
        try:
            assert messages==req[call]['messages'],'independent_prompt_vs_C'
            body=json.loads(dt.body());assert body['messages']==messages,'D_prompt_vs_independent'
            assert set(body)=={'model','messages','max_completion_tokens'} and body['model']=='gpt-6-luna' and body['max_completion_tokens']==4096
            response=resp[call]['response'];choice=response['choices'][0]
            content=choice['message']['content'];before=state.copy();expected=oracle_prefix[step+1]
            action,error=parse(content)
            if action:state.update(action['writes'])
            n_after=sum(state.get(k)!=v for k,v in expected.items())+int(action is None)
            correct=n_after==0
            if not correct and error is None:error='numeric'
            if choice['finish_reason']=='length' and not correct:error='truncated_'+str(error)
            record=dict(step=step,review=review,action=action,content=content,before=before,after=state.copy(),expected=expected,
                        n_before=0 if review==0 else sum(before.get(k)!=v for k,v in expected.items()),n_after=n_after,correct=correct,error_type=error)
            actual=dt.apply(response)
            for key,value in record.items():
                assert actual[key]==value,('D_action',key)
                assert old[key]==value,('C_action',key)
            z=local_counts[int(step>=16)]
            if review==0:start=before.copy();z[0]+=1;z[2]+=int(not correct)
            if review==2:z[3]+=1;z[4]+=int(correct)
            kinds[str(error)]+=1;review_counts[review]+=1;actions+=1
            if correct:step+=1;review=0;start=None;last=None
            elif review==2:z[1]+=1;fatal=True
            else:review+=1;last=content
            assert dt.step==step and dt.review==review
            assert dt.state==state and dt.counts==local_counts
            assert dt.terminal==(fatal or step==64)
        except Exception as ex:
            errors.append(dict(file=p.name,call=call,error=repr(ex)));break
    endpoints={str(L):'K' if step>=L else ('F' if fatal else 'U') for L in (8,16,32,64)}
    if saved['length']==64:
        if fatal or step==64:full64_cells[bisect.bisect_right([8,16,32,64],step)]+=1
        else:full64_missing+=1
    details.append(dict(file=p.name,seed=seed,length=saved['length'],actions=actions,completed=step,fatal=fatal,endpoints=endpoints,counts=local_counts))

result=dict(resources=resource_info(),files=len(files),actions=sum(x['actions'] for x in details),errors=errors,
            error_types=dict(kinds),review_counts=dict(review_counts),full64_cells=full64_cells,
            full64_missing=full64_missing,details=details,
            note='短 C 题完成计划长度后停止；L 以外标 U，不把旧短题成功伪造成 D64 成功。C 的服务停止不作致命。')
dump_file('independent_replay.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False))
