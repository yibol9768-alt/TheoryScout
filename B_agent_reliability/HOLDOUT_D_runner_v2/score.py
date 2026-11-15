#!/usr/bin/env python3
"""只读离线评分：原始 WAL →确定性动作→resolved 机制分母→G0–G4。"""
import argparse
import base64
import bisect
import json
import math
from pathlib import Path
from protocol import ROOT, Task, dump, sha, guard, GuardError
from evidence import version, validate_smoke

DESIGN = json.loads((ROOT/'HOLDOUT_D_DESIGN.json').read_text())
SEEDS = DESIGN['seeds']

def read_wal(path):
    raw=Path(path).read_bytes()
    if not raw or not raw.endswith(b'\n'):raise ValueError('empty_or_torn_WAL')
    events=[];head='0'*64
    for line in raw.splitlines():
        e=json.loads(line);digest=e.pop('digest')
        if e['seq']!=len(events)+1 or e['prev']!=head or sha(dump(e).encode())!=digest:
            raise ValueError('WAL_chain_mismatch')
        e['digest']=digest
        if dump(e).encode()!=line:raise ValueError('noncanonical_WAL_bytes')
        events.append(e);head=digest
    return events

def cell(c, fatal):
    if type(c) is not int or not 0<=c<=64 or (fatal and c==64) or (not fatal and c!=64):
        raise ValueError('invalid_terminal_c')
    return bisect.bisect_right([8,16,32,64],c)

def resolved_counts(records):
    """初次错但尚未修复/致命的推进完全不进入最终 n,d,e,m2,s2。"""
    counts=[[0]*5 for _ in range(2)];pending={}
    for r in records:
        key=(r['seed'],r['step'])
        if r['review']==0:pending[key]=[]
        if key not in pending or r['review']!=len(pending[key]):raise ValueError('resolved_review_order')
        pending[key].append(r)
        if r['correct'] or r['review']==2:
            group=pending.pop(key);z=counts[int(r['step']>=16)]
            z[0]+=1;z[1]+=int(not r['correct']);z[2]+=int(not group[0]['correct'])
            z[3]+=int(r['review']==2);z[4]+=int(r['review']==2 and r['correct'])
    return counts

def rebuild(events):
    if events[0]['kind']!='init' or any(e['kind']=='init' for e in events[1:]):raise ValueError('init_order')
    if events[0].get('protocol_source_sha256')!=sha((ROOT/'experiments/real/reliability_core.py').read_bytes()):
        raise ValueError('WAL_protocol_source_mismatch')
    if events[0].get('runner_version')!=version():raise ValueError('WAL_runner_version_mismatch')
    tasks={s:Task(s) for s in SEEDS};requests={};pending={};expected={};actions=set();records=[]
    fixture=events[0].get('fixture') or {};attempts=fixture.get('attempts',fixture.get('success',0));success=fixture.get('success',0)
    halt=None;guard_fail=False;smoke=False;issues=[];pause_until=0.;quota_ok=False;quota_success=-1;last_quota=None
    live=events[0]['mode']=='live';launch=False
    for i,e in enumerate(events):
        k=e['kind']
        if k=='reserve':
            seed=e['seed'];t=tasks[seed];body=base64.b64decode(e['body'],validate=True)
            if t.terminal or body!=t.body() or sha(body)!=e['body_sha256'] or e['rid']!=t.rid or e['step']!=t.step or e['review']!=t.review:
                raise ValueError('WAL_prompt_or_position')
            if e['call']!=attempts+1 or e['attempt']!=1+sum(r['rid']==e['rid'] for r in requests.values()):raise ValueError('WAL_attempt_sequence')
            if any(r['rid']==e['rid'] for r in pending.values()):raise ValueError('duplicate_pending_request')
            if halt:issues.append('send_after_halt')
            if live:
                if seed not in SEEDS[:2] and not smoke:issues.append('send_without_human_smoke')
                if not launch:issues.append('send_without_launch_audit')
                if not quota_ok or quota_success//100!=success//100 or e['utc']<pause_until:issues.append('quota_or_service_barrier')
                if success%100+len(pending)>=100:issues.append('hundred_boundary_barrier')
            attempts+=1;requests[e['call']]=e;pending[e['call']]=e
            if attempts>12000 or success+len(pending)>9000 or len(pending)>2:raise ValueError('WAL_budget_or_concurrency')
        elif k=='response':
            req=pending.pop(e['call']);raw=base64.b64decode(e['raw'],validate=True)
            if sha(raw)!=e['raw_sha256']:raise ValueError('WAL_response_hash')
            if 200<=e['status']<300:
                success+=1
                try:
                    r=tasks[req['seed']].apply(guard(raw));expected[e['call']]=r;records.append(r)
                except GuardError:guard_fail=True
            elif e['status']==0 or e['status']==408 or 500<=e['status']<=599:
                pause_until=max(pause_until,e['finished']+300);quota_ok=False
            else:halt='other_http'
        elif k=='action':
            if e['call'] in actions or expected.get(e['call'])!=e['record']:raise ValueError('WAL_action_mismatch')
            actions.add(e['call'])
        elif k=='smoke_approved':
            validate_smoke(e['approval'],events[:i],version())
            if not live:raise ValueError('mock_human_approval_forbidden')
            smoke=True
        elif k=='halt':halt=e['reason']
        elif k=='quota':
            if pending:issues.append('quota_with_pending')
            if e['ok']:
                if not (type(e.get('five_hour')) in (int,float) and type(e.get('week')) in (int,float)
                        and 0<=e['five_hour']<.8 and 0<=e['week']<.9):raise ValueError('quota_values')
                if e['checked']<pause_until:issues.append('quota_before_wait')
                quota_ok=True;quota_success=success;last_quota=e
            else:quota_ok=False;pause_until=max(pause_until,e['checked']+300)
        elif k=='launch':
            z=e['environment']
            launch=(e['live'] is True and z.get('memory_max')=='2147483648' and z.get('swap_max')=='0'
                    and z.get('stdin')=='/dev/null' and z.get('tty') is False and z.get('sid')!=z.get('original_sid')
                    and all(z.get('threads',{}).get(v) in ('1','2') for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')))
            if not launch:issues.append('launch_contract')
        elif k=='recovery':quota_ok=False;launch=False
        elif k=='reconcile':
            if halt=='unknown_inflight':halt=None
        elif k not in ('init','pause','smoke_wait'):raise ValueError('unknown_WAL_event')
    if guard_fail:issues.append('guard_failed_return')
    if set(expected)!=actions:issues.append('response_without_durable_action')
    if pending:issues.append('unresolved_inflight')
    return tasks,records,dict(attempts=attempts,successful=success,pending=len(pending),halt=halt,
                              issues=sorted(set(issues)),live=live,launch=launch,smoke=smoke)

def mechanism(per_task):
    import numpy as np
    x=np.asarray(per_task,dtype=np.int64);tot=x.sum(axis=0);req=DESIGN['mechanism']['minimum_eligibility']
    base=dict(counts=tot.tolist(),eligible=False,passed=None,valid=0,intervals=None)
    if any(tot[:,3]<req['r2_attempts_each_segment']) or any(tot[:,1]<req['fatal_each_segment']):
        return dict(base,status='信息不足')
    rng=np.random.default_rng(DESIGN['future_mechanism']['seed']);diff=[]
    for _ in range(100):
        # 每批100个配对整题重采样，限制内存；同一索引用于前后段。
        sums=x[rng.integers(0,144,size=(100,144))].sum(axis=1)
        valid=np.all(sums[:,:,0]>0,axis=1)&np.all(sums[:,:,3]>0,axis=1)
        a=sums[valid];diff.extend(np.column_stack((a[:,0,1]/a[:,0,0]-a[:,1,1]/a[:,1,0],
                                                  a[:,1,4]/a[:,1,3]-a[:,0,4]/a[:,0,3])).tolist())
    base['valid']=len(diff)
    if len(diff)<req['bootstrap_valid']:return dict(base,status='信息不足')
    ci=np.quantile(diff,[.025,.975],axis=0,method='linear').T.tolist()
    passed=all(pair[0]>0 for pair in ci)
    return dict(base,eligible=True,passed=passed,intervals=ci,status='通过' if passed else '未达到机制门槛')

def primary_pass(T):
    return T>DESIGN['comparison']['cutoff']['critical']

def describe(tasks,records):
    counts=[0]*5
    for t in tasks.values():
        if t.terminal:counts[cell(t.step,t.status!='success')]+=1
    bounds={}
    for L in (8,16,32,64):
        K=sum(t.step>=L for t in tasks.values());F=sum(t.terminal and t.step<L for t in tasks.values());U=144-K-F
        pred=DESIGN['predictions'][str(L)];lo,hi=pred['count99_D']
        valid=[k for k in range(lo,hi+1) if abs(k/144-pred['S_D'])<=.10]
        bounds[str(L)]=dict(K=K,F=F,U=U,success_count_bound=[K,K+U],effective_acceptance=[min(valid),max(valid)],
                            robust_incompatible=(K>max(valid) or K+U<min(valid)))
    by_seed={s:[] for s in SEEDS}
    for r in records:by_seed[r['seed']].append(r)
    resolved=[resolved_counts(by_seed[s]) for s in SEEDS]
    censored={}
    for name in ('M_D','A'):
        total=0.
        for task_counts in resolved:
            for i,(n,d,*_) in enumerate(task_counts):
                model=DESIGN['models'][name];h=(model['E' if i==0 else 'T'] if name=='M_D' else model)['h']
                total+=(n-d)*math.log1p(-h)+d*math.log(h)
        censored[name]=total
    return dict(five_cells=counts,terminal=sum(counts),prefix_bounds=bounds,resolved_per_task=resolved,
                censored_log_likelihood_descriptive=censored)

def frozen_check():
    errors=[]
    for name in ('HOLDOUT_D_PREREG.sha256','HOLDOUT_D_AMEND1.sha256','HOLDOUT_D_AMEND2.sha256'):
        p=ROOT/name
        if not p.is_file():errors.append('missing:'+name);continue
        for line in p.read_text().splitlines():
            digest,rel=line.split('  ',1);q=(ROOT/rel).resolve()
            if not q.is_relative_to(ROOT) or not q.is_file() or sha(q.read_bytes())!=digest:errors.append('freeze:'+rel)
    return errors

def score(path):
    events=read_wal(path);tasks,records,engineering=rebuild(events);d=describe(tasks,records)
    complete=d['terminal']==144
    freeze_errors=frozen_check()
    G0=(not freeze_errors and not engineering['issues'] and engineering['live'] and engineering['launch']
        and engineering['smoke'] and not events[0].get('fixture') and events[0].get('runner_version')==version()
        and events[0].get('freeze_verified') is True)
    G1=complete and not engineering['pending'] and engineering['halt'] is None
    eligible=G0 and G1
    weights=DESIGN['comparison']['weights_log_D_over_A']
    # 仅完整144题提供描述 T；G0/G1 未过则不执行确认判决。
    T=sum(n*w for n,w in zip(d['five_cells'],weights)) if complete else None
    G2=all(not v['robust_incompatible'] for v in d['prefix_bounds'].values()) if eligible else None
    G3=primary_pass(T) if eligible else None
    mech=mechanism(d['resolved_per_task']) if eligible else dict(status='信息不足',passed=None,reason='G0/G1未通过')
    if eligible and G2 and G3 and mech['passed']:
        decision='在此反馈协议下，分段修复风险假说获得一次新批次支持'
    elif not eligible:decision='信息不足'
    elif G2 and G3:decision='预测相容且相对A获支持，分段修复机制证据未完成'
    else:decision='未满足全部确认门槛'
    return dict(**d,T=T,T_role='完整144题描述值；仅G0/G1通过才判决' if complete else '不完整，禁止完成子集评分',
                weights=weights,strict_cutoff=0.541,secondary_T_gt_log40=(T>math.log(40) if eligible else None),
                G0=G0,G1=G1,G2=G2,G3=G3,G4=mech,decision=decision,
                engineering=engineering,freeze_errors=freeze_errors,wal_head=events[-1]['digest'],
                tasks=[dict(seed=s,c=t.step,status=t.status,terminal=t.terminal) for s,t in tasks.items()])

def c_replay():
    details=[];counts=[0]*5;actions=0
    for path in sorted((ROOT/'experiments/real/holdout_c_real').glob('sheet_d0_*.jsonl')):
        events=[json.loads(line) for line in path.read_text().splitlines()]
        saved=next(e['task'] for e in events if e['event']=='task');t=Task(saved['seed'])
        assert t.t['prefix'][:saved['length']+1]==saved['prefix']
        requests={e['call']:e for e in events if e['event']=='request'}
        responses={e['call']:e for e in events if e['event']=='response'};records=[]
        for e in events:
            if e['event']!='action':continue
            assert json.loads(t.body())['messages']==requests[e['call']]['messages']
            r=t.apply(responses[e['call']]['response'])
            for key in ('before','after','expected','content','correct','error_type','n_after','n_before','step','review','action'):
                assert r[key]==e[key],(path,key)
            records.append(r);actions+=1
        if saved['length']==64 and t.terminal:counts[cell(t.step,t.status!='success')]+=1
        details.append(dict(file=path.name,length=saved['length'],c=t.step,terminal64=t.terminal,resolved=resolved_counts(records)))
    return dict(files=len(details),actions=actions,full64_cells=counts,details=details,
                confirmation=False,note='C重放仅核验协议；短题成功不充当D64终局，不改C原判。')

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False);ap.add_argument('wal',type=Path,nargs='?')
    ap.add_argument('--c-replay',action='store_true');ap.add_argument('--out',type=Path)
    z=ap.parse_args()
    if not z.c_replay and z.wal is None:ap.error('需要WAL或--c-replay')
    try:result=c_replay() if z.c_replay else score(z.wal)
    except (ValueError,KeyError,TypeError,IndexError,AssertionError) as ex:
        result=dict(G0=False,G1=False,decision='信息不足',integrity_error=str(ex));code=2
    else:code=0
    if z.out:
        p=z.out.resolve()
        if not p.is_relative_to(ROOT/'HOLDOUT_D_runner_v2'):raise ValueError('score_output_must_be_inside_v2')
        with p.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    else:print(json.dumps(result,ensure_ascii=False,indent=2))
    return code

if __name__=='__main__':raise SystemExit(main())
