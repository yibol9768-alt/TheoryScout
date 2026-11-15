"""Offline audit and pre-specified pilot estimands. Makes no network requests."""
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import beta
from pilot import ROOT, MODEL, Executor, Task, parse_action
from pilot2 import hashes

OUT=ROOT/'pilot2_real'
BOOT=10000

def cp(x,n):
    if not n: return None
    return [0.0 if x==0 else float(beta.ppf(.025,x,n-x+1)),
            1.0 if x==n else float(beta.ppf(.975,x+1,n-x))]

def rate(rows):
    # (family, seed, numerator, denominator). Strata preserve each executor.
    x=sum(a[2] for a in rows); n=sum(a[3] for a in rows)
    groups=defaultdict(list)
    for f,s,a,b in rows:
        if b: groups[f].append((a,b))
    rng=np.random.default_rng(260511)
    samples=[]
    for _ in range(BOOT):
        sx=sn=0
        for group in groups.values():
            for j in rng.integers(0,len(group),len(group)):
                sx+=group[j][0]; sn+=group[j][1]
        if sn: samples.append(sx/sn)
    task_n=sum(bool(a[3]) for a in rows)
    task_bad=sum(a[3]>0 and a[2]<a[3] for a in rows)
    task_any=sum(a[2]>0 for a in rows if a[3])
    return dict(x=x,n=n,estimate=x/n if n else None,cp95_iid_reference=cp(x,n),
                cluster_bootstrap95=np.quantile(samples,[.025,.975]).tolist() if samples else None,
                bootstrap_degenerate=bool(samples and min(samples)==max(samples)),
                exposed_tasks=task_n,tasks_any_numerator=task_any,
                tasks_any_numerator_cp95=cp(task_any,task_n),
                tasks_with_any_complement=task_bad,
                tasks_any_complement_cp95=cp(task_bad,task_n))

def paired(rows):
    if not rows: return dict(tasks=0,estimate=None)
    groups=defaultdict(list)
    for f,s,d in rows: groups[f].append(d)
    vals=[v for vs in groups.values() for v in vs]
    rng=np.random.default_rng(260511)
    bs=[]
    for _ in range(BOOT):
        draw=[]
        for group in groups.values(): draw.extend(rng.choice(group,len(group)))
        bs.append(float(np.mean(draw)))
    d=float(np.mean(vals)); h=math.sqrt(2*math.log(40)/len(vals))
    return dict(tasks=len(vals),values=rows,estimate=d,
                cluster_bootstrap95=np.quantile(bs,[.025,.975]).tolist(),
                bootstrap_degenerate=min(bs)==max(bs),
                hoeffding95=[max(-1,d-h),min(1,d+h)])

def main():
    summary_path=ROOT/'FINAL_SUMMARY_PILOT2.json'
    if not summary_path.exists(): summary_path=ROOT/'SUMMARY_PILOT2.json'
    if not summary_path.exists(): summary_path=OUT/'summary.json'
    summary=json.loads(summary_path.read_text())
    assert 'finished_utc' in summary, 'batch still running'
    from datetime import datetime
    mock=json.loads((ROOT/'pilot2_mock/summary.json').read_text())
    assert datetime.fromisoformat(mock['finished_utc']) < datetime.fromisoformat(summary['started_utc'])
    assert mock['calls']==408 and len(mock['results'])==24 and all(r['success'] for r in mock['results'])
    if 'continuation' in summary:
        from resume_pilot2 import extra_hashes
        ext=json.loads((ROOT/'FREEZE_RESUME_PILOT2.json').read_text())
        continuation=summary['continuation']
        base=ROOT/'pilot2_real/summary.json'
        assert hashlib.sha256(base.read_bytes()).hexdigest()==continuation['base_summary_sha256']
        assert extra_hashes()==ext['sha256']==continuation['continuation_sha256']
        assert hashlib.sha256((ROOT/'AMENDMENT_PILOT2.md').read_bytes()).hexdigest()==ext['amendment_sha256']==continuation['amendment_sha256']
        cm=ROOT/'pilot2_resume_mock/summary.json'
        assert hashlib.sha256(cm.read_bytes()).hexdigest()==ext['mock_summary_sha256']
        assert datetime.fromisoformat(ext['frozen_utc']) < datetime.fromisoformat(continuation['started_utc'])
        assert datetime.fromisoformat(json.loads(cm.read_text())['finished_utc']) < datetime.fromisoformat(continuation['started_utc'])
        assert continuation['prior_calls']+continuation['calls']==json.loads((ROOT/'SUMMARY_PILOT2.json').read_text())['calls']
    if 'final_continuation' in summary:
        from finish_pilot2 import source_hashes
        ff=json.loads((ROOT/'FREEZE_FINISH_PILOT2.json').read_text())
        fc=summary['final_continuation']
        assert source_hashes()==ff['sha256']==fc['source_sha256']
        assert hashlib.sha256((ROOT/'AMENDMENT2_PILOT2.md').read_bytes()).hexdigest()==ff['amendment_sha256']==fc['amendment_sha256']
        assert hashlib.sha256((ROOT/'SUMMARY_PILOT2.json').read_bytes()).hexdigest()==ff['base_summary_sha256']==fc['base_summary_sha256']
        fm=ROOT/'pilot2_finish_mock/summary.json'
        assert hashlib.sha256(fm.read_bytes()).hexdigest()==ff['mock_summary_sha256']
        assert datetime.fromisoformat(json.loads(fm.read_text())['finished_utc']) < datetime.fromisoformat(ff['frozen_utc']) < datetime.fromisoformat(fc['started_utc'])
        assert summary['calls']==fc['prior_calls']+fc['calls']
    for planned, observed in zip(summary['schedule'],summary['results']):
        assert all(planned[k]==observed[k] for k in planned)
    freeze=json.loads((ROOT/'FREEZE_PILOT2.json').read_text())
    assert hashes()==freeze['sha256']==summary['frozen_sha256']
    assert datetime.fromisoformat(freeze['frozen_utc']) < datetime.fromisoformat(summary['started_utc'])
    assert hashlib.sha256((ROOT/'PREREG_PILOT2.md').read_bytes()).hexdigest()==summary['prereg_sha256']
    records=[]; usage=Counter(); reasoning=0; models=Counter(); checks=0
    counts=Counter(); examples=[]
    for r in summary['results']:
        events=[json.loads(line) for line in (ROOT/r['trace']).read_text().splitlines()]
        req=[e for e in events if e['event']=='request']; res=[e for e in events if e['event']=='response']
        ts=[e for e in events if e['event']=='transition']
        assert len(req)==r['calls']<=20
        assert r['work']==len(ts)
        assert r['repair_events']==sum(t['mode']=='repair' for t in ts)
        assert r['injections']==sum(e['event']=='injection' for e in events)
        assert r['recoveries']==sum(e['event']=='commit' and e['repaired'] for e in events)
        assert r['progress']==sum(e['event']=='commit' for e in events)
        assert r['within_5L']==(r['work']<=5*r['L'])
        # Transport failures can leave one unanswered request; do not discard it.
        assert len(res)<=len(req)
        local=Counter()
        for a,b in zip(req,res):
            assert a['call']==b['call']
            assert a['model_requested']==MODEL and a['max_completion_tokens']==2048
            rr=b['response']; models[rr['model']]+=1
            assert rr['model']==MODEL
            u=rr['usage']
            assert u['total_tokens']==u['prompt_tokens']+u['completion_tokens']
            assert all(type(u[k]) is int and u[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens'))
            for k in ('prompt_tokens','completion_tokens','total_tokens'):
                usage[k]+=u[k]; local[k]+=u[k]
            reasoning+=(u.get('completion_tokens_details') or {}).get('reasoning_tokens',0)
            if r['status'] not in ('provider_token_limit','completion_length_limit'):
                assert u['prompt_tokens']<=2048 and u['completion_tokens']<=2048
            assert (u.get('completion_tokens_details') or {}).get('reasoning_tokens',0)<=u['completion_tokens']
            checks+=1
        assert all(local.get(k,0)==r['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens'))
        task=Task(r['family'],r['seed'],r['L']); ex=Executor(task,r['K'])
        assert events[0]['task']==task.spec() and events[0]['K']==r['K']
        last=None
        for e in events:
            if e['event']=='response': last=e
            if e['event']=='transition':
                assert e['before']==ex.snapshot()
                try: action=parse_action(last['response']['choices'][0]['message']['content'])
                except (KeyError,IndexError,TypeError,ValueError): action=None
                assert action==e['model_action']
                shadow=copy.deepcopy(ex)
                from pilot import Stop
                try:
                    v=shadow.apply(action); stopped=None
                except Stop as err:
                    v=False; stopped=str(err)
                assert v==e['model_action_valid'] and shadow.n()==e['model_action_n_after']
                assert stopped==e['model_action_stop']
                applied=copy.deepcopy(action)
                if e['injection']:
                    assert r['injected'] and e['step'] in (4,10) and e['mode']=='advance'
                    applied['value']+=7
                    assert e['injection']['original']==action and e['injection']['applied']==applied
                assert applied==e['action']
                assert ex.apply(applied)==e['valid']
                assert ex.snapshot()==e['after'] and ex.n()==e['n_after']
                if e['mode']=='repair': assert e['n_before']==Executor_n_before(e,task)
                checks+=1
            if e['event']=='commit':
                assert ex.n()==0 and e['checkpoint']==task.prefix[e['step']+1]
                ex.commit()
                checks+=1
        assert ex.final()==r['final']
        assert all(row['state']==task.prefix[i+1] for i,row in enumerate(r['final']['ledger']))
        assert events[-1]['event']=='final' and events[-1]['status']==r['status']
        assert (ROOT/r['trace']).stat().st_size==r['trace_bytes']
        # History reset is checked from actual requests, rather than assumed from source.
        for e in req:
            p=json.loads(e['messages'][1]['content'])
            assert p['legal_targets']==list(task.initial)
            if e['mode']=='advance': assert p['untrusted_history']==[]
        natural=[]; injected=[]; episode=None
        for t in ts:
            if t['mode']=='advance':
                episode='injected' if t['injection'] else 'natural'
            else:
                (injected if episode=='injected' else natural).append(t)
            bad=t['model_action_stop'] is not None or t['model_action_n_after']>0
            if bad:
                a=t['model_action']
                if t['model_action_stop']: kind=t['model_action_stop']
                elif a is None: kind='unparseable_json'
                elif not t['model_action_valid']: kind='invalid_schema'
                elif a['op']=='undo': kind='undo_incomplete'
                elif a['value']!=task.prefix[t['step']+1][a['target']]:
                    kind='wrong_value' if a['target']==task.operations[t['step']]['target'] else 'wrong_target_and_value'
                else: kind='incomplete_obligations'
                counts[(r['family'],t['mode'],kind)]+=1
                examples.append(dict(family=r['family'],seed=r['seed'],K=r['K'],injected=r['injected'],
                                     step=t['step']+1,mode=t['mode'],kind=kind,
                                     operation=task.operations[t['step']],
                                     checkpoint=task.prefix[t['step']],expected=task.prefix[t['step']+1],
                                     current=t['before']['state'],model_action=t['model_action'],
                                     actual_action=t['action'],model_action_n_after=t['model_action_n_after'],
                                     trace=r['trace']))
        records.append(dict(result=r,events=events,transitions=ts,natural_repairs=natural,injected_repairs=injected))
    assert sum(r['result']['calls'] for r in records)==summary['calls']<=500
    assert len(records)==len(summary['schedule']) or summary['halt']
    metrics={}
    for family in ('all','kv','sheet'):
        subset=[rec for rec in records if family=='all' or rec['result']['family']==family]
        m={}
        for name,selector,num in (
            ('natural_seed_clean',lambda rec:[t for t in rec['transitions'] if t['mode']=='advance'] if not rec['result']['injected'] else [],lambda t:int(t['model_action_stop'] is not None or t['model_action_n_after']>0)),
            ('seed_original_injected_branch',lambda rec:[t for t in rec['transitions'] if t['mode']=='advance'] if rec['result']['injected'] else [],lambda t:int(t['model_action_stop'] is not None or t['model_action_n_after']>0)),
            ('detect_prompt_injected',lambda rec:rec['injected_repairs'],lambda t:int(isinstance(t['model_action'],dict) and t['model_action'].get('detect') is True)),
            ('detect_prompt_natural',lambda rec:rec['natural_repairs'],lambda t:int(isinstance(t['model_action'],dict) and t['model_action'].get('detect') is True)),
            ('repair_zero_injected',lambda rec:rec['injected_repairs'],lambda t:int(t['model_action_valid'] and t['model_action_stop'] is None and t['model_action_n_after']==0)),
            ('repair_zero_natural',lambda rec:rec['natural_repairs'],lambda t:int(t['model_action_valid'] and t['model_action_stop'] is None and t['model_action_n_after']==0)),
            ('repair_down_injected',lambda rec:rec['injected_repairs'],lambda t:int(t['model_action_n_after']<t['n_before'])),
        ):
            rows=[]
            for rec in subset:
                ts=selector(rec); r=rec['result']
                rows.append((r['family'],r['seed'],sum(num(t) for t in ts),len(ts)))
            m[name]=rate(rows)
        # First prompted detection per observed error episode, separate from repeated attempts.
        first_rows=[]
        for rec in subset:
            seen=set(); ts=[]
            for t in rec['injected_repairs']:
                if t['step'] not in seen: ts.append(t); seen.add(t['step'])
            r=rec['result']; first_rows.append((r['family'],r['seed'],sum(t['model_action'].get('detect') is True if isinstance(t['model_action'],dict) else False for t in ts),len(ts)))
        m['detect_first_injected']=rate(first_rows)
        # Paired post-recovery next steps and jointly contaminated repair exposure.
        paired_post=[]; paired_exposure=[]; identical_post_prompts=post_prompt_pairs=0
        pairs=defaultdict(dict)
        for rec in subset:
            r=rec['result']; pairs[(r['family'],r['seed'])][r['injected']]=rec
        for (f,s),p in pairs.items():
            if set(p)!={False,True}: continue
            clean={t['step']:t for t in p[False]['transitions'] if t['mode']=='advance'}
            inj={t['step']:t for t in p[True]['transitions'] if t['mode']=='advance'}
            recovered={e['step'] for e in p[True]['events'] if e['event']=='commit' and e['repaired']}
            bad=lambda t:int(t['model_action_stop'] is not None or t['model_action_n_after']>0)
            ds=[bad(inj[i+1])-bad(clean[i+1]) for i in (4,10)
                if i in recovered and i+1 in inj and i+1 in clean]
            if ds: paired_post.append((f,s,float(np.mean(ds))))
            cr={e['state']['index']:e['messages'] for e in p[False]['events'] if e['event']=='request' and e['mode']=='advance'}
            ir={e['state']['index']:e['messages'] for e in p[True]['events'] if e['event']=='request' and e['mode']=='advance'}
            for i in (4,10):
                if i in recovered and i+1 in inj and i+1 in clean:
                    post_prompt_pairs+=1
                    identical_post_prompts+=cr[i+1]==ir[i+1]
            ds=[]
            for i in (4,10):
                repairs=[t for t in p[True]['injected_repairs'] if t['step']==i]
                if repairs and i in clean: ds.append(float(np.mean([bad(t) for t in repairs]))-bad(clean[i]))
            if ds: paired_exposure.append((f,s,float(np.mean(ds))))
        m['post_recovery_paired_difference']=paired(paired_post)
        m['post_recovery_prompt_pairs']=post_prompt_pairs
        m['identical_post_recovery_prompt_pairs']=identical_post_prompts
        m['joint_contamination_paired_difference']=paired(paired_exposure)
        ts=[t for rec in subset for t in rec['transitions']]
        m['all_raw_delta_counts']=dict(Counter(str(t['delta']) for t in ts))
        m['repair_delta_counts']=dict(Counter(str(t['delta']) for t in ts if t['mode']=='repair'))
        m['natural_error_self_report']=dict(n=sum(t['model_action_n_after']>0 for t in ts if t['mode']=='advance' and not t['injection']),
            x=sum(t['model_action_n_after']>0 and isinstance(t['model_action'],dict) and t['model_action'].get('detect') is True for t in ts if t['mode']=='advance' and not t['injection']))
        metrics[family]=m
    result=dict(source_freeze_ok=True,prereg_hash_ok=True,audit_checks=checks,
                calls=summary['calls'],responses=sum(models.values()),unanswered_requests=summary['calls']-sum(models.values()),models=dict(models),usage=dict(usage),reasoning_tokens=reasoning,
                successes=sum(rec['result']['success'] for rec in records),runs=len(records),
                actual_injections=sum(rec['result']['injections'] for rec in records),
                visited_n_after=dict(Counter(str(t['n_after']) for rec in records for t in rec['transitions'])),
                max_active_undo_slots=max(rec['result']['peak_undo_slots'] for rec in records),
                max_prompt_tokens=max(e['response']['usage']['prompt_tokens'] for rec in records for e in rec['events'] if e['event']=='response'),
                max_completion_tokens=max(e['response']['usage']['completion_tokens'] for rec in records for e in rec['events'] if e['event']=='response'),
                completion_length_responses=sum(e['response']['choices'][0]['finish_reason']=='length' for rec in records for e in rec['events'] if e['event']=='response'),
                recoveries=sum(rec['result']['recoveries'] for rec in records),
                action_error_types=[dict(family=f,mode=m,kind=k,count=c) for (f,m,k),c in counts.items()],
                examples=examples,metrics=metrics,
                note='Action CP intervals assume iid for reference only; cluster bootstrap may degenerate at zero events. Unvisited natural repair kernel is unidentified.')
    (ROOT/'ANALYSIS_PILOT2.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('audit_checks','calls','responses','models','usage','reasoning_tokens','successes','runs','actual_injections','recoveries','action_error_types')},ensure_ascii=False,indent=2))

def Executor_n_before(t,task):
    expected=task.prefix[t['step']+1]
    return sum(t['before']['state'][k]!=v for k,v in expected.items())+int(t['before']['invalid'])

if __name__=='__main__': main()
