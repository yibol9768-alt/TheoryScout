"""Additional OFFLINE audit and descriptive statistics; no network or model client."""
import collections
import hashlib
import json
from pathlib import Path
import random
from datetime import datetime
from scipy.stats import beta

ROOT=Path(__file__).resolve().parent

def ci(e,n):
    if not n:return '不可估计'
    lo=0 if e==0 else float(beta.ppf(.025,e,n-e+1))
    hi=1 if e==n else float(beta.ppf(.975,e+1,n-e))
    return f'[{lo:.4f}, {hi:.4f}]'

def main():
    out=ROOT/'difficulty_real'
    summary=json.loads((out/'summary.json').read_text())
    assert 'finished_utc' in summary, 'wait for completed batch'
    rows=[]
    timeline=[]
    checks=0
    for result in summary['results']:
        p=out/(result['id']+'.jsonl')
        events=[json.loads(x) for x in p.read_text().splitlines()]
        requests={x['call']:x for x in events if x['event']=='request'}
        by_logic=collections.defaultdict(list)
        for x in requests.values():
            by_logic[(x['step'],x['review'])].append(x)
            assert x['model']=='gpt-6-luna'
            timeline.append((datetime.fromisoformat(x['utc']),1,x['call']))
            checks+=1
        for x in events:
            if x['event'] in ('response','service_error'):
                timeline.append((datetime.fromisoformat(x['utc']),-1,x['call']))
            if x['event']=='response':
                response=x['response'];u=response['usage']
                assert response['model']=='gpt-6-luna'
                assert 0<=u['completion_tokens']<=4096 and 0<=u['prompt_tokens']<=16000
                assert u['prompt_tokens']+u['completion_tokens']==u['total_tokens']
                assert 0<=(u.get('completion_tokens_details') or {}).get('reasoning_tokens',0)<=u['completion_tokens']
                checks+=1
        for xs in by_logic.values():
            assert len(xs)<=2
            if len(xs)==2:
                xs.sort(key=lambda x:x['attempt'])
                assert [x['attempt'] for x in xs]==[0,1]
                assert xs[0]['messages']==xs[1]['messages']
                checks+=1
        task_spec=events[0]['task']
        aa=[x for x in events if x['event']=='action']
        advances=[x for x in aa if x['review']==0]
        repairs=[x for x in aa if x['review']>0]
        wrong=[x for x in advances if not x['correct']]
        # Prove each advance starts from canonical prefix, with no automatic overwrite.
        for a in advances:
            assert a['before']==task_spec['prefix'][a['step']]
            checks+=1
        for prev,nxt in zip(aa,aa[1:]):
            assert prev['after']==nxt['before']
            checks+=1
        first_reviews=[x for x in repairs if x['review']==1]
        second_reviews=[x for x in repairs if x['review']==2]
        after_first=[x for x in advances if wrong and x['step']>wrong[0]['step']]
        # Independently check memory prompts expose initial state once, no checkpoint.
        if result['family']=='memory':
            for req in requests.values():
                if req['review']==0:
                    pld=json.loads(req['messages'][-1]['content'])
                    assert 'checkpoint' not in pld and 'applied_state' not in pld
                    assert ('initial' in pld)==(req['step']==0)
                    assert req['messages'][1]['role']=='user'
                    assert json.loads(req['messages'][1]['content'])['initial']==task_spec['initial']
                    checks+=1
        rows.append(dict(id=result['id'],family=result['family'],level=result['level'],
                         n=len(advances),e=len(wrong),types=dict(collections.Counter(x['error_type'] for x in wrong)),
                         first_reviews=len(first_reviews),first_detect=sum(bool(x['action'] and x['action']['detect']) for x in first_reviews),
                         first_fixed=sum(x['correct'] for x in first_reviews),
                         second_reviews=len(second_reviews),second_fixed=sum(x['correct'] for x in second_reviews),
                         review_wrong=sum(not x['correct'] for x in repairs),
                         subsequent_steps=len(after_first),subsequent_errors=sum(not x['correct'] for x in after_first),
                         error_task=int(bool(wrong))))
    active=set(); peak=0
    for _,delta,call in sorted(timeline):
        if delta==1:
            assert call not in active
            active.add(call)
            peak=max(peak,len(active))
        else:
            assert call in active
            active.remove(call)
        checks+=1
    assert peak<=2 and not active
    budget=[json.loads(x) for x in (out/'budget.jsonl').read_text().splitlines()]
    quotas=[x for x in budget if x['event']=='quota']
    assert [x['calls'] for x in quotas]==[0]+list(range(100,summary['calls']+1,100))
    assert all(max(x['five_hour'],x['week'])<80 for x in quotas) or summary['halt']=='quota_80'
    checks+=len(quotas)
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['family'],r['level'])].append(r)
    table=[]
    lines=['','## 补充离线审计：复查、再错与整题不确定性','',
           '以下由独立离线 audit_difficulty.py 生成，不修改冻结运行器或预注册。自然种子分母全部确认动作前为正确前缀；相邻动作状态连续，排除了悄悄重置错误状态。memory推进提示确认仅首轮含初始值，并保留完整历史。', '',
           '|变体/等级|有错题/题数（95%精确区间）|首复查detect/首复查|首复查修好/首复查|二复查修好/二复查|复查再错次数|首次错后后续自然错/推进|错误类型|',
           '|---|---|---|---|---|---|---|---|']
    rng=random.Random(644901)
    for f in ('sheet','indirect','memory'):
        for level in range(4):
            rs=groups[(f,level)]
            if not rs:continue
            observed=[r for r in rs if r['n']>0]
            total=lambda key:sum(r[key] for r in rs)
            # Task cluster bootstrap of pooled rate, include variable exposure as observed.
            boot=[]
            for _ in range(10000):
                chosen=[rng.choice(observed) for _ in observed]
                n=sum(r['n'] for r in chosen)
                if n:boot.append(sum(r['e'] for r in chosen)/n)
            boot.sort()
            bci=[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]] if boot else None
            types=collections.Counter()
            for r in rs:types.update(r['types'])
            row=dict(family=f,level=level,observed_tasks=len(observed),task_cluster_bootstrap_95=bci,types=dict(types),
                     **{k:total(k) for k in ('n','e','error_task','first_reviews','first_detect','first_fixed','second_reviews','second_fixed','review_wrong','subsequent_steps','subsequent_errors')})
            table.append(row)
            lines.append(f"|{f}/{level}|{total('error_task')}/{len(observed)} {ci(total('error_task'),len(observed))}|{total('first_detect')}/{total('first_reviews')}|{total('first_fixed')}/{total('first_reviews')}|{total('second_fixed')}/{total('second_reviews')}|{total('review_wrong')}|{total('subsequent_errors')}/{total('subsequent_steps')}|{dict(types)}|")
    lines+=['','任务风险分母只含至少观察到一次推进动作的题，完全服务失败题不冒充无错误题；这些区间针对各题实际观察窗口，不是完整L=16的失败风险区间。每格整题簇bootstrap（10000次，固定种子644901，只重采样有动作暴露的题）见 [AUDIT_DIFFICULTY.json](AUDIT_DIFFICULTY.json)。只有至多4题时全零/全一bootstrap会退化，不能用窄区间宣称机制通过；服务缺失的不确定性不由bootstrap自动解决。后续再错率是错误后的条件描述，受到复查反馈与记忆状态暴露的干预，不能当成历史因果效应。', '',
            f'额外完成{checks}项请求、重试同提示、状态连续性、初始信息与并发复核；观测并发峰值{peak}。']
    (ROOT/'AUDIT_DIFFICULTY.json').write_text(json.dumps(dict(checks=checks,peak_concurrency=peak,rows=rows,cells=table),ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'DIFFICULTY.md').open('a') as fp:fp.write('\n'.join(lines)+'\n')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [ROOT/'DIFFICULTY.md',ROOT/'ANALYSIS_DIFFICULTY.json',ROOT/'AUDIT_DIFFICULTY.json',
               ROOT/'finalize_difficulty.py',ROOT/'FREEZE_DIFFICULTY.json',ROOT/'PREREG_DIFFICULTY.md',
               ROOT/'difficulty.py',ROOT/'test_difficulty.py',ROOT/'report_difficulty.py',
               ROOT/'TEST_DIFFICULTY.txt',ROOT/'ANALYSIS_DIFFICULTY_MOCK.json',
               Path(__file__).resolve(),*sorted(out.glob('*'))] if p.is_file()}
    (ROOT/'ARTIFACTS_DIFFICULTY.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,peak_concurrency=peak,cells=table),ensure_ascii=False))

if __name__=='__main__':main()
