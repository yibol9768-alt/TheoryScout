"""Offline independent replays and Chinese reports; no model calls."""
import argparse
import collections
import json
import math
import random
from datetime import datetime
import reliability_core as d
import reliability_r2 as r

def exact(e, n):
    if not n:
        return None
    def tail(p, lower):
        ks = range(e, n+1) if lower else range(e+1)
        return math.fsum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)
                                  +k*math.log(p)+(n-k)*math.log1p(-p)) for k in ks)
    bounds = []
    for lower in (True, False):
        if lower and e == 0:
            bounds.append(0.)
            continue
        if not lower and e == n:
            bounds.append(1.)
            continue
        lo, hi = 0., 1.
        for _ in range(48):
            mid = (lo+hi)/2
            if (tail(mid, lower) < .025) == lower:
                lo = mid
            else:
                hi = mid
        bounds.append((lo+hi)/2)
    return bounds

def interval(x):
    return '不可估计' if x is None else f'[{100*x[0]:.2f}%, {100*x[1]:.2f}%]'

def rate(e, n):
    return '不可估计' if not n else f'{100*e/n:.2f}%'

def oracle(t):
    regs = [t['initial'][f'r{i}'] for i in range(t['knob'] if t['family']=='memory' else (6 if t['family']=='sheet' else 8))]
    cursor, checksum = t['initial'].get('cursor', 0), 0
    prefixes = [t['initial'].copy()]
    for op in t['operations']:
        if t['family'] == 'sheet':
            for x in op['instructions']:
                a,b,c = x['a'],x['b'],x['c']
                regs[a] = divmod(regs[a]+x['m']*regs[b]-regs[c]+x['d'],997)[1]
                regs[c] = divmod(regs[c]+regs[a],997)[1]
        elif t['family'] == 'indirect':
            cursor = (cursor+op['start'])%8
            for delta in op['delta']:
                pos = cursor
                nxt = t['table'][str(pos)][regs[pos]&1]
                regs[pos] = divmod(regs[pos]+regs[nxt]+delta,997)[1]
                regs[nxt] = divmod(regs[nxt]+2*regs[pos],997)[1]
                cursor = (nxt+regs[pos]%3)%8
        else:
            a,b,c = op['a'],op['b'],(op['a']+op['shift'])%len(regs)
            regs[a] = divmod(regs[a]+op['m']*regs[b]+op['d'],997)[1]
            regs[c] = divmod(regs[c]-regs[a]+regs[b],997)[1]
            checksum = divmod(7*checksum+regs[a]+3*regs[c]+regs[b],997)[1]
        gold = {f'r{j}':v for j,v in enumerate(regs)}
        if t['family']=='sheet':
            f0=(regs[0]+2*regs[1]-regs[2])%997
            f1=(3*f0+regs[3]-regs[4])%997
            gold.update(F0=f0,F1=f1,F2=(f1+2*regs[5])%997)
        if t['family']=='indirect':
            gold['cursor']=cursor
        if t['family']=='memory':
            gold['checksum']=checksum
        prefixes.append(gold)
    return prefixes

def inspect(out, phase=None):
    summary = json.loads((out/'summary.json').read_text())
    budget = [json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
    reserves = [x for x in budget if x['event']=='reserve']
    assert [x['call'] for x in reserves] == list(range(1, summary['calls']+1))
    assert summary['calls'] <= summary['cap']
    if phase:
        r.verify(phase)
        assert summary['frozen_sha256'] == r.hashes(phase)
    rows, events, transitions, examples = [], [], [], []
    timeline = []
    checks = 0
    for result in summary['results']:
        trace = [json.loads(s) for s in (out/(result['id']+'.jsonl')).read_text().splitlines()]
        t = trace[0]['task']
        assert oracle(t) == t['prefix']
        assert t == d.task({k:t[k] for k in ('family','level','knob','seed')} | ({'length':t['length']} if 'length' in t else {}))
        checks += len(t['operations'])
        requests = {x['call']:x for x in trace if x['event']=='request'}
        responses = {x['call']:x for x in trace if x['event']=='response'}
        acts = [x for x in trace if x['event']=='action']
        state = t['initial'].copy()
        seen_error = False
        first_n = first_e = after_n = after_e = 0
        for x in trace:
            if x['event']=='request':
                timeline.append((x['utc'], 1, x['call']))
                assert x['model'] == d.MODEL
                if x['attempt']:
                    prev = requests[x['call']]  # locate preceding identical logical request in this trace
                    candidates = [v for v in requests.values() if v['call']<x['call']
                                  and (v['step'],v['review'])==(x['step'],x['review'])]
                    prev = max(candidates, key=lambda v:v['call'])
                    assert prev['messages'] == x['messages'] and prev['attempt']+1 == x['attempt']
                    assert x['attempt'] <= (4 if phase else 1)
                checks += 1
            if x['event'] in ('response','service_error'):
                timeline.append((x['utc'], -1, x['call']))
            if x['event']=='response':
                assert x['response']['model'] == d.MODEL
            if x['event']=='action':
                assert x['before']==state
                if x['review']==0:
                    assert state==t['prefix'][x['step']]
                    if not seen_error:
                        first_n += 1
                        first_e += int(not x['correct'])
                    else:
                        after_n += 1
                        after_e += int(not x['correct'])
                    seen_error |= not x['correct']
                assert x['content']==responses[x['call']]['response']['choices'][0]['message']['content']
                parsed, _ = d.parse(x['content'], d.required(t,t['operations'][x['step']]))
                assert parsed == x['action']
                if parsed:
                    state.update(parsed['writes'])
                n_after = sum(state.get(k)!=v for k,v in t['prefix'][x['step']+1].items())+int(parsed is None)
                assert state==x['after'] and n_after==x['n_after'] and (n_after==0)==x['correct']
                transitions.append(dict(id=result['id'],step=x['step'],review=x['review'],
                                        n_before=x['n_before'],n_after=x['n_after'],correct=x['correct'],
                                        detect=bool(parsed and parsed['detect']),error_type=x['error_type']))
                checks += 1
        natural = [x for x in acts if x['review']==0]
        reviews = [x for x in acts if x['review']>0]
        assert len(natural)==result['advances'] and sum(not x['correct'] for x in natural)==result['seed_errors']
        for a in natural:
            req = requests[a['call']]
            if t['family']=='memory':
                assert len(req['messages'])==1+2*a['step']+2*sum(x['review']>0 and x['step']<a['step'] for x in acts)+1
                if a['step']>0:
                    assert 'initial' not in json.loads(req['messages'][-1]['content'])
            if not a['correct'] and not any(x['family']==t['family'] for x in examples):
                examples.append(dict(family=t['family'],id=result['id'],step=a['step'],operation=t['operations'][a['step']],
                                     before=a['before'],action=a['action'],expected=a['expected'],
                                     reviews=[{k:x[k] for k in ('review','action','correct','n_after')} for x in reviews if x['step']==a['step']]))
        row = dict(result, n=len(natural), e=sum(not x['correct'] for x in natural),
                   first_n=first_n,first_e=first_e,after_n=after_n,after_e=after_e,
                   error_task=int(seen_error),observed_task=int(bool(natural)),
                   first_reviews=sum(x['review']==1 for x in reviews),
                   first_detect=sum(x['review']==1 and bool(x['action'] and x['action']['detect']) for x in reviews),
                   first_fixed=sum(x['review']==1 and x['correct'] for x in reviews),
                   second_reviews=sum(x['review']==2 for x in reviews),
                   second_fixed=sum(x['review']==2 and x['correct'] for x in reviews),
                   types=dict(collections.Counter(x['error_type'] for x in natural if not x['correct'])))
        rows.append(row)
        events.extend(trace)
    active = set()
    peak = 0
    for _,delta,call in sorted(timeline):
        if delta==1:
            assert call not in active
            active.add(call)
            peak = max(peak,len(active))
        else:
            assert call in active
            active.remove(call)
        checks += 1
    # Fatal HTTP/JSON guards may leave an open request: report rather than fabricate a response.
    assert peak<=2
    assert not active or summary['halt']
    assert len([x for x in events if x['event']=='request'])==summary['calls']
    quotas = [x for x in budget if x['event']=='quota']
    if phase and summary['mode']=='real' and summary['halt']!='quota_unavailable':
        assert [x['calls'] for x in quotas] == [0]+list(range(100,summary['calls']+1,100))
        assert all(max(x['five_hour'],x['week'])<80 for x in quotas) or summary['halt']=='quota_80'
    if phase=='calib1':
        assert max(collections.Counter(x['task'] for x in reserves).values(),default=0)<=48
        assert max(collections.Counter(x['config'] for x in reserves).values(),default=0)<=2400
    services = [x for x in events if x['event']=='service_error']
    responses = [x for x in events if x['event']=='response']
    usage = collections.Counter()
    models = collections.Counter()
    for x in responses:
        resp=x['response']
        models[resp['model']]+=1
        usage.update({k:resp['usage'][k] for k in ('prompt_tokens','completion_tokens','total_tokens')})
        usage['reasoning_tokens']+=(resp['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens',0)
    return dict(summary=summary,rows=rows,checks=checks,peak_concurrency=peak,quotas=quotas,
                usage=dict(usage),models=dict(models),service_errors=len(services),
                retries=sum(x['event']=='request' and x['attempt']>0 for x in events),
                service_exhausted=sum(x['status']=='service_retry_exhausted' for x in rows),
                cooldowns=[x for x in budget if x['event']=='cooldown'],
                service_boundaries=[x for x in budget if x['event']=='service_boundary'],
                transitions=transitions,examples=examples)

def aggregate(rows):
    keys = ('n','e','first_n','first_e','after_n','after_e','error_task','observed_task',
            'first_reviews','first_detect','first_fixed','second_reviews','second_fixed','repair_reerrors')
    totals = {k:sum(x[k] for x in rows) for k in keys}
    totals.update(tasks=len(rows),complete=sum(x['status']=='success' for x in rows),
                  statuses=dict(collections.Counter(x['status'] for x in rows)))
    totals['iid_95']=exact(totals['e'],totals['n'])
    totals['task_error_95']=exact(totals['error_task'],totals['observed_task'])
    rng=random.Random(752911)
    boot=[]
    if rows:
        for _ in range(10000):
            sample=rng.choices(rows,k=len(rows))
            n=sum(x['n'] for x in sample)
            if n:
                boot.append(sum(x['e'] for x in sample)/n)
    boot.sort()
    totals['cluster_95']=[boot[int(.025*(len(boot)-1))],boot[int(.975*(len(boot)-1))]] if boot else None
    totals['valid_bootstrap']=len(boot)
    return totals

def report(phase, mock=False):
    name = 'DIFFICULTY_R2' if phase=='r2' else 'CALIB1'
    out = d.ROOT/(('difficulty_r2_' if phase=='r2' else 'calib1_')+('mock' if mock else 'real'))
    a=inspect(out,phase)
    summary=a['summary']
    groups=collections.defaultdict(list)
    for row in a['rows']:
        groups[(row['family'],row['level'])].append(row)
    cells=[]
    for (family,level),rows in sorted(groups.items(),key=lambda item:(list(d.KNOBS).index(item[0][0]),item[0][1])):
        cell=dict(family=family,level=level,knob=d.KNOBS[family][level],**aggregate(rows))
        cell['r2_eligible']=len(rows)==4 and cell['complete']==4 and cell['n']==64 and .02<=cell['e']/cell['n']<=.10
        cells.append(cell)
    a['cells']=cells
    if phase=='r2':
        eligible=[s for s in cells if s['r2_eligible']]
        conclusion=f"R2正式合格格数{len(eligible)}；完成{sum(s['complete'] for s in cells)}/{len(summary['schedule'])}题。"
        if not mock:
            old=inspect(d.ROOT/'difficulty_real')
            a['combined_selection']=r.selection()
            a['combined_cells']=[]
            for family in d.KNOBS:
                for level in range(4):
                    rows=[x for x in old['rows']+a['rows'] if (x['family'],x['level'])==(family,level)]
                    a['combined_cells'].append(dict(family=family,level=level,**aggregate(rows)))
            conclusion+=f"校准机械筛选：{a['combined_selection']['reason']}，候选{[(s['family'],s['level']) for s in a['combined_selection']['selected']]}。"
        conclusion+=' 本阶段不能直接进入容量/长度留出，需独立校准与容量定义。'
    else:
        a['admission']=[]
        for s in cells:
            ci=s['cluster_95']
            strong=bool(ci and ci[1]-ci[0]>.08)
            history=bool(s['first_n']>=20 and s['after_n']>=20 and abs(s['first_e']/s['first_n']-s['after_e']/s['after_n'])>.10)
            gates={'1_complete_50':s['tasks']==50 and s['complete']==50 and s['n']==1600,
                   '2_rate_2_10':bool(s['n'] and .02<=s['e']/s['n']<=.10),
                   '3_at_least_30_errors':s['e']>=30,
                   '4_separate_detection_repair_with_success':s['first_fixed']+s['second_fixed']>0,
                   '5_replay_passed':True,
                   '6_capacity_mapping_defined':False}
            a['admission'].append(dict(family=s['family'],level=s['level'],gates=gates,
                                       strong_cluster_warning=strong,history_warning=history,admit=False))
        conclusion='不能进入容量/长度留出实验。原子批事务尚未定义可迁移的容量K，自发检测/修复核也没有无标签对照；各项定量准入门槛如下。'
    lines=[f'# {name} 结果','',conclusion,'',
           f"唯一模型 `{d.MODEL}`；HTTP尝试{summary['calls']}/{summary['cap']}（含冒烟、重试），全局停止原因 `{summary['halt']}`。运行 {summary['started_utc']} 至 {summary.get('finished_utc',summary['updated_utc'])}。",'',
           '|变体/等级/旋钮|题数/完整|自然错误 e/n（率）|动作IID 95%参考|整题簇95%|首次错前 e/n|首错后 e/n|',
           '|---|---|---|---|---|---|---|']
    for s in cells:
        lines.append(f"|{s['family']}/{s['level']}/{s['knob']}|{s['tasks']}/{s['complete']}|{s['e']}/{s['n']} ({rate(s['e'],s['n'])})|{interval(s['iid_95'])}|{interval(s['cluster_95'])}|{s['first_e']}/{s['first_n']}|{s['after_e']}/{s['after_n']}|")
    lines+=['','动作前均为正确前缀；首次错误前风险含首次错误。实际错误写入保留，修好才继续；早停未观察尾部绝不补零。动作IID区间只作参考；整题簇bootstrap含零暴露题，不能消除服务或修复早停删失。纯自然指无人工注入，明确失败反馈仍是辅助干预。','',
            '|配置|至少一错题/有暴露题（95%）|首复查detect|首复查修好|二复查修好|复查仍错|终止原因|',
            '|---|---|---|---|---|---|---|']
    for s in cells:
        lines.append(f"|{s['family']}/{s['level']}|{s['error_task']}/{s['observed_task']} {interval(s['task_error_95'])}|{s['first_detect']}/{s['first_reviews']}|{s['first_fixed']}/{s['first_reviews']}|{s['second_fixed']}/{s['second_reviews']}|{s['repair_reerrors']}|{s['statuses']}|")
    lines+=['','反馈后detect与正确修复分开；不能估计自发q。memory复查暴露当前状态，后续错误率混合了反馈历史与选择，不能解释为历史因果效应。全量逐题率、错误类型、逐次n_before→n_after（schema非法时含额外失败标志）见分析JSON。','',
            f"服务失败{a['service_errors']}次，服务重试{a['retries']}次，耗尽重试题{a['service_exhausted']}；冷却{len(a['cooldowns'])}次。无返回请求token未知，服务失败不计模型错误。已知usage={a['usage']}；reasoning已包含在completion中，不重复加。已返回模型={a['models']}。",'',
            f"额度检查={ [{k:x[k] for k in ('calls','five_hour','week')} for x in a['quotas']] }。并发峰值{a['peak_concurrency']}。服务边界检查={a['service_boundaries']}。",'']
    if phase=='r2':
        lines+=['逐项对照预注册：',
                f"- 冻结/独立前缀与动作重放：通过，{a['checks']}项；旧冻结未改。",
                f"- 题数：尝试{len(a['rows'])}/48，完整{sum(s['complete'] for s in cells)}/48；删失原因见表。",
                '- 动作、提示与复查协议：继承原执行器，无注入，最多两次复查。',
                f"- HTTP硬限1200/并发≤2/额度整百检查：审计通过；停止={summary['halt']}。",
                f"- 最多4次服务重试与连续5次冷却：实现及故障模拟检验通过；本批服务失败{a['service_errors']}次。",
                f"- 四题全部成功+64推进+[2%,10%]：合格{sum(s['r2_eligible'] for s in cells)}格，不放宽门槛。",
                '- 整题不确定性、首次错前风险、反馈检测与修复：均分别报告。','']
        if not mock:
            lines+=['旧轮+R2合并（包含所有可评分删失轨迹）：','',
                    '|配置|错误/推进|合并率|校准完整性|', '|---|---|---|---|']
            for s in a['combined_selection']['cells']:
                lines.append(f"|{s['family']}/{s['level']}|{s['e']}/{s['n']}|{rate(s['e'],s['n'])}|{s['complete']}|")
            lines+=['',f"机械筛选详情：`{json.dumps(a['combined_selection']['selected'],ensure_ascii=False)}`；理由 `{a['combined_selection']['reason']}`。后备选择不等于R2正式达标。",'']
    else:
        selection=summary['selection']
        lines += [f"冻结选择原因：`{selection['reason']}`，筛选依据是旧轮+R2，校准估计仅用本阶段新题。",'']
        labels={'1_complete_50':'①50题完整、无删失','2_rate_2_10':'②自然错误点估计2%–10%',
                '3_at_least_30_errors':'③至少30个自然错误片段','4_separate_detection_repair_with_success':'④检测/修复分开且观察成功修复',
                '5_replay_passed':'⑤独立重放、连续性、schema审计','6_capacity_mapping_defined':'⑥跨历史容量K映射'}
        for item in a['admission']:
            lines.append(f"配置 {item['family']}/{item['level']}：")
            for k,passed in item['gates'].items():
                lines.append(f"- {labels[k]}：{'通过' if passed else '未通过'}。")
            lines+= [f"- 簇相关/异质性警示：{item['strong_cluster_warning']}；历史依赖警示：{item['history_warning']}。",'']
        lines+=['每题48、每配置2400、总4800、并发≤2、整百额度检查与10/25/50服务检查均按冻结规则核验；失败题保留不替换。自然错误少于30不拟合自然核；反馈核不可冒充无提示自然核。','']
    for ex in a['examples']:
        lines += [f"真实样例 {ex['id']}，step={ex['step']}：",'```json',json.dumps(ex,ensure_ascii=False,indent=2),'```','']
    lines += [f"原始轨迹与完整提示：[{out.name}/summary.json]({out.name}/summary.json)。冻结：[{r.FREEZES[phase]}]({r.FREEZES[phase]})。逐题及错格变化：[ANALYSIS_{name}.json](ANALYSIS_{name}.json)。",
              '冒烟人工检查记录在各输出目录 SMOKE_APPROVED.json；冒烟包含在正式分母。所有本地执行命令使用用户systemd scope的2G内存、零swap限制。']
    suffix='_MOCK' if mock else ''
    (d.ROOT/(name+suffix+'.md')).write_text('\n'.join(lines)+'\n')
    (d.ROOT/('ANALYSIS_'+name+suffix+'.json')).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(phase=phase,calls=summary['calls'],checks=a['checks'],peak=a['peak_concurrency'],
                          cells=cells,admission=a.get('admission')),ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['r2','calib1'])
    ap.add_argument('--mock',action='store_true')
    args=ap.parse_args()
    report(args.phase,args.mock)
