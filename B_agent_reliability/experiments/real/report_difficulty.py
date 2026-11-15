"""Offline trace re-score and difficulty report; never makes model requests."""
import collections
import argparse
import hashlib
import json
from pathlib import Path
import random
from scipy.stats import beta
import difficulty as d

def interval(e, n):
    if not n:
        return '不可估计'
    lo = 0 if e == 0 else beta.ppf(.025, e, n-e+1)
    hi = 1 if e == n else beta.ppf(.975, e+1, n-e)
    return f'{100*lo:.2f}%–{100*hi:.2f}%'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['mock','real'],default='real')
    args=ap.parse_args()
    out = d.ROOT/('difficulty_'+args.mode)
    summary = json.loads((out/'summary.json').read_text())
    cells = collections.defaultdict(list)
    events = []
    examples = []
    models = collections.Counter()
    usage = collections.Counter()
    checks = 0
    for r in summary['results']:
        rows = [json.loads(s) for s in (out/(r['id']+'.jsonl')).read_text().splitlines()]
        t = rows[0]['task']
        # Replay using independent array implementation, distinct from executor dict path.
        regs = [t['initial'][f'r{i}'] for i in range(t['knob'] if t['family']=='memory' else (6 if t['family']=='sheet' else 8))]
        cursor = t['initial'].get('cursor', 0)
        checksum = 0
        for i, op in enumerate(t['operations']):
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
            assert gold == t['prefix'][i+1], (r['id'],i)
            checks += 1
        request_by_call = {x['call']: x for x in rows if x['event']=='request'}
        responses = {x['call']:x for x in rows if x['event']=='response'}
        actions = [x for x in rows if x['event']=='action']
        for x in rows:
            if x['event']=='response':
                response=x['response']
                models[response.get('model')]+=1
                assert response.get('model')==d.MODEL
                for key in ('prompt_tokens','completion_tokens','total_tokens'):
                    usage[key]+=response.get('usage',{}).get(key,0)
                usage['reasoning_tokens']+=(response.get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens',0)
        for a in actions:
            response=responses[a['call']]['response']
            assert a['content']==response['choices'][0]['message']['content']
            parsed, error=d.parse(a['content'],d.required(t,t['operations'][a['step']]))
            assert parsed==a['action']
            actual=a['before'].copy()
            if parsed:
                actual.update(parsed['writes'])
            assert actual==a['after']
            n=sum(actual.get(k)!=v for k,v in t['prefix'][a['step']+1].items())+int(parsed is None)
            assert n==a['n_after'] and (n==0)==a['correct']
            a.update(id=r['id'],family=r['family'],level=r['level'],operation=t['operations'][a['step']])
            if a['review']==0 and not a['correct']:
                a['reviews']=[x for x in actions if x['step']==a['step'] and x['review']>0]
                examples.append(a)
            checks+=1
        assert r['seed_errors']==sum(not a['correct'] for a in actions if a['review']==0)
        assert r['advances']==sum(a['review']==0 for a in actions)
        cells[(r['family'],r['level'])].append(r)
        events.extend(rows)
    reserves=[json.loads(s) for s in (out/'budget.jsonl').read_text().splitlines()]
    call_ids=sorted(x['call'] for x in reserves if x['event']=='reserve')
    assert call_ids==list(range(1,summary['calls']+1)) and summary['calls']<=1200
    assert len([x for x in events if x['event']=='request'])==summary['calls']
    frozen=json.loads((d.ROOT/'FREEZE_DIFFICULTY.json').read_text())
    assert frozen['sha256']==d.hashes()==summary['frozen_sha256']
    services=[x for x in events if x['event']=='service_error']
    retries=[x for x in events if x['event']=='request' and x['attempt']==1]
    lines=['# B：自然动作错误的难度扫描','',
           f"模型严格为 `{d.MODEL}`；完成 {len(summary['results'])}/48 题，成功 {sum(r['status']=='success' for r in summary['results'])} 题。真实 HTTP 尝试 {summary['calls']}/1200，已返回 {sum(models.values())}；停止原因 `{summary['halt']}`。",
           f"运行时间 {summary['started_utc']} 至 {summary.get('finished_utc',summary['updated_utc'])}。并发≤2；没有注入、动作改写或其他模型。",'',
           '每步是一笔原子批事务，不能将本轮每步率与 PILOT2 单键写入率直接等同。干净指动作前状态正确；修复后的后续步骤仍含真实错误历史，单独报告首次错误前风险。', '',
           '## 各变体与难度', '',
           '|变体|等级/旋钮|已尝试题/完整题|自然种子 e/n|率|95%精确参考区间|首次错误前 e/n|',
           '|---|---|---|---|---|---|---|']
    stats=[]
    for f in d.KNOBS:
        for level,k in enumerate(d.KNOBS[f]):
            rs=cells[(f,level)]
            n=sum(r['advances'] for r in rs); e=sum(r['seed_errors'] for r in rs)
            first_e=first_n=0
            for r in rs:
                aa=sorted([x for x in events if x.get('event')=='action' and x.get('review')==0 and x.get('id')==r['id']],key=lambda x:x['step'])
                for a in aa:
                    first_n+=1
                    if not a['correct']:
                        first_e+=1
                        break
            eligible=len(rs)==4 and all(r['status']=='success' for r in rs) and n==64 and .02<=e/n<=.10
            stat=dict(family=f,level=level,knob=k,e=e,n=n,eligible=eligible,first_e=first_e,first_n=first_n)
            stats.append(stat)
            rate=f'{100*e/n:.2f}%' if n else '不可估计'
            lines.append(f"|{f}|{level}/{k}|{len(rs)}/{sum(r['status']=='success' for r in rs)}|{e}/{n}|{rate}|{interval(e,n)}|{first_e}/{first_n}|")
    selected=sorted([s for s in stats if s['eligible']],key=lambda s:(abs(s['e']/s['n']-.06),s['level'],list(d.KNOBS).index(s['family'])))
    recommended=[]
    for s in selected:
        if s['family'] not in [x['family'] for x in recommended]:
            recommended.append(s)
        if len(recommended)==2:
            break
    lines+=['','区间为动作独立同分布假设下的 Clopper–Pearson 95%参考区间；同题动作相关，每格仅4个独立新题。点估计选中不代表真实率已被证明处于2%–10%。未完整格因难题或服务失败而删失，不作合格配置；不能把早停低错误率当作容易。','',
            '## 真实错误与复查','']
    for f in d.KNOBS:
        candidates=[x for x in examples if x['family']==f]
        if not candidates:
            lines.append(f'{f}：没有观察到自然错误，不能编造样例。')
        for a in candidates[:2]:
            simple={k:a[k] for k in ('id','step','operation','before','action','expected','error_type','n_after')}
            simple['reviews']=[{k:x[k] for k in ('review','action','correct','n_after')} for x in a['reviews']]
            lines+=['',f"真实样例 `{a['id']}`，step={a['step']}（从0开始）：",'```json',json.dumps(simple,ensure_ascii=False,indent=2),'```']
    total=lambda key:sum(r[key] for r in summary['results'])
    errors=total('seed_errors'); attempts=total('repair_attempts')
    lines+=['',f"自然错误 {errors} 次；验证器触发复查动作 {attempts} 次，其中自报 detect=true {total('error_detects')} 次、该次完成修复 {total('repaired')} 次、复查仍错误 {total('repair_reerrors')} 次。首个自然错误动作自报 detect 不等于发现自己的当前错误；detect 的定义是上一动作有错。完整逐动作数据见原始 JSONL。",
            '检测率属于验证器明确告知失败后的提示检测，不能估计自发 q。复查给当前实际状态，memory 因而获得额外状态信息；这是干预，相关历史可能改变后续错误率，首次错误前列用于敏感性检查。两次复查仍错即停题，绝不自动纠正状态。', '',
            '## 服务、额度与资源', '',
            f"传输/HTTP5xx失败 {len(services)} 次；重试 {len(retries)} 次（同一逻辑请求最多1次）；耗尽重试题数 {sum(r['status']=='service_retry_exhausted' for r in summary['results'])}。未知 token 请求 {len(services)} 次，独立计数，不计模型种子错误。已返回模型计数 {dict(models)}。",
            f"已知输入 {usage['prompt_tokens']}、输出 {usage['completion_tokens']}、合计 {usage['total_tokens']} token；reasoning {usage['reasoning_tokens']} 已含于输出，不重复加。服务失败部分用量未知。",
            f"额度检查：{[{k:v for k,v in x.items() if k in ('calls','five_hour','week')} for x in reserves if x['event']=='quota']}。两路并发中停止信号出现时已发请求可返回；不会再保留新调用号。", '',
            '## 推荐与后续校准', '']
    if recommended:
        lines.append('按运行前规则推荐：'+ '；'.join(f"`{s['family']}` 等级{s['level']}、旋钮{s['knob']}（{s['e']}/{s['n']}）" for s in recommended)+'。这些是探索性筛选配置，正式校准必须用全新种子。')
    else:
        lines.append('没有完整格符合预注册的2%–10%点估计条件，不能宣称找到了合格配置。本轮到此停止，不临时追加等级或种子。')
    lines+=['', '下一轮先为每个入选配置登记50个独立新题、L=32、纯自然轨迹；每个错误最多2次明确反馈复查，单题最多48调用、每配置硬上限2400调用，两配置合计≤4800（含服务重试，触顶即停），并发≤2，仍只用gpt-6-luna。该预算为建议，不在本轮执行。以任务为簇估计种子率、首次错误前风险、反馈检测与修复/再错，多步/多单元格变化全部保留；自然错误过少就不拟合自然核。若要估计自发检测，须另设固定频率、与oracle判错独立的无标签复查对照，不能沿用本轮反馈检测。',
             '', '只有独立校准确认非零自然错误、有足够自然修复片段、跨历史状态的定义可用后，才登记K∈{2,3,4}的容量实验；本轮批写与原执行器单槽容量不同，尚未定义可比较的K，不能直接推H50或容量收益。正式长度/容量留出仍须冻结参数、传播整题不确定性、比较无容量基线并计端到端服务风险。', '',
             '## 可复核证据', '',
             f'离线复核 {checks} 项（独立数组oracle前缀、响应内容与严格动作重评分、实际状态、全局调用号和冻结SHA）。冻结文件：[FREEZE_DIFFICULTY.json](FREEZE_DIFFICULTY.json)；运行前设计：[PREREG_DIFFICULTY.md](PREREG_DIFFICULTY.md)；原始任务、完整提示、响应、usage、状态：[difficulty_real/summary.json](difficulty_real/summary.json)。', '',
             '离线重现：`/home/liuyibo/venvs/theory/bin/python experiments/real/report_difficulty.py`。']
    (d.ROOT/('DIFFICULTY.md' if args.mode=='real' else 'DIFFICULTY_MOCK.md')).write_text('\n'.join(lines)+'\n')
    analysis=dict(stats=stats,recommended=recommended,usage=dict(usage),models=dict(models),service_errors=len(services),
                  retries=len(retries),checks=checks,freeze_verified=True)
    (d.ROOT/('ANALYSIS_DIFFICULTY.json' if args.mode=='real' else 'ANALYSIS_DIFFICULTY_MOCK.json')).write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')
    print(d.dump(analysis))

if __name__=='__main__':
    main()
