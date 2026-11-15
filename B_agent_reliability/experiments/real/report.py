"""Read-only trace audit, sample rendering, and Chinese tiny-pilot report."""
import hashlib
import json
from pathlib import Path
from pilot import ROOT

def pretty(x): return json.dumps(x,ensure_ascii=False,indent=2)
def block(x): return '```json\n'+pretty(x)+'\n```\n'
def main():
    s=json.loads((ROOT/'tiny_real/summary.json').read_text())
    checks=[]; tokens={k:0 for k in ['prompt_tokens','completion_tokens','total_tokens']}
    models=set(); violations=[]; sample=['# 人工可读轨迹样例\n\n完整逐事件 JSONL 保留所有提示、原始响应、usage、状态、注入、检测、修复、提交和最终核对；以下摘录不替代原始轨迹。\n']
    total_requests=0; total_responses=0
    for r in s['results']:
        path=ROOT/r['trace']; events=[json.loads(x) for x in path.read_text().splitlines()]
        req=[e for e in events if e['event']=='request']; resp=[e for e in events if e['event']=='response']
        final=[e for e in events if e['event']=='final']
        total_requests+=len(req); total_responses+=len(resp)
        checks.append({'trace':r['trace'],'request_count_ok':len(req)==r['calls'],
                       'final_record_ok':len(final)==1 and final[0]['status']==r['status'],
                       'response_count_ok':len(resp)==len(req),
                       'trace_bytes_ok':path.stat().st_size==r['trace_bytes'],
                       'ledger_prefix_replay_ok':True})
        from pilot import Task
        t=Task(r['family'],r['seed'],r['L'])
        assert r['final']['expected']==t.prefix[-1]
        assert all(x['state']==t.prefix[i+1] for i,x in enumerate(r['final']['ledger']))
        assert all(sum(e['response']['usage'][k] for e in resp)==r['usage'][k] for k in tokens)
        for e in resp:
            rr=e['response']; models.add(rr.get('model'))
            for k in tokens: tokens[k]+=rr['usage'][k]
            if rr['usage']['prompt_tokens']>2048 or rr['usage']['completion_tokens']>128:
                violations.append({'trace':r['trace'],'call':e['call'],'usage':rr['usage']})
        sample.append(f"## {r['family']} / seed={r['seed']}（真实）\n\n状态 `{r['status']}`，调用 {r['calls']}，推进 {r['progress']}/16。完整轨迹：[{path.name}]({r['trace']})。\n")
        # First two actual prompt/reply/transition bundles, plus final verification.
        for e in events:
            if e['event'] in ['request','response','transition','injection','commit','stop'] and e.get('call',1)<=2:
                if e['event'] in ['transition','commit'] and e.get('step',0)>0: continue
                sample.append(block(e))
        if any(x['event']=='injection' for x in events):
            sample.append('真实注入、修复与恢复完整片段：\n')
            for i,x in enumerate(events):
                if x['event']=='injection':
                    for y in events[max(0,i-2):]:
                        sample.append(block(y))
                        if y['event']=='commit' or y['event']=='stop': break
                    break
        sample.append('最终核对：\n'+block(r['final']))
    mock=ROOT/'tiny_mock/sheet_2207.jsonl'
    e=[json.loads(x) for x in mock.read_text().splitlines()]
    sample.append('## 工作表注入—修复—恢复（mock，不是真实模型证据）\n\n真实运行未必到达注入位置；这里展示已通过的完整功能路径。完整轨迹：[sheet_2207.jsonl](tiny_mock/sheet_2207.jsonl)。\n')
    for i,x in enumerate(e):
        if x['event']=='injection' and x['step']==4:
            for y in e[max(0,i-2):]:
                sample.append(block(y))
                if y['event']=='commit': break
            break
    (ROOT/'SAMPLE_TRACES.md').write_text('\n'.join(sample))
    assert all(all(v for k,v in c.items() if k!='trace') for c in checks)
    assert total_requests==s['calls']<=150
    assert all(r['calls']<=32 for r in s['results'])
    assert all(r['L']==16 and r['K']==3 for r in s['results'])
    audit={'checks':checks,'requests':total_requests,'responses':total_responses,'models':sorted(models),
           'usage':tokens,'token_limit_violations':violations,'all_checks_passed':True,
           'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
    (ROOT/'AUDIT.json').write_text(pretty(audit)+'\n')
    rows=['| 执行器/种子 | 注入计划 | 状态 | 推进 | 调用 | 输入/输出/合计 token | W | 槽峰值 | 活动JSON字节峰值 | 轨迹磁盘字节 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in s['results']:
        u=r['usage']
        rows.append(f"| {r['family']}/{r['seed']} | {'第5、11步' if r['injected'] else '无'} | {r['status']} | {r['progress']}/16 | {r['calls']} | {u['prompt_tokens']}/{u['completion_tokens']}/{u['total_tokens']} | {r['work']} | {r['peak_undo_slots']} | {r['peak_live_json_bytes']} | {r['trace_bytes']} |")
    report=f'''# 真实 LLM 微型试跑

本轮仅执行两类各2个预定任务（L=16、K=3）；没有扩大到40次先导。真实模型成功 {sum(r['success'] for r in s['results'])}/{len(s['results'])}。该结果用于工程故障定位，不能据此拒绝或支持一维机制。

## 实现与协议

`pilot.py` 包含确定性随机表键值累加、带写入前像及可执行 undo 的工作表、独立前缀 oracle、逐提交义务账本、最终完整状态校验、mock 和真实 chat/completions 客户端。输入含固定表、干净检查点、当前操作、当前状态和最多2条未验证历史。错误暂停推进，恢复后清历史。第5、11步对错误组的实际写入加7并作为声称正确的污染历史反馈；是否真正触发见轨迹，计划不算观测。完整实施定义见 [DESIGN_TINY.md](DESIGN_TINY.md)。

n 是错误单元格数加格式义务；K限制义务触顶和活动前像槽。顺序撤销日志与审计日志额外计存储，不把全部开销说成O(K)。独立任务种子1105/2207；生成器可保持长度/容量配对前缀，微型本身不作干净/污染配对因果估计。oracle 强制检测，detect自报不等于自然 q。注入不能用于估计自然种子率。

真实调用仅 `{s['endpoint']}`，请求与每次响应 model 都保存。观测 model：`{', '.join(sorted(models))}`；{'未发现模型字段不符。' if models=={'gpt-6-luna'} else '发现模型不符，已立即停整批，未尝试剩余任务。'} 串行并发1，无自动重试、其他模型或其他推理服务。密钥仅由指定命令读入环境，未写入轨迹、代码或报告；HTTP异常不保存响应体或鉴权头，拒绝重定向并绕过系统代理。

## 测试

单元测试14项通过，覆盖四种mock全流程组合、undo恢复、真实槽容量限制、非法动作、调用/全局硬停、model不符停批、实际usage超限、最终错误抵消、账本篡改、配对前缀、污染历史清理、轨迹禁止覆盖。独立 mock 批次4/4成功，共68次假调用（干净各16、注入各18），不计入真实调用数。见 [TEST_RESULTS.txt](TEST_RESULTS.txt)、[tiny_mock/summary.json](tiny_mock/summary.json)。

离线审计通过：请求/响应计数、逐次模型字段、token汇总、最终记录与summary一致、磁盘大小、每个账本提交独立oracle重放。见 [AUDIT.json](AUDIT.json)。审计通过表示证据自洽，不代表真实运行成功。

## 真实结果与成本

时间：{s['started_utc']} 至 {s['finished_utc']}（UTC）。

{chr(10).join(rows)}

真实请求 {total_requests} 次、收到响应 {total_responses} 次，低于总硬上限150；各运行≤32次。返回 usage 累计：输入 **{tokens['prompt_tokens']}**、输出 **{tokens['completion_tokens']}**、合计 **{tokens['total_tokens']}** token（输出包括服务端报告的 reasoning token，原始细项完整保留）。不查询价格，未编造美元成本。W是实际应用的动作数，API超限响应在应用前停止，因此调用数可能大于W。W≤5L的任务数 {sum(r['within_5L'] for r in s['results'])}/{len(s['results'])}，但预算内成功数仅 {sum(r['success'] and r['within_5L'] for r in s['results'])}；不能把提前失败的低成本当作效率证据。活动JSON字节是序列化载荷大小，不是进程RSS；磁盘字节单独列出。

实际触发注入 {sum(r['injections'] for r in s['results'])} 次，恢复 {sum(r['recoveries'] for r in s['results'])} 次；均来自 kv/2207，工作表两次运行在第5步之前停止，未触发计划注入。成功任务18个动作全部通过逐步/最终双重核对，两次repair的n均从1降为0。

实际输出token超过128的响应有 {len(violations)} 次。客户端确实发送 max_completion_tokens=128，服务端usage仍可能超过该值，相关运行按失败停止。首个任务超限后初版管理逻辑提前停整批；随后 `finish_tiny.py` 仅完成同批剩余3个从未尝试的槽位，未重跑第一个任务、未改提示/参数、未按效果换种子。新增单次续跑标记防止重复，原始首轮summary已由继续标记记录原因和调用数；逐次JSONL保留原始事实。后续超限同样判该运行失败，model不符仍停整批。这是管理逻辑修正，不能隐去。

所有轨迹见 [tiny_real/summary.json](tiny_real/summary.json) 及其 JSONL 引用；人工可读样例见 [SAMPLE_TRACES.md](SAMPLE_TRACES.md)。没有重试直至成功，没有追加或替换运行、扩样或阈值修改。

## 发现的问题与是否放大

**目前不应直接放大到40次预注册先导。** mock正确证明本地语义和恢复路径可执行，真实微型暴露了接口/提示约束问题：示例target写成“cell/key”，模型出现字面 `cell/k6` 而非允许的 `k6`；模型没有获得明确的合法键列表和有效JSON示例。格式错误被真实记录为义务，没有静默替换键名。需要在新数据前明确动作schema和合法target示例，保持严格校验，不在评分时自动纠正模型。

同时必须解决本机反代对输出128限制的实际执行：确认它支持的参数与reasoning开销口径，证明128总输出上限能生效；当前guard只能在收到usage后发现超限，不能撤回已消耗token。输入使用o200k_base估算并留148token余量，实际usage逐次复核；这不是gpt-6-luna精确tokenizer证明。所有修改均应先mock验证并冻结新实现，再另获授权采新真实数据。本轮到此结束，不再调用API。

需要补充的科学设计：正式40次须至少覆盖人工正确/错误历史，并做同题配对；加入状态相关、残余污染、隐藏旁路负对照的运行配置（当前执行器/轨迹结构可承载，尚未实现完整统计检验）。增加同n不同历史的可比较转移样本及恢复后残余历史风险记录。记录0跳和多项跳，不靠拆分事件制造出生死亡链；自然检测、自然种子与强制oracle检测/人工注入应分开。槽容量与顺序日志仍不是纯O(K)的全部资源模型，扩容量时要量真实存储。不能从4条短轨迹冻结可靠rho，不能声称再生、theta平稳或容量留出预测通过。

保留 FORMULATION 与 RESULTS_WEEK1 原判据：概率误差 max(0.03,5SE)、H50两倍、恢复后残余历史效应0.03等全部不变；没有因微型失败调整门槛。40次先导本轮未运行，小K参数外推及K=6/10/16留出任务均未运行。
'''
    (ROOT/'PILOT_TINY.md').write_text(report)
    print(pretty({'requests':total_requests,'usage':tokens,'statuses':[r['status'] for r in s['results']], 'audit':'passed'}))

if __name__=='__main__': main()
