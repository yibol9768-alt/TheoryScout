"""Render the Chinese pilot report from audited observations; no API calls."""
import json
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from pilot import ROOT

def ci(x): return '不可估计' if x is None else f'[{x[0]:.4f}, {x[1]:.4f}]'
def rr(m): return f"{m['x']}/{m['n']}"+('（不可估计）' if not m['n'] else f" = {m['estimate']:.4f}；95% {ci(m['cp95_iid_reference'])}")

def main():
    s=json.loads((ROOT/'FINAL_SUMMARY_PILOT2.json').read_text())
    a=json.loads((ROOT/'ANALYSIS_PILOT2.json').read_text())
    freeze=json.loads((ROOT/'FREEZE_PILOT2.json').read_text())
    metrics=a['metrics']; m=metrics['all']; rs=s['results']
    duration=(datetime.fromisoformat(s['finished_utc'])-datetime.fromisoformat(s['started_utc'])).total_seconds()
    lines=['# B 真实动作小先导 PILOT2','',
      f"本轮完成 **{a['runs']}/24 运行、{a['successes']} 次成功**；两个执行器各6个新题，每题干净/注入配对，K=2/3各3题，L=16。真实请求 **{a['calls']}** 次，返回 **{a['responses']}** 次；仅观测到 `{', '.join(a['models'])}`。人工注入 {a['actual_injections']} 次，片段恢复 {a['recoveries']} 次。自然错误必须与人工改写分开解释。",'',
      f"真实时间 {s['started_utc']} 至 {s['finished_utc']}（UTC），墙钟 {duration/60:.2f} 分钟。先导不是正式容量/长度留出验证；本轮没有以测试结果重新拟合预测、追加种子或重跑失败槽；存在明示的管理续跑修订。",'',
      '## 工程修复、验证与冻结','',
      '输出请求字段改为 `max_completion_tokens=2048`，推理 token 包含在 completion_tokens 中；返回 usage 超2048或 finish_reason=length 均记本运行失败，保留耗费，不应用动作、不重试。输入实际上限仍2048。每个响应 model 严格核对，不符即停整批。并发1；单次运行上限20调用、计划24×20=480、全局硬上限500。密钥仅在环境中读取，未输出或写盘。', '',
      '动作提示列出 kv 的 k0–k7 和 sheet 的 A1/A2/A3/B1/B2/B3，逐题给 legal_targets，并给两个有效 JSON 示例和操作语义。评分拒绝别名前缀、额外/重复字段、非法类型，undo 要求 detect=true及可撤销记录；无静默纠正。原始动作另在 shadow 上评分，shadow 不决定执行转移。', '',
      '[单元测试](TEST_PILOT2.txt) 49次执行全部通过（含继承/发现重复执行，不是49项独立行为），覆盖实际HTTP请求字段、含推理输出超限/截断、严格schema、500硬停、模型不符整批停止、mock配对、注入与自然错误分离。冻结后 [mock整批](pilot2_mock/summary.json) 24/24成功、408假调用，不计真实调用。真实启动核对代码与预注册哈希，离线审计再次核对。', '',
      f"冻结时间 `{freeze['frozen_utc']}`；[完整SHA-256清单](FREEZE_PILOT2.json)：", '',
      '| 文件 | SHA-256 |','|---|---|']
    for name,digest in freeze['sha256'].items(): lines.append(f'| {name} | `{digest}` |')
    lines+=['',f"运行前 [PREREG_PILOT2.md](PREREG_PILOT2.md) SHA-256：`{s['prereg_sha256']}`。原 tiny_real 及旧报告保留，本轮不将旧轨迹并入估计。",'',
      '## 原管理停批与透明续跑', '', '原批次在第9运行kv/3323/K3/clean的第11请求发生90秒传输错误，依原预注册停止：8成功、1失败、147请求/146返回。之后在任何续跑真实请求前写[AMENDMENT_PILOT2.md](AMENDMENT_PILOT2.md)，对原计划剩余15个未尝试槽一次性续跑；原失败槽不重跑、不替换，不修改原预注册或原轨迹。这是对原传输错误停批管理规则的公开偏离，必须保留在评估中。第一次续跑又在kv/3323/K3/injected第12请求传输失败并停批；累计159请求/157返回，8成功/2失败。随后在第三管理批次真实数据前另写[AMENDMENT2_PILOT2.md](AMENDMENT2_PILOT2.md)，公开撤销首次修订中‘不再第三批’的管理限制，以逐槽记录失败、继续尚未尝试槽的方式完成剩余14槽。两次偏离都不能隐去；两个失败槽均未重试。', '', '续跑管理器经过4项约束测试、15/15完整mock（256假调用），最终冻结清单见[FREEZE_RESUME_PILOT2.json](FREEZE_RESUME_PILOT2.json)。核心实现SHA不变。共享计数继承147，累计仍≤500（剩余最坏147+15×20=447）。最终管理器另经4项约束测试、14/14完整mock（238假调用）后冻结，见[FREEZE_FINISH_PILOT2.json](FREEZE_FINISH_PILOT2.json)；继承159，最坏159+14×20=439≤500。最终管理器只对未尝试槽各尝试一次，传输/HTTP失败记该槽失败后推进；model不符/usage缺失/全局硬限立即全停。遇model不符绝不续跑。原始失败与已知/未知消耗均保留，不把失败改成成功。三批真实调用共享同一个500上限，没有将新管理批次当新预算。', '', '## 成本、预算成功率与完整运行','',
      f"实际已返回响应的输入 **{a['usage']['prompt_tokens']}**、输出 **{a['usage']['completion_tokens']}**、合计 **{a['usage']['total_tokens']}** token；未返回请求 **{a['unanswered_requests']}** 次，其token未知，故上述是可核对的已知用量、不能冒充完整提供商账单。输出中的 reasoning_tokens 合计 **{a['reasoning_tokens']}**，不重复加总。单响应实际最大输入{a['max_prompt_tokens']}、最大输出{a['max_completion_tokens']}，截断响应{a['completion_length_responses']}次。没有现行费率，因此不虚构美元成本；若每百万输入/输出价为 p_in/p_out，则已知响应部分费用为 `{a['usage']['prompt_tokens']/1e6:.6f} p_in + {a['usage']['completion_tokens']/1e6:.6f} p_out`。",'',
      '| 执行器/seed | K | 条件 | 状态/推进 | 调用/W | 已知输入/输出/合计token | 槽峰/活动字节/轨迹字节 |','|---|---:|---|---|---|---|---|']
    for r in rs:
        u=r['usage']
        response_count=sum(json.loads(line)['event']=='response' for line in (ROOT/r['trace']).read_text().splitlines())
        unknown=r['calls']-response_count
        missing=f'；另{unknown}次未知' if unknown else ''
        lines.append(f"| {r['family']}/{r['seed']} | {r['K']} | {'注入' if r['injected'] else '干净'} | {r['status']} {r['progress']}/16 | {r['calls']}/{r['work']} | {u['prompt_tokens']}/{u['completion_tokens']}/{u['total_tokens']}{missing} | {r['peak_undo_slots']}/{r['peak_live_json_bytes']}/{r['trace_bytes']} |")
    lines+=['',f"单次调用硬限内成功 {sum(r['success'] for r in rs)}/{len(rs)}；W≤5L且成功 {sum(r['success'] and r['within_5L'] for r in rs)}/{len(rs)}。本轮20调用硬限严于5L=80，W门槛不是独立效率证据；提前失败的低耗费不算高效。状态分布 `{dict(Counter(r['status'] for r in rs))}`。传输/HTTP等非动作失败也计端到端失败，不能与自然动作种子混在一起或从成功率分母删除。完整原始提示、响应、usage、动作、注入与账本在 [最终合并summary及JSONL引用](FINAL_SUMMARY_PILOT2.json)，[原批次](pilot2_real/summary.json)、[首次续跑](pilot2_resume_real/summary.json)和[最终管理批次](pilot2_finish_real/summary.json)分别保留。活动字节是写后、追加当前history前的state/history/checkpoint序列化采样值，非完整活动内存峰值或进程RSS；顺序撤销日志和磁盘归档另计，不能声称整体存储O(K)。",'',
      '## 真实模型动作错误与人工可读样例','',
      f"干净轨迹自然种子错误率 **{rr(m['natural_seed_clean'])}**。注入分支的**未改写原动作**错误率 **{rr(m['seed_original_injected_branch'])}**。前者为主估计，后者仅敏感性描述；不能把人工加7计成模型错误。",'']
    if a['action_error_types']:
        lines+=['| 执行器 | 模式 | 类型 | 次数 |','|---|---|---|---:|']
        for e in a['action_error_types']: lines.append(f"| {e['family']} | {e['mode']} | {e['kind']} | {e['count']} |")
        for e in a['examples'][:6]:
            lines += ['',f"真实错误：{e['family']}/{e['seed']}，K={e['K']}，第{e['step']}步，{e['mode']}，{e['kind']}。完整轨迹 [{Path(e['trace']).name}]({e['trace']})。",'```json',json.dumps(e,ensure_ascii=False,indent=2),'```']
    else:
        total=sum(r['work'] for r in rs)
        lines += [f"全部 **{total}** 次已评分原始模型动作均完成了当前事务（超时请求没有可评分动作）；自然数值错误、非法键/单元格、schema/JSON错误、undo误用均为 **0/{total}**（混合模式分母仅作动作质量描述）。自然 advance 错误0次与{a['actual_injections']}次人工注入是两类事实。没有真实错误样例可展示，不能编造；旧版 `cell/k6` 只属 PILOT_TINY 历史错误，本轮频率为0。以下展示真实原动作、人工改写及真实修复，明确标注人工来源。"]
    # Always show actual paired injection/recovery, rather than a mock example.
    for fam in ('kv','sheet'):
        for r in rs:
            if r['family']!=fam or not r['injected']: continue
            es=[json.loads(x) for x in (ROOT/r['trace']).read_text().splitlines()]
            ts=[e for e in es if e['event']=='transition']
            t=next((e for e in ts if e['injection']),None)
            if not t: continue
            repair=next((e for e in ts if e['step']==t['step'] and e['mode']=='repair'),None)
            request=next(e for e in es if e['event']=='request' and e['state']['index']==t['step'])
            p=json.loads(request['messages'][1]['content'])
            sample=dict(family=fam,seed=r['seed'],step=t['step']+1,operation=p['operation'],checkpoint=p['checkpoint'],
                        original_model_action=t['model_action'],original_action_n_after=t['model_action_n_after'],
                        artificially_applied_action=t['action'],n_after_injection=t['n_after'],
                        real_repair_action=repair['model_action'] if repair else None,
                        n_after_repair=repair['n_after'] if repair else None)
            lines += ['',f"{fam} 实际注入样例（[原始轨迹]({r['trace']})）：",'```json',json.dumps(sample,ensure_ascii=False,indent=2),'```']
            break
    lines += ['',f"实际全部转移 Δn 计数：`{m['all_raw_delta_counts']}`；repair Δn 计数：`{m['repair_delta_counts']}`。0跳和多项变化原样保留。n_before=0 的 advance 是‘操作尚未提交’基线；Δn=0的正确推进不等于修复。",'',
      f"实际n_after访问计数：`{a['visited_n_after']}`，活动前像槽峰值 `{a['max_active_undo_slots']}`。这些观测不覆盖高n转移。", '', '## 参数与不确定性','',
      '| 估计量 | kv | sheet | 合并描述 |','|---|---|---|---|']
    for key,title in (('natural_seed_clean','自然种子率（干净advance）'),('seed_original_injected_branch','注入分支原始advance错误率'),
                       ('detect_first_injected','注入后首个提示检测率'),('detect_prompt_injected','注入后全部提示检测率'),
                       ('repair_zero_injected','注入后一次动作完成修复率'),('repair_down_injected','注入后义务下降率'),
                       ('detect_prompt_natural','自然错误后的提示检测率'),('repair_zero_natural','自然错误后修复率')):
        lines.append('| '+title+' | '+' | '.join(rr(metrics[f][key]) for f in ('kv','sheet','all'))+' |')
    lines += ['', '上表是动作级 Clopper–Pearson 95%区间，只是独立同分布参考。任务内动作相关，不能用动作数替代12个独立题的机制功效；超时任务只提供已观察的推进，存在不等暴露和配对缺失。按执行器分层、整题保留配对的10000次簇bootstrap及完整分母见 [ANALYSIS_PILOT2.json](ANALYSIS_PILOT2.json)；全零/全一时bootstrap退化，不代表没有不确定性。任务级精确区间提供另一种描述；它们针对各题实际观察窗口，不是完整L=16风险的上界：','']
    for f in ('kv','sheet','all'):
        z=metrics[f]['natural_seed_clean']; d=metrics[f]['repair_zero_injected']
        lines.append(f"- {f}：出现自然错误的任务 {z['tasks_any_numerator']}/{z['exposed_tasks']}，95% {ci(z['tasks_any_numerator_cp95'])}；出现注入修复失败的任务 {d['tasks_with_any_complement']}/{d['exposed_tasks']}，95% {ci(d['tasks_any_complement_cp95'])}。")
    lines += ['', '检测率是已收到 oracle 的 repair 提示之后的自报 detect，不能当自然检测率 q。干净种子事件若为0，自然检测、修复、上下跳比 rho 都不可识别；注入仅识别本协议n=1、已知有错、可见检查点情况下的修复。自然q、Gamma异质性、eta及高n核不拟合、不补伪计数。即使注入下观察到纯下降，也不能用“rho=∞”认证高容量外推。', '',
      '| 历史效应（按独立题先平均） | kv | sheet | 合并描述 |','|---|---|---|---|']
    for key,title in (('post_recovery_paired_difference','清历史恢复后下一步差'),('joint_contamination_paired_difference','污染repair对同题干净步骤差')):
        cells=[]
        for f in ('kv','sheet','all'):
            z=metrics[f][key]
            cells.append('不可估计' if not z['tasks'] else f"{z['estimate']:.4f}（{z['tasks']}题）；保守95% {ci(z['hoeffding95'])}")
        lines.append('| '+title+' | '+' | '.join(cells)+' |')
    lines += ['', f'配对只用双方都可观察且实际恢复后的步骤；超时导致的缺失不是0差，不能以未观察步骤补齐。配对bootstrap区间若为[0,0]只是小样本退化；表中Hoeffding区间针对每题差∈[-1,1]、题目独立给出保守不确定性。不能排除±0.03。恢复后每个 advance 实际收到空历史；可配对的下一步输入逐字一致 {m['identical_post_recovery_prompt_pairs']}/{m['post_recovery_prompt_pairs']}。在逐请求无状态的API协议下，再生主要由恢复检查点和清空输入历史施加，不能把零差当作模型内生遗忘机制的证据，更不能推广到保留错误历史的 agent；repair污染对照同时改变状态和oracle提示，也不是纯历史效应因果估计。', '',
      '## 预测与原判据','',
      f"运行前工程基准为24/24成功、408调用、24次注入并修复；本轮实际为{a['successes']}/24成功、{a['calls']}调用、{a['actual_injections']}次注入、{a['recoveries']}次恢复。提示schema问题是否消失以逐动作严格评分为准。存在传输失败时，“24/24成功、408调用”的完整工程预测未实现；不能把没有已评分动作错误改写成全批工程成功。输出/schema修复的验证也不等于一维机制通过。",'',
      'RESULTS_WEEK1 的判据全部保留：足够计数的测试格概率误差>max(0.03,5SE)、H50误差超过两倍则拒绝；恢复后残余效应>0.03且有限增广无法消除则停止一维路线；足够计数仍须每格成功和失败各≥30。本轮每个family×K只有3个独立题，全部运行也仅24，不能达到该标准；没有多长度数据，H50不可检验。没有显著拒绝不能写成支持。', '',
      '校准不确定性已通过动作参考区间、整题bootstrap和配对保守界传播；但自然修复转移核没有访问，无法合法产生有用的容量/长度预测区间。因此不把点估计0错误当作p_seed=0确定值、不把人工修复率填成自然rho，也不生成假的H50。预注册的公式敏感性传播以核可识别为前提，本轮未满足。', '',
      '## 能否进入正式容量/长度留出实验及具体后续设计','',
      '**不足以启动正式容量/长度预测验证。** 当前只能检查这套短任务协议的执行与失败记录。若没有自然错误片段，K从2增到3几乎未受到压力，K≥4所需高n转移完全未观察；同题并未跨K配对。不能以简单任务全成功证明容量收益，也不能借人工注入得出自然总体风险。', '',
      '按预注册选择规则，候选K={2,3,4}、L={16,32,64,128,256,512,1024,2048}，先要求自然核/硬预算预测可识别，再选择预测中位成功率0.2–0.8且95%校准区间宽≤0.4的格。本轮自然核不可识别，合格格为0，不冻结任何未经数据支持的测试格。本轮到此停止API调用。', '',
      '后续设计分成以下两个有预算的阶段，须另行登记；不是本轮授权内的扩样：', '',
      '1. **补充自然校准**：两个执行器各50新任务，K=2/3各25，L=32，每题干净/注入配对，共200运行，每次≤40调用，共≤8000调用，输入/输出各≤16384000 token。先保持本轮提示和评分；若仍没有足够自然错误，就停止该执行器配置，而不是把L任意拉长或混入人工种子。可另登记更有自然认知负担的任务版本（如更多依赖的工作表批事务），但须重新验证状态定义、oracle信息成本及动作schema。要估计高n，必须另登记定向真实动作采样；人工初始化状态单独标记，不能替代自然核。', '',
      '2. **正式留出协议的具体模板**：两个执行器各200个全新独立题；以相同种子跨容量和长度配对，固定2×2格（两个容量来自{2,3,4}，两个长度来自候选网格），共1600运行，主验证全部是干净轨迹；人工注入只在独立校准/诊断中，不能混进自然成功率。校准期间不见测试题。只在四格都符合0.2–0.8、校准区间宽≤0.4、预计每格成功/失败各≥30并且核的相关行可识别时选格，顺序为离0.5最近、L更短、K更小；若不能选出矩形四格，不启动该模板。为控制费用，本模板另限定L≤128，每运行调用上限ceil(1.25L)≤160，总硬上限**256000调用**，输入/输出各≤**524288000 token**，并发≤2。若按先导规则选出的格需要L>128，必须另写预算，不能沿用此上限。实际格一经新校准选出须在正式请求前写明并冻结。', '',
      '拟议验证包括：同L的容量干预、同K的长度留出；训练/测试任务隔离；从K≤3校准向K=4的预测只有在高状态转移可识别、或明确登记并受检验的结构假设下才有资格作外推。不能用测试数据补核。必须另校准传输/服务失败这一端到端风险（本轮未知token请求不能视为自然动作错误）；完整核在可识别状态上作带调用计数的吸收递推，按独立题簇bootstrap/参数集合传播校准误差，保留0跳、多跳、槽溢出及旁路；同时报告点预测误差、95%预测区间覆盖、max(0.03,5SE)及H50两倍原判据，并与不使用K的基线比较。SE包含独立校准与测试的整题不确定性；不可识别项给界而非填值。负对照（状态依赖、残余历史、隐藏旁路）须单独登记真实动作协议；可预留每family×3类各200新题、L=32、干净/干预配对，共2400运行×40=96000调用，输入/输出各≤196608000 token。它们能否访问所需状态仍须先检查，不能用随机硬币伪造机制证据。补充校准8000+主验证256000+该负对照96000的总条件性上限为360000调用、输入/输出各737280000 token；不是本轮500调用授权内的工作。所有后续token上界均按服务遵守每请求2048输入/输出计算，是名义资源上界；客户端usage事后guard不能撤回违约消耗，未返回请求须另从服务端用量账单核对。硬可实施的客户端约束是调用计数、并发和请求参数。', '',
      '每格200题在p=0.2时预期40个少数结局，满足计数只是期望并非保证。2×2模板也不能自动认证历史效应≤0.03；最坏分布、题差∈[-1,1]时，Hoeffding 95%半宽≤0.03需要每family约8198个独立配对题，远超本轮6题。若以全零不一致事件做较窄的精确界，仍需独立新题和预先指定的等价检验，不能靠退化bootstrap宣称通过。', '',
      '本轮对研究决策的含义：输出/schema修复得到已返回响应的验证，完整端到端工程预测因传输失败未通过；审计要求的“有真实自然错误、可迁移状态、冻结容量留出预测”尚未交付。在增加调用前先排查传输/反代服务稳定性（当前异常标签不足以确诊根因），然后再投入自然错误与可识别性校准，或转向B′的外推不可识别/样本复杂度问题；不扩大经典可靠性公式的实证叙事。', '',
      '## 离线证据复核','',
      f"[ANALYSIS_PILOT2.json](ANALYSIS_PILOT2.json) 完成{a['audit_checks']}项逐响应、逐动作、逐提交复核；核对请求/返回模型、输出字段、逐次usage及汇总、严格原动作重评分、+7干预、每次实际状态转移、独立oracle前缀和最终账本、所有advance空历史、轨迹字节及冻结哈希。审计自洽不等于科学机制通过。",'',
      '文件散列、依赖版本及原始轨迹清单见 [ARTIFACTS_PILOT2.json](ARTIFACTS_PILOT2.json)。', '', '离线重现（不会调用模型）：', '```bash',
      '/home/liuyibo/venvs/theory/bin/python experiments/real/analyze_pilot2.py',
      '/home/liuyibo/venvs/theory/bin/python experiments/real/report_pilot2.py','```','']
    (ROOT/'PILOT2.md').write_text('\n'.join(lines))

if __name__=='__main__': main()
