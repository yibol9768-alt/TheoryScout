"""Offline final delivery; no model calls and no changes to frozen files."""
import collections
import json
from pathlib import Path

import holdout_b as h
import report_holdout_b_fixed as report


def prompt_audit(summary):
    checks = 0
    for result in summary['results']:
        trace = report.events(h.OUT / (result['id'] + '.jsonl'))
        task = trace[0]['task']
        responses = {x['call']: x['response'] for x in trace if x['event'] == 'response'}
        actions = {(x['step'], x['review']): x for x in trace if x['event'] == 'action'}
        for request in (x for x in trace if x['event'] == 'request'):
            step, review = request['step'], request['review']
            op = task['operations'][step]
            keys = h.d.required(task, op)
            start = task['prefix'][step]
            initial = dict(step=step, operation=op, required_writes=keys,
                           checkpoint=start, table=task['table'])
            expected = [{'role': 'system', 'content': h.d.COMMON + h.d.RULES['sheet']},
                        {'role': 'user', 'content': h.d.dump(initial)}]
            if review:
                previous = actions[(step, review - 1)]
                content = responses[previous['call']]['choices'][0]['message']['content']
                feedback = dict(mode='validator_triggered_review',
                                instruction='The applied transaction failed validation. Recompute and correct the required writes. State whether you detect a previous error.',
                                step=step, operation=op, required_writes=keys,
                                applied_state=previous['after'], checkpoint=start, table=task['table'])
                expected += [{'role': 'assistant', 'content': content or ''},
                             {'role': 'user', 'content': h.d.dump(feedback)}]
            assert request['messages'] == expected, 'prompt_replay_mismatch'
            checks += 1
    return checks


def pct(value):
    return '信息不足' if value is None else f'{value:.2%}'


def ci(values):
    return '无有效区间' if not values else f'[{pct(values[0])}, {pct(values[1])}]'


def main():
    frozen = h.verify()
    before = json.loads((h.ROOT / 'CONTINUATION_HOLDOUT_B.json').read_text())
    assert h.sha(h.ROOT / 'FREEZE_HOLDOUT_B.json') == before['freeze_manifest_sha256']
    assert h.sha(h.ROOT / 'report_holdout_b_fixed.py') == before['report_fixed_sha256']
    assert h.sha(h.ROOT.parent.parent / 'REVIEW_B.md') == before['review_sha256']
    for name, digest in before['smoke_sha256'].items():
        assert h.sha(h.OUT / name) == digest
    a = report.report()
    prompt_checks = prompt_audit(a['summary'])
    cal = json.loads((h.ROOT / 'PREDICTIONS_HOLDOUT_B.json').read_text())
    pre = json.loads((h.ROOT / 'PRECHECK_FORMAL_HOLDOUT_B.json').read_text())
    assert pre['calibration_replay_checks'] == cal['calibration_replay_checks']
    assert frozen == h.verify()
    evidence = dict(utc=h.d.now(), frozen_manifest_unchanged=True,
                    frozen_sources_and_calibration_unchanged=True, smoke_traces_unchanged=True,
                    disclosed_report_amendment_unchanged=True, prompt_replay_checks=prompt_checks,
                    existing_independent_checks=a['checks'], calibration_replay_checks=pre['calibration_replay_checks'])
    h.write(h.ROOT / 'FINAL_AUDIT_HOLDOUT_B.json', evidence)
    lines = ['# HOLDOUT_B 正式结果', '', '**' + a['conclusion'] + '**', '',
             '本次按冻结的方案 (b) 完成长度留出检验；两道冒烟题均保留在原分母，没有替换、补题或重新拟合。' if a['G2'] else
             '本次按冻结停止规则保留全部计划槽及两道冒烟题；未完成部分不填补。', '',
             '|长度|正式终局/计划|成功|观测成功率|冻结主预测|绝对偏差|冻结97.5%预测成功数|G3|',
             '|---|---|---|---|---|---|---|---|']
    if a['summary']['halt'] == 'quota_unavailable':
        lines[5:5] = [f"第{a['summary']['calls']}次HTTP尝试边界的额度检查失败，按冻结规则立即停止新请求。40个计划槽中，26槽进入执行，20题获得模型终局，5题服务重试耗尽、1题额度守卫停止，14题未执行。因此G2未通过，G3不可判；两道冒烟失败已计入模型终局，绝不删除。", '']
    for length, g in a['groups'].items():
        p = g['prediction']
        difference = abs(g['observed'] - p['main']) if g['observed'] is not None else None
        lines.append(f"|{length}|{g['terminal_known']}/20|{g['successes']}|{pct(g['observed'])}|{pct(p['main'])}|{pct(difference)}|{p['predictive']['counts']}|{'通过' if g['G3'] else '不通过' if g['G2'] else '证据不足'}|")
    lines += ['', '## 冻结门槛逐条核对', '',
              f"- **G0：通过。** 冻结文件和全部校准输入SHA256一致；50题CALIB1再次独立重放{pre['calibration_replay_checks']}项通过，283个自然错误≥30；冻结整题bootstrap有效{cal['bootstrap_valid']}/10000≥9900。"]
    for length, g in a['groups'].items():
        interval = g['prediction']['calibration95']
        lines.append(f"  L={length}预测的校准95%区间为{ci(interval)}，宽度{pct(interval[1]-interval[0])}≤40%。")
    lines += [f"- **G1：{'通过' if a['G1'] else '未通过'}。** 两道冒烟均有可评分动作。人工审阅人手算核对sheet_d0_5260001第3步F1=42、F2=543；用户明确转述审阅通过并同意继续，已据此建立SMOKE_APPROVED.json，未将代理审计当作人工审阅。",
              f"- **G2：{'通过' if a['G2'] else '未通过，正式结论为证据不足'}。** 终局必须为success或unrepaired_after_2_reviews；各长度终局、缺失和成功率界见下表。",
              f"- **G3：{'通过' if a['G3'] else '未通过' if a['G2'] else '不可判'}。** 两个长度均须满足绝对偏差≤0.20且成功数在各自冻结中央97.5%预测区间内，端点包含；具体数值见首表。",
              f"- **G4：{'触发局部迁移旗标' if a['G4_drift'] else '未触发局部迁移旗标'}。** 判定同时要求绝对差>0.10及两份整题簇95%区间不相交；任一必要分母<20标记信息不足。下表同时给出校准与留出区间。",
              f"- **G5：通过已发生部分的完整审计。** HTTP {a['summary']['calls']}/2000，峰值并发{a['peak_concurrency']}，唯一返回模型{a['models']}；{a['checks']}项独立检查及额外{prompt_checks}项完整提示重建一致。实际写入、独立数组oracle、评分和终局均重放通过。两道冒烟原始轨迹及所有冻件哈希保持不变。", '',
              '|长度|终局状态计数|缺失|计划分母成功率最坏/最好界|成功率精确95%区间|',
              '|---|---|---|---|---|']
    for length, g in a['groups'].items():
        lines.append(f"|{length}|{g['statuses']}|{g['unknown']}|{ci(g['success_bounds'])}|{ci(g['task_success_exact95'])}|")
    for length, g in a['groups'].items():
        low, high = g['prediction']['predictive']['counts']
        possible = [g['successes'], g['successes'] + g['unknown']]
        robust = possible[1] < low or possible[0] > high
        lines.append(f"\nL={length}将全部缺失任意补为成功/失败时，可能成功数为{possible}，冻结预测计数区间为{[low, high]}；稳健预测不符={'是' if robust else '否（范围仍相交，不能认证通过）' if not g['G2'] else '否'}。")
    lines += ['', '两次复查后仍错属于模型终局失败，保留在分母；服务耗尽、硬限或安全停止属于缺失，不补零、不补题。', '',
              '## 局部参数迁移', '',
              '校准为50题整题簇bootstrap；留出分别按各长度已进入执行的题为簇bootstrap，均10000次。未执行槽没有局部率数据。各率仅使用实际可观测暴露；未观测尾部不补零。', '',
              '|长度|量|CALIB1 分子/分母、点估计|CALIB1 簇95%|留出 分子/分母、点估计|留出 簇95%|冻结诊断|',
              '|---|---|---|---|---|---|---|']
    names = {'error': '自然错率p', 'q_fb': '首反馈自报发现q_fb', 'r1': '首修成功率r1',
             'r2': '二修条件成功率r2', 'reerror_next': '修好后下一步再错率',
             'next_clean': '正确推进后下一步错率', 'before': '首次错前风险', 'after': '首次错后风险'}
    for length, g in a['groups'].items():
        for name, r in g['rates'].items():
            c = cal['rates'][name]
            diagnostic = '信息不足' if not r['sufficient'] else '漂移旗标' if r['drift'] else '未触发旗标'
            lines.append(f"|{length}|{names[name]}|{c['e']}/{c['n']}，{pct(c['point'])}|{ci(c['cluster95'])}|{r['e']}/{r['n']}，{pct(r['point'])}|{ci(r.get('cluster95'))}|{diagnostic}|")
    lines += ['', 'q_fb来自明确失败反馈后的自报，不能解释为自然检测率，也不额外乘入修复概率。未触发旗标不等于证明稳定；修复、再错和历史分组存在条件化。服务缺失还使局部率存在删失：例如L=64共81次自然错误，但只有80次首修获得可评分响应，未返回的首修不能补记为修好或没修好。两个长度的二修分母均<20，机制迁移信息不足。', '',
              '|长度|位置（0起）|推进暴露|错误|错误率|', '|---|---|---|---|---|']
    for length, g in a['groups'].items():
        for block in range(int(length) // 16):
            n = g['counts'].get(f'block_{block}_n', 0)
            e = g['counts'].get(f'block_{block}_e', 0)
            lines.append(f"|{length}|{16*block}–{16*block+15}|{n}|{e}|{pct(e/n if n else None)}|")
    lines += ['', '后续位置只包含存活到该位置的题，不能据区块差异声称位置因果效应或历史污染。检测×修好联合表、复查错格上跳/零跳/下降及具体跳幅原样保留在ANALYSIS_HOLDOUT_B.json中各组counts。', '',
              '## 额度、服务与工作成本', '',
              f"HTTP尝试{a['summary']['calls']}（含原冒烟15次及全部重试）；有效响应{a['responses']}，服务失败{a['service_errors']}，重试{a['retries']}，600秒冷却{a['cooldowns']}。全局停止原因：{a['summary']['halt']}；最终阶段：{a['summary']['stage']}。服务失败的无返回token耗用未知，绝不算模型动作错误。",
              '', '|累计尝试边界|五小时额度|周额度|', '|---|---|---|']
    for q in a['quotas']:
        lines.append(f"|{q['calls']}|{q['five_hour']}%|{q['week']}%|")
    if a['summary']['halt'] == 'quota_unavailable':
        lines += [f"|{a['summary']['calls']}|额度检查失败，无有效数值|按冻结守卫全局停止|", '',
                  '最后一次有效额度值不能替代失败边界的检查；不能据此前额度较低继续发请求。本次未绕过停止守卫，也未重启或补跑未执行题。']
    lines += ['', '启动前及每100次边界均由冻结执行器调用指定plus_quota.py；任一额度≥80%或检查失败即停新请求。服务退避沿用同一动作最多4重试、10/30/60/120秒，连续5服务失败冷却600秒；实际是否触发见上文计数。', '',
              f"已知usage：{a['usage']}。reasoning已包含在completion，不重复计费；不推算未观测token或美元成本。所有计算通过MemoryMax=2G、MemorySwapMax=0的systemd用户scope运行。SOLKEY仅从环境读入内存，未打印或写文件。", '',
              '|任务|L|终局|完成步/推进调用|自然错|复查调用|实际动作W|成功且W≤5L|',
              '|---|---|---|---|---|---|---|---|']
    for row in sorted(a['rows'], key=lambda r: (r['seed'] % 10000, r['length'])):
        lines.append(f"|{row['id']}|{row['length']}|{row['status']}|{row['steps']}/{row['advances']}|{row['seed_errors']}|{row['repair_attempts']}|{row['W']}|{row['within_5L_and_success']}|")
    seen = {row['seed'] for row in a['rows']}
    for row in h.schedule():
        if row['seed'] not in seen:
            lines.append(f"|sheet_d0_{row['seed']}|{row['length']}|未执行（冻结守卫停止）|0/0|未观测|0|0|未判定|")
    lines += ['', 'W=推进动作+复查动作，服务失败不算动作；提示、日志存储另计。两次复查是时间预算，不是义务容量K。', '',
              '## 冻结对照与理论边界', '',
              '主模型固定为S(L)=(1−19/1279)^L；q_fb不额外相乘。没有以本轮终点或中间结果重新拟合任何预测。', '',
              '|冻结模型|L=16 Brier / log loss|L=64 Brier / log loss|两长度等权 Brier / log loss|',
              '|---|---|---|---|']
    for name, rows in a['scores'].items():
        per = {x['length']: f"{x['brier']:.6f} / {x['log_loss']:.6f}" for x in rows}
        equal = f"{sum(x['brier'] for x in rows)/len(rows):.6f} / {sum(x['log_loss'] for x in rows)/len(rows):.6f}"
        lines.append(f"|{name}|{per.get(16, '无数据')}|{per.get(64, '无数据')}|{equal}|")
    lines += ['', 'no_repair是另一协议的反事实；terminal_only由旧31/50整题终点外推；history_two_state使用冻结首次错前/后参数。分数仅作事前对照，不据此改选主模型；接近分数不能区分机制。存在服务缺失时，分数只覆盖已观测的模型终局，可能有选择偏差，不能当作完整留出评分。', '',
              '|旧认证门槛/主张|本轮判断|', '|---|---|',
              '|成功与失败各≥30|每长度仅20题，不可能满足；没有通过旧科学认证。|',
              '|week1 max(0.03,5SE)拒绝阈值|保留为辅助项，不能替代本轮G3或据此放宽门槛。|']
    for length, g in a['groups'].items():
        lines.append(f"|L={length}旧概率拒绝辅助项|触发={g['legacy_probability_rejection']}。|")
    lines += ['|H50两倍、容量K收益、历史残余0.03|本轮未检验，不能认证。|',
              '|自然q、rho、Gamma异质性、原生±1生灭链/容量相图|本轮不识别，不能由长度预测推出。|', '',
              '本轮判断仅适用于sheet/0、knob=2、明确失败反馈、最多两次复查的固定协议，以及本次20题/长度的粗精度。预测不符时应保留否决结果；预测相容也不构成恒定风险、再生、一维充分状态或B原理论完整经验桥梁的证明。', '',
              '## 可核查材料', '',
              '- [冻结预注册](experiments/real/PREREG_HOLDOUT_B.md)、[冻结清单](experiments/real/FREEZE_HOLDOUT_B.json)、[冻结预测](experiments/real/PREDICTIONS_HOLDOUT_B.json)。',
              '- [完整分析](experiments/real/ANALYSIS_HOLDOUT_B.json)、[最终审计](experiments/real/FINAL_AUDIT_HOLDOUT_B.json)、[运行日志](experiments/real/RUN_FORMAL_HOLDOUT_B.log)、[请求计数及额度日志](experiments/real/holdout_b_real/budget.jsonl)。',
              '- [人工放行记录](experiments/real/holdout_b_real/SMOKE_APPROVED.json)、[续跑前哈希记录](experiments/real/CONTINUATION_HOLDOUT_B.json)。每题完整提示、原始响应、实际写入、期望状态和判分均在holdout_b_real同名jsonl。',
              '- 已披露的报告格式勘误沿用[REPORTING_ERRATUM_HOLDOUT_B.md](experiments/real/REPORTING_ERRATUM_HOLDOUT_B.md)及独立fixed报告器；冻结原报告器未回写。本交付脚本只离线审计和呈现，不改评分、参数或门槛。', '',
              '离线复现：`systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 python3 experiments/real/finalize_holdout_b_results.py`。']
    destination = h.ROOT.parent.parent / 'RESULTS_HOLDOUT_B.md'
    destination.write_text('\n'.join(lines) + '\n')
    print('RESULTS', destination.name, 'PROMPT_CHECKS', prompt_checks)


if __name__ == '__main__':
    main()
