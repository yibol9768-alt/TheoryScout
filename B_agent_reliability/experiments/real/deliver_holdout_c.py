"""Offline final delivery, retaining every frozen implementation and raw trace."""
import json

import holdout_c as h
import finalize_holdout_c as finalizer
import supplementary_holdout_c as supplementary


def number(x):
    return '不可估计' if x is None else f'{x:.4f}'


def interval(x):
    return '不可估计' if x is None else '[' + ', '.join(number(v) for v in x) + ']'


def deliver():
    h.verify()
    amendment = json.loads((h.ROOT / 'FREEZE_AMENDMENT_408_HOLDOUT_C.json').read_text())
    assert h.sha(h.ROOT / 'FREEZE_HOLDOUT_C.json') == amendment['original_freeze_sha256']
    assert all(h.sha(h.ROOT / name) == digest for name, digest in amendment['sha256'].items())
    summary = json.loads((h.OUT / 'summary.json').read_text())
    assert summary['stage'] == 'finished' and summary['halt'] is None
    assert len(summary['results']) == 40
    assert all(row['status'] in ('success', 'unrepaired_after_2_reviews') for row in summary['results'])
    replay_checks = supplementary.replay(h.OUT)
    obj = finalizer.finalize()
    assert obj['G2'] is True and obj['correction_operational_audit'] is True
    cal = json.loads((h.ROOT / 'PREDICTIONS_HOLDOUT_C.json').read_text())
    lines = ['', '## 完成后补充核验与数值', '',
             f'全部40题的完整提示及响应确定性重放通过，共{replay_checks}个模型动作；'
             '原始冻结、校准输入、B报告及B原始汇总、408修正代码的SHA256再次匹配。', '',
             'G5严格确认性门槛不通过：408实施偏差和修正哈希时间顺序限制保留。'
             '修正后操作审计通过不等于原冻结协议无偏差；G3结论只按预定数值门槛解释。', '',
             '|长度|成功/计划|成功率|精确95%区间|冻结主预测|绝对差|',
             '|---|---|---|---|---|---|']
    for length, group in obj['groups'].items():
        lines.append(f"|{length}|{group['successes']}/20|{number(group['observed'])}|"
                     f"{interval(group['task_success_exact95'])}|{number(group['prediction']['main'])}|"
                     f"{number(abs(group['observed']-group['prediction']['main']))}|")
    lines += ['', '各率均按题簇bootstrap；以下并列校准与留出95%区间。'
              '分母小于20按原规则为信息不足，即使区间看起来很窄也不能认证稳定。', '',
              '|长度|率|CALIB1点估计及簇95%|C点估计及簇95%|有效重采样数|判定|',
              '|---|---|---|---|---|---|']
    for length, group in obj['groups'].items():
        for name, rate in group['rates'].items():
            original = cal['rates'][name]
            verdict = '漂移旗标' if rate['drift'] else '信息不足' if not rate['sufficient'] else '未触发旗标'
            lines.append(f"|{length}|{name}|{number(original['point'])} {interval(original['cluster95'])}|"
                         f"{number(rate['point'])} {interval(rate.get('cluster95'))}|{rate.get('valid', 0)}|{verdict}|")
    labels = {'main': '主预测', 'no_repair': '无修复（不同协议反事实）',
              'terminal_only': '仅旧终点', 'history_two_state': '首次错两状态'}
    lines += ['', '冻结对照按题评分，各长度等权；不据此重新选择主模型或声称机制优越。', '',
              '|预测|L16 Brier / log loss|L64 Brier / log loss|等权Brier|等权log loss|',
              '|---|---|---|---|---|']
    equal_scores = {}
    for name, scores in obj['scores'].items():
        assert {row['length'] for row in scores} == {16, 64}
        by_length = {row['length']: row for row in scores}
        mean = {key: sum(row[key] for row in scores)/2 for key in ('brier', 'log_loss')}
        equal_scores[name] = mean
        cells = [f"{number(by_length[length]['brier'])} / {number(by_length[length]['log_loss'])}" for length in (16, 64)]
        lines.append(f"|{labels[name]}|{cells[0]}|{cells[1]}|{number(mean['brier'])}|{number(mean['log_loss'])}|")
    lines += ['', '|长度|W总数|每题W均值|W范围|成功且W≤5L|', '|---|---|---|---|---|']
    for length in (16, 64):
        rows = [row for row in obj['rows'] if row['length'] == length]
        work = [row['W'] for row in rows]
        lines.append(f"|{length}|{sum(work)}|{sum(work)/len(work):.2f}|{min(work)}–{max(work)}|"
                     f"{sum(row['within_5L_and_success'] for row in rows)}/20|")
    lines += ['', '按位置的自然错误/实际推进暴露（后段有幸存者选择）：', '',
              '|长度|0–15|16–31|32–47|48–63|', '|---|---|---|---|---|']
    for length, group in obj['groups'].items():
        counts = group['counts']
        cells = [f"{counts.get('block_'+str(i)+'_e', 0)}/{counts.get('block_'+str(i)+'_n', 0)}" for i in range(4)]
        lines.append('|' + '|'.join([length] + cells) + '|')
    lines += ['', '408原始失败为1次额外服务尝试，未计模型动作；'
              '原状态恢复重发单列，冻结报告的attempt>0重试数不包含这次特殊恢复。'
              '全部失败服务尝试的token未知，不用有效响应usage推算。', '',
              'G1放行依据为HUMAN_RELEASE_HOLDOUT_C.md与本轮用户明确确认；'
              'SMOKE_APPROVED.json为这份既有授权的运行器门禁记录，没有新增自动审阅冒充人工审阅。', '',
              '完整复现（离线，不发送模型请求）：', '',
              '```bash',
              'systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 python3 -B experiments/real/deliver_holdout_c.py',
              '```', '']
    restart = h.ROOT / 'RESTART_AUDIT_HOLDOUT_C.json'
    if restart.exists():
        recovery = json.loads(restart.read_text())
        assert len(recovery['terminal']) == 3
        assert recovery['calls'] == 166 and recovery['successful'] == 164 and recovery['failed'] == 2
        archive = h.OUT / 'RESTART_ORIGINAL_20261106'
        assert all(h.sha(archive/name) == digest for name, digest in recovery['original_sha256'].items())
        assert h.sha(h.ROOT/'restart_holdout_c.py') == recovery['adapter_sha256']
        assert supplementary.replay(h.OUT) == replay_checks
        lines += ['', '## 机器重启中断与原状态恢复', '',
                  '正式进程因机器重启中断。原始日志中call166已于2026-11-06 16:36:56 UTC记录transport失败，'
                  '随后额度查询失败并每300秒暂停；最后等待记录为17:07:22 UTC。恢复前共有166次服务尝试、'
                  '164个成功响应、2次服务失败（含原408）。重启没有另记模型失败或虚构token。', '',
                  f"恢复审计时间为{recovery['utc']}。原始轨迹确认3题已有模型终局，全部保留、不重跑；"
                  '第4题sheet_d0_7740002的65个已记录动作在内存确定性重放，逐条核对提示、响应、实际写入、'
                  '状态与计数，不重复请求或落盘旧动作。只在step50/review0恢复原call166的同一序列化请求，'
                  '首次重发attempt递增；真实额度合格后继续。', '',
                  '重建前原始summary/budget/四题轨迹逐字节归档至holdout_c_real/RESTART_ORIGINAL_20261106/，'
                  '其哈希及恢复适配器哈希记录于RESTART_AUDIT_HOLDOUT_C.json；最终复核全部匹配。'
                  '原始冻结和408修正冻结均未改动，40题槽位、种子、参数、终局评分和报告门槛未变。'
                  '全部40题完成后的独立oracle审计和完整提示重放再次通过；网络失败始终按原规则暂停等待。', '']
        obj['restart_recovery'] = recovery
    restart2 = h.ROOT / 'RESTART2_AUDIT_HOLDOUT_C.json'
    if restart2.exists():
        recovery2 = json.loads(restart2.read_text())
        archive2 = h.OUT / 'RESTART2_ORIGINAL_20261106'
        assert all(h.sha(archive2/name) == digest for name, digest in recovery2['original_sha256'].items())
        assert h.sha(h.ROOT/'restart2_holdout_c.py') == recovery2['adapter_sha256']
        assert len(recovery2['terminal']) == 3
        assert (recovery2['calls'], recovery2['successful'], recovery2['failed']) == (168, 164, 4)
        lines += ['', '## 会话结束导致的第二次中断与脱离会话续跑', '',
                  '首次恢复运行器是上一Codex会话的子进程，随会话结束被连带终止；'
                  '当时本机反代未监听，call167及随后call168均已持久化为transport失败，'
                  '成功返回仍为164次。中断本身未另计模型失败或请求。', '',
                  f"第二次恢复审计时间为{recovery2['utc']}；启动时间、用户scope及进程隔离验证见"
                  'ENGINEERING_HOLDOUT_C.md与RUN_ENV_RESTART2_FORMAL_HOLDOUT_C.txt。'
                  '以setsid/nohup和独立systemd用户scope脱离会话，MemoryMax=2GiB、MemorySwapMax=0。'
                  '保留3题终局，第4题65个动作仅在内存重放，从step50/review0原样续跑；'
                  '请求与call168正文相同，attempt递增，既有成功响应不重复请求。'
                  '第二入口仅替换归档与审计文件名，原恢复脚本和首次审计保持原字节。'
                  '第二次恢复前数据及哈希分别保存于RESTART2_ORIGINAL_20261106/和'
                  'RESTART2_AUDIT_HOLDOUT_C.json，最终复核匹配。', '']
        obj['session_restart_recovery'] = recovery2
    obj['full_prompt_replay_checks'] = replay_checks
    obj['scores_equal_length'] = equal_scores
    obj['original_freeze_and_amendment_hashes_verified'] = True
    h.write(h.ROOT / 'ANALYSIS_HOLDOUT_C.json', obj)
    for path in (h.ROOT / 'HOLDOUT_C.md', h.ROOT.parent.parent / 'RESULTS_HOLDOUT_C.md'):
        text = path.read_text()
        first_conclusion_end = text.index('**', text.index('**') + 2) + 2
        text = text[:first_conclusion_end] + '\n\n**解释限制：G5存在HTTP408实施偏差及修正哈希时间顺序限制；本批不能声称无偏差的确认性认证。**' + text[first_conclusion_end:]
        text = text.replace('python3 experiments/real/report_holdout_c.py', 'python3 -B experiments/real/deliver_holdout_c.py')
        path.write_text(text + '\n'.join(lines))
    manifest = {str(path.relative_to(h.ROOT.parent.parent)): h.sha(path) for path in
                [h.ROOT / 'deliver_holdout_c.py', h.ROOT / 'finalize_holdout_c.py',
                 h.OUT / 'summary.json', h.OUT / 'budget.jsonl', h.OUT / 'SMOKE_APPROVED.json',
                 h.ROOT / 'HUMAN_RELEASE_HOLDOUT_C.md', h.ROOT / 'ANALYSIS_HOLDOUT_C.json',
                 h.ROOT.parent.parent / 'RESULTS_HOLDOUT_C.md']}
    if restart.exists():
        for path in (restart,h.ROOT/'restart_holdout_c.py',h.ROOT/'ENGINEERING_HOLDOUT_C.md'):
            manifest[str(path.relative_to(h.ROOT.parent.parent))] = h.sha(path)
    if restart2.exists():
        for path in (restart2, h.ROOT/'restart2_holdout_c.py',
                     h.ROOT/'RUN_ENV_RESTART2_FORMAL_HOLDOUT_C.txt'):
            manifest[str(path.relative_to(h.ROOT.parent.parent))] = h.sha(path)
    h.write(h.ROOT / 'DELIVERY_HOLDOUT_C.json', dict(utc=h.d.now(),sha256=manifest,
            full_prompt_replay_checks=replay_checks,completed=40,G2=True,G3=obj['G3'],
            G5_strict=False,correction_operational_audit=True))
    print('DELIVERY: 40 terminal tasks; full replay and corrected operational audit passed.')


if __name__ == '__main__':
    deliver()
