"""从本轮重跑的实际输出生成中文验收记录。"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'y3d'
def read(path):return json.loads(path.read_text())
execution=read(ROOT/'Y3D_EXECUTION.json');assert execution['passed']
protocol=read(OUT/'protocol_rerun/results.json')
edges=read(OUT/'final_rerun/results.json')
r3=read(OUT/'r3_rerun/results.json')
faults=read(OUT/'audit_rerun/independent_faults/results.json')
false=read(OUT/'audit_rerun/false_confirmation/results.json')
clobber=next(r for r in faults['results'] if r['case']=='report_mutation_before_accept')
assert all(read(OUT/n)['passed'] for n in ['synthetic_y3d_tests.json','synthetic_rerun_tests.json','protocol_rerun/results.json','final_rerun/results.json','r3_rerun/results.json'])
assert not false['accepted'] and not clobber['accepted']
errors={}
for label,path in [('R3a',OUT/'audit_rerun/false_confirmation/formal'),('R3b',OUT/'audit_rerun/independent_faults/report_mutation_before_accept/formal')]:
    errors[label]=read(path/'launch_status.json')['publication_errors']
text=f'''# Y3D 测试与冻结验收记录

只做确定性执行协议修复与测试；没有正式抽样、训练/查询随机流生成或 LLM API 调用。Python 执行全部使用指定解释器及 systemd 用户 scope，MemoryMax=3G、MemorySwapMax=0、四项线程环境变量均为 2，并通过原 environment() 核验。阶段、实际返回码、CPU/墙钟与日志见 `Y3D_EXECUTION.json`。

|本轮重跑|实际结果|证据|
|---|---|---|
|原 59 项合成检查，独立两次|59/59，各次通过，标签与 Y3C 原结果逐项一致|`y3d/synthetic_y3d_tests.json`、`synthetic_rerun_tests.json`|
|完整协议及原持久化审计运行夹具|{len(protocol['checks'])} 项检查、{len(protocol['outcomes'])} 个提交结果；全部 {protocol['audit_fixtures']} 个旧夹具重放|`y3d/protocol_rerun/results.json`|
|最终发布边界与真实退出注入|{len(edges['checks'])} 项通过|`y3d/final_rerun/results.json`|
|独立审计故障及阳性对照|{len(faults['results'])} 个结果符合修复后的预期|`y3d/audit_rerun/independent_faults/results.json`|
|三个失败终点仅改候选判定|分析实际返回 {false['analyzer_returncode']}，拒绝提交及采信|`y3d/audit_rerun/false_confirmation/results.json`|
|独立重算、样本资格及覆盖分支|{len(r3['checks'])} 项通过|`y3d/r3_rerun/results.json`|

R3a：原三个候选判定均为 false，仅把 PENDING 的 candidate_decisions 改为 true，NPZ、事件、状态和全部 CSV 不变。新监管器拒绝，不产生 FINAL_ACCEPTED，verify_accepted() 拒绝。实际拒绝原因：`{errors['R3a']!r}`。

R3b：真实分析子进程写完 PENDING 完成标记后、正常返回前，只把 all_completed.csv 清成表头；实际返回 {clobber['analyzer_returncode']}，候选仍披露 {clobber['candidate_complete_cells']} 格、预期 2880 行，实际 {clobber['reported_data_rows']} 行。新监管器拒绝，不产生 FINAL_ACCEPTED，采信验证拒绝。实际拒绝原因：`{errors['R3b']!r}`。

原分析 main() 的真实子进程退出注入为 {json.dumps(edges['injections'],ensure_ascii=False)}。保留运行器真实无抽样 SIGKILL、异常状态、缺日志、非法事件、提交哈希不符、预算超限、禁止覆盖及发布后修改检查。运行器 SIGKILL 使用无抽样进程及已有状态组合，不声称运行过正式运行器。

完整计划阳性对照、合法部分软停与空软停都可以提交并验证；统计不满足资格时不通过。另有全部三个统计判定为 false 的合法执行阳性对照，最终凭据三个标志仍均为 false，避免以一律拒绝制造通过。

R3 分支测试还覆盖缺一行、等量重复行、端点/诊断样本数、端点资格、诊断等价判定和候选样本摘要。为隔离 verify_accepted() 的独立重算分支，三个测试会同步重签测试凭据里的相关哈希，再分别注入错误候选判定、覆盖或样本数；验证仍拒绝。这三个分支隔离测试不是单故障攻击模型，R3a/R3b 的单故障回归独立保留。

测试移植仅做版本环境名、隔离输出路径及修复后拒绝预期的映射。final_edges 改为读取本轮完整成功 protocol_rerun，历史“先前测试辅助程序失败已保存”的检查改为“本轮完整协议成功”；其余边界检查继续执行。历史 resume_protocol 是原失败测试的证据恢复辅助程序，本轮直接完整成功运行，无需执行该历史恢复操作。复制来的历史结果没有计作本轮通过证据。

本轮执行作用域在父进程退出前的 CPU 账本为 `{execution['scope_cpu_stat_before_parent_exit'].strip().replace(chr(10),'; ')}`。所有阶段实收 CPU 合计 {sum(r['cpu_seconds'] for r in execution['stages']):.6f} 秒，包含测试、真实子进程、辅助脚本及归档；不包含目录复制、前置快照、报告生成、冻结和核验。没有把历次修订成本累计伪称为低于原 200 秒准备额度；正式及准备预算配置均逐字节保留。

冻结程序将重新全量比对 `Y3D_BASELINE.sha256` 中既有非 log 文件，检查 15 组逐字节不变性及科学函数 AST，结果写 `Y3D_FREEZE_AUDIT.json`，源代码差异写 `Y3D_SOURCES.diff`。不向既有日志打开写入；环境会话记录器可能自动追加日志，因此不将非 log 比对声称为全部日志零变化。

本轮全部展开夹具封装于 `y3d/Y3D_test_fixtures.tar`；源代码、摘要、归档和本说明列入 `Y3D_FROZEN.sha256`。原复制历史归档保留。冻结后真实 `y3d/launch.py --verify-only` 的输出保存在 `Y3D_verify_only.log`，该后置执行日志不递归纳入冻结清单。启动命令和必要采信条件见 `Y3D_READY.md`。测试凭据不是正式结果，也不构成科学候选的新确认或否证。
'''
with (ROOT/'Y3D_TEST_REPORT.md').open('x') as f:f.write(text)
print(json.dumps(dict(report_written=True,protocol_checks=len(protocol['checks']),edge_checks=len(edges['checks']),R3_checks=len(r3['checks']))))
