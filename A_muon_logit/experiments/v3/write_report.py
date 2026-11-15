"""Render the frozen outcomes; never fits a model or changes a gate."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'experiments/v3'


def read(p):
    return json.loads(p.read_text())


def f(x, digits=2):
    return '不可识别' if x is None else f'{x:.{digits}f}'


def main():
    full = read(HERE/'full/summary.json')
    smoke = read(HERE/'smoke/summary.json')
    audit = read(HERE/'audit.json')
    assert audit['status'] == 'PASS'
    runs = [read(HERE/'full'/f'{rid}.json') for rid in full['run_ids']]
    strata = read(HERE/'aggregate.json')
    digest = hashlib.sha256((ROOT/'PREREG_V3_OFFLINE.md').read_bytes()).hexdigest()
    assert digest == full['prereg_sha256'] == (ROOT/'PREREG_V3_OFFLINE.sha256').read_text().split()[0]
    events = sum(r['metrics']['event_windows'] for r in runs)
    exactevents = sum(r['metrics']['exact_event_windows'] for r in runs)
    eligible = sum(r['metrics']['G2_identifiable'] for r in runs)
    finalerror = max(r['audit']['final_weight_error'] for r in runs)
    probeerror = max(r['audit']['probe_error'] for r in runs)
    direction = max(r['audit']['max_direction_op'] for r in runs)
    violation = max(max(r['audit']['one_step_max_violation'], r['audit']['multistep_max_violation']) for r in runs)
    total = sum(x['elapsed_seconds'] for x in (full, smoke, audit))
    memory = max(x['peak_RSS_bytes'] for x in (full, smoke, audit))
    lines = [
        '# V3 A′ 离线结果：保留条件覆盖定理，停止作为主会主线', '',
        '**结论：A′ 目前不能撑起主会论文。** 谱更新约束给出的覆盖包络在全部18条旧轨迹上通过数值审计，但少量见证的实用证书只在 isotropic、η=.01 通过登记门槛；下一窗口刷新时刻预测没有通过。低/中学习率没有足够覆盖失败事件，高学习率有事件时经验几何时钟明显劣于常数/仅间隔基线。按冻结止损，A 以条件理论和负结果技术报告收束，不通过增加头、候选数、窗口或更松阈值继续追认。', '',
        'P-V2-1 的38/72头空间正结果仍成立；本轮既没有推翻它，也没有以它代替动态预测。此判断只针对这里登记的 A′ 路线，不是证明所有见证方法都不可能。', '',
        '| 关卡 | 事前门槛（细节见预注册） | 本轮判定 |',
        '| --- | --- | --- |',
        '| G0 正确性 | 18/18重放、谱约束、一步/多步包络、0证书违反 | **通过，18/18** |',
        '| G1 少量且有用 | 各条件/η至少2/3 seed：H中位数≥5、H≥5比例≥.5、扫描≤.2、覆盖≥.99 | **整体失败；仅1/6层通过，3/18运行通过** |',
        f'| G2 刷新时刻 | 事件≥10/seed；对3基线各改善≥20%，事件MAE≤5，误报≤.1 | **0/6层；{eligible}/18可识别者均失败，其余不可识别** |',
        '| G3 更替结构 | 至少2/3 seed：历史/未来赢家更替率Spearman≥.3且几何风险AUROC≥.7 | **0/6层；AUROC全部不可识别，不能给通过** |', '',
        '## 数据、冻结和信息边界', '',
        f'预注册 [PREREG_V3_OFFLINE.md](PREREG_V3_OFFLINE.md) 在第一次冒烟前冻结，SHA256=`{digest}`；正式、审计及本报告均重新核验一致。研究者已知此前汇总，因此这是旧数据的提前冻结分析，不是新的确认实验。无联网、无LLM/API、无新训练。', '',
        '主集18条 muon0/base，condition∈{isotropic,anisotropic}、η∈{.01,.03,.1}、seed=11/23/47；全部72头。每头固定128×8=1024探针对，见证top4、ρ=.9。沿原float32方向重放全部600步，权重为float64。100..149仅用于起点150的历史50步赢家特征；评分从150开始。拟合起点150/175/200/225/250/275；测试300/325/…/575，每起点预测后25步，参数在300冻结。', '',
        '每次起点允许观察全部分数和q/k范数；之后预测仅使用该锚点与公开的谱更新上界。未来方向仅参与重放及事后审计，不作为预测输入。主事件T为top4首次不能达到完整最大绝对logit的.9；无事件编码26。此截尾编码MAE不是无限时域生存时间误差。窗口/头不当作独立种子。输入漂移恒0，不能声称检验了非零漂移理论。', '',
        '## G0：数学包络正确，但正确不等于有预测效用', '',
        f'最大最终权重误差 **{finalerror:.3g}**（门槛1e-5），最大原探针Mabs误差 **{probeerror:.3g}**（1e-4）；全部保存方向的最大谱范数 **{direction:.10f}**，低于1+1e-6的存储误差裕量。一步与测试起点1..25步包络的最大“真实偏差−半径”为 **{violation:.3g}≤1e-8**，证书违反为0。', '',
        f'独立审计重新哈希{audit["source_files"]}个原输入，复核{audit["independent_origin_rows"]}个拟合/测试origin/head的事件标签与证书/经验预测，及{audit["independent_policy_steps"]}个在线策略覆盖步，均通过。采用与主实现不同的向量化检查，未import主运行脚本。原输入未修改。', '',
        '另有2100个含非零漂移的随机谱几何检查、150个枚举区间选择检查、300个尖锐期限/预算及attention反例检查，以及5个独立LaTeX问题的结构检查。它们辅助排错，既不是训练外实证，也不是形式化证明。正文给出了数学证明；未运行Lean/Isabelle或外部证明服务，未声称生成编译PDF。', '',
        '## G1：只有低学习率单一条件提供有用保证', '',
        '下表每个三元组按seed11/23/47；H是测试origin/head的安全前缀中位数，扫描比例来自151..600在线策略。覆盖率全部为1，但扫描本身会使覆盖恢复，因此必须和扫描成本一起看。', '',
        '| 条件 | η | H中位数 | H≥5比例 | 扫描比例 | 分数评估成本比 | G1通过seed |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for st in strata:
        v = st['values']
        triple = lambda key, digits=2: ' / '.join(f(x, digits) for x in v[key])
        lines.append(f'| {st["condition"]} | {st["eta"]} | {triple("H_median",1)} | {triple("H_ge5_fraction")} | {triple("scan_fraction",3)} | {triple("score_cost_ratio",3)} | {st["gates"]["G1"]}/3 |')
    lines += ['',
        'η=.1多数起点连下一步都无法认证，在线策略需要96.5%–100%的步重新全扫描；加上每步4个见证评估，部分成本比超过1。isotropic/.01则为约15%–17%扫描，确有局部节省。异质包络存储全部1024对的锚点分数和q/k范数，并对遗漏集合计算上界，成本比只计分数评估，不是CPU/attention内核加速。', '',
        '理论证书只保证安全前缀；到期表示现有信息不再足以保证，不能解释为下一步一定发生覆盖失败。统一谱约束忽略共享Q/K更新对候选间的耦合，这正是保守性的来源之一。', '',
        '## G2：失败事件不足与真正预测失败分别报告', '',
        f'864个测试origin/head中有 **{events}** 个主事件、**{exactevents}** 个完整赢家出top4事件。η=.01/.03共576个测试窗口的主事件为0；另isotropic/.1/seed47仅1个。故13/18运行不满足“≥10事件”的识别门槛。低/中学习率通常几何模型、常数26、仅间隔模型的MAE都为0，这不是几何预测获胜；基线MAE=0时改善记缺失，不除以epsilon造改善。', '',
        '经验模型只把理论半径乘登记网格gamma，参数按前窗MAE选定；gamma<1没有路径保证。以下列出η=.1全部seed，MAE均为截尾T误差（步），误报只在无主事件窗口计算。', '',
        '| 条件 | seed | 主事件/48 | 几何MAE | 常数MAE | 更替率MAE | 仅间隔MAE | 事件MAE | 无事件误报率 | G2可识别 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in runs:
        if r['eta'] != .1:
            continue
        m = r['metrics']
        lines.append(f'| {r["condition"]} | {r["seed"]} | {m["event_windows"]} | {f(m["MAE"]["geometry"])} | {f(m["MAE"]["constant"])} | {f(m["MAE"]["rate"])} | {f(m["MAE"]["gap"])} | {f(m["event_MAE"])} | {f(m["false_alarm_rate"],3)} | {"是" if m["G2_identifiable"] else "否"} |')
    lines += ['',
        '五个事件充足的运行，几何模型对历史更替率基线有改善，但全部劣于常数和仅间隔基线；无事件误报率均为100%，远超≤.1门槛。即使isotropic/.1的两个seed事件MAE≤5，也不能跳过其余联合门槛。isotropic/.03/seed47在没有事件时仍误报7/48（.146），同样不属于成功。', '',
        '所以本轮没有得到“间隔与谱可达几何增量能预测下一窗刷新时刻”的支持；不能通过只比较较弱的赢家率基线改写结论。gamma模型失效也不否定所有可能的可达集模型，但冻结止损不允许据本轮再加参数/换窗口补票。', '',
        '## G3：部分赢家更替率可预测，尚不能转成覆盖风险', '',
        '| 条件 | η | 全路径赢家更替率11/23/47 | 前50步→后25步Spearman | ≥.3的seed | AUROC |',
        '| --- | --- | --- | --- | --- | --- |']
    for st in strata:
        v = st['values']
        rho = v['winner_rate_spearman']
        lines.append(f'| {st["condition"]} | {st["eta"]} | {" / ".join(f(x,3) for x in v["winner_change_rate"])} | {" / ".join(f(x,3) for x in rho)} | {sum(x is not None and x >= .3 for x in rho)}/3 | 全部不可识别 |')
    lines += ['',
        'η=.01的两条件和isotropic/.03，赢家更替率相关分量分别有至少2/3 seed达到.3；不能说完全没有更替结构。但低/中学习率没有主失败标签，高学习率多数T_cert恒为1（风险分数常数），另一个seed只有1个事件，所有AUROC按登记均不可识别。不能填成.5或给通过。精确赢家换人也不等于.9覆盖失效，不用T_exact补主事件的识别门槛。', '',
        '## 理论成果与论文判断', '',
        '[FORMULATION_V3.md](FORMULATION_V3.md) 给出：带输入漂移和衰减的多步谱可达包络；含锚点选择误差的区间覆盖与最小预算公式；统一速度下尖锐的间隔/刷新期限及条件扫描代价；可由固定输入attention实现的不可覆盖反例；输入网充分条件；自适应刷新下同时误差的并集保证。五个自足问题位于 [lemmas_v3](lemmas_v3/)。', '',
        '必要下界明确限定在独立分数/全扫描访问模型，谱外包络的预算不冒充真实共享Q/K可达集的必要预算。固定有限宇宙全保留当然可覆盖；紧致域有Lipschitz条件时近似有限网也可覆盖。这些限定防止“任何有限候选都不可能”的错误泛化。', '',
        '与QK-Clip的区别是扫描后见证覆盖和必要刷新，不是提出新裁剪器，也没有K2停用时长预测。与logit增长/谱最大间隔工作的区别是极值选择误差和刷新信息代价。所有文献差异仅据本地NOVELTY，版本/定理/优先权及kinetic data structures、safe screening、动态top-k等近邻方向均标为需联网核实，未完成新一轮查新。', '',
        '**主会判断为否。** 理论是可证明的条件性结果，但与经典Lipschitz筛选/事件触发方法重叠风险很高；实验没有形成新颖、可靠的跨窗预测，非零输入漂移与统计校准也没有外部验证。一个稳定条件上的节省不足以承接A原先的瞬态峰主张。保留为严谨技术报告/负结果，不恢复A主会主线，不扩大训练。本轮不把它误称为“同一V2反馈模型连续两次失败”；这是不同窄路线未通过自己预先写死的判据。', '',
        '## 执行、资源与复现', '',
        f'先1轨迹/1头冒烟（{smoke["elapsed_seconds"]:.2f}秒），再18轨迹/72头正式（{full["elapsed_seconds"]:.2f}秒），最后独立审计（{audit["elapsed_seconds"]:.2f}秒）；三阶段合计 **{total:.2f}秒**（{total/60:.2f}分钟），峰值RSS **{memory/1024**2:.1f} MiB**。报告生成另记 `experiments/v3/report_metadata.json`，计算总量远低于5400秒。这里报告的是计算进程墙钟，不把人工阅读/编码时长当训练时间。', '',
        '各阶段均通过 `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0`，指定 `/home/liuyibo/venvs/theory/bin/python`，OMP/OPENBLAS/MKL/NUMEXPR=4，NumPy CPU；不import torch、不创建GPU张量。scope及invocation ID保存在日志。未发生运行失败或结果后改门槛；冒烟前的历史窗口缓存修正在IMPLEMENTATION_LOG中记录。原文件36个哈希再次通过，预注册未修改。', '',
        '- [复现命令及数据字段](experiments/v3/README.md)、[实现记录](experiments/v3/IMPLEMENTATION_LOG.md)。',
        '- [正式关卡汇总](experiments/v3/full/summary.json)、[逐层未舍入指标](experiments/v3/aggregate.json)、[独立审计](experiments/v3/audit.json)。',
        '- 每条运行JSON和同名NPZ位于 [experiments/v3/full](experiments/v3/full/)，保存所有origin、候选、拟合网格、误报及缺失。',
        '- [冻结哈希](PREREG_V3_OFFLINE.sha256)、[最终产物校验](experiments/v3/ARTIFACTS.sha256)。', '']
    (ROOT/'RESULTS_V3_OFFLINE.md').write_text('\n'.join(lines))
    metadata = dict(elapsed_seconds=time.monotonic()-START,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        previous_stages_seconds=total, api_calls=0, prereg_sha256=digest)
    (HERE/'report_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
