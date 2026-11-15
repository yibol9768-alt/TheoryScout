"""Render the Chinese evidence report without changing any frozen decision."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from util import HERE, ROOT, DATA, sha, save, verify_prereg, metadata


def pct(x,d=1):
    return '不可识别' if x is None else f'{100*x:.{d}f}%'


def num(x,d=3):
    return '缺失' if x is None else f'{x:.{d}f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    start=time.monotonic();checksum=verify_prereg();out=HERE/'full'
    stats=json.loads((out/'statistics.json').read_text())
    mc=json.loads((out/'calibration_summary.json').read_text())
    audit=json.loads((out/'audit.json').read_text())
    supplement=json.loads((out/'dense_ce_supplement.json').read_text())
    replaymanifest=json.loads((out/'replay_manifest.json').read_text())
    feedback=[json.loads(p.read_text()) for p in sorted((out/'feedback').glob('*.json'))]
    replays=[json.loads(p.read_text()) for p in sorted((out/'replays').glob('*.json'))]
    candidates=[c for r in feedback for c in r['candidates']]
    identifiable=[c for c in candidates if c['enough_variation']]
    syncfit=sum(c['positive_at_freeze'] and c['synchronization']['fit']['gate'] for c in candidates)
    synchold=sum(c['positive_at_freeze'] and c['synchronization']['holdout']['gate'] for c in candidates)
    syncboth=sum(c['positive_at_freeze'] and c['synchronization']['fit']['gate'] and c['synchronization']['holdout']['gate'] for c in candidates)
    con=sum(c['contraction']['gate'] for c in candidates)
    rho=[c['synchronization']['holdout']['innovation_QK_correlation'] for c in candidates]
    ratios=[v for r in feedback for v in r['sigma_over_block_noise_RMS_SD'] if v is not None]
    fwer=[r['independent_FWER'] for r in mc['runs']]
    power2=[r['power']['0.2'] for r in mc['runs']];power4=[r['power']['0.4'] for r in mc['runs']]
    threshold=[r['critical_95pct'] for r in mc['runs']]
    varratio=[v for r in feedback for v in r['residual_variance_max_min_ratio']]
    spatial=stats['spatial'];opt=stats['optimizer']
    allcp=[c for r in replays if r['primary'] for c in r['checkpoints']]
    meanrows=[]
    for s in stats['feedback']:
        v=s['groups']['varying_candidates'];a=s['groups']['all_candidates']
        runs=sorted([r for r in feedback if r['condition']==s['condition'] and r['eta']==s['eta']],key=lambda r:r['seed'])
        persistence=[]
        for r in runs:
            diagnostic=r['varying_candidates']
            persistence.append(1-diagnostic['window_MSE_feedback']/diagnostic['window_MSE_persistence'] if diagnostic.get('identifiable') and diagnostic['window_MSE_persistence']>0 else None)
        meanrows.append([s['condition'],s['eta'],' / '.join(pct(i) for i in v['improvements']),
                         ' / '.join(pct(i) for i in persistence),
                         f'{v["mean_pass_seeds"]}/3',f'{v["transition_pass_seeds"]}/3',
                         f'{a["mean_pass_seeds"]}/3'])
    optrows=[]
    for s in opt['strata']:
        sig=s['modes']['matched_fro']['signsgd'];adam=s['modes']['matched_fro']['adamw']
        optrows.append([s['condition'],s['eta'],s['weak_masked_candidates'],num(sig['median_ratio_masked']),
                        num(adam['median_ratio_masked']),f'{sig["same_direction_seeds"]}/3，{adam["same_direction_seeds"]}/3',
                        f'{sig["predictor_positive_seeds"]}/3，{adam["predictor_positive_seeds"]}/3',
                        '通过' if s['gate'] else '失败'])
    cerows=[]
    for maincount in stats['CE']['counts']:
        extra=next(c for c in supplement['counts'] if c['comparator']==maincount['comparator'] and c['target']==maincount['target'])
        cerows.append([maincount['comparator'],maincount['target'],f'{maincount["matched_pairs"]}/18',
                       num(maincount['median_Mabs_ratio_muon_other']),f'{extra["matched_pairs"]}/18',num(extra['median_Mabs_ratio_muon_other'])])
    mctable=[]
    for r in mc['runs']:
        run=next(s for s in feedback if s['run_id']==r['run_id'])
        mctable.append([run['condition'],run['eta'],run['seed'],num(r['observed_T']),num(r['critical_95pct']),
                        pct(r['independent_FWER'],2),pct(r['power']['0.2']),pct(r['power']['0.4']),
                        num(r['approximate_Gaussian_p_Holm']),num(r['exchangeability_sensitivity_p_Holm'])])
    geotable=[]
    for name in ('muon0','signsgd','adamw'):
        original=[c['comparisons']['original_k0'][name]['amplitude'] for c in allcp]
        matched=[c['comparisons']['matched_fro'][name]['amplitude'] for c in allcp]
        delta=[c['comparisons']['matched_fro'][name]['abs_logit_delta'] for c in allcp]
        geotable.append([name,num(float(np.median(original))),num(float(np.median(matched))),num(float(np.median(delta)),4)])
    lengthrows=[]
    for optname in ('muon0','muon09','adamw','signsgd'):
        for eta in (.01,.03,.1):
            runs=[r for r in replays if r['optimizer']==optname and r['eta']==eta]
            selected=[next(s for s in r['same_step'] if s['step']==300) for r in runs]
            losses=[]
            for r in runs:
                original=json.loads((DATA/'runs'/f'{r["run_id"]}.json').read_text())
                losses.append(next(v['ce'] for v in original['train'] if v['step']==300))
            lengthrows.append([optname,eta,num(float(np.median(losses))),
                               num(float(np.median([s['Mabs'] for s in selected]))),
                               num(float(np.median([s['cumulative_actual_length'] for s in selected]))),
                               num(float(np.median([s['cumulative_direction_length'] for s in selected])))])
    counts=[]
    for condition in ('isotropic','anisotropic'):
        for eta in (.01,.03,.1):
            rows=[r for r in spatial['rows'] if r['condition']==condition and r['eta']==eta]
            counts.append([condition,eta,sum(r['spatial_gate'] for r in rows),sum(r['falling'] for r in rows),
                           sum(r['falling'] and r['spatial_gate'] for r in rows)])
    resources=[replaymanifest['resources'],stats['resources'],mc['resources'],supplement['resources'],audit['resources']]
    elapsed=sum(r['elapsed_seconds'] for r in resources)
    peak=max(r['peak_RSS_bytes'] for r in resources)
    overall=(datetime.now(timezone.utc)-datetime(2026,11,6,7,47,52,tzinfo=timezone.utc)).total_seconds()
    # A contradiction with the report's declared outcome must stop publication.
    assert not opt['gate']
    assert all(not r['closure_identified'] for r in feedback)
    assert audit['status']=='PASS'
    text=f'''# A 第三轮离线结果：修改模型，封存当前 V2 机制桥梁的负结果

**结论：不继续把 V2 同步标量递推作为 E1 瞬态峰或 Muon 特异性的解释；A 需要修改模型，当前桥梁保留为负结果。** 第 150 步固定见证集合的空间正结果仍成立（38/72 头）。但只有 eta=.01 的单步条件均值和相干下降得到跨种子支持；同步性、完整状态闭合、以及匹配更新范数后的特异性门槛没有组成证据链。P-V2-5 按登记跳过，未模拟或宣称未来极值/裁剪风险预测成功。

建议 A 不恢复为主会主线，也不扩大训练。可保留的正证据是“真实八 key 梯度残差对下一步方向均值有部分预测力”；可以继续的窄方向是 §8 的 A′ 候选覆盖/刷新问题，而不是继续给 (3.4) 加隐藏阶段变量。这不是对所有 Muon 机制的否定。

| 关卡 | 冻结门槛 | 本轮判断 |
| --- | --- | --- |
| P-V2-1（既有） | C>=.8、R²>=.7、变化 RMS>=.1，原跨种子/学习率条件 | 沿用通过；回落头中仅 {spatial['falling_spatial_pass']}/{spatial['falling_heads']} 通过空间门槛 |
| P-V2-2 | 均值 MSE 改善>=20%、至少2/3 seed；相干预测下降>=30%且观测同向；收缩兼容；同步阈值20%/.8 | **完整机制未通过**；单步均值和转折仅低学习率通过，同步双窗 {syncboth}/{len(candidates)} |
| P-V2-3 | 20000定阈+20000独立验证，FWER 4%..6%，rho=.2/.4功效>=80%；主模型需完整状态 bootstrap | **主检验不可检验/未通过验证**；附项 FWER 校准通过，但 rho=.2 功效不足 |
| P-V2-4 | 同状态、匹配范数、事前遮蔽组中位幅度比>=1.5，2/3 seed同向且可由遮蔽预测 | **失败**；六个条件/学习率分层均未通过 |
| P-V2-5 | 前四关形成可识别闭合反馈后才做未来预测 | **跳过**；P2/P3/P4 未组成闭环 |

## 数据、冻结与重放边界

预注册 [PREREG_ROUND3.md](PREREG_ROUND3.md) 在第三轮冒烟前冻结，SHA256 为 `{checksum}`；本轮结束再次核验一致。它是提前冻结分析规格的旧数据研究，不是新种子确认。所有主集为 18 条 muon0/base，seed=11/23/47、两条件、eta=.01/.03/.1；另 54 条 E1 仅同状态对照。不把 72 头或 864 个候选检查点当独立训练重复。

未保存的每步权重、原始梯度/Adam 矩、训练概率残差和密集 CE 均已用原方向廉价重建。方向 t 对应 W[t-1]；P4 的 150/300/450 检查点为更新后状态，梯度取原下一 minibatch。原始训练文件未修改；{audit['original_files_unchanged']} 个输入文件哈希核验通过，18 条主运行还与第二轮记录的哈希一致。

原方向只存 float32，故重放不是原 float64 检查点的位级复制。最大最终权重误差 {audit['max_validation_errors']['final_weight_max_error']:.3g}，最大探针 Mabs 误差 {audit['max_validation_errors']['probe_Mabs_max_error']:.3g}；分别低于冻结的 1e-5、1e-4。投影递推误差 {audit['max_validation_errors']['scalar_recurrence_error']:.3g}，完整训练梯度投影独立核对误差 {audit['max_validation_errors']['full_gradient_projection_error']:.3g}，式(7.2)误差 {audit['max_validation_errors']['logit_identity_error']:.3g}。重放未执行优化器训练更新。

## P-V2-2：单步均值有局部正证据，尚无长程反馈

用真实八 key 平滑目标 .9125/.0125 重算每样本概率残差，经 128 个固定 32 样本块得到投影均值和噪声；未对最大 logit 使用 sigmoid。m=clip(beta*g,-B,B)，B=真实输入范数/k0；beta 只在 151..300 拟合，301..600 不重拟合。{len(identifiable)}/{len(candidates)} 候选满足预定方向均值变化条件；不变候选不自动通过。以下均值 MSE 是三个留出 100 步窗口的均值误差，主结果仅纳入有变化候选，另列全候选对照，种子顺序 11/23/47。

{table(['条件','eta','对固定均值改善 11/23/47','对持续性改善 11/23/47','固定基线改善>=20%种子','下降>=30%且观测同向种子','全候选改善>=20%种子'],meanrows)}

两个条件的 eta=.01 均值门槛均为 3/3，转折门槛分别为 3/3、2/3。eta=.03 与 .1 的有变化候选均未达到至少2/3种子这一门槛。全候选 isotropic/.1 的均值门槛为2/3，但变化筛选后仅1/3，不能用稳定候选替代有信号候选补票。空间通过头的独立分层也保存在逐运行 JSON 中。

**这些是当前实测完整状态上的单步条件均值，再按100步聚合评分，不是从窗口起点预测未来100步。** 留出 g 由当时真实的完整权重/八 key 激活计算；若将它当作已知未来输入递推，就会泄漏未来状态。仅 a,b 不确定这些多 key 梯度；本轮未识别并验证一个可从预测 a,b 自行产生 g 的闭合律。缺的不是可以补存的一张梯度表，而是经过验证的状态生成模型。沿原方向重放只能恢复已发生路径；重模拟原完整训练过程来造新状态不属于本轮离线授权。

同步门槛在拟合窗通过 {syncfit}/{len(candidates)}、留出窗通过 {synchold}/{len(candidates)}、两窗均通过 {syncboth}/{len(candidates)}。留出创新 Q/K 相关的中位数为 {np.median(rho):.3f}、范围 {min(rho):.3f}..{max(rho):.3f}，要求>=.8；不能把 ab 改写为 a²。局部二维收缩诊断有 {con}/{len(candidates)} 候选满足预测区间位于[0,1)并与观测兼容，也没有跨候选的完整闭环。区间按时间块近似估计，不是逐步独立置信区间。拟合 sigma/原分块噪声 SD 的中位数 {np.median(ratios):.2f}，该经验比例不验证 uniform 噪声定律。

## P-V2-3：不显著不能支持独立性

统计量一次审查每运行 4 个100步窗、16候选、Q/K、lag1..20，共2560个 ACF 读数的最大绝对值；FFT 与独立直接求和实现的误差低于1e-12。每次高斯附项模拟重新拟合同一个受约束 beta；同刻32维协方差整行保留。每运行20000次定阈、另20000次独立验证，rho=.2/.4各20000次；另1999次整行共同置换。18个运行分别对两种 p 作 Holm 校正。

主反馈模型没有闭合状态/噪声生成律，不能进行 §7.3 所要求的完整状态模拟、重新拟合及参数置信集最不利校准，因此主检验**不可检验**。已完成的是预注册的固定设计、同方差高斯 **plug-in 附项**，不能替代主检验；交换性置换也仅为敏感性分析。实际残差窗口方差最大/最小比的中位数 {np.median(varratio):.2f}、最大 {max(varratio):.2f}，未建立同方差/交换性。

附项临界值范围 {min(threshold):.3f}..{max(threshold):.3f}；独立误拒率 {pct(min(fwer),2)}..{pct(max(fwer),2)}，{sum(r['calibration_gate'] for r in mc['runs'])}/18 落在4%..6%。rho=.2 的功效仅 {pct(min(power2))}..{pct(max(power2))}（80%门槛通过 {sum(r['power_gate']['0.2'] for r in mc['runs'])}/18）；rho=.4 为 {pct(min(power4))}..{pct(max(power4))}。所以至少不能凭“不显著”排除 .2 相关。Holm 后高斯附项拒绝 {mc['auxiliary_Holm_rejections']['approximate_Gaussian_p']}/18、置换敏感性拒绝 {mc['auxiliary_Holm_rejections']['exchangeability_sensitivity_p']}/18；这不证明鞅差条件，更不证明真实状态噪声独立。

{table(['条件','eta','seed','实测T','临界值','验证FWER','rho=.2功效','rho=.4功效','高斯Holm p','置换Holm p'],mctable)}

## P-V2-4：匹配范数后未出现登记的危险方向放大

全部72条来源轨迹在三个共同状态比较同一个原 minibatch 梯度的 polar/sign/Adam。Adam 矩来自该来源轨迹此前重算的原始梯度历史，包含当前梯度；未采用零矩或用 Muon 方向反推梯度。原 k0 与每个参数矩阵的 Frobenius 匹配结果均保存；匹配范数跨优化器最大差 {audit['max_matched_fro_spread']:.3g}。

主集共864个候选检查点，事前由梯度定义的“双路弱且被遮蔽”组68个。主门槛要求该组两种幅度比的中位数均>=1.5、2/3种子同向，且与事前遮蔽指标正向关联；下表没有改用事后胜出的子组。

{table(['条件','eta','弱遮蔽数/144','Muon/sign匹配幅度比','Muon/Adam匹配幅度比','同向seed sign，Adam','遮蔽正相关seed sign，Adam','联合判定'],optrows)}

所有主候选的描述性中位比，Muon/sign 为原尺度 {opt['all_primary']['original_k0']['signsgd']['median_all']:.3f}、匹配后 {opt['all_primary']['matched_fro']['signsgd']['median_all']:.3f}；Muon/Adam 为原尺度 {opt['all_primary']['original_k0']['adamw']['median_all']:.3f}、匹配后 {opt['all_primary']['matched_fro']['adamw']['median_all']:.3f}。原尺度上的部分差别不能称作方向机制。实际 logit 一步值包含交叉项和 eta² 项，直接前向验证了恒等式；幅度大也不自动代表绝对 logit 增长。

{table(['反事实方向','原k0幅度A中位数','匹配范数幅度A中位数','匹配后绝对logit增量中位数'],geotable)}

CE 主审计严格密测0..50；其后仅采用原完整训练CE观测，不插值。目标固定1.0/.8，单运行差<=.005、配对差<=.01。主表精确匹配 {audit['exact_CE_pair_count']}/108，其余 {audit['missing_CE_pair_count']} 缺失，不能把夹住目标的稀疏两点写成精确匹配。

为满足“能重建就重建”的数据完整性要求，另在计算前冻结 [补充登记](experiments/round3/SUPPLEMENT_DENSE_CE.md)，对原稀疏穿越区间补全每一整数步，给出真正相邻的上下两步及 logit 端点。补充表仍沿用原 CE 数值容差；**它不替换预注册主表、不改变 P4 失败判定**。相邻两步均离目标太远仍为不可匹配，端点区间不约束未发生的分数步状态。

{table(['比较方','CE目标','主表精确匹配','主表 Mabs 比','补充审计精确匹配','补充 Mabs 比'],cerows)}

补充审计共精确匹配 {audit['supplemental_CE_pair_count']}/108 对，其余 {108-audit['supplemental_CE_pair_count']} 对仍不可精确匹配；不把一次更新跨过目标时的上下步伪装成目标上的状态。

同训练步与实际累计更新长度也不混为相同进度。下表为 t=300、每优化器/学习率跨两个条件与3 seed的描述性中位数（完整同条件数据保存在 JSON）：

{table(['优化器','eta','完整训练CE','探针Mabs','累计实际位移（含decay）','累计方向长度'],lengthrows)}

不根据这些局部反事实和CE配对推导完整训练因果排序；这些检验针对登记的候选与梯度遮蔽定义。本轮没有执行可选的20次正交旋转实验。

## 空间对象与 §8 止损

只取160..600内的完整头峰，晚期501..600中位数下降>=10%且有连续两点低于.9峰，得到32个回落头，其中8个空间通过；54个越过8的头中27个空间通过。下表每格总数为12头，不包含第150步基线峰的动态成功。

{table(['条件','eta','空间通过/12','明显回落/12','回落且空间通过'],counts)}

获得单步反馈正证据的 eta=.01 在两个条件共24个头中，没有一个满足此处>=10%回落定义；因此这部分成功不能代替解释明显回落。

逐条对照 §8：

1. “所有明显回落/越阈配置都无法覆盖”没有整体触发：仍有8个回落头空间通过；但多数回落头未过空间门槛，不能用稳定区正结果解释全部尖峰。A′ 的覆盖/更替方向有明确数据动机。
2. 第二条不能整体宣告触发：eta=.01 的条件均值有正证据，不能说真实残差毫无预测力。eta=.03/.1 的有变化主分析未过20%改善/至少2个seed门槛，部分种子还劣于基线，故停止将当前模型解释这些配置；这不等于所有种子均值都不优于基线。低学习率缺的是可用预测状态更新的闭合反馈，而不是已经证明所有多 key 约化都不可能。同步原型不适用，本轮不给 (3.4) 的 E1 解释通过。
3. 同状态匹配步长后登记的危险放大门槛失败，故停止把 L1 反例宣传为此 E1 训练的 Muon 特异性解释。遮蔽并非从未出现，但没有达到预定的可预测放大门槛；CE匹配也不能补足完整轨迹因果排序。
4. 本轮是 V2 的一次提前冻结执行，不把第二轮探索或本轮CE补充冒充“连续两次反馈模型修改失败”。因此连续两次规则尚未触发，不用它夸大停题理由。
5. 本轮不能建立“空间充分性—可测闭合反馈—冻结未来预测”链，也没有新增超过经典性质的理论结果；不据此承诺十周主会论文。两个研究周的期限尚未在本轮被检验，不能假称时间条件已经触发。

**行动判断是修改模型，并封存当前同步桥梁/特异性桥梁的负结果。** 下一步仅建议 A′ 的旧数据分析：先冻结赢家更替、候选间隔和基于真实一步几何的刷新时刻预测，再看其能否跨窗口预测；明确给出不可覆盖反例和必要刷新代价。此建议未在本轮执行，不追加 rank、头、窗或更松阈值来延长当前 V2。若 A′ 仍不能预测下一窗，则以技术报告结束 A。未经闭合和校准，不执行 P-V2-5，也不输出看似有效的裁剪停用时间或风险认证。

## 执行、审计与产物

先1 seed/1头冒烟，再18主运行+54对照全量；附项每运行4组20000模拟（总1440000条450步单步模型样本序列）及1999整行置换。模拟的是统计模型，不是重新训练。全部经 `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0`，指定解释器，CPU、BLAS/OMP/MKL=4；正式分析不导入 torch、不创建GPU张量。启动时独立检查 torch 线程=4、CUDA不可用。无网络/LLM API，调用和费用均为0。

正式重放、统计、校准、CE补充与审计实测合计约 {elapsed/60:.2f} 分钟；这些阶段测得峰值 RSS {peak/1024**2:.1f} MiB，低于3G。包括阅读、编码、冒烟、等待在内截至生成本报告墙钟约 {overall/60:.1f} 分钟，低于约3小时。首次冒烟在结果写出遇到重复 step 字段异常，保留失败日志并修复后重跑，未修改门槛；其约10秒失败用时不混入正式阶段计时。最终审计为 **{audit['status']}**。

- [逐运行重放和密集 CE](experiments/round3/full/replay_manifest.json)，[主统计与全部 CE 缺失](experiments/round3/full/statistics.json)。
- [单步反馈明细](experiments/round3/full/feedback/)，[校准汇总与逐运行 p/功效](experiments/round3/full/calibration_summary.json)，原始模拟 T 数组位于同目录 calibration/。
- [后段真实相邻步 CE 补充](experiments/round3/full/dense_ce_supplement.json)，[输入不可变与数值审计](experiments/round3/full/audit.json)。
- [复现命令](experiments/round3/README.md)，[实现修复记录](experiments/round3/IMPLEMENTATION_LOG.md)。源码、冻结文档与结构化输出的哈希另存 experiments/round3/ARTIFACTS.sha256；原训练输入哈希在重放 JSON 中。
'''
    (ROOT/'RESULTS_ROUND3.md').write_text(text)
    save(out/'report_metadata.json',dict(created_UTC=datetime.now(timezone.utc).isoformat(),
         prereg_sha256=checksum,report_sha256=sha(ROOT/'RESULTS_ROUND3.md'),
         stage_elapsed_seconds=elapsed,total_wall_seconds=overall,peak_RSS_bytes=peak,
         resources=metadata(start)))
    print('Wrote RESULTS_ROUND3.md',flush=True)


if __name__=='__main__':
    main()
