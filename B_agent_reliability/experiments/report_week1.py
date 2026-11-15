"""Render Chinese report from completed, frozen experiment outputs."""
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/week1_results"
load = lambda name: json.loads((OUT / name).read_text())
e1, e2, e3, e4, manifest = [load(name + ".json") for name in ("E1", "E2", "E3", "E4", "manifest")]
sections = []
add = sections.append
fmt = lambda x: "不可估" if x is None else f"{x:.6g}"


def table(headers, rows):
    add("| " + " | ".join(headers) + " |")
    add("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        add("| " + " | ".join(str(c) for c in row) + " |")
    add("")


add("## 执行概况与结论\n")
add(f"E1–E4 全部完成，未减样本量、未调用外部 API、未联网、未下载模型。从首次启动到最终完成墙钟 **{manifest['elapsed_seconds']:.1f} 秒（{manifest['elapsed_seconds']/60:.2f} 分钟）**，含首次失败与修复间隔；活动计算约 **{manifest.get('active_compute_seconds',manifest['elapsed_seconds']):.1f} 秒**，峰值 RSS **{manifest['max_rss_mib']:.1f} MiB**；地址空间硬限 3,500,000,000 字节。指定解释器、OMP=3、torch CPU 线程=3 均已记录，CUDA 未初始化。CPU 逐事件与统计运算显著快于文档保守预算，不需要用满预算。阶段耗时不含前期实现与报告撰写。\n")
if (OUT / "delivery.json").exists():
    delivery=load("delivery.json")
    add(f"上述6.55分钟对应 E1–E4 主程序首次启动至最终成功结束；包括随后补全H比值／曲率、原始数据审计和报告，距本会话首次记录开工时间共 **{delivery['session_elapsed_seconds']/60:.2f} 分钟**，距主程序首次启动共 **{delivery['since_experiment_start_seconds']/60:.2f} 分钟**，均小于文档55分钟和用户75分钟上限。后处理不新增生成数据、不调整预测或阈值。时间记录见 delivery.json。\n")
table(["阶段", "实际秒", "硬限秒"], [[name, f"{manifest['stage_times'][name]:.2f}", limit] for name, limit in (("E1",480),("E2",480),("E3",900),("E4",1080))])
if "E4_all_attempts_seconds" in manifest:
    add(f"E4 表中为最终成功运行；加上首次中断的 E4 共 {manifest['E4_all_attempts_seconds']:.2f} 秒，仍在该阶段硬限内。全部实现、验证与交付从会话开工起也在用户 75 分钟内。\n")
add("**判断：有限继续到真实执行器先导；尚未达到继续投入主会论文的最低证据门槛。** 合成公式和统计层通过，强违例诊断有效；但仍没有独立真实 agent 轨迹。不能把本轮结果写成已经验证经验桥梁，更不能声称解释 METR。第 2 周先验证状态可观测性、恢复再生和容量干预；若失败且有限增广状态不能修复，应止损该一维机制路线。\n")

add("## E1：公式校验与冻结容量外推\n")
add("设置：rho∈{0.5,1,1.5,2}，K∈{2,4,8,12,16}，种子 11/23/37，各格 20,000 片段；另每个 rho/种子在 K=3 校准 20,000 片段，在独立种子 K=4/8/12 测试。共 2,160,000 片段。Wilson 同时区间 Bonferroni 族错误率 0.01，96 格；片段联合直方图 bootstrap 200 次。\n")
errh = max(abs(r['linear_h']/r['exact_h']-1) for r in e1['formula'])
errw = max(abs(r['linear_w']/r['exact_w']-1) for r in e1['formula'])
add(f"独立三对角求解与闭式的最大相对误差：h={errh:.3g}、w={errw:.3g}。公式格概率 {e1['probability_passes']}/60、工作 {e1['work_passes']}/60 通过；冻结参数外推概率 {e1['extrap_probability_passes']}/36、工作 {e1['extrap_work_passes']}/36 通过。**支持公式与实现正确；对真实机制仍不确定。**\n")
table(["真 rho", "三种子 rho_hat 范围", "K16 实测 h 范围", "K16 平均工作范围", "K16 工作 p99 范围", "超过100事件比例范围"],
      [[rho,
        f"{min(r['rho_hat'] for r in e1['calibration'] if r['true_rho']==rho):.5f}–{max(r['rho_hat'] for r in e1['calibration'] if r['true_rho']==rho):.5f}",
        *[f"{min(r[key] for r in e1['formula'] if r['rho']==rho and r['k']==16):.6g}–{max(r[key] for r in e1['formula'] if r['rho']==rho and r['k']==16):.6g}" for key in ('failure_rate','mean_work','work_p99','over_100')]] for rho in (.5,1.,1.5,2.)])
rare = sum(r['rare_failure'] for r in e1['heldout'])
add(f"留出格中有 {rare} 格失败数<30；这些格仅报告概率区间，不以零／极少失败拟合指数。所有格的 p95/p99、工作 bootstrap 区间、100 事件截断成功率与 M/T 种子率见 E1.json。概率外推这里没有额外并入校准参数的不确定性，故通过只作冒烟结果。\n")

add("## E2：混合律、留出长度与旁路负对照\n")
add("设置：三个 Gamma 族各 100,000 任务，均值 theta=0.1，T=100 独立种子计数校准，R=8。21 个长度 10..10^6、6 种容量、3 种 eta，共 54 配置、907,200,000 个条件运行。任务 theta 跨重复及长度共享。先用预注册 Adam／三初值拟合 Gamma–Poisson；三种曲线仅用前 11 个长度，后 10 点留出。所有 bootstrap 按整任务行抽样 200 次。\n")
table(["真 alpha,beta", "计数 MLE alpha,beta", "矩估计 alpha,beta"],
      [[f"{f['alpha']},{f['beta']}", f"{f['fit']['alpha']:.6f},{f['fit']['beta']:.6f}", f"{f['fit']['moment_alpha']:.6f},{f['fit']['moment_beta']:.6f}"] for f in e2['families']])
maxerr = max(r['heldout_max_abs_error'] for r in e2['configurations'])
meanerr = np.mean([r['heldout_max_abs_error'] for r in e2['configurations']])
add(f"独立计数校准的冻结 Gamma 预测，54 配置后 10 点最大绝对成功率误差的均值为 {meanerr:.6g}，最坏为 {maxerr:.6g}。以下 Brier 为各配置先平均留出点、再对 54 配置等权平均；二元运行 Brier 按成功次数精确计算。\n")
table(["预测来源", "平均留出 Brier"], [["独立计数校准 Gamma", fmt(float(np.mean([r['frozen_heldout_brier'] for r in e2['configurations']])))]] +
      [[name, fmt(float(np.mean([r['curve_models'][model]['heldout_brier'] for r in e2['configurations']])))] for name, model in (("短长度单指数","exponential"),("短长度自由 log-logistic","loglogistic"),("短长度 Gamma 曲线拟合","gamma"))])
add("同一生成族下，机制参数独立校准和仅拟合短长度曲线是两项不同检查；短长度可能没有覆盖曲线弯曲，拟合参数未必可识别。E2 未预注册整体显著性门槛，结论按误差描述，不事后创造过关标准。\n")
phase = [next(r for r in e2['configurations'] if r['alpha']==1 and r['kind']=='log' and r['capacity']==a and r['eta']==0) for a in (.5,1.,2.)]
table(["a", "拟合量", "预注册渐近指数", "实测有限长度指数", "同点精确指数", "任务bootstrap 95%区间"],
      [[r['capacity'], r['tail_quantity'], f"{abs(1-r['capacity']*math.log(2)):.6f}", fmt(r['tail_exponent']), fmt(r['finite_length_exact_exponent']), str([round(v,6) for v in r['tail_exponent_ci']]) if r['tail_exponent_ci'] else '不可估'] for r in phase])
add("取整 K 引起台阶／振荡，且长度有限；应对照同点精确斜率。与渐近指数有偏差不构成否定。稀有格自动遵守成功、失败各≥100 的指数拟合门槛。\n")
table(["alpha=1, eta", "H50(K4)", "H50(K8)", "H50(K12)", "旁路上界"],
      [[eta, *[fmt(next(r['H50'] for r in e2['configurations'] if r['alpha']==1 and r['kind']=='fixed' and r['capacity']==k and r['eta']==eta)) for k in (4,8,12)], '无有限上界' if eta==0 else fmt(10/eta)] for eta in (0.,.001,.01)])
add("上表 H50 是冻结机制的精确预测，不冒充有限任务样本的观测中位数。固定 K 下理论 H80/H50 只由 alpha 决定；分别为 "+"、".join(f"alpha={a}: {(.8**(-1/a)-1)/(2**(1/a)-1):.6f}" for a in (.5,1.,2.))+"。式(13)在 alpha=1 时局部斜率恒为1；alpha≠1 时随长度弯曲，各配置完整斜率见 JSON。\n")
if (OUT / "E2_derived.json").exists():
    derived=load("E2_derived.json")["configurations"]
    add("从已保存任务行补全预注册的观测 H 比值和 logit 曲率：在固定长度网格目标成功率两侧，对 logit(S) 与 log L 插值，不外推；bootstrap 仍为原种子按任务抽样200次。这是既有结果后处理，未修改预测或判定。K=8、eta=0 的代表数字：\n")
    table(["alpha", "观测 H80/H50", "bootstrap 95%区间", "理论比值", "相同网格精确插值比值"],
          [[r['alpha'], fmt(r['observed_H80_over_H50']),str([round(x,6) for x in r['ratio_bootstrap_ci']]),fmt(r['exact_formula_ratio']),fmt(r['exact_grid_ratio'])] for r in derived if r['k']==8 and r['eta']==0])
    slope_rows=[]
    for dr in derived:
        if dr['k']!=8 or dr['eta']!=0:
            continue
        source=next(r for r in e2['configurations'] if r['alpha']==dr['alpha'] and r['kind']=='fixed' and r['capacity']==8 and r['eta']==0)
        ids=dr['valid_indices']
        picks=[min(ids,key=lambda j:abs(source['observed'][j]-target)) for target in (.8,.2)]
        slope_rows.append([dr['alpha'], *[f"{dr['finite_difference_logit_slope_observed'][j]:.5f} / {dr['finite_difference_logit_slope_exact'][j]:.5f}" for j in picks]])
    table(["alpha", "S最接近0.8时局部斜率：观测/同点精确", "S最接近0.2时局部斜率：观测/同点精确"],slope_rows)
    add("eta=0.01、alpha=1 的观测 H50："+"、".join(f"K={r['k']}: {r['observed_H50']:.3f}（95% {r['H50_bootstrap_ci'][0]:.3f}..{r['H50_bootstrap_ci'][1]:.3f}）" for r in derived if r['alpha']==1 and r['eta']==.01)+"；地板与理论1000上界相容。全27个固定容量格的观测H50、H比值、区间和有限差分logit斜率已存 E2_derived.json；网格插值有有限分辨率偏差，bootstrap 不覆盖该确定性离散偏差。\n")
add("另以独立逐事件任务运行测工作成本（每格 2,000 任务，eta=0，L 为该格理论 H50，硬限 W≤5L）：\n")
table(["alpha", "K", "成功率", "预算内成功率", "超预算率", "E[W]/L", "p99(W/L)"],
      [[r['alpha'],r['k'],*[fmt(r[key]) for key in ('success','budget_success','over_budget','mean_work_over_L','p99_work_over_L')]] for r in e2['cost_diagnostics']])
add("**对照：混合统计层、容量收益与 eta 地板的方向得到支持；有限长度相图与预注册渐近预测相容。对真实执行成本和真实异质性仍不确定。** 成本表必须与成功率一起看，不能把平均工作界当作每次硬预算保证。\n")
add("E[W]/L 小于1是因为失败会提前停止正常推进，并不表示成功完成一个长度 L 的任务少于 L 工作。a随长度调整容量时，Gamma的精确预测必须使用变化的 g_L；常数g的两参数曲线对照没有该容量输入，因此不应把汇总 Brier 差异解释为同参数数目下的纯链接函数优劣。\n")

add("## E3：三种机制压力测试\n")
add("设置：每环境 K=3 校准 30,000 片段，K=6/10/16 各 30,000 独立片段。常数 rho、状态 rho_n 和显式 eta 模型均冻结；状态模型在 n≥3 不可识别，未用测试数据填参。任务级各格 10,000 运行，在常数模型预测 H50 处检验；非再生环境保留 Z、使用独立暂态传播联合片段分布加速。拒绝阈值保持 max(0.03,5SE) 或 H50 超过两倍。\n")
def explicit_verdict(env,r):
    cal=env['calibration']
    modelH=math.log(2)/(cal['theta_hat_clean']*r['predicted_g_explicit'])
    ratio=max(modelH/r['H50_true_or_MC'],r['H50_true_or_MC']/modelH)
    if r['task_rejected_explicit'] or ratio>2:
        return '拒绝'
    sufficient=r['task_n']*min(r['observed_task_success'],1-r['observed_task_success'])>=30
    return '未拒绝' if sufficient else 'H50未拒绝；概率计数不足'
table(["环境", "K", "片段实测失败率", "常数预测失败率", "预测H50处实测任务成功率", "H50误差倍数", "常数机制判定", "显式eta预测成功率／判定"],
      [[env['environment'], r['k'], fmt(r['failure_rate']), fmt(r['predicted_h_constant']), fmt(r['observed_task_success']), fmt(r['H50_ratio']), '拒绝' if r['mechanism_rejected'] else '未拒绝', fmt(r['predicted_task_success_explicit'])+'／'+explicit_verdict(env,r)] for env in e3['environments'] for r in env['tests']])
add("环境1为 n≤3 时 rho=2、n≥4 时 rho=0.4；必须拒绝的常数外推负对照全部拒绝，**支持诊断有效，否定该环境中的常数 rho 简化**。环境2的单片段 h 基本不变，任务生存发生偏移，说明只检查片段会漏掉非再生；环境3的 eta=0.01 在大容量显著破坏无旁路收益，显式 eta 模型可以利用可观测零事件失败估计旁路。\n")
for env in e3['environments']:
    add(f"环境{env['environment']}校准：rho_hat={env['calibration']['rho_hat']:.6f}，eta_hat={env['calibration']['eta_hat_from_zero_events']:.6f}，干净种子率={env['calibration']['theta_hat_clean']:.6f}。")
    if env['history']:
        hh=env['history']; add(f"非再生序列总体种子率={hh['overall_rate']:.6f}，低/高污染四分位率={hh['low_Z_rate']:.6f}/{hh['high_Z_rate']:.6f}，Z 与间隔相关={hh['gap_Z_correlation']:.6f}。")
    add("")
add("显式旁路的识别依赖事件数可观测；如果连零内部事件都隐藏，仅靠单容量吸收率不能把 eta 与 rho 唯一分开。非再生 H50 来自预先固定网格／模拟死亡分位数，区间及暂态传播尾质量见 JSON；普通环境 H50 对照是已知生成环境的精确值。**本轮仍是合成压力测试，不证明真实 agent 可被一维状态描述。**\n")
add("显式旁路模型也并非全部过关：环境3在K=6时被原阈值拒绝。校准eta_hat=0.011733高于真值0.01，rho_hat也有误差；本轮诊断没有传播校准不确定性，不能事后换种子或放宽阈值消掉这一拒绝。大K任务出现0或极少成功的格，概率检查计数不足；表中另以预注册H50两倍标准判定，不能将计数不足的未拒绝称为支持。下一轮若改变校准样本量或传播不确定性，须在新数据前另行预注册。\n")

add("## E4：本地公开数据相容性与同题重试\n")
add(f"本地 time-horizon-1-1 快照共 {e4['data']['rows']} 行，跳过 {e4['data']['skipped']} 行不适用记录。实际存在 task_id/task_family、人类分钟数和逐运行二元结果。源文件 SHA256：`{e4['data']['sha256']}`。README 声称参见 LICENSE，但快照没有 LICENSE；许可未核实，不重新分发源数据。任务版本列表见 E4.json；未将公开快照当作全量任务。按别名字典序选择前三个合格模型，五折按家族隔离，评估每任务平均的运行对数损失和 10 箱 ECE。拟合也采用任务等权的归一化二项似然，与该评估目标一致；家族 bootstrap 200 次评估固定折外预测，不重训全流程。\n")
table(["公开模型", "任务/家族/运行", "指数loss", "Gamma loss", "log-logistic loss", "Gamma−log-logistic 95%区间"],
      [[r['alias'],f"{r['tasks']}/{r['families']}/{r['runs']}",*[fmt(r['fit_results'][name]['cv_log_loss']) for name in ('exponential','gamma','loglogistic')],str([round(v,6) for v in r['gamma_minus_loglogistic_ci']])] for r in e4['models']])
table(["模型", "指数 ECE", "Gamma ECE", "log-logistic ECE"],
      [[r['alias'],*[fmt(r['fit_results'][name]['ECE']) for name in ('exponential','gamma','loglogistic')]] for r in e4['models']])
add("**结论：仅可判断边缘曲线相容性，级联机制仍不确定。** Gamma 与自由 log-logistic 的损失差及区间直接报告，不按微小差异宣称因果胜出。L∝人类工时是敏感性假设；没有动作轨迹，未估计 rho 或 K。\n")
table(["alpha", "N", "精确H50", "H50/logN", "极限常数", "MC与积分最大绝对差"],
      [[r['alpha'],r['N'],fmt(r['H50']),fmt(r['H50_over_logN']),fmt(r['asymptotic_H50_over_logN']),fmt(r['max_MC_integral_abs'])] for r in e4['retries'] if r['N'] in (1,8,128)])
r = next(r for r in e4['retries'] if r['alpha']==1 and r['N']==128)
li = 4
add(f"共享难度与每次重抽难度差异很大：alpha=1、N=128、L={r['lengths'][li]:.0f} 时，共享任务精确覆盖率={r['integral'][li]:.6f}，错误重抽基线={r['redrawn_task_baseline'][li]:.6f}。**支持同题重试的精确式及向对数标度收敛方向**；N≤128 的 H50/logN 与极限仍有差距，不宣称已达到渐近极限。没有计算大 N 交错和。\n")

add("## 继续／止损与下一步最该做的三件事\n")
add("本轮没有发现足以因公式或诊断实现失败而立即止损的证据；允许有限继续做真实机制先导。文档的最低继续门槛要求独立执行轨迹的冻结参数干预预测通过，目前**未达到**。因此暂停扩大理论叙事、扩大模型规模或以 METR 拟合代替机制验证。若第 2 周状态定义不可重复或恢复后残余历史错误率效应>0.03且有限增广状态不能消除，停止一维路线；第 4 周前仍无两类执行任务证据，则转技术笔记或停止主会方向。\n")
add("1. **先建两个可验证本地执行器并做真实轨迹先导。** 固定随机表键值累加、带撤销日志的工作表编辑；提交时逐义务和最终状态双重核对，记录错误注入、义务数、检测、修复、污染历史、真实存储和事件/token工作；同一任务配对长度与容量，先验证 n 是否充分以及恢复是否再生。\n2. **冻结 K=3 校准，再做 K=6/10/16 与留出长度干预。** 保持本轮拒绝阈值，加入状态相关、残余污染、隐藏旁路负对照；同时报告 W≤5L、调用数硬限下的成功率与超限率，不把平均成本当成硬预算保证。\n3. **完成新颖性全文对照和投入门槛审查。** 对照 TraceToChain、分支过程／检查点／可靠性理论的容量干预、工作量与旁路边界；若已有相当理论及实证覆盖则止损。METR 后续只做独立样本／工时映射敏感性，不作为桥梁替代。\n")
add("## 下一轮真实 LLM API 设计与预算（本轮未调用）\n")
add("先导：两类执行器，各 20 个独立任务，K=3，固定短长度 L=16，至少覆盖人工注入的正确／错误历史；40 次运行，每次最多 32 次模型调用，每调用输入最多 2048、输出最多 128 token（结构化动作），相同上下文规则贯穿先导和主实验。上限 **1,280 次调用、2,621,440 输入 token、163,840 输出 token**。按实际返回记录，不在失败后反复重试直到通过；干净种子率、转移、零事件旁路可观测性、工作与再生诊断单独校准。若任何运行碰调用硬限，按失败并报告。\n")
add("先导通过后主实验：每类 20 个新任务，长度 {16,32,64}、测试容量 {6,10,16}、各 2 次共享任务难度重复，共 **720 次运行**。每次最多 128 次调用，为最长64步任务保留纠错余量；上限 **92,160 次调用、188,743,680 输入 token、11,796,480 输出 token**。小容量参数仅来自先导，不以主实验重拟合；按任务／家族留出，保留所有状态与提交义务。成本配对比较必须同时包括调用数、token、工作、存储与超限失败。该规模是上限设计，低稀有失败计数只报告区间，不保证足以验证极小 h。\n")
add("不查询实时价格。若供应商价格为输入 p_in、输出 p_out 美元／百万 token，则先导上界 **2.62144 p_in + 0.16384 p_out 美元**，主实验上界 **188.74368 p_in + 11.79648 p_out 美元**。示例价格仅作算术假设 p_in=1、p_out=5，分别约 $3.44 与 $247.73；不是现行报价。建议先导设置 $25／2小时硬停，主实验设置 $300／24小时硬停；这些是硬上限，实际延迟或费率可能使样本无法全部完成，须报告中断和已完成数。按实际供应商价格不足时先缩小预注册样本，不能中途看效果调样本。并发最多2，不访问无关外部系统；真实价格、提供商、模型版本和费率在下一轮运行前再冻结。\n")

add("## 交付与复核\n")
add("代码：[week1.py](experiments/week1.py)，启动：[run_week1.sh](experiments/run_week1.sh)，复现说明：[README_WEEK1.md](experiments/README_WEEK1.md)，冻结实施约定：[preregistration_week1.md](experiments/preregistration_week1.md)。原始 JSON／JSON.gz 在 `experiments/week1_results/raw/`，完整统计为 `E1.json`–`E4.json`，实际资源与散列为 `manifest.json`。\n")
add("![E1](experiments/week1_results/figures/E1.png)\n\n![E2](experiments/week1_results/figures/E2.png)\n\n![E3](experiments/week1_results/figures/E3.png)\n\n![E4](experiments/week1_results/figures/E4.png)\n")
if (OUT / "verification.json").exists():
    ver=load("verification.json")
    add(f"结果审计通过 {ver['checks']} 项检查：样本量、事件守恒、原始 JSON 与汇总一致性、按任务标准误、家族折隔离、原文／冻结约定／执行代码散列不变、CPU／线程／内存／时限，以及四张 PNG 解码。审计只复核既有输出，没有重选模型或修改判定标准。\n")
original = (ROOT / "experiments/preregistration_week1.md").read_text().split("## 结果")[0]
amendment = (ROOT / "experiments/amendment_week1.md").read_text() if (ROOT / "experiments/amendment_week1.md").exists() else ""
(ROOT / "RESULTS_WEEK1.md").write_text(original + "\n## 执行中实现勘误（E4 重跑前记录）\n\n" + amendment.replace("# E4 内存修正，重跑前记录", "") + "\n" + "\n".join(sections), encoding="utf-8")
print("Wrote RESULTS_WEEK1.md")
