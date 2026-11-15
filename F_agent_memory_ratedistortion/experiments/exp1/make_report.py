"""Produce the Chinese report from final outputs, preserving frozen thresholds."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,math,resource,hashlib
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
root=Path(__file__).resolve().parent; project=root.parent.parent
s=json.loads((root/'summary.json').read_text()); e0=s['e0']; resourceinfo=s['resources']
records=[json.loads(x) for x in (root/'formal/results.jsonl').read_text().splitlines()]
def f(x,d=4):return '不可评估（空适用域）' if x is None else f'{x:.{d}f}'
def pct(x):return '不可评估（空适用域）' if x is None else f'{100*x:.1f}%'
lines=[]
def add(x=''):lines.append(x)
add('# EXP1 第一轮验证结果')
add()
add('**结论：指定通道下的整数优化与有限数学预测获得支持；本轮没有达到冻结的模型外初步支持标准。几何检索的相界不能视为已验证。** 严格适用域检验未通过时，测试MAE较低或个别排序正确不能替代原通过条件。以下逐条使用原门槛，未事后放宽；失败、未完成和不可评估分开记录。')
add()
add(f'预注册SHA-256：`{resourceinfo["prereg_sha256"]}`；原文件与校验和保持不变。代码、原始条件结果、校准成功计数、配对逐查询向量与汇总见 `experiments/exp1/`。所有结果限于本地合成bit源/冻结检索器，不是自然语言QA，也不证明新颖性。')
add()
add('## 执行范围与资源')
add()
add(f'先完成两个256查询冒烟，再完成E0与正式E1/E2。正式主程序耗时 {resourceinfo["elapsed_seconds"]:.2f} 秒（{resourceinfo["elapsed_seconds"]/60:.2f} 分钟），峰值RSS {resourceinfo["peak_rss_mib"]:.2f} MiB；E0 {e0["elapsed_seconds"]:.3f} 秒。解释器 `/home/liuyibo/venvs/theory/bin/python`，单进程、线程1，均在 `systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0` 下运行；外层3540秒timeout，Python地址空间1.9GB。后处理同样受scope限制。无GPU、LLM API、联网、安装软件或跨项目数据导入。')
add()
add('| 层 | 完成配置 | 预定但未运行 | 独立检索校准 | 经验候选扫描 |')
add('|---|---:|---:|---:|---|')
for layer in ('E1','E2'):
    st=s.get('status',{}).get(layer,{})
    add(f'| {layer} | {s[layer]["configurations"]} | {st.get("NOT_RUN_RESOURCE_DEADLINE",0)} | {s[layer]["unique_calibrations"]["homogeneous"]["n"]} | '+('指定通道，DP全局整数最优' if layer=='E1' else f'均匀族全部可行(n,s)，排列数量{s[layer]["permutations"]}；Zipf阈值网格及32随机方案')+' |')
add()
add('每正式条件历史8192、检索校准4096（拟合/验证各2048）、独立正确证据reader校准4096、测试8192；seed=0,1,2。全部预算与噪声核心条件优先，阶段时间用完后停止新条件；未运行清单见 `formal/status.json`，不能将其记作通过。不同epsilon/reader会共享检索校准，独立校准数列单独统计，避免重复计数。')
add()
for layer in ('E1','E2'):
    sizes={g:sum(d['layer']==layer and d['G']==g for d in records) for g in (16,32,64)}
    add(f'{layer}实际规模覆盖：{sizes}（数值为配置数）。')
add('G=64未运行；本轮不能声称已验证该规模的外推。')
invalid=json.loads((root/'invalid_calibration_run/metadata.json').read_text())
add()
add(f'执行纠错：首次运行的校准索引可能同时放入同段摘要与原文，已整轮作废并保留在invalid_calibration_run/，耗时{invalid["elapsed_seconds"]:.2f}秒。修正为按经历互斥采样后重新冒烟与正式运行；本报告只统计修正后的formal/。两个正式运行累计{(invalid["elapsed_seconds"]+resourceinfo["elapsed_seconds"])/60:.2f}分钟，仍低于一小时；预注册始终未改。')
add()
add('QED只读记录：**仅L3明确PASS**；L1/L2/L4/L5/L6有proof.md但未发现最终验证PASS，均记未确认，详见 `QED_STATUS.md`。本轮E0是有限数值检查，不能替代符号证明。')
add()
add('## 对照冻结门槛')
add()
add(f'1. **E0通过。** {e0["budget_dp_cases"]} 个DP/穷举问题最大优势差 {e0["max_dp_gain_gap"]:.3g}；{e0["uniform_integer_phase_cases"]} 个均匀整数相图问题最大差 {e0["max_uniform_phase_gain_gap"]:.3g}，均严格小于 `1e-10`。{e0["fractional_vertex_cases"]} 个顶点和{e0["dual_support_certificate_cases"]}个对偶支持证书通过，取整界通过。额外1008个逐策略Gumbel参考检查通过。G5/k5/b6台阶确为(0,5)。')
add()
for layer in ('E1','E2'):
    q=s[layer]
    add(f'{layer} 的全部主策略/预算测试行数 {q["all"]["n"]}：MAE {f(q["all"]["mae"])}，95%分位误差 {f(q["all"]["p95"])}，最大误差 {f(q["all"]["max"])}。这些总体数字包括所有校准失败条件；显式增加预算或访问能力的pad/cold/route另列配对比较。')
    add()
q=s['E1']; add(f'**E1独立校准准确率检查并非全部通过。** 同质拟合预测有 {q["threshold_exceedances"]}/{q["all"]["n"]} 行超过 `max(0.015,4SE)`；预注册两参数扩展有 {q["extended_threshold_exceedances"]}/{q["extended_all"]["n"]} 行超过同一门槛。不得称这些越界为模型外发现。已用逐策略直接Gumbel采样参考检查排查实现；oracle生成参数的误差MAE {f(q["oracle"]["mae"])}，最大 {f(q["oracle"]["max"])}，支持有限校准误差和对异质通道误用同质拟合是主要原因。独立校准预测与已知生成参数预测严格分开，后者不替代前者。')
diag=json.loads((root/'diagnostic_examples.json').read_text()) if (root/'diagnostic_examples.json').exists() else {}
add()
add('| E1生成通道 | 测试行数 | 同质预测MAE/越界数 | 两参数预测MAE/越界数 | 已知参数越界数 |')
add('|---|---:|---|---|---:|')
for name in ('E1_homogeneous_generator','E1_heterogeneous_generator'):
    if name in diag.get('strata',{}):
        z=diag['strata'][name];add(f'| {name} | {z["strategy_rows"]} | {z["homogeneous_mae"]:.4f}/{z["homogeneous_exceedances"]} | {z["extended_mae"]:.4f}/{z["extended_exceedances"]} | {z["oracle_exceedances"]} |')
add()
add('已知生成参数仍有少量越界时，不改门槛：每seed只生成一份有限32-bit源，固定源的各经历实际摘要保真度会波动；Zipf重加权会放大它。仅以8192次查询的二项SE不能表示跨源不确定性。reader校准也是有限样本；这些是必须在后续预注册改善的误差归因，不能将本轮越界改成通过。')
add()
add('2. **适用域诊断失败。** 门槛为留出Recall最大绝对残差≤0.03，等负载子集/构成目标命中差≤0.04。')
add()
add('| 层/拟合 | 通过/独立校准数 | 判定 |')
add('|---|---:|---|')
for layer in ('E1','E2'):
    for model in ('homogeneous','extended'):
        v=s[layer]['unique_calibrations'][model];add(f'| {layer}/{model} | {v["passes"]}/{v["n"]} | '+('通过' if v['passes']==v['n'] and v['n'] else '失败')+' |')
add()
add('原始cell残差、样本数及二项SE见 `calibration_cells.csv`；不剔除低样本cell，不用置信区间替换点残差门槛。一个校准条件拆分到多个M/类型/构成/子集后，每cell样本量偏小。E1已知通道的事后参数bootstrap仅诊断这个问题，见 `calibration_sampling_diagnostic.json`，不改变本轮失败判定，也不充当模型外支持。校准背景索引按经历互斥选择0/s/r，原文目标包括同段其他字段；M=1为单证据探针，不是可行原文策略。超过目标类型可行最大M时记录实际负载。')
add()
add('3. **模型外初步支持未通过。** 原标准必须同时满足适用域内MAE≤0.025、95%误差分位≤0.05、候选集经验regret≤0.025、预测差≥0.02的端点排序正确率≥80%，且至少两个seed、两个可区分噪声区间出现边界两侧。')
add()
q=s['E2'];add(f'适用域内可评估行数 {q["domain"]["n"]}，MAE {f(q["domain"]["mae"])}，95%分位 {f(q["domain"]["p95"])}；DP候选集regret最大 {f(q["dp_regret_domain"]["max"])}；适用域内排序 {pct(q["endpoint_ordering_domain"]["rate"])}（n={q["endpoint_ordering_domain"]["n"]}）。全部条件的端点排序率为 {pct(q["endpoint_ordering"]["rate"])}（{q["endpoint_ordering"]["correct"]}/{q["endpoint_ordering"]["n"]}），只作诊断。')
add()
add('| 模型外支持的必要项 | 原门槛 | 结果 | 判定 |')
add('|---|---|---|---|')
for label,limit,observed in [('适用域内MAE',.025,q['domain']['mae']),('适用域内95%误差分位',.05,q['domain']['p95']),('适用域内DP候选集regret最大值',.025,q['dp_regret_domain']['max'])]:
    verdict='通过' if observed is not None and observed<=limit else ('不可评估，未通过' if observed is None else '失败')
    add(f'| {label} | ≤{limit} | {f(observed)} | {verdict} |')
rate=q['endpoint_ordering_domain']['rate']; verdict='通过' if rate is not None and rate>=.8 else ('不可评估，未通过' if rate is None else '失败')
add(f'| 适用域内有效端点排序 | ≥80% | {pct(rate)} | {verdict} |')
boundary=[x for x in q['boundaries'] if x['domain_pass'] and x['eligible']]
crossing_seeds=[seed for seed in (0,1,2) if any(x['observed_difference']>0 for x in boundary if x['seed']==seed) and any(x['observed_difference']<0 for x in boundary if x['seed']==seed)]
noise_values=sorted(set(x['channel'][1] for x in boundary)); boundary_pass=len(crossing_seeds)>=2 and len(noise_values)>=2
add(f'| 固定β核心族边界两侧 | ≥2 seed且≥2噪声区间 | 两侧seed={crossing_seeds}；可评估σ={noise_values} | '+('通过' if boundary_pass else '信息不足，未通过')+' |')
add()
badseeds=sorted(set(x['seed'] for x in q['conditions_over_05']))
add(f'4. **按预注册：适用域多数失败时否定共同增量通道作为本轮几何检索器的已校准近似。** 由于E1自身也会被小cell样本的严格诊断拒绝，必须同时保留“校准设计辨别力不足”的解释，不能将全部失败直接归因于几何难负例。适用域通过且测试MAE或候选集regret>0.05的seed={badseeds}；'+('达到多个seed反证迁移预测的条件。' if len(badseeds)>=2 else '未达到多个seed这一更强反证条件；支持标准仍未满足。')+'不声称已证伪模型内A/B/C。')
add()
add('5. **配对与seed报告要求通过；总体显著性不作结论。** `paired_seed_differences.csv` 提供seed级原始配对均值和95%区间；NPZ保存逐查询端点正确向量。区间条件于固定key与源，仅三个seed，不能把8192次噪声查询当8192个独立语料/用户。')
add()
add('## 机制、相界及误差归因')
add()
add('下表为G32、epsilon=4/15、reader0、均匀查询的核心端点；Δ为摘要accuracy减原文accuracy。临界预测差不足0.02的条件仅报告差与区间，不判排序。')
add()
add('| 层 | seed | 通道 | 校准λ | 摘要accuracy | 原文accuracy | 配对Δ [95%区间] |')
add('|---|---:|---|---:|---:|---:|---|')
for layer in ('E1','E2'):
    for row in s[layer]['boundaries']:
        if layer=='E1' and row['channel'][1:]!=[1.,1.]:continue
        if layer=='E1' and row['channel'][0] not in (31.,112.,255.):continue
        ci=row['ci95'];add(f'| {layer} | {row["seed"]} | {row["channel"]} | {f(row["lambda_hat"],2)} | {f(row["summary_accuracy"])} | {f(row["raw_accuracy"])} | {f(row["observed_difference"])} [{f(ci[0])}, {f(ci[1])}] |')
add()
add('![端点准确率差](experiments/exp1/endpoint_accuracy.png)')
add()
add('细线为三个固定key的seed实现，粗线为seed均值，虚线为独立校准预测；不表示新语料总体置信带。完整可导出矢量图见endpoint_accuracy.svg。')
add()
add('几何检索的差异有结构：τ=.3时三个seed都在σ=1到2之间出现端点反转，预测曲线也较接近；但σ=2未通过冻结适用域诊断，不能算作合格迁移证据。τ=1时高噪声下原文仍明显占优，单一共同增量预测失配更大；不能用τ=.3的较好结果掩盖这个反例。')
add()
add('E1事前数字：λ31摘要/原文=.652381/.600629，λ112=.735417/.735417，λ255=.767596/.834204；这些指定通道数值不是用QA结果回拟合得到。E2校准λ只是模型拟合坐标，不能把σ直接等同于λ。')
add()
for layer in ('E1','E2'):
    q=s[layer];add(f'- {layer}：等候选填充式(26)的优势残差绝对值MAE {f(q["padding_residual_gain"]["mae"])}、最大 {f(q["padding_residual_gain"]["max"])}；独立reader的端点Recall比较 {q["reader_recall_comparisons"]} 次，最大变化 {f(q["reader_recall_max_difference"])}；冷归档逐查询完全一致={q["cold_archive_identical"]}。')
add()
add('P1/P2的数值比较见逐条件预测/实际优势与配对表；原方案没有为这些比值另给显著性通过门槛，因此不事后创造门槛。E1两侧反转支持指定通道机制；E2若不满足等负载关系，先检查摘要/原文目标margin与同段难负例差异，不改λ迎合测试答案。P4的Recall不变是实现独立性的检查，不能单凭它证明相界迁移。P5由穷举通过，P6由逐查询相等通过。')
add()
add('| E1 λ=31、β=.6机制 | 三seed配对差均值（摘要减原文） |')
add('|---|---:|')
core=[d for d in records if d['layer']=='E1' and d['G']==32 and d['dist']==0 and d['epsilon']==4/15 and d['nu_s']==0 and d['nu_r']==0 and d['channel']==[31.,1.,1.]]
for label,key in [('未填充','full_summary-full_raw'),('填充96项后','padded_summary-full_raw')]:
    vals=[d['paired'][key]['mean'] for d in core]
    if vals:add(f'| {label} | {np.mean(vals):.6f} |')
add()
mech=json.loads((root/'mechanisms_results.json').read_text())
add('预注册同熵补充：p=(.5,.5,0)与p′=(.7729078,.1135461,.1135461)均为H(I)=1；β=.6、b=1时理论风险分别.35和.268128。源与reader固定规则相同，各seed实际风险如下（单活动项检索必命中；没有另拟合相界）。')
add()
add('| seed | p的实际风险 | p′的实际风险 |')
add('|---:|---:|---:|')
for seed in (0,1,2):
    rr=[x for x in mech['same_entropy'] if x['seed']==seed];add(f'| {seed} | {rr[0]["actual_risk"]:.6f} | {rr[1]["actual_risk"]:.6f} |')
add()
add('P1补充固定目标通道、加零查询概率项：λ31、G32，d=32/96时预测优势比分别63/95=.663158和63/159=.396226；各seed实际比值（积分掉公平猜测噪声）见mechanisms_results.json。这些使用已知生成参数的机制补充不能替代独立校准E1门槛。')
add()
add('经历级原文路由作为独立增强架构列出：全部原文载荷、G个活动key、每经历8-byte偏移表、一次字段随机访问；pad/cold也分别记录多出的包/冷归档bytes。各行含实际存储bytes、活动key bytes、覆盖质量、缺失率、条件Recall、实际accuracy、独立校准预测及两参数扩展预测。所有Zipf regret都仅相对于所扫候选集；均匀族也只扫10个固定分配排列，不能声称枚举3^G个几何索引。经验最佳在同测试集选取，regret带选择偏差。')
add()
add('假设范围：ε=.1、ν_r=.1、ν_s=0时真实β=.85/.8=1.0625，超出定理C的β≤1范围，B的u_s<u_r也可能不成立；这些条件只检验一般风险式/整数DP及reader轴，不能作为压缩拥塞相界的支持证据。模型外边界两侧的判定单独使用固定β=.6核心族。适用域诊断是有限平均Recall检验，不能证明逐经历、全策略统一误差η。')
add()
diag=json.loads((root/'diagnostic_examples.json').read_text()) if (root/'diagnostic_examples.json').exists() else {}
if diag.get('largest_uniform_E2_regret'):
    ex=diag['largest_uniform_E2_regret']; dp=ex['selected_dp']; best=ex['best_count_scan']
    add(f"描述性反例（事后按最大经验regret选取，不是新检验）：{ex['id']}，预算{dp['budget']}，模型DP选(n,s)=({dp['n']},{dp['s']})，预测accuracy={dp['predicted_accuracy']:.4f}、实际={dp['accuracy']:.4f}；计数扫描已有(n,s)=({best['n']},{best['s']})、排列{best['permutation']}，实际accuracy={best['accuracy']:.4f}。该条件原记录regret={dp['regret']:.4f}。完整信息见diagnostic_examples.json。")
    add()
add('## 主定理判断与下一轮')
add()
add('A（整数DP）及C（模型内整数相图）获得有限实例支持，B（顶点/对偶/取整）获得有限实例支持且已有QED L3 PASS；这些结果均不证明自然检索器遵循共同增量分母。模型外迁移预测本轮不获支持。完整网格因阶段上限而未执行的部分不补成成功；真实evidence未运行，因为本题目录内无现成数据，不跨项目读取。')
add()
add('下一轮先独立预注册校准样本分配：保持0.03/0.04标准，减少诊断cell或增加每cell样本，并用已知E1生成通道先验证诊断的误拒率。随后用新冻结key与新测试集，区分摘要/原文目标margin和同段/异段难负例；允许不同λ_s、λ_r或块结构竞争时必须重推风险与求解，不能原样引用单分式DP/两阈值。保留经历级路由强基线，若可靠免费路由消除了拥塞，应研究访问成本，而不是继续把摘要优势归于候选减少。')
add()
add('当前最大未解风险：共同增量、逐字段竞争与擦除reader是过强的可解性假设；有限校准cell噪声又妨碍模型适用域判定。应先提升检验的辨别力，再判断需要哪种最小模型扩展。')
(project/'RESULTS_EXP1.md').write_text('\n'.join(lines)+'\n')
print('Wrote RESULTS_EXP1.md')
