"""Assemble the report from frozen gates and the read-only proof audit; no new test."""
import csv,hashlib,json,math,time
from pathlib import Path
from collections import Counter
from design import ROOT,OUT

def readcsv(name):return list(csv.DictReader((OUT/name).open()))
def truth(value):return value is True or value=='True'
def writecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=json.loads((OUT/'analysis_summary.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text())
assert audit['prereg_and_all_declared_hashes_match']
p2=readcsv('P2_gates.csv');models=readcsv('model_gates.csv');noise=readcsv('P3_noise_gates.csv');old=readcsv('old_76_retest.csv')
new=[]
for r in p2:
    if r['d']!='512':continue
    rr=[v for v in models if v['kind']=='P2' and all(v[k]==r[k] for k in ('original_d','d','kappa','name','metric'))]
    zero=next(v for v in rr if v['model']=='zero');supported=[v['model'] for v in rr if v['model']!='zero' and truth(v['compatible']) and truth(v['calibration_b_significant'])]
    anymodel=any(truth(v['compatible']) for v in rr)
    correction=not truth(zero['compatible']) and bool(supported)
    noiseok=truth(r['direction_confirmed']) and truth(r['noise_compatible_1pp'])
    if correction:label='有限修正支持，速率'+('未分辨' if len(supported)>1 else supported[0])
    elif not anymodel:label='冻结修正族不足'
    elif noiseok:label='噪声解释相容到1pp'
    else:label='未分辨'
    new.append({key:r[key] for key in ('original_d','d','kappa','name','metric','mean','SE','target')}|
               dict(direction_confirmed=truth(r['direction_confirmed']),noise_gate_with_direction=noiseok,
                    compatible_models=','.join(v['model'] for v in rr if truth(v['compatible'])),classification=label))
writecsv('P2_attribution.csv',new)
text=(ROOT/'RESULTS_EXP2.md').read_text()
text=text.replace('512查询、16新种子','512查询、112新种子')
intro_end=text.index('\n\n## G0')
old_intro=text[:intro_end]
old_noise=Counter(r['noise_class'] for r in old)
old_nonshrinking_noise=Counter(r['noise_class'] for r in old if not truth(r['pass_trend']))
main=(f"# EXP2 验证结果\n\n结论：P2 原四个失败条件在112个全新种子下方向确认 {sum(truth(r['direction_confirmed']) for r in p2 if r['d']==r['original_d'])}/4，d=512的24种子复制确认 {sum(truth(r['direction_confirmed']) for r in p2 if r['d']=='512')}/4；高维归因门槛为 {dict(Counter(r['classification'] for r in new))}。"
      f"两步Sign的无拟合 B/√d 与Adam代表的冻结 K/d 权重误差预测通过 {summary['mechanism_pass']}/{summary['mechanism_total']} 格，但曲线偏差不能据此归因。P3原76项失败有 {summary['old76_retest_pass']} 项在新样本下缩小，仍有 {76-summary['old76_retest_pass']} 项未缩小；全240曲线的新趋势失败 {240-summary['curve_trends_pass']} 项。"
      f"旧76项的去噪归因为 {dict(old_noise)}；全部d256分类为 {summary['P3_low_noise']}。这些结果支持的噪声/有限维结论有明确精度边界，未分辨项保留，EXP1判决不改。QED核心证明已核查并补齐M组装稿，本轮未发现关键环节的明确断裂；尚需统一全文、完整独立复审和有限维修正定理。当前交付仍不足以直接宣称成熟主会论文。")
text=main+text[intro_end:]
analysis_resource=json.loads((OUT/'analysis_resource.json').read_text())
peak=max(summary['metadata']['max_rss_MiB'],analysis_resource['max_rss_MiB'],audit['max_rss_MiB'])
text=text.replace('峰值RSS ', '冒烟／正式阶段峰值RSS ',1)
text=text.replace('## 事前功效和实际精度',f'另记录汇总阶段自身RSS峰值{analysis_resource["max_rss_MiB"]:.2f} MiB、独立重放审计峰值{audit["max_rss_MiB"]:.2f} MiB；冒烟、正式、汇总和独立审计四阶段有记录的峰值为{peak:.2f} MiB，而非只报正式生成阶段。事前两次规划计算同样使用2GiB硬限制，但未单独保留RSS峰值，故不声称这个数覆盖规划峰值。汇总亦在2GiB/零交换/2线程scope内完成，见analysis_resource.json。\n\n## 事前功效和实际精度')
g0point='## G0：冻结、实现与资源\n\n'
text=text.replace(g0point,g0point+'首次规划版因三项P2事前功效不足80%，在任何EXP2数据生成前将四个原失败条件定向增至112种子；旧版及其散列完整保存在predata_v1，最终方案随后冻结。冒烟和正式数据都只按最终方案生成，没有看新结果后追加种子。\n\n')
needle='## G2：可预测的有限维修正'
attribution='完整128→256→512三维配对差、实际种子数与SE见P2_dimension_gradient.csv；中间维度仅作描述，没有新增门槛。\n\n### P2归因门槛\n\nG3支持修正时优先归入修正；否则G4必须同时满足方向确认与相对极限误差的1个百分点等效性。“相容”不等于证明旧失败的唯一成因。\n\n| 原d | kappa | 方法/指标 | d512归类 |\n|---:|---:|---|---|\n'
for r in new:attribution+=f"| {r['original_d']} | {r['kappa']} | {r['name']}/{r['metric']} | {r['classification']} |\n"
text=text.replace(needle,attribution+'\n'+needle)
powers=json.loads((OUT/'power.json').read_text())
highpower='高维24种子的事前功效保守沿用d256方差分量，四条件依次为：'+', '.join(f"原d{p['original_d']}/k{p['kappa']}/{p['metric']}={p['power']:.3f}" for p in powers if p['kind']=='P2' and p['d']==512)+'。低维112种子的功效不能直接转移到高维24种子的归因门槛；G4的去噪平方差等效性也没有被承诺为逐格80%功效。\n\n'
text=text.replace('## G1：P2四个旧失败的独立复核',highpower+'## G1：P2四个旧失败的独立复核')
# Old two mean-rank failures are a primary interpretability focus, not a new significance family.
rankfocus=[r for r in old if r['metric']=='mean_rank']
focus='### 原两个平均秩非单调项\n\n| kappa | 方法 | 新d128绝对差 | 新d256绝对差 | 新趋势 | G4 |\n|---:|---|---:|---:|---|---|\n'
for r in rankfocus:focus+=f"| {r['kappa']} | {r['name']} | {float(r['value128']):.7f} | {float(r['value256']):.7f} | {'缩小' if truth(r['pass_trend']) else '未缩小'} | {r['noise_class']} |\n"
text=text.replace('## 分布诊断与解释边界',focus+'\n## 分布诊断与解释边界')
trends=readcsv('all_trends.csv')
zero_ties=[r for r in trends if r['metric'] in ('mean_rank','top0.01','top0.1','top0.5') and not truth(r['pass_trend']) and float(r['value128'])==float(r['value256'])==0.]
text=text.replace('### 原两个平均秩非单调项',f'新曲线趋势失败中有{len(zero_ties)}项是两维都测到0的非首步指标；冻结规则仍将0=0记作未缩小，没有改判。\n\n### 原两个平均秩非单调项')
text=text.replace('### 原两个平均秩非单调项',f'原76项按G4分类：{dict(old_noise)}；其中新趋势仍未缩小的25项分类：{dict(old_nonshrinking_noise)}。这些噪声相容结论都带0.5个百分点精度边界。\n\n### 原两个平均秩非单调项')
levels=[]
for r in noise:
    n=int(r['seeds']);mean=float(r['mean']);A=float(r['A']);v=float(r['query_noise_variance'])
    levels.append({key:r[key] for key in ('d','kappa','name','metric','seeds')}|dict(mean_bias=mean,query_variance=v,full_dataset_gap_second_moment=A,between_dataset_variance_untruncated=n/(n-1)*(A-mean*mean)+v/(n-1)))
writecsv('sampling_levels.csv',levels)
text=text.replace('## G5：EXP1冻结趋势原样复核','需区分三个层次：G3检验跨数据集平均的有符号系统偏差；A检验查询抽样以外的完整有限数据集差；后者即使显著，也可能主要是不同数据集的随机环境波动，不能自动叫作确定性的b/d修正。sampling_levels.csv给出未经截零的数据集间方差估计，仅作描述，不追加门槛。\n\n## G5：EXP1冻结趋势原样复核')
placeholder='本节的QED源码核查、缺口和投稿判断由研究员在读取允许的证明证据后补全；不得把此占位句当作最终交付。'
proof='''已按用户点名的只读例外读取外部QED结果，本轮五份problem.tex都与lemmas_v2逐字一致；源文件与SHA-256保存在qed_readonly/source_manifest.json。L1、L2的proof_2、L3、L5详细核验均为PASS；L4有完整证明正文但未见分解详细PASS标签。本轮未调用QED/LLM/API，未写外部目录，也未执行/import其代码。

正文核查显示，L2已经用Hermite奇函数协方差算子界、真实训练候选高斯回归、三候选联合行CLT及交换性CDF方差完成代理路线；L3已经用Sign首次分歧计数、Adam可达历史商恒等式与O_P(log d)的Kronecker框架界完成矩阵/完整训练对角/错误候选转移。尤其beta2=0<beta1没有被偷偷排除。不能继续把L2/L3本身当作完全未解目标。

当前缺口分为三层：第一，需把代理函数类、q_T>0、固定T/e参数及随机查询归一化写进统一M；本轮M_assembly.md给出Adam全局Lipschitz接合、q_T>0、Sign代理函数类及L4的组装证明；proof_audit.md核对随机查询转移，现有链条已有闭环的数学内容，尚未交付完成第二次完整独立复审的论文全文。第二，EXP2的B/√d和K/d系数、余项及曲线有符号偏差尚没有相应定量定理；L3的Sign上界d^-1/3不矛盾于更快的经验d^-1/2，但也不证明它。第三，top1极值、T增长、e趋0、长尾及单数据集召回率集中不属于已接合的M。没有发现明确反例不等于所有扩展成立。

两条具体推进路线是：条件联合CLT计算Z在±alpha附近与残差扰动的联合密度，保留训练自作用以求跳点首项；或先平滑门控、定量留一/留二控制得分和候选CDF，再给出去平滑误差回到原Sign。平方权重差的d^-1/2首项不自动等于有符号曲线首项；阈值扰动的对称抵消可能使后者从d^-1开始，需推导而非从拟合误差选速率。

主会判断：当前可以形成较扎实的研究报告和真实有限步坐标优化器理论候选，尚不能仅凭自动PASS与这些有限维表格宣称论文已经成熟。若完成现有核心证明的独立复审、统一M全文与本轮已补全的真正Adam接合，并按本目录NOVELTY所列最接近工作逐条全文核对差异，则“真实重复训练高斯键下的有限步Sign/Adam秩律及显式步长反转”有继续冲主会的理由。仅有标量阈值、近白化直觉或相对秩图表不够；当前没有重新外部查新，也没有证明新的最优top1容量。详细核查及不依赖实验的推导见proof_audit.md。'''
assert placeholder in text;text=text.replace(placeholder,proof)
text+='\n## 最终独立审计\n\n'
text+='另以全矩阵X/Y重放一个已存在的正式数据集（跨两个8192列流块），独立原始W单位Adam及稠密512×n候选比较，1536个秩与正确分数一致，另验证正缩放秩不变；没有生成新种子。\n\n'
text+=f"逐配置重新构造全部预定方法集合和查询RNG，核对 {audit['completed_configurations']}/{audit['expected_configurations']} 个配置、{audit['actual_rows']}/{audit['expected_rows']} 条记录、全部原始秩与嵌套查询指标、逐查询对角MSE、线程、cgroup硬限制和零OOM。声明于注册正文的散列与frozen_inputs.json逐项一致；详情 independent_audit.json。\n"
text=text.replace('## 分布诊断与解释边界','关于非单调的机制性推断：Sign权重平方误差若为d^(-1/2)，得分扰动RMS约d^(-1/4)。在固定阈值反集中近似下，指示差的discordance约d^(-1/4)，仅查询噪声对种子曲线绝对差的贡献就可能约为d^(-1/8)/sqrt(M)，维度翻倍只下降约8%。因此增加查询和种子也不保证每格严格单调。这只是基于冻结机制的解释性推断，未检验新的指数、未设新的门槛，也不能代替G3/G4归因。\n\n## 分布诊断与解释边界')
# Complete all numerical work, then record its actual total budget use.
used=time.time()-json.loads((OUT/'budget_start.json').read_text())['start_unix'];assert used+7<3600
(OUT/'completion_budget.json').write_text(json.dumps(dict(elapsed_total_seconds=used,limit_seconds=3600,planning_seconds_upper_bound=7,with_planning_upper_bound_seconds=used+7,
    numerical_work_complete=True,no_additional_sampling=True),indent=2)+'\n')
text+=f"\n从冒烟启动至最终数值汇总共 {used:.2f} 秒（{used/60:.2f} 分钟）；事前两次规划计算合计约6.35秒，按7秒上界相加仍小于60分钟。此后只做文稿与文件摘要整理，不再生成实验数据。\n"
text=text.replace('(8)已经验验证','式(8)已经数值验证').replace('既有P1/P4成功仍不等于正确方差、经验CDF与真实迭代转移已经完成数学证明。','既有P1/P4实验成功本身不能证明正确方差、经验CDF与真实迭代转移；证明候选链的当前进展另见下一节。')
(ROOT/'RESULTS_EXP2.md').write_text(text)
(OUT/'final_summary.json').write_text(json.dumps(dict(P2_original_direction_confirmed=sum(truth(r['direction_confirmed']) for r in p2 if r['d']==r['original_d']),P2_high_direction_confirmed=sum(truth(r['direction_confirmed']) for r in p2 if r['d']=='512'),P2_high_attribution=dict(Counter(r['classification'] for r in new)),P3_old76_noise=dict(old_noise),P3_old76_nonshrinking_noise=dict(old_nonshrinking_noise),P3_old76_shrinking=summary['old76_retest_pass'],P3_new_curve_failures=240-summary['curve_trends_pass'],mechanism_pass=summary['mechanism_pass'],mechanism_total=summary['mechanism_total'],recorded_peak_RSS_MiB=peak,total_wall_seconds=used,total_wall_with_planning_upper_bound=used+7),ensure_ascii=False,indent=2)+'\n')
print(main);print('total numerical seconds',used)
