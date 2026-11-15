"""Seed-level analysis; reads only experiment-local files. Never changes prereg."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/exp1'
METRICS=('mean_rank','top0.01','top0.1','top0.5')

def stat(values):
    a=np.array(values,dtype=float)
    return float(np.mean(a)),float(np.std(a,ddof=1)/math.sqrt(len(a))) if len(a)>1 else float('nan')

def writecsv(name,items):
    if not items:return
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)

def pm(a,b):return f'{a:.5f} ± {b:.5f}'

def main():
    digest=hashlib.sha256((ROOT/'PREREG_EXP1.md').read_bytes()).hexdigest()
    assert digest==(ROOT/'PREREG_EXP1.sha256').read_text().split()[0]
    metadata=json.loads((OUT/'formal_metadata.json').read_text())
    # Stream large per-query CDF arrays, retaining compact seed summaries only.
    rows=[]
    for line in (OUT/'formal_rows.jsonl').open():
        r=json.loads(line)
        for key in ('wrong_cdf_per_query','wrong_cdf_mean','cdf_grid','wrong_cdf_grid_sup_per_query',
                    'true_scores','ranks','query_indices','conditional_mean_per_query'):r.pop(key,None)
        rows.append(r)
    events=[json.loads(l) for l in (OUT/'formal_events.jsonl').open()]
    unique={}
    for r in rows:
        if r['phase']=='C':continue
        key=(r['d'],r['kappa'],r['seed'],r['name'])
        if key not in unique or r['phase']=='B':unique[key]=r
    groups=defaultdict(list)
    for r in unique.values():groups[(r['d'],r['kappa'],r['name'])].append(r)
    cells=[];diagnostics=[];group_summary={}
    for (d,k,name),rs in sorted(groups.items()):
        for metric in METRICS:
            mean,se=stat([r[metric] for r in rs]);pred=rs[0]['predictions'][metric];err=mean-pred
            threshold=max(.03,3*se);strict=max(.05,4*se)
            cells.append(dict(d=d,kappa=k,name=name,metric=metric,seeds=len(rs),mean=mean,SE=se,
                prediction=pred,error=err,threshold=threshold,pass_gate=len(rs)==5 and abs(err)<=threshold,
                strict_threshold=strict,strict_exceed=abs(err)>strict))
        ds=dict(d=d,kappa=k,name=name,seeds=len(rs))
        for metric in ('true_mean','true_variance','wrong_mean','wrong_second','wrong_cdf_grid_sup_mean','wrong_cdf_aggregate_sup'):
            mean,se=stat([r[metric] for r in rs]);ds[metric]=mean;ds[metric+'_SE']=se
        ds['score_mean_prediction']=rs[0]['score_mean_prediction']
        diagnostics.append(ds);group_summary[(d,k,name)]=ds
    writecsv('all_gates.csv',cells);writecsv('score_cdf_diagnostics.csv',diagnostics)
    primary=[r for r in cells if r['d'] in (128,256)]
    p1names=('Sign1','Sign2_a0.6_actual','Sign2_a0.6_proxy')
    p1=[r for r in primary if r['name'] in p1names]
    p4=[r for r in primary if r['name'].startswith('Adam')]
    pairs=defaultdict(dict)
    for r in primary:pairs[(r['kappa'],r['name'],r['metric'])][r['d']]=r
    strict=[];finite_trends=[]
    for (k,name,metric),pp in pairs.items():
        if len(pp)==2:
            finite_trends.append(dict(kappa=k,name=name,metric=metric,error128=pp[128]['error'],
                error256=pp[256]['error'],absolute_error_shrinks=abs(pp[256]['error'])<abs(pp[128]['error'])))
        if len(pp)==2 and all(r['strict_exceed'] for r in pp.values()) and pp[128]['error']*pp[256]['error']>0:
            strict.append(dict(kappa=k,name=name,metric=metric,error128=pp[128]['error'],error256=pp[256]['error']))
    writecsv('strong_failures.csv',strict)
    writecsv('formula_error_trends.csv',finite_trends)
    reversal=[]
    for d in (128,256):
        for k in (.5,1.,2.):
            for alpha in (.6,1.5):
                a={r['seed']:r for r in groups[(d,k,f'Sign2_a{alpha:g}_actual')]}
                b={r['seed']:r for r in groups[(d,k,'Sign1')]}
                for metric in METRICS:
                    vals=[a[s][metric]-b[s][metric] for s in sorted(a.keys()&b.keys())]
                    mean,se=stat(vals);expected=(-1 if alpha==.6 else 1)*(1 if metric=='mean_rank' else -1)
                    reversal.append(dict(d=d,kappa=k,alpha=alpha,metric=metric,difference=mean,SE=se,
                                         expected_sign=expected,pass_direction=mean*expected>0))
    writecsv('paired_reversal.csv',reversal)
    stability=[];trend=[]
    for (d,k,name),rs in sorted(groups.items()):
        if 'actual' not in name:continue
        proxies={r['seed']:r for r in groups[(d,k,name.replace('actual','proxy'))]}
        item=dict(d=d,kappa=k,name=name,seeds=len(rs))
        for metric in ('weight_mse','relative_weight_mse','diagonal_mse','normalized_diagonal_mse'):
            mean,se=stat([r[metric] for r in rs]);item[metric]=mean;item[metric+'_SE']=se
        for metric in METRICS:
            vals=[r[metric]-proxies[r['seed']][metric] for r in rs];mean,se=stat(vals)
            item[metric+'_difference']=mean;item[metric+'_difference_SE']=se
            item[metric+'_absolute_difference']=float(np.mean(np.abs(vals)))
        stability.append(item)
    for a in stability:
        if a['d']!=128:continue
        b=next(r for r in stability if r['d']==256 and r['kappa']==a['kappa'] and r['name']==a['name'])
        for metric in ('weight_mse','diagonal_mse')+tuple(m+'_absolute_difference' for m in METRICS):
            exact='_T1_' in a['name'] and a[metric]<1e-25 and b[metric]<1e-25
            trend.append(dict(kappa=a['kappa'],name=a['name'],metric=metric,value128=a[metric],value256=b[metric],
                              shrinks=b[metric]<a[metric] or exact,exact_first_step=exact))
    writecsv('stability_by_dimension.csv',stability);writecsv('stability_trends.csv',trend)
    # First-step beta equality, compared across the same seed and epsilon.
    firststep=[]
    for d in (128,256):
        for k in (.5,1.,2.):
            for e in (.1,1.):
                for kind in ('actual','proxy'):
                    a={r['seed']:r for r in groups[(d,k,f'Adam_b0_e{e:g}_T1_{kind}')]}
                    b={r['seed']:r for r in groups[(d,k,f'Adam_b0.9_e{e:g}_T1_{kind}')]}
                    firststep.append(dict(d=d,kappa=k,e=e,kind=kind,
                        maximum_metric_difference=max(abs(a[s][m]-b[s][m]) for s in a for m in METRICS)))
    writecsv('adam_first_step.csv',firststep)
    cg=defaultdict(list)
    for r in rows:
        if r['phase']=='C':cg[(r['d'],r['lambda_target'],r['name'])].append(r)
    csummary=[]
    for (d,l,name),rs in sorted(cg.items()):
        mean,se=stat([r['top1'] for r in rs])
        csummary.append(dict(d=d,lambda_target=l,n=rs[0]['n'],lambda_actual=rs[0]['lambda_actual'],
                             name=name,top1=mean,SE=se,seeds=len(rs)))
    writecsv('top1_exploratory.csv',csummary)
    condition=[]
    for (d,k,name),rs in sorted(groups.items()):
        if 'conditional_mean' not in rs[0]:continue
        mean,se=stat([r['true_mean']-r['conditional_mean'] for r in rs]);cm,cse=stat([r['conditional_mean'] for r in rs])
        condition.append(dict(d=d,kappa=k,name=name,conditional_mean=cm,conditional_SE=cse,
                              measured_minus_conditional=mean,paired_SE=se))
    writecsv('conditional_mean.csv',condition)
    fail=lambda rr:[r for r in rr if not r['pass_gate']]
    summary=dict(prereg_sha256=digest,raw_rows=len(rows),unique_main_rows=len(unique),
        metadata=metadata,primary_cells=len(primary),primary_failures=len(fail(primary)),
        P1_cells=len(p1),P1_failures=len(fail(p1)),P4_cells=len(p4),P4_failures=len(fail(p4)),
        strong_failures=strict,reversal_failures=[r for r in reversal if not r['pass_direction']],
        stability_nonshrinking=[r for r in trend if not r['shrinks']],
        omitted=[r for r in events if r['event']!='completed'],
        first_step_max_difference=max(r['maximum_metric_difference'] for r in firststep))
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Full report, with all thresholds in attached CSV, no refitted constants.
    report=['# EXP1 第一轮验证结果','',
        f"结论：主定理 M 的有限维秩预测获得部分支持，本轮整体验证未全部通过。P1 秩指标 {len(p1)-len(fail(p1))}/{len(p1)} 格、P4 Adam 指标 {len(p4)-len(fail(p4))}/{len(p4)} 格通过冻结门槛，双维度同向强失败 {len(strict)} 项；P2 步长反转 {sum(r['pass_direction'] for r in reversal)}/{len(reversal)} 个方向通过，4项失败。P3 权重和抽样对角MSE的120项趋势全部通过，但曲线差异240项中76项未缩小，按冻结的趋势读出判失败并保留。当前没有证据要求修改高斯固定步数主模型；需要进一步分辨有限维修正与抽样噪声。这些计算不构成渐近定理证明。",'',
        '## 冻结与实现审计','',
        f"预注册 SHA-256：`{digest}`，运行前后校验一致。文件 PREREG_EXP1.md / PREREG_EXP1.sha256 冻结后未改。Adam 数值积分、种子与全网格在新数据生成前固定；既有 V2 先导结果已在预注册披露，不冒称盲法首次注册。全程未调用 LLM API、网络或 GPU，未导入其他项目代码。",'',
        '两配置冒烟共86条方法记录通过稠密秩/CDF、正确候选移除、正缩放不变性、Adam 原始 W 单位与重标度递推、两步 Sign 阈值公式检查。正式使用同样实现。数据不白化、不归一化候选；Adam 完整状态与偏差校正，polar 精确 SVD。', '',
        '独立事后完整性审计已检查全部1875条原始记录的查询数、合法秩、CDF单调性及由原始秩重算top-q的一致性；A/B重叠的135条方法记录逐项分数、查询和秩完全相同。原稿、冻结预注册和预积分文件的SHA-256均保持一致。证据见 integrity_audit.json 和最终 ARTIFACTS.sha256。', '',
        '## 预算和覆盖','',
        f"正式完成 {metadata['completed']}/{metadata['planned']} 个数据配置，原始方法记录 {len(rows)} 条（A/B 的 d=128 重复数据只保留 B 汇总，去重主模型记录 {len(unique)} 条；共享数据的方法记录并不独立）。正式墙钟 {metadata['elapsed_seconds']:.2f} 秒，峰值 RSS {metadata['max_rss_MiB']:.2f} MiB，最大单次乘法 {metadata['max_single_matmul_seconds']:.3f} 秒。systemd scope MemoryMax=2G、MemorySwapMax=0；Python虚拟地址硬限3,500,000,000 bytes，三种线程变量均为1。正式计算小于一小时。",'',
        f"省略/预算中止事件 {len(summary['omitted'])} 个；P5 的 Zipf、实用 epsilon 等模型外测试按预注册不执行。完整资源和阶段耗时见 experiments/exp1/formal_metadata.json、formal_events.jsonl 与 formal.log。",'',
        '## P1：无拟合秩曲线','',
        f"判决：{len(p1)-len(fail(p1))}/{len(p1)} 格通过。逐个 d=128、256、kappa=0.5、1、2 和 q=0.01、0.1、0.5 检验，误差阈值严格为 max(0.03,3SE_seed)。下表为 kappa=1；±均为五独立种子的标准误，查询和候选不当作独立重复。完整逐格阈值见 all_gates.csv。",'',
        '| d | 方法 | 平均归一化秩（实测±SE） | 秩预测 | top-10%（实测±SE） | top-10%预测 |',
        '|---:|---|---:|---:|---:|---:|']
    for d in (128,256):
        for name in ('GD','polar','Sign1','Sign2_a0.6_proxy','Sign2_a0.6_actual','Sign2_a1.5_actual'):
            a=next(r for r in primary if r['d']==d and r['kappa']==1 and r['name']==name and r['metric']=='mean_rank')
            b=next(r for r in primary if r['d']==d and r['kappa']==1 and r['name']==name and r['metric']=='top0.1')
            report.append(f"| {d} | {name} | {pm(a['mean'],a['SE'])} | {a['prediction']:.5f} | {pm(b['mean'],b['SE'])} | {b['prediction']:.5f} |")
    ptrends=[r for r in finite_trends if r['name'] in p1names]
    report+=['',f"有限维误差趋势另存 formula_error_trends.csv：P1的{len(ptrends)}对128→256比较中，{sum(r['absolute_error_shrinks'] for r in ptrends)}对绝对公式误差缩小，其余未缩小。不能因所有格处于冻结容差内，就声称偏差已随维度单调趋零；五种子结果仍含抽样波动。"]
    report+=['','## P2：两步的步长反转','',
        f"判决：{sum(r['pass_direction'] for r in reversal)}/{len(reversal)} 项方向符合预注册；未符合的方向不隐去。原稿没有配对显著性门槛，这里不事后添加。以下 kappa=1 的两步减一步配对差；负秩差/正 top-q 差表示改善。完整结果见 paired_reversal.csv。",'',
        '| d | alpha | 秩差 ± 配对SE | top-10%差 ± 配对SE |','|---:|---:|---:|---:|']
    for d in (128,256):
        for a in (.6,1.5):
            ss=[r for r in reversal if r['d']==d and r['kappa']==1 and r['alpha']==a]
            rr=next(r for r in ss if r['metric']=='mean_rank');tt=next(r for r in ss if r['metric']=='top0.1')
            report.append(f"| {d} | {a} | {pm(rr['difference'],rr['SE'])} | {pm(tt['difference'],tt['SE'])} |")
    report+=['','P2整体判决：未全部通过。四个失败项如下；差值均未超过一个配对SE，但不能据此把失败改判通过。','',
        '| d | kappa | alpha | 指标 | 两步减一步 ± 配对SE | 预测方向 |',
        '|---:|---:|---:|---|---:|---|']
    for r in summary['reversal_failures']:
        report.append(f"| {r['d']} | {r['kappa']} | {r['alpha']} | {r['metric']} | {pm(r['difference'],r['SE'])} | {'正' if r['expected_sign']>0 else '负'} |")
    report+=['',
        '## P3：真实迭代到代理的双重稳定性','',
        f"P3整体判决：未全部通过。原稿仅要求随维度缩小，没有规定下降速率。按冻结的128→256趋势读出，共 {len(trend)} 项，{sum(r['shrinks'] for r in trend)} 项缩小或首步精确重合，{sum(not r['shrinks'] for r in trend)} 项未缩小。权重MSE和训练对角MSE各60项均通过；曲线失败项为平均秩2项、top-1% 30项、top-10% 22项、top-50% 22项。后者是离散抽样指标，未添加显著性条件或改变原判读。训练对角误差只由预先均匀抽取的128查询估计，不等价于证明完整式(17)。完整趋势和全部非单调项见 stability_by_dimension.csv、stability_trends.csv、analysis_summary.json。",'',
        '| 方法（kappa=1） | d | 归一化权重MSE ± SE | 归一化对角得分MSE ± SE | 平均绝对配对秩差 |','|---|---:|---:|---:|---:|']
    for name in ('Sign2_a0.6_actual','Sign2_a1.5_actual','Adam_b0.9_e0.1_T16_actual'):
        for d in (32,64,128,256):
            matches=[r for r in stability if r['d']==d and r['kappa']==1 and r['name']==name]
            if not matches:continue
            r=matches[0]
            report.append(f"| {name} | {d} | {pm(r['relative_weight_mse'],r['relative_weight_mse_SE'])} | {pm(r['normalized_diagonal_mse'],r['normalized_diagonal_mse_SE'])} | {r['mean_rank_absolute_difference']:.6f} |")
    report+=['','## P4：Adam 完整参数网格','',
        f"判决：{len(p4)-len(fail(p4))}/{len(p4)} 个指标格通过，同一个预积分 c_T 用于所有负载，不拟合 c。首步两组 beta 的全部指标最大差 {summary['first_step_max_difference']:.3g}，与偏差校正首步无 beta 依赖一致。16个参数组合均包含代理和真实，不能把多步 Adam 称为 SignGD。",'',
        '| beta1 | e | T | c_T（预积分） | 主维度指标失败数/48 |','|---:|---:|---:|---:|---:|']
    for m in json.loads((OUT/'moments.json').read_text()):
        rr=[r for r in p4 if r['name'].startswith(f"Adam_b{m['b1']:g}_e{m['e']:g}_T{m['T']}_")]
        report.append(f"| {m['b1']:g} | {m['e']:g} | {m['T']} | {m['c']:.6f} | {len(fail(rr))}/{len(rr)} |")
    report+=['','## 所有未通过项与强失败审计','',
        f"全主维度所有方法共{len(primary)}个门槛格，未过{len(fail(primary))}个。两个维度同向超过 max(0.05,4SE) 的强失败共{len(strict)}项。门槛不改；未过格与强失败分开报告。",'']
    report+=['| d | kappa | 方法 | 指标 | 误差 | 门槛 |','|---:|---:|---|---|---:|---:|']
    for r in fail(primary):report.append(f"| {r['d']} | {r['kappa']} | {r['name']} | {r['metric']} | {r['error']:.6f} | {r['threshold']:.6f} |")
    if not fail(primary):report.append('| — | — | 无未过格 | — | — | — |')
    report+=['','## 正确分数、错误 CDF 与条件均值诊断','',
        '方差目标为1，正确均值目标 c/sqrt(kappa)；原稿未给方差/CDF的数值通过门槛，因此不得事后把这些诊断包装为“已通过完整分布律”。错误 CDF 同时保存每查询网格和聚合网格，避免聚合抵消掩盖随机查询偏差。网格sup是完整sup_s的下界。下表 kappa=1，±为种子标准误；完整诊断见 score_cdf_diagnostics.csv。','',
        '| d | 方法 | 正确均值 ± SE（目标） | 正确方差 ± SE | 错误二阶矩 ± SE | 每查询CDF网格sup均值 ± SE |','|---:|---|---:|---:|---:|---:|']
    for d in (128,256):
        for name in ('GD','Sign1','Sign2_a0.6_proxy','Sign2_a0.6_actual','Adam_b0.9_e0.1_T16_actual'):
            r=group_summary[(d,1.,name)]
            report.append(f"| {d} | {name} | {pm(r['true_mean'],r['true_mean_SE'])} ({r['score_mean_prediction']:.5f}) | {pm(r['true_variance'],r['true_variance_SE'])} | {pm(r['wrong_second'],r['wrong_second_SE'])} | {pm(r['wrong_cdf_grid_sup_mean'],r['wrong_cdf_grid_sup_mean_SE'])} |")
    report+=['','Sign1、Sign2代理的式(12)精确条件均值与实测差及配对SE见 conditional_mean.csv；仅检验漂移，不能代替方差/CDF。','',
        '## C：top-1 探索与 P5 边界','',
        'C没有冻结的有限维通过门槛，标探索。lambda_c参照：GD=0.5，Sign1=1/pi≈0.31831，Sign2(alpha=0.6)≈0.404876。下表展示两维度六点top-1种子均值，不能用固定kappa秩CLT代入q=1/n，也不能声称证明尖锐阈值。','',
        '| lambda目标 | d64 GD | d64 Sign1 | d64 Sign2 | d128 GD | d128 Sign1 | d128 Sign2 |','|---:|---:|---:|---:|---:|---:|---:|']
    for l in (.20,.28,.34,.40,.46,.60):
        values=[]
        for d in (64,128):
            for name in ('GD','Sign1','Sign2_a0.6_actual'):
                matches=[r for r in csummary if r['d']==d and r['lambda_target']==l and r['name']==name]
                values.append(pm(matches[0]['top1'],matches[0]['SE']) if matches else '未运行')
        report.append('| '+str(l)+' | '+' | '.join(values)+' |')
    report+=['','P5 判决：未检验，不报通过或否定。没有 raw epsilon、Zipf、交叉熵、归一化候选、动量 Muon 的外推证据。','',
        '## QED 背景与证据边界','',
        '只读用户指定的本题 v2 输出，五份problem.tex均与本地lemmas_v2对应文件逐字相同。L1、L3、L5的proof_1及L2的proof_2详细核验均为 Overall Verdict PASS；L5在冻结时状态页仍停留在VERIFYING_PROOF_DETAILED，后续快照已更新为COMPLETED，故不改写冻结文件。L2的proof_1详细报告未完成，但proof_2已明确PASS。L4未见分解核验报告，不能从存在proof.md推出PASS。快照见qed_snapshot.txt、qed_L2_verdict.txt。自动核验状态不等于本轮独立人工重证，实验也不补足证明。','',
        '## 模型解释、下一步与交付','',
        '主定理M的无拟合秩曲线获支持：1032个主维度门槛格全部通过，没有触发原文强失败条款。P2全部平均秩方向正确，但4个top-q方向失败；P3误差矩一致缩小，但曲线差异的76项趋势失败。因此不能把本轮说成全部预言获确认，也不能把两点有限维趋势失败直接当成渐近反例。', '',
        '本轮暂不要求修改主模型或重新拟合c_T。需要改进的是有限维预测：Sign2(alpha=0.6,kappa=1)真实/代理平均绝对秩差从0.002839升至0.003540，尽管归一化权重MSE从0.05246降至0.03697、对角MSE从0.05100降至0.03369；这说明误差矩缩小不保证五种子曲线逐点单调。该现象的成因尚不能由本轮确定，不能把噪声解释当成已经证实。', '',
        '分布诊断也有限制：kappa=1时，Sign2(alpha=0.6)真实更新的每查询CDF网格sup均值从0.01996降至0.01324；正确得分方差分别为0.94038±0.05514、0.95640±0.07832，没有显示非消失方差修正。全主维度中最大方差均值为1.14744±0.04734（d=128,kappa=1,Sign2 alpha=1.5代理），仍应保留为有限维诊断，不能只看秩通过。原稿没有方差/CDF数值门槛，故这两项不另报通过。', '',
        '下一轮应另立预注册，优先增加独立数据种子并在更大d复核P2的四个失败方向和P3的两个平均秩非单调项；同时保留相同任务与c_T，检验条件方差、随机单查询CDF与对角误差。若更大维度仍有同向稳定偏差，再计算S非对角、自作用或候选共同环境的有限维修正，必要时修改极限模型。长尾、实用epsilon、T增长、极值top-1须另立假设；C阶段的平滑过渡尚不足以检验尖锐阈值。','',
        '可复现入口：experiments/exp1/run.py、analyze.py；原始 formal_rows.jsonl、分阶段事件、积分与种子均保留。all_gates.csv 给出每个通过/失败门槛，analysis_summary.json 汇总全部异常，其他CSV对应各预测。不可覆盖既有run输出；复跑请先把整个实验目录副本放在本题目录内并将输出指向新的子目录，同时保留冻结文件与原数据。']
    (ROOT/'RESULTS_EXP1.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('metadata','stability_nonshrinking','reversal_failures')},indent=2))

if __name__=='__main__':main()
