"""Fill the review with results from the strict single-thread run only."""
import csv
import json
import math
from independent_mc import ROOT, OUT, resources, write_json

def read(name):
    return json.loads((OUT/name).read_text())

summary=read('analysis_summary.json')
rank=read('rank_checks.json')
cells=[row for d in (64,128,256,512) for row in read(f'mc_d{d}.json')['results']]
fits=list(csv.DictReader((OUT/'fits.csv').open()))
moments=list(csv.DictReader((OUT/'moment_checks.csv').open()))
keys=[(v['r'],v['method'],v['task']) for v in read('mc_d64.json')['results']]
def fitrow(key,power):
    return next(v for v in fits if (float(v['r']),v['method'],v['task'])==key and float(v['power'])==power)

lines=['### 6.1 缩放偏差的全部 16 格结果','',
       f"独立物理响应积分得到的系数，与原系数表的最大绝对差为 **{summary['max_coefficient_discrepancy']:.3g}**。"
       '这仅验证系数代数及响应导数约定，不能代替余项证明。', '',
       '下表是 `d(E recall−Rinf)`，单位是召回概率乘 d，**不是百分点**；± 后为一个外层组 Monte Carlo 标准误。', '',
       '|r|方法|任务|候选 B_R|d=64|d=128|d=256|d=512|',
       '|---|---|---|---:|---:|---:|---:|---:|']
for key in keys:
    row=[v for v in cells if (v['r'],v['method'],v['task'])==key]
    lines.append('|'+ '|'.join([str(key[0]),key[1],'旧' if key[2]=='old' else '新',f"{row[0]['BR']:.5f}"]+
                              [f"{v['scaled_bias']:.5f} ± {v['scaled_se']:.5f}" for v in row])+'|')
lines += ['', '### 6.2 外推拟合有明显的模型敏感性', '',
          '两种拟合均使用上述全部四个维度及其 Monte Carlo 标准误，未改幅度。下表截距 ± 一个标准误；'
          '这个误差只含模拟误差，不含拟合模型偏差。完整斜率、残差和描述性统计在 `review18_checks/strict/fits.csv`。','',
          '|r|方法|任务|B_R|b+c/√d 截距|b+c/d 截距|',
          '|---|---|---|---:|---:|---:|']
for key in keys:
    fs=[fitrow(key,p) for p in (.5,1.)]
    lines.append('|'+ '|'.join([str(key[0]),key[1],'旧' if key[2]=='old' else '新',f"{float(fs[0]['BR']):.5f}"]+
                              [f"{float(v['intercept']):.5f} ± {float(v['se']):.5f}" for v in fs])+'|')
max_standardized=max(abs(float(v['intercept'])-float(v['BR']))/float(v['se']) for v in fits if float(v['power'])==1.)
lines += ['',
          '**必须披露的反向结果：** 强制把 `O(d^−1/2)` 当成一条固定斜率直线，会得到看似否定候选的截距。'
          '例如 r=−.5 旧任务 GD 的 `b+c/√d` 截距约 −3.45866，而候选为 −3.27428，相差约 7.05 个纯 Monte Carlo 标准误。'
          '但同一批数据的 `b+c/d` 截距约 −3.29154，标准误 .01618；它的残差平方和统计量约 .0878（2 自由度），'
          '前一种约 5.2106。后者允许的有限维曲率足以解释前者的截距偏移。', '',
          f'对 `b+c/d` 诊断，16 格截距距候选的最大值为 {max_standardized:.3f} 个标准误。'
          '这与候选相容，却不能据此把真实余项升级为已证明的 O(d^−2)。'
          '反过来，前一种直线的截距偏移也不构成对原 O(d^−3/2) 定理的反例：原定理没有要求余项在 1/√d 坐标上严格线性。', '',
          '图中画的是 `d(E recall−Rinf)−B_R`；误差棒为 1.96 个模拟标准误，横轴为 1/√d。', '',
          '![独立缩放偏差诊断](review18_checks/strict/scaled_bias.png)', '',
          '可导出的图：`review18_checks/strict/scaled_bias.pdf`。', '',
          '### 6.3 闭合量与三阶绝对余项的数值压力检查', '',
          '除 recall 外，主程序直接累计了 A、B、AB、投影能量及 lambda 的二、三阶量；全部记录见 '
          '`review18_checks/strict/moment_checks.csv`。两个 d=512 例子如下，均为描述性点估计：','',
          '|r、方法、任务|d Var A / 目标|d Var B / 目标|d Cov(A,B) / 目标|d E R / 目标|d E(lambda−mu)² / 目标|',
          '|---|---:|---:|---:|---:|---:|']
for key in [(0.,'Sign','old'),(-.5,'Adam1','new')]:
    v=next(v for v in moments if int(v['d'])==512 and (float(v['r']),v['method'],v['task'])==key)
    fields=[('varA_coefficient','target_varA'),('varB_coefficient','target_varB'),
            ('covAB_coefficient','target_covAB'),('projection_coefficient','target_projection'),
            ('lambda_second_coefficient','target_lambda_second')]
    lines.append('|'+f"{key[0]:g}，{key[1]}，{'旧' if key[2]=='old' else '新'}|"+
                 '|'.join(f"{float(v[a]):.5f} / {float(v[b]):.5f}" for a,b in fields)+'|')
lines += ['',
          '特别地，Sign r=0 旧任务的 `d E(lambda−mu)²` 从 d=64 的约 4.13158 降至 d=512 的约 4.01642，目标为 4；'
          '`d^(3/2) E|lambda−mu|³` 由约 13.53460 至 12.85703，没有呈现维度爆炸。'
          '这支持所检查范围内的量级，但不是高阶 Lp 引理的数值证明。', '',
          '随机阈值替换差及其分组标准误也逐格保存于 `mc_summary.csv`。主结果使用实际随机阈值；'
          '没有先将阈值设成 z 再把得到的结果称作真实 recall。', '',
          '## 7. 完整训练数据的独立排名检查', '',
          '生成完整 X、Y0、YE，计算真实非线性 U，并用全部候选直接比较排名：'
          '`d=20,n=257` 共 4096 份数据，每份 8 个查询；`d=64,n=1024` 共 1024 份数据，每份 8 个查询。'
          '四种方法、两个 r、两任务全部报告，合计 655360 个直接排名指标。'
          '这些检查不依赖“模拟根公式自身”来验证根公式的统计含义。','']
maxrow=max(rank['results'],key=lambda v:abs(v['difference']/v['se']))
maxz=abs(maxrow['difference']/maxrow['se'])
lines += [f"32 格中，直接排名与残差积分的最大配对差标准化值为 **{maxz:.3f}**；"
          f"该格为 d={maxrow['d']}, r={maxrow['r']}, {maxrow['method']}, {maxrow['task']}，"
          f"差 {maxrow['difference']:.6f}、标准误 {maxrow['se']:.6f}。"
          '这是多格实现诊断，不按单格显著性选择结果。', '',
          f"对同一实际数据直接计算残差能量，式 (6) 的最大逐样本误差为 **{rank['max_score_identity']:.3g}**。"
          f"另在 {rank['event_checks']} 组物理可行的条件参数及高斯坐标上检查平方不等式选支，"
          f"事件不一致数为 **{rank['event_mismatch']}**。完整结果在 `review18_checks/strict/rank_checks.json`，含全部 32 格，未删样本。", '',
          '判断：未发现精确积分的分布反例或选支错误。此检查不能证明全参数渐近余项。', '',
          '## 8. d=512 区间、原始 formal 数据和统计含义', '',
          '### 8.1 端点算术与统计不等式', '',
          '我直接从 `z2b_checks/Z2B_outer_rows.csv` 重算无偏样本方差及四个单侧端点，'
          '四个端点与作者 JSON 的 float64 数值差均为 0；详见 `review18_checks/strict/interval_arithmetic.json`。', '',
          '经验 Bernstein 引用已对照原论文 '
          '[Maurer–Pontil (2009), Theorem 4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf)：'
          '将取值从 [−1,1] 缩放至 [0,1] 后再缩回，方差项半径为 `sqrt(2 V log(2/delta_s)/B)`，'
          '有界范围项为 `14 log(2/delta_s)/(3(B−1))`。使用 B=2048 而不是 BQ，'
          '两个 r 各两个单侧界、各分配 .0025 失败概率，联合覆盖率至少 .99（理想实数运算下）。', '',
          '### 8.2 从保存的逐查询 success/rank 重新汇总', '',
          '读取 `y3d/formal/d512_r{1,2}_s*.npz` 的全部 64 个文件，独立核对 '
          '`success == (rank≤ceil(.1n))`，然后按 seed 先平均，再计算配对差及标准误。'
          '结果与 formal 的 diagnostics、secondary 汇总完全一致。以下单位全部为百分点：', '',
          '|r|作者 99% 区间|formal GD|formal Sign|formal GD−Sign ± 1 SE|覆盖 formal 均值|本轮独立积分 ± 1 SE|',
          '|---|---|---:|---:|---:|---|---:|']
for v in summary['formal']:
    lines.append(f"|{v['r']:g}|[{100*v['author_lower']:.5f}, {100*v['author_upper']:.5f}]|"
                 f"{100*v['GD']:.5f}|{100*v['Sign']:.5f}|{100*v['difference']:.5f} ± {100*v['se']:.5f}|是|"
                 f"{100*v['mc_mean']:.5f} ± {100*v['mc_se']:.5f}|")
lines += ['',
          'r=0 有 32 个 seed、每个 512 个查询；r=−.5 有 32 个 seed、每个 8192 个查询。'
          '两行 formal 的描述性 95% 配对均值区间分别约为 [42.07432,43.68008]pp 与 '
          '[22.92680,23.27528]pp，也都包含在作者的较宽积分区间内。'
          '它们是独立实验均值的 t 区间，不能与作者的 99% 积分覆盖率混成同一个概率声明。', '',
          'r=0 作者积分中心约 42.97937pp，本轮为 43.02137pp，相差约 2.20 个合并模拟标准误；'
          'r=−.5 对应约 .14 个标准误。必须披露这一偏移，不能描述为所有点估计在一个标准误内逐位复现。'
          '但前者不足以推翻其保守 99% 区间，也不是渐近系数错误的证据；本轮没有按这个偏移重选随机种子。', '',
          '**覆盖对象必须写清：** r=0 只有 20/32 个单 seed 差落入作者区间，r=−.5 为 32/32。'
          '这不构成区间失效，因为它覆盖的是总体期望，不是承诺每个 seed 有 99% 的观测覆盖。'
          '同理，“两个观测均值都在区间内”只是一致性对照，不能验证区间在重复计算中的标称覆盖率。', '',
          '### 8.3 这些区间没有单独识别 B_R', '',
          '作者区间都包含未修正极限差 [43.42185pp,23.74510pp] 对应的两行值，也包含修正预测 '
          '[43.02841pp,23.09634pp]。原文对此有明确说明。'
          '因此：有限维期望差的统计区间 **PASS**；将这两条覆盖结果当成首阶系数的独立认证 **FAIL**。'
          '主定理的依据应是上述解析引理链。', '',
          '普通 float64 没有向外舍入，严格实数端点认证仍未交付；对最末小数不得使用“机器证明”的说法。', '',
          '## 9. 复现、资源及文件范围', '',
          '正式产物统一位于 `review18_checks/strict/`；根目录下同名数值文件是首次试算记录。'
          '首次试算仅限制每个 BLAS 库为 2 线程，没有记录进程总线程数；NumPy、SciPy 分别加载 BLAS 时，'
          '这不足以证明满足用户的总线程约束。因此正式结果全部使用相同随机种子、相同样本量重新以单线程计算；'
          '没有根据先前输出改变科学设置，也未删除试算记录。', '',
          '正式复现命令（排他创建输出，已有 strict 结果时需将子目录名及脚本输出路径一致改成新名字）：', '',
          '```bash',
          'systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \\',
          '  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\',
          '  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 REVIEW18_RUN_SUBDIR=strict \\',
          '  bash review18_checks/strict_run.sh',
          '```', '',
          '该脚本顺序运行四个主模拟、完整数据排名核验及汇总；没有并行运行数值进程。'
          '图形配置缓存也显式放在本轮输出目录内。']
runrecords=[read(f'mc_d{d}.json') for d in (64,128,256,512)]
peak=max(v['resources']['peak_rss_kib'] for v in runrecords+[rank,summary])
maxthreads=max(v['resources']['kernel_threads'] for v in runrecords+[rank,summary])
lines += ['',
          f"正式四维主计算共 {summary['simulation_seconds']:.2f} 秒墙钟、"
          f"{summary['simulation_cpu_seconds']:.2f} CPU 秒；完整排名核验 {rank['wall_seconds']:.2f} 秒。"
          f"主计算、排名及汇总的最高峰值 RSS 为 {peak/1024:.2f} MiB；"
          f"这些进程记录的内核总线程数最大为 **{maxthreads}**，两套 BLAS 各为 1。"
          '每个正式进程均验证 cgroup `memory.max=3221225472`、`memory.swap.max=0`。', '',
          '没有对其他项目进行文件读写，本轮显式写操作没有涉及本项目原有文件；未进行全目录全文件哈希审计，'
          '也不为自动追加日志或其他并发进程的写入作不变性保证。'
          '`input_hashes.json` 保存全部 64 个 formal 原始文件及主要证明、代码、表格的 SHA256，'
          '`final_audit.json` 再核对这些输入及被审证明的作者最终散列。'
          '独立程序仅相互导入本轮新代码，未导入被审代码或 formal 分析程序。', '',
          '需保留的主要交付：', '',
          '- `independent_mc.py`：实际非线性条件模拟、独立响应求积和二次不等式积分。',
          '- `independent_rank.py`：完整数据直接排名、残差能量恒等式和 40 万组选支检查。',
          '- `strict/mc_d*.npz`：外层组原始累计量，可重算模拟标准误。',
          '- `strict/mc_summary.csv`、`strict/fits.csv`、`strict/moment_checks.csv`：全部格子与两类拟合。',
          '- `strict/formal_comparison.csv`、`strict/interval_arithmetic.json`：formal 原始记录对照及作者端点重算。',
          '- `strict/rank_checks.json`、`strict/analysis_summary.json`、`strict/final_audit.json`：完整核验和资源记录。', '',
          '## 10. 最终总判定', '',
          '**PASS：固定参数、充分大 d 的期望召回首阶系数，以及原文已限定的有限维统计积分主张。**'
          '旧的实际非线性闭合及非光滑阈值接口在新路径上已补上，未发现被假设掉的关键独立性。'
          '建议明确写出 `n>d+1`，并在描述数值“吻合”时同时披露点估计标准误与拟合模型敏感性。'
          '全参数统一余项、d=512 的显式渐近误差常数、单数据集保证及机器舍入证书仍是 FAIL 的更强解释，'
          '它们并非本文已经证明的内容。', '']

path=ROOT/'PROOF_REVIEW18.md'
text=path.read_text()
marker='<!-- REVIEW18_NUMERICAL_RESULTS -->'
assert marker in text
path.write_text(text.replace(marker,'\n'.join(lines)))
write_json('report_build.json',dict(strict_results_only=True,resources=resources()))
