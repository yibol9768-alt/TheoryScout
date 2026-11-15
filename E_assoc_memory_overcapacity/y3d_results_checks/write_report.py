"""只新建中文审读报告；数值来自独立重算产物。"""
import csv
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'y3d_results_checks'
summary = json.loads((OUT/'summary.json').read_text())
execution = json.loads((OUT/'execution.json').read_text())
with (OUT/'diagnostics72_recomputed.csv').open(newline='') as f:
    cells = list(csv.DictReader(f))
with (OUT/'old45_residuals.csv').open(newline='') as f:
    old45 = list(csv.DictReader(f))
# 旧绝对 recall 的官方次要区间经平移应与独立残差区间相同。
with (ROOT/'y3d/formal/analysis/secondary_old45.csv').open(newline='') as f:
    official45 = list(csv.DictReader(f))
for row in old45:
    other = next(x for x in official45 if all(x[k] == row[k] for k in ('d','ir','method')))
    for end in ('lower','upper'):
        assert abs(float(row[end]) - (float(other[end])-float(other['corrected']))) < 1e-12

def pp(x):
    return f'{100*float(x):+.4f}'

def ci(row):
    return f'[{pp(row["lower"])}, {pp(row["upper"])}]'

def lossfmt(x):
    return f'{float(x):.8g}'

text = '''**结论：按冻结的预注册，三维旧任务、三维新任务及指定的 d512、r=−0.5 旧任务 GD−Sign 三个预测平方损失改善终点均通过家族 3 的 Bonferroni 判据；full72 实用等价未成立（仅 10/72 格通过）。full72 未通过的主要直接瓶颈是区间精度：62 个未等价格中有 58 格的 CI 全宽至少为 0.010，无法容纳于 ±0.005 等价带。现有结果不足以把 full72 未通过归因为普遍、稳定的修正后系统偏差，也不能据此排除局部偏差。**

描述上确有方向一致的局部偏离，尤其 d256、r=0.5 的旧绝对 recall 五方法均低于修正预测，GD 的次要族 45 区间排除了预测值；这些发现须与三个确认终点分开陈述。召回系数及联合闭合仍是候选，非光滑概率余项未获证明。

审读日期：2026-11-06。只读取本项目资料，未调用 LLM API、未新增训练或查询随机流、未修改已有实验产物。本文所称“通过”均沿用冻结判据，没有删格、改权重、换终点、放宽等价带或追加样本。

**预注册意义上的确认结论**

依据 [Y3D_PREREG.md](Y3D_PREREG.md) §1、§6，以及 [Y3D_READY.md](Y3D_READY.md) 的历史名称映射，Y3B/y3b 指向本次 Y3D/y3d。正式采信凭据为 [FINAL_ACCEPTED.json](y3d/formal/analysis/FINAL_ACCEPTED.json)，其 analysis_complete、confirmation_valid 均为 true，publication_state 为 accepted；用户已完成 `--verify-accepted`。本轮另核对该凭据绑定的全部证据哈希、冻结的预注册/预测/配置/READY 哈希，并从 NPZ 独立计算统计量。冻结清单 SHA256 为 `472659c022b06d17bc1c436ad8460f45b879271c00a536a3a9e77902fd84a4ae`。

全部 288 个数据集完成，组成 32 个完整区组；每个 (d,r) 格有 32 个独立数据集，目标格每份 8192 查询，其余每份 512 查询。三个终点的统计单位均为 seed/区组，未把逐查询指示量当作独立 seed。旧汇总固定平均 27 个配对格（GD−Sign、Adam2−Adam.1、Adam.1−Sign），新汇总固定平均 45 个方法格（含 Adam1），所有格等权。Adam1 的旧绝对 recall 属次要结果，不进入三个主要旧配对。

每格损失改善定义为 ℓ(y;p₀,p₁)=(y−p₀)²−(y−p₁)²，p₀ 为冻结极限预测、p₁ 为冻结有限维修正预测。旧任务 y 是同一数据集的方法 recall 配对差；新任务 y 是绝对 recall。正值表示修正预测的平方误差较小，单位是 recall²，不能读成方法本身的召回率提升。区间为 L̄ ± t(1−0.05/6,31)·SE，临界值 2.5309258452；下界严格大于零才通过。原始 p 均为双侧，调整 p=min(1,3p)。

|确认终点|N|平均损失改善|SE|家族 3 同时 95% CI|双侧原始 p|Bonferroni p|冻结判定|
|---|---:|---:|---:|---|---:|---:|---|
'''
names = ['三维旧任务配对差汇总', '三维新任务绝对 recall 汇总', 'd512、r=−0.5 旧 GD−Sign']
for name, row in zip(names, summary['endpoints']):
    text += f'|{name}|{row["n"]}|{lossfmt(row["mean"])}|{lossfmt(row["se"])}|[{lossfmt(row["lower"])}, {lossfmt(row["upper"])}]|{lossfmt(row["p_two_sided"])}|{lossfmt(row["p_bonferroni"])}|通过|\n'

target = next(r for r in cells if (r['d'],r['ir'],r['task'],r['label']) == ('512','2','old','GD-Sign'))
text += f'''
因此可采用预注册措辞：“两项主检验及指定 d512 对比均支持。”主汇总满足至少 24 个完整区组条件，指定 d512 终点满足全部 32 份目标数据条件；正常退出与正式采信资格有效。该 d512 对比的观测均值为 {float(target['observed']):.10f}，极限预测 {float(target['uncorrected']):.10f}，修正预测 {float(target['corrected']):.10f}。这仅确认指定对比的损失改善，不替代全部 d512 格或修正系数精确性的检验。Student-t 区间沿用预注册的常规近似，并非有限样本精确保证。

**full72：未通过主要受精度限制，局部偏离仍应披露**

残差统一定义为“观测−修正预测”：正号表示修正预测偏低，负号表示修正预测偏高；旧任务残差针对方法差，新任务针对绝对 recall。家族 72 的临界值为 t(1−0.05/144,31)=3.7672574813。全族实用等价要求全部 72 格的残差区间严格落入 [−0.005,+0.005] 内，本次仅 10 格满足，因此冻结判定是 **full72_equivalent=false**。CI 包含零并不满足等价条件。

下表及后续残差表以百分点（pp）表示，0.005 recall=0.5 pp；CI 的“全宽”应与等价带全宽 1.0 pp 比较，“半宽”才与 0.5 pp 比较。

|范围|等价/格数|CI 全宽≥1.0 pp|残差点估计在 ±0.5 pp 内|CI 排除零|残差正/负|CI 全宽最小/中位/最大（pp）|平均绝对残差（pp）|
|---|---:|---:|---:|---:|---:|---|---:|
'''
groups = [('全部', summary['full72'])]
groups += [('d'+d, summary['by_dimension'][d]) for d in ('128','256','512')]
groups += [('旧配对差', summary['by_task']['old']), ('新绝对 recall', summary['by_task']['new'])]
for name, row in groups:
    text += (f'|{name}|{row["equivalent"]}/{row["n"]}|{row["width_at_least_band"]}|'
             f'{row["mean_inside_band"]}|{row["excludes_zero"]}|{row["positive"]}/{row["negative"]}|'
             f'{100*row["width_min"]:.4f} / {100*row["width_median"]:.4f} / {100*row["width_max"]:.4f}|'
             f'{100*row["mean_abs_residual"]:.4f}|\n')

text += '''
58/62（93.5%）个未等价格的全宽至少为 1.0 pp，即使把区间中心移到零，也不能严格装入等价带；这包括全部 40 个未等价的新任务格。其余四格的区间虽略窄于等价带，仍因中心偏离而轻微越界，详见下表。这里的“精度不足/低功效”是依据实际区间几何形状的诊断，没有以观测均值代入事后功效来替代预注册判定，也没有证明其真实残差为零。

|d|r|旧配对|残差（pp）|家族 72 CI（pp）|全宽（pp）|
|---:|---:|---|---:|---|---:|
'''
for row in summary['narrow_non_equivalent']:
    text += f'|{row["d"]}|{row["r"]:g}|{row["label"]}|{pp(row["mean"])}|{ci(row)}|{100*row["width"]:.4f}|\n'

text += '''
全部 72 个同时区间均包含零，没有任何一格的区间完整处于等价带之外。58/72 个残差点估计已在带内，14 格在带外，但这些点估计不构成排除零或排除等价带的区间证据。正/负残差为 28/44，方法之间共用数据和查询，不能把 72 个符号视作独立重复来检验偏差。

通过等价的十格具体为：d256、r=−0.5 的旧 Adam.1−Sign；d512、r=0.5 的旧 Adam.1−Sign；以及 d512、r=−0.5 的全部三个旧配对和全部五个新方法。其中目标格 (d512,r=−0.5) 每份查询数是其他格的 16 倍，正好贡献其中 8 格等价，这与更高精度相符，但不能据此量化加查询的因果效果。

维数趋势也不支持简单的“所有格存在稳定同向系统偏差”：平均绝对残差依次为 0.3361、0.3797、0.0918 pp，d256 略大于 d128、d512 较小，未呈逐级单调下降；d512 的 24 个点估计全部落在等价带内。新任务 r=0.5 五方法在 d128 均为正（+0.5701 至 +0.6468 pp），d256 均为负（−0.9041 至 −0.3254 pp），d512 靠近零（−0.0072 至 +0.2272 pp）。这更像需要继续辨认的局部共同波动/偏差，而非已确立的跨维同向误差。d512 的精度与其目标格查询数增加混杂，不能把等价数增加全部归于维数增大，也不能由三个维度拟合或验证余项幂次。

**描述性发现：修正后仍偏离的格、方法与方向**

以下均为对固定全部格的描述，不新增确认性声明。旧配对中 r=0.5 的 GD−Sign 在三个维度残差均为负（−0.0681、−0.1514、−0.2175 pp），Adam.1−Sign 也均为负（−0.0793、−0.1234、−0.1241 pp）：修正后的方法差仍偏高，但同时 CI 均包含零，且后者 d512 已满足实用等价。r=0 与 r=−0.5 的多数旧配对残差在 d128、d256 间反号；最突出的两格 GD−Sign 分别从 −0.4663、−0.7887 pp 变为 +0.8282、+0.9106 pp，d512 为 −0.1512、+0.0047 pp。没有稳定的同向维数延续。

新任务除 r=0.5 上述反号外，r=0 的 GD 和 Adam2 在三个维度均为负（GD：−0.3353、−0.2721、−0.0483 pp；Adam2：−0.3493、−0.0838、−0.0396 pp），r=−0.5 的 GD 也均为负（−0.0561、−0.1874、−0.0403 pp），表示这些冻结预测在点估计上偏高；其余方法没有整齐一致的跨维符号。特别是 r=−0.5 的新 GD 冻结修正为极小正位移，而观测相对修正的残差三维均为负，本轮没有删去这些不利格。

损失改善也不是逐格都为正：例如 d256 的旧 r=0 三配对及 r=−0.5 的 GD−Sign、Adam2−Adam.1，其平均损失改善为负；d128、r=0.5 的新 GD、Adam2 也为负。r=−0.5 的新 GD 三维均为负。三项汇总/指定对比通过不能扩写为每格均受益；全部 72 格的平均损失改善保留于独立 CSV。

旧任务**绝对 recall** 的另一个共同偏离更明显：r=0.5 时五方法在 d128 均高于修正预测（+0.3871 至 +0.5172 pp），d256 均低于（−1.3276 至 −1.1762 pp），d512 又均高于（+0.2660 至 +0.5599 pp）。d256 GD 的观测 0.7276611328，修正预测 0.7409371247，残差 −1.3276 pp；原次要族 45 同时 recall 区间减去修正预测后为 [−2.6277,−0.0275] pp，排除了零。Adam2 的相应上界为 +0.0017 pp，其他方法的区间也均包含零。这个局部共同方向可能在旧配对差中相互抵消，因此不能用旧配对改善声称旧绝对 recall 逐方法准确。

这里的族 45 比较沿用冻结次要 recall 区间，平移后检查其是否覆盖固定修正预测；它没有成为第四个确认终点。原 `secondary_old45.csv` 中的 p 值检验的是 recall 均值对零，不能将这些 p 值当作修正预测偏差的显著性。上述单格区间偏离值得披露，但跨维符号反转、同格五方法相关和较宽区间，使其仍不足以确立某种跨维稳定的遗漏项或普遍失配机制。

以下给出旧绝对 recall 的全部 45 个残差点估计，含 Adam1；全精度残差区间另存于 [old45_residuals.csv](y3d_results_checks/old45_residuals.csv)。正为预测偏低，负为预测偏高。

|r|方法|d128 残差（pp）|d256 残差（pp）|d512 残差（pp）|
|---:|---|---:|---:|---:|
'''
for ir in range(3):
    for method in ('GD','Adam2','Adam1','Adam.1','Sign'):
        rr = [next(x for x in old45 if x['d']==str(d) and x['ir']==str(ir) and x['method']==method) for d in (128,256,512)]
        text += f'|{(.5,0.,-.5)[ir]:g}|{method}|' + '|'.join(pp(x['mean']) for x in rr) + '|\n'

text += '''
**不能声称的内容**

- 不能称 full72 全族实用等价成立，或把 CI 含零写成“预测已验证准确”。仅 10/72 格满足冻结等价标准。
- 不能称全部 d512 格、每个方法或每个单格都改善；指定 d512 确认仅针对 r=−0.5 的旧 GD−Sign。
- 不能称 recall 的 B_R 系数、联合闭合 J 或 1/d 幂次已获证明或精确验证。模型是在看过 X1 后选择的候选，本次新数据的损失改善不消除其选择历史，也不证明系数恰等于冻结数值。
- 不能称真实非线性高阶迹控制、非光滑概率余项已经解决；分数均值与错误 RMS 平方均值首项的数学支持不能自动补齐 recall 推导缺口。
- 不能把 full72 未通过解释成已经发现普遍系统偏差，也不能把宽区间解释成已排除偏差。局部方向只能描述，不能据此改预测、拟合系数、换格或重判本次确认结论。
- 不能把规划的 80%/90% 条件功效当成本次实测功效或全族等价功效，不能将查询数视为独立样本数，也不能据本次审读追加 seed。

**论文可用的一句话结论建议（事实陈述）**

“在冻结的独立实验中，有限维修正预测在三维旧任务配对差、新任务召回率及预指定的 d512、r=−0.5 旧任务 GD−Sign 对比上均通过了 Bonferroni 校正的预测平方损失改善检验，但全 72 格实用等价未成立，仅 10 格的同时置信区间落入 ±0.005 等价带。”

**独立重算及复核证据**

[recompute.py](y3d_results_checks/recompute.py) 未导入项目的 analyze、predict、common、publication 或任何旧分析实现。它从全部 288 份 NPZ 的 success 指示量独立计算 2880 个 dataset×method×task recall，另以 ranks 与阈值一致性复核 success；预测只读取冻结 CSV。损失使用直接两个平方之差实现，随后独立形成 32 个区组的 27/45 格均值和目标格 32 份损失，独立计算样本 SD、SE、t 区间和双侧 p。在估计全部完成后才读取正式表比较。

三个确认终点的均值、SE、临界值、区间及 p 与正式表最大绝对数值差为 8.68×10⁻¹⁸（四舍五入上界）；72 格诊断的对应统计量与正式表逐项一致；2880 个 recall 与正式 all_completed 表逐项一致。全部原始精度数值见 [endpoints_recomputed.csv](y3d_results_checks/endpoints_recomputed.csv)、[block_losses.csv](y3d_results_checks/block_losses.csv)、[diagnostics72_recomputed.csv](y3d_results_checks/diagnostics72_recomputed.csv)、[recall_from_npz.csv](y3d_results_checks/recall_from_npz.csv) 和 [summary.json](y3d_results_checks/summary.json)。311 个本项目实际读取输入的哈希及读后不变复核记录见 [input_hashes.json](y3d_results_checks/input_hashes.json)。此处是读取输入不变的核验，不是对其他同时运行任务的全项目状态承诺。

独立重算命令如下，日志新建于 [recompute.log](y3d_results_checks/recompute.log)：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \\
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \\
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \\
  /home/liuyibo/venvs/theory/bin/python -B y3d_results_checks/recompute.py
```

'''
text += (f'实际 cgroup 为 3 GiB、swap=0，NumPy/SciPy 两份 BLAS 动态库实际线程均为 2；'
         f'Python 3.12.3、NumPy {execution["numpy"]}、SciPy {execution["scipy"]}。'
         f'独立重算耗用 CPU {execution["cpu_seconds"]:.3f} 秒、墙钟 {execution["wall_seconds"]:.3f} 秒、'
         f'峰值 RSS {execution["peak_rss_mib"]:.2f} MiB，见 [execution.json](y3d_results_checks/execution.json)。'
         '这些是本轮离线审读成本，未并入或改写正式实验成本。产物以排他创建方式保存；重算脚本再次运行会拒绝覆盖。\n\n')

text += '''**全部 72 格残差、同时区间与区间全宽**

下表逐格披露，单位均为 pp；每格 N=32，统一家族 72。表中“是”才满足严格等价条件，符号针对“观测−修正预测”。d 的顺序和逐格数据同时给出维数趋势；没有剔除 Adam1、新 GD 的微小位移格或 d512 格。

|d|r|任务|方法/配对|残差（pp）|同时 CI（pp）|全宽（pp）|等价|
|---:|---:|---|---|---:|---|---:|---|
'''
for row in cells:
    text += (f'|{row["d"]}|{float(row["r"]):g}|{"旧差" if row["task"]=="old" else "新 recall"}|{row["label"]}|'
             f'{pp(row["mean"])}|{ci(row)}|{100*float(row["width"]):.4f}|'
             f'{"是" if row["equivalent"]=="True" else "否"}|\n')

# 对三项通过、十格等价和四格窄区间未等价再次做报告层一致性检查。
assert len(cells) == 72 and sum(r['equivalent']=='True' for r in cells) == 10
assert len(summary['narrow_non_equivalent']) == 4
assert all(x['lower'] > 0 for x in summary['endpoints'])
for rel, h in json.loads((OUT/'input_hashes.json').read_text()).items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == h
with (ROOT/'Y3D_RESULTS.md').open('x') as f:
    f.write(text)
with (OUT/'report_validation.json').open('x') as f:
    json.dump(dict(report_sha256=hashlib.sha256(text.encode()).hexdigest(),
                   all_three_confirmatory_pass=True, full72_equivalent=False,
                   equivalent_cells=10, diagnostic_rows=72, old_absolute_rows=45,
                   secondary45_shifted_intervals_match=True,
                   reviewed_inputs_unchanged=True), f, ensure_ascii=False, indent=2)
    f.write('\n')
print('已排他创建 Y3D_RESULTS.md；72 格诊断、45 格旧绝对 recall、三确认终点及输入哈希复核通过。')
