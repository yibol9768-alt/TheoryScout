# 随投稿提供的自包含复算包

本包复算 exact polar 方差极限和 MSR 交点（定理 6.2）、两步 Adam 稳定项排序与 `u=2/5` 链、历史反转阈值与反例，以及 MR 和有限维修正确认实验。所有输入均已包含在本目录；不读取原工程目录，不访问网络，不调用任何模型服务。数学范围与算术假设见 [MATHEMATICS.md](MATHEMATICS.md)。

环境实测：**Python 3.12.3、numpy 2.5.4、scipy 1.18.1、mpmath 1.3.0，x86_64**。只需要这三个第三方库和 Python 标准库；精确版本列在 `requirements.txt`、`versions.json`。不需要绘图库、符号库、LaTeX 或原工程。Python 3.12 为已验证版本；其他版本没有作完整验证。polar 部分要求 IEEE-754 binary64 的正确舍入四则运算、平方根及渐进下溢。

下列命令均从本目录运行，`python` 指向装有上述依赖的解释器。输出目录必须不存在，重复运行请换目录名。脚本将线程环境设为 1，并禁止写入字节码缓存。Linux 上进行与本次实测相同的资源限制时，给每条 `python ...` 命令加前缀：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B
```

前缀中的 `python -B` 替换命令开头的 `python`。审稿人直接使用 Python 即可复算；systemd 仅提供 Linux 的资源限制，不是 Python 依赖。

## 结论、命令与预期结果

每条命令的标准输出、退出码和实际秒数均写在新输出目录的 `command_*.log`、`timings.json`。本次实际运行日志另附于 `logs/`；表中耗时为本机量级，不是跨机器保证。

|结论或检查范围|命令|预计耗时|预期输出|
|---|---|---|---|
|包内文件完整性|`python verify_package.py`|小于 1 秒|`PASS`，文件哈希全部一致|
|全部证书从零生成，并独立检查全部覆盖和符号|`python reproduce.py certificates --output recomputed/certificates`|约 4–5 分钟|polar、1027 格、全部根及历史证书生成；最后两个检查器均 `PASS`|
|定理 6.2、15 个 MSR 交点及方差包络|`python reproduce.py polar --output recomputed/polar`|约 30–45 秒|`polar_variance.json`；`polar_check.json` 中 15 个交点、90 条严格端点不等式通过，论文表格包含关系通过|
|两步 Adam 的 1027 格、64 边界格、匹配根、Sign/GD 链和 e=1 历史消融|`python reproduce.py adam --output recomputed/adam`|约 3–4 分钟|`stabilizer_ordering.json`、`matched_chain.json`、`moment_history.json` 均 `PASS`；全域 N 下界约 .17234974，J 下界约 .00005315197|
|历史机制的阈值定位和具体反例|`python reproduce.py history --output recomputed/history`|约 45 秒|`history_reversal.json` 为 `PASS`；H(1.05)>0、H(1.10)<0，e=.5 的旧信号差为正，e=2 时为负|
|独立复核全部覆盖和全部格、边界、根的符号，不重新求积分|`python reproduce.py check-adam --output recomputed/independent_decisions`|约 1 秒|1027 格精确覆盖；所有符号、首次根、`u=2/5` 链、历史消融与反例通过|
|独立重积分抽样：首格、末格；仍检查全部覆盖和符号|`python reproduce.py check-adam-integrals --output recomputed/independent_sample`|约 2–3 秒|`reintegrated_region_cells=2`；末格细分，实际子格数 3|
|独立重积分全部区域、边界、根、固定方向、消融、阈值与反例|`python reproduce.py check-adam-all --output recomputed/independent_full`|约 8–15 分钟|`independent_reintegration=all`、`reintegrated_region_cells=1027`、`PASS`；两个 JSONL 文件保存独立积分包络|
|MR 与有限维修正确认实验的代表性实际训练重放|`python reproduce.py replay --output recomputed/replay`|约 5–6 分钟|MR 12 个、确认实验 18 个数据集；全部 300 个方法×任务数据集召回值与原结果相等，`max_recall_error=0`|
|从原始数据集测量值重算论文表格与 90 条预测|`python reproduce.py tables --output recomputed/tables`|约 1 秒|18 个主要方向、7 个预测覆盖、22/30 个新任务等价；3 个确认端点、10/72 个诊断等价，`manuscript_tables_reproduced=true`|
|从查询级 NPZ 独立重建测量值并重走表格管线（小规模不作确认）|`python reproduce.py query-tables --output recomputed/query_tables`|约 2 秒|独立读取 30 个查询归档、生成 300 个测量值；统计管线通过，确认端点保持不合格|
|完整实验重跑设计与规模检查|`python reproduce.py full-plan --output recomputed/full_plan`|约 1 秒|MR 1152 个数据集、确认实验 288 个数据集和完整查询预算|

快速审查可运行完整性检查、独立符号检查和表格重算；完整数学复算使用 `certificates` 加 `check-adam-all`。独立检查器默认检查随包的 `two_step_adam_certificate/reference/`。检查刚刚从零生成的证书时，在独立检查命令中加 `--certificates recomputed/certificates`。

## 三部分的交付内容

`polar_certificate/polar_variance.py` 是最小生成器；`check_polar.py` 读取精确十六进制端点，并用 `mpmath.iv` 检查所有原始包络的组合和根端点判定。`reference/polar_variance.json` 保存 k 的每批网格积分包络、所有效率积分的 A/Q 包络、有理根端点和来源哈希。4096² 个格由生成命令全部重新计算，不在磁盘逐格保存。polar 检查器不另写第二套二维求积；其普通 SciPy 均值积分比较只是诊断，不能替代区间证明。

`two_step_adam_certificate/` 内的四个定理/命题生成器调用 `adam_integrals.py` 和 `integer_intervals.py`。后者使用固定精度整数外向舍入。`reference/` 保存全部区域积分和余项、边界、根端点、整段单调性、固定方向积分及历史反转的原始包络。`check_adam.py` **不导入这些生成器或区间核心**：覆盖用精确有理数，全部判号用 `mpmath.iv`；全量重积分用另写的“幂级数的双数系数”实现。遇到区间依赖导致的宽包络，检查器精确二分参数格，所有子格必须通过。独立包络的端点编码为精确二进制 `(sign,mantissa,exponent)`，表示 `(-1)^sign × mantissa × 2^exponent`；不把显示小数当作证书。

`experiments_replay/` 的 `replay_retrieval.py` 实际生成 Gaussian 数据、执行真实残差更新、对全部训练候选排名，并保存查询级 NPZ。两个设计采用原实验的正式随机数种子、流、形状、调用顺序和查询预算。小规模只取各格前两个正式种子，不替代原有统计确认。`population_model.py` 从积分和首次匹配方程重新生成总体参数与 90 条候选有限维修正预测；冻结参数和预测仅作一致性检查。

`matched_retrieval_measurements.jsonl` 保存 MR 的每个数据集、每种方法的原始召回；`finite_dimension_measurements.csv` 保存确认实验每个数据集的召回、D/V 均值和查询内方差协方差。它们是从查询级结果导出的数据集测量值，**不是已经计算好的组间区间或检验结论**。导出时独立遍历全部 1440 个原始 NPZ，检查查询预算、排名与成功标记、全部测量值，导出误差为零；归档内容哈希和验证结果见 `measurement_provenance.json`。本包不附整个大型原始查询库；代表性查询级数据可用重放命令从零生成。

`measurements_from_queries.py` 可从重放或完整重跑的 NPZ 独立提取测量值；`query-tables` 演示从随包代表性查询归档开始的完整管线。表格命令生成 MR 主表及其 Markdown、30 个匹配检查、3 个确认端点、72 格诊断与汇总、45 个旧任务区间、90 格机制测量，以及重新积分的预测。重复单位始终为数据集；完整区组确认与 Bonferroni 家族大小保持原设计。`statements/` 附上证书生成时读取的论文公式/结论快照和历史机制的解析证明，`manuscript_hashes.json` 记录该论文快照的原始内容哈希；生成的证书另外校验包内实际源码和摘录的内容哈希。

## 完整实验重跑

完整科学设计已固定，不重新规划样本数、不拟合新参数；小规模与完整规模使用同一个训练和排名实现。完整规模可执行：

```bash
python reproduce.py full-experiments --output recomputed/full_experiments
```

它在同一个资源限制中并行执行 MR 的 1152 个数据集和确认实验的 288 个数据集，每个工作进程一个数值线程；预计约 1–2 小时。每个数据集的观测均与随包原测量逐项核对，所有查询级数组保存在新目录。完成后从新测量值重算表格：

```bash
python experiments_replay/recompute_tables.py \
  --matched-measurements recomputed/full_experiments/matched/measurements.csv \
  --confirmation-measurements recomputed/full_experiments/confirmation/measurements.csv \
  --output recomputed/full_experiments/tables
```

**本次完整设计通过断点续算完成。** 首次完整重跑因机器重启中断，没有可用退出码，不能记作成功的单次执行。重启前完整写出的查询归档经过逐数组检查后直接复用；只补算缺失的数据集，没有重做已验证的数据集，也没有删除中断证据。两个恢复命令均正常退出，最终覆盖 MR 的全部 1152 个及确认实验的全部 288 个数据集。

恢复工具为 `experiments_replay/resume_retrieval.py`，参数为 `--design`、`--existing`（旧查询目录）与 `--output`（必须不存在的新目录）。工具不覆盖旧归档；每个复用归档都检查随机查询索引、预算、排名、成功标记、数值有限性及全部召回，并记录 SHA-256。它只用于已有断点的恢复；从零复算仍使用上面的完整重跑命令。

从全部 1440 个查询归档再次独立提取 14400 个数据集×方法×任务测量值，与恢复生成的测量逐项一致；随后分别重算论文表格，两套表格文件逐字节相同。恢复耗时、命令和资源记录见 `VALIDATION.md`、`logs/continued_run_summary.json`、`logs/full_replay_continuation.json`。全样本新测量与表格随包附在 `experiments_replay/reference/full_replay/`。

本地的 `full_experiments_recomputation/`、`full_experiments_continuation/` 和 `query_pipeline_validation/` 保留运行证据。为控制投稿体积，投稿归档不包含这些可选运行目录；所有默认输入、最小生成器、原始数学包络、代表性查询归档、全样本测量与验证日志均包含在归档中。清单只核对投稿文件，运行证据不会被删除。

论文摘录保持证书生成时的快照。续作期间检测到原论文文件另有变化，本包没有修改这些原文件，也不据此声称已经核验后续新增的解析定理。有限维修正实验仍按其原始设计和历史解释重放。
