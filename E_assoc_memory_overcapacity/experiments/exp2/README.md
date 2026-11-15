# EXP2 复现与证据入口

从本题根目录执行；不导入其他项目代码，不调用API。最终冻结设计见 `../../PREREG_EXP2.md`，中文结论见 `../../RESULTS_EXP2.md`。

`predata_v1/` 保存第一版规划、代码和功效；未生成任何EXP2数据时发现P2三个原失败条件功效不足80%，仅将四个P2条件加到112种子后再次冻结最终版。最终SHA-256为 `0ca323ffd25775447df29609d7d0482841bc2a207d65b1e61d57fd76076b23b9`。不将这个事前样本量修订隐藏为“唯一一次规划”。正式只有一次，原始数据不可覆盖。

执行记录：

```bash
sha256sum -c PREREG_EXP2.sha256
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 timeout 125s /home/liuyibo/venvs/theory/bin/python experiments/exp2/run.py smoke
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 timeout 3150s /home/liuyibo/venvs/theory/bin/python experiments/exp2/run.py formal
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python experiments/exp2/analysis_driver.py
```

analysis_driver.py仅包装原冻结analyze.py并记录汇总阶段自身资源，不改门槛；另运行audit.py、gradients.py、finalize.py完成独立重放审计、描述梯度和文稿组装，均使用同一scope/线程/Python约束。analysis/finalize会写汇总及报告，归档后不要原地覆盖本轮文稿。

不要直接重跑smoke/formal：脚本拒绝覆盖。`prepare.py` 也不应重跑覆盖冻结预测；规划修订脚本 `refine_predata.py` 仅为事前修改记录，现在已因存在budget_start而拒绝再执行。复跑应在本题目录内另建独立实验目录，保留原文件，并将ROOT/OUT及预注册单独配置；不得向其他项目目录拷贝/导入。

- `forecasts.json`：全部240 P3及4 P2的EXP1校准系数、校准SE、固定预测，不由EXP2拟合。
- `power.json`：独立种子/查询随机效应分解、规划功效、MDE、所需样本量；高维保守沿用d256方差分量。
- `mechanism.json`：EXP1校准的Adam16 K；Sign B直接由参数计算。
- `formal_rows.jsonl`：每方法512个完整候选秩、查询索引、正确分数，参数及实际单位、全候选错误得分一二阶矩；真实方法另含权重和逐查询对角误差。
- `formal_events.jsonl`、`formal_metadata.json`、`formal.log`：全部配置、耗时、RSS、cgroup硬限制、线程与异常。
- `P2_gates.csv`：G1全部原条件和d512读出。
- `P2_attribution.csv`、`P2_dimension_gradient.csv`：P2冻结归因门槛与128/256/512三维描述读出；中间维度没有追加检验。
- `mechanism_gates.csv`、`diagonal_gates.csv`：G2固定预测的等效性门槛。
- `model_gates.csv`：G3无偏差/平方根/倒维度三模型逐项判决。
- `P3_noise_gates.csv`、`P3_seed_decomposition.csv`：G4查询噪声底、未截零的去噪平方差、所有不确定项。
- `all_trends.csv`、`old_76_retest.csv`：G5全部旧式趋势及原76项逐条复核。
- `score_diagnostics.csv`：正确得分方差与均值、错误得分矩；没有事后添加通过门槛。
- `integrity_audit.json`：由原始秩重算指标、配对查询、首步一致性、冻结散列等独立检查。
- `independent_audit.json`：528个配置精确覆盖、查询RNG重新生成、1536个秩的独立稠密重放及原始W单位Adam复核，全部通过。
- `analysis_resource.json`、`completion_budget.json`、`final_summary.json`：汇总自身资源峰值、总预算及最终归因统计。含规划上界共约51.34分钟；有记录运行/分析阶段的RSS峰值492.62 MiB。
- `proof_audit.md`：本轮指定QED正文只读核查、关键估计复核、L4与误差转移的独立条件证明、当前缺口与投稿边界。
- `M_assembly.md`：以现有L2/L3为输入，补全Sign/Adam代理函数类、全局Lipschitz与q_T>0，接成真实迭代M的组装证明。
- `qed_readonly/`：指定外部QED正文/详细核验/题面的本轮快照，源路径、时间与散列见source_manifest.json；未在源目录执行或写入。

本轮随机数据生成、冒烟、正式和分析统一从budget_start计时；正式截止起点+3120秒，全部数值工作≤3600秒。原始数据和图表不能代替主定理证明。所有模型判决和旧式趋势判决并存，不互相改写。

最终RESULTS_EXP2.md在自动汇总后作过文字润色和证明说明补全，没有改变任何门槛、数值或归因。最终文件摘要保存在ARTIFACTS.sha256，可在本题根目录执行 `sha256sum -c experiments/exp2/ARTIFACTS.sha256` 检查。
