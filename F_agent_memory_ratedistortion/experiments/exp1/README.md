# EXP1 复现

从本题根目录运行 `experiments/exp1/run.sh --smoke`（输出目录必须不存在），再运行 `experiments/exp1/run.sh`。入口拒绝覆盖已有结果并检查预注册SHA256。

既有输出保留，重跑请在本题目录内使用新的输出目录并显式修改入口的输出位置；不要删除既有数据。

`e0_checks.py` 为本题 E0 的目录内副本。`core.py` 实现独立检索校准、源bit与reader、预算DP、LP顶点向下取整、所有计数/10排列扫描、机制消融和配对bootstrap；`run.py` 定义网格/优先级/阶段时间截断。SciPy HiGHS仅用于CPU上的LP，不进行学习。

正式原始结果逐条件写 `formal/results.jsonl`；每条件 `.npz` 保存五个端点逐查询正确向量（int8）和E2均匀族候选计数扫描结果。校准拟合/留出cell成功数、样本数在JSONL中，不保存大分数张量。`status.json` 列出每个完成/未运行配置。

汇总命令（亦须置于同样scope内）：

```
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python experiments/exp1/summarize.py
```

`metrics.csv` 每行一个条件/预算/策略；MAE主统计排除显式超出主架构的pad/cold/route，对其另报配对差。regret相对于同测试集的候选集经验最高准确率，具有向上选择偏差，不是总体/全策略空间regret。`calibration.csv` 每个独立检索条件/模型一行，避免将共享校准的不同epsilon/reader冒充独立实验。

`paired_seed_differences.csv` 是逐seed/配置配对差及条件于固定key的95%bootstrap区间；不代表新语料/用户总体不确定性。所有失败条件都进入总体误差，不能只展示通过的cell。

首个正式运行因校准索引未强制同段算子互斥而作废，完整保留在 `invalid_calibration_run/`（含早期派生统计）；后续 `formal/` 才是修正后正式结果。此前中间汇报中的E2误差数字随该无效运行作废。修正后冒烟为 `smoke_exclusive/`。耗时报告保留首轮约1037秒，并将其纳入累计一小时计算约束。

`dp_policies.jsonl` 由冻结数值输入及同一DP算法重建每个正式条件/预算的完整0/s/r模式向量；`export_dp.py` 校验模式数量和预测值与原始记录完全一致，可断点追加。最终完整性检查同时核对所有DP向量是否齐全。主统计不得混入无效校准运行。
