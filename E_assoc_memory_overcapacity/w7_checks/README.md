# W7 的小规模检索核查

理论和完整证明见 ../W7_FT_RANK.md。只检查实际一步微调，不导入项目实验代码，不调用任何 LLM API，不访问其他项目。

运行命令（在本项目根目录）：

    systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python w7_checks/check_ft_rank.py > w7_checks/run.log 2>&1
    systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python w7_checks/report.py > w7_checks/report.log 2>&1

两脚本在导入数值库前设置 OpenBLAS/OMP/MKL/NumExpr 线程数为 2。模拟脚本还从 cgroup 文件核实 2 GiB 内存及零 swap 限制，并从实际加载的 NumPy/SciPy OpenBLAS 动态库查询线程数。记录在 metadata.json。绘图缓存留在本目录 .mplconfig/。

实验设计：

- 6 组键/值相关性、3 个维数（32、64、128）、每格 4 个独立 seed，共 72 个数据集。
- 每个数据集实际做一步 GD、Sign、Adam(e=.1)、Adam(e=1)，其中 e 是除以残差标准差后的 stabilizer；使用真实梯度残差 Z1-Z0 S1。
- 每个数据集抽取 256 个不同训练查询，同一数据集的所有方法使用同一组查询。
- 先用总体首次分支公式匹配新任务 c1=r+.35(1-r)，再计算理论旧/新 rank CDF，参数不从模拟 recall 反推。
- 六组相关性包括完全共享键，以及相同 r 但不同 r_x、r_y 的模型。

文件：

| 文件 | 内容 |
|---|---|
| check_ft_rank.py | 独立模拟和预测公式 |
| report.py | 汇总和绘图 |
| predictions.json | 各方法的总体匹配参数、q、c0、c1 |
| rows.csv | 每个 seed/方法/任务的正确分数、错误 CDF、rank、权重误差 |
| rank_cdf.csv | 9 个预先声明比例上的实际/预测 rank CDF |
| summary.csv | 合并四个 seed 的逐格摘要 |
| results.md | 人可读误差表和同新检索比较 |
| aggregate.json | 按维数合并的误差摘要 |
| rank_cdf.png、rank_cdf.pdf | d=128、共享键三种相关性的 rank CDF 图 |
| metadata.json | cgroup、实际 BLAS 线程、版本、耗时、峰值内存和随机 seed |
| run.log、report.log | 标准输出和错误日志 |

误差口径：正确均值/方差的合并样本量为 4×256；rank-grid 误差取九点的最大值；错误 CDF KS 每个 seed 检查前 8 个查询的所有错误候选。不同查询在有限维中相关；这些表是描述性核查，没有独立查询的置信区间、显著性或等价性结论。有限维新任务 recall 不被强制精确相等，小方法差距也不保证在本样本量中分辨。
