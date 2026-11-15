# Review 6 独立复现

三个脚本均为本轮独立编写。没有读取或导入 `w7_checks/`、作者实验脚本或以前的审稿代码，没有网络请求或 LLM API 调用。只读取四份指定审稿材料；脚本写出的文件全部位于本目录。

计算统一使用：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python review6_checks/check_review6.py \
  > review6_checks/run.log 2>&1

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python review6_checks/check_edges.py \
  > review6_checks/edges.log 2>&1

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python review6_checks/summarize_review6.py \
  > review6_checks/summary.log 2>&1
```

`metadata.json` 和 `edges_metadata.json` 从实际进程的 cgroup 读取 `memory.max`、`memory.swap.max`，并直接调用两份 OpenBLAS 库的线程查询符号，确认每份均为 2 线程。无需安装额外软件。`PYTHONDONTWRITEBYTECODE=1` 避免生成缓存。

第一次环境探测发现没有安装 `threadpoolctl`，随后改为直接检查 OpenBLAS；首次主脚本的资源检查还发现 NumPy 的线程查询符号带 `64_`，补充符号后才开始模拟。边界检查的辅助切点例子一度误选无有限极大值的参数，已更正为 `r=-.2` 并完整重跑；最终文件来自修正后的运行。这些是审稿程序的修正，与被审稿定理无关。

## 设计与统计单位

- 主实验：`d=64,128`，`n=d²/2`，六组 `(rx,ry)` 为 `(0,.8),(1,0),(1,.5),(sqrt(.5),sqrt(.5)),(1,-.5),(sqrt(.5),-sqrt(.5))`。共 192 个独立数据集，每个 160 个不同训练查询。四种实际一步更新共享同一数据集，故方法之间也不当作独立复制。
- 固定总体新任务信号目标 `u=r+.28(1-r)`；GD、Sign、Adam(e=.2)、Adam(e=2) 分别直接求幅度曲线的首次根，没有用 W7 的闭式根。Adam 的物理归一化 epsilon 为 `e*sqrt(2(1-r))`。实际梯度始终是 `Z0 S1-Z1`。
- `per_dataset.csv` 是每数据集诊断；`joint_scores.csv` 记录两任务相关系数、协方差及联合 recall；`query_samples.npz` 保留抽样正确分数和 rank；`pooled.csv` 在同一配置内合并 16 个数据集；`summary.csv` 是各格误差的平均。
- 错误 CDF：每数据集前 8 个查询使用所有 `n-1` 个错误训练候选，报告精确阶梯 CDF 与标准正态 CDF 的 KS 距离；另外记录阈值 `-1,0,1` 处 CDF。
- rank 分位：`.1,.5,.9`，理论值 `Phi(-mu+Phi^{-1}(p))`；rank CDF 另在 `.01,.05,.1,.25,.5,.75,.9` 七个固定点比较。
- `stability.csv`：实际与代理矩阵误差及完整正确分数 MSE；另检查四步 GD、Sign、Adam(.9,.99)、Adam(.9,0)。四步检查仅针对稳定性，没有声称数值验证了完整四步 CLT。
- `regression.csv`：在两份键条件下直接组装完整协方差和回归矩阵，检查最小特征值、`||P||`、`d||P||`；GD 协方差修正另外由 Wick 公式计算。
- `branch_grid.csv`：30 个匹配目标、四个方向；覆盖负相关、负目标、两种可达几何。`edges_metadata.json` 记录晚分支排序反例及负旧均值的有限切点例子。
- `full_dataset.csv`：另外 48 个独立数据集，完全共享键，`ry=0,-.5`；实际一步 Sign 的两个任务都计算所有训练查询与所有候选的 rank。采用分块矩阵乘法，未分配 `n*n` 矩阵。`concentration_summary.csv` 的 SD 是 12 个独立数据集的 SD。

只把独立数据集作为复制单位；不对相关查询套用独立样本显著性检验。两档维度只能暴露显著反例或提供有限数值诊断，不能证明 CLT、严格排序、收敛速率或 recall 集中。

完整表格见 [results.md](results.md)。数学审稿见 [PROOF_REVIEW6.md](../PROOF_REVIEW6.md)。

`verify_artifacts.py` 用相同 systemd/线程命令运行，检查输出数量、样本形状、有限性、分支、脚本语法和资源限制，并生成 `verification.json`。主模拟前后四份输入哈希相同；最终检查发现背景 `fine_tuning.tex` 被外部进程改动，已重新读取、在审稿报告中说明，并在 verification 中保留初始/最终哈希。W7、M、G 的哈希仍未变化；本轮未写入任何既有文件。
