# EXP1

先读项目根目录的 `PREREG_EXP1.md` 与 `PREREG_EXP1.sha256`。预注册在任何本轮冒烟/正式计算之前冻结；历史 `smoke_v2.py` 只作为先导参考，本实验不 import 它。

* `proxy.py`：精确 A+B+V、六组显式迭代/MC、有限网格前沿、等损失插值。
* `matrix.py`：共同有限教师、逐步新数据、SGD/oracle/薄 SVD Muon、独立调参、局部留出响应及噪声协方差。
* `run.py`：冻结校验、单线程环境、阶段预算、逐配置落盘、RSS 与源文件哈希。
* `analyze.py`：只读正式数据的独立完整性检查和汇总；不重新选择拟合区间。
* `render.py`：从已审计数据生成补充表格、PNG 和 PDF 科学图；不重新计算或筛选实验结果。

所有命令从本题根目录执行，必须用指定解释器并保留 systemd 内存约束。输出目录需尚不存在；复现时另取本题目录内的新目录，保留原结果。

```bash
sha256sum -c PREREG_EXP1.sha256
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 timeout 180s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp1/run.py --smoke --out experiments/exp1/smoke_reproduce
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 timeout 3300s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp1/run.py --out experiments/exp1/formal_reproduce
```

本轮已有正式数据的只读审计和重新制图：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp1/analyze.py
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp1/render.py
```

这两个汇总脚本固定读取 `formal/`；分析其他复现目录时应先修改汇总脚本的输入路径，并保留原始分析结果。重新审计会改写耗时字段，重新绘图会改写导出文件，因而最终 `ARTIFACTS.sha256` 需重新生成；冻结的 `PREREG_EXP1.sha256` 始终不改。

原始日志 `smoke.log`、`formal.log` 含 scope 名称与完整进度；每个输出目录的 `manifest.json` 记录时间、解释器、线程、代码哈希、冻结哈希、进程峰值 RSS 和完成状态。`formal/scope_properties.txt` 记录运行中的实际 systemd 属性。

`proxy.json` 保留全部预算、A/B/V/n/b/T/h/Ta_n、边界与相邻差分；大整数在 JSON 中保持十进制整数，JavaScript 消费者应避免将它们自动转换为 Number。`calibration.json` 保留全部96条候选训练和锁定学习率；`matrix.json` 保留全部432条正式训练；`matrix_frontiers.json` 为按 nD/实测墙钟分别选取的有限网格 Pareto 曲线。墙钟前沿不包括调参费用，调参总量和 SVD 秒数单独报告，不能隐去后当总成本。

`local.json` 的 rates/se 是未乘学习率的每步谱响应；每个 `.npz` 保存均值、完整128×128更新噪声协方差和 rates/se。固定残差快照与在线训练有不同控制变量；局部结果不能直接代入代理的 D=bT 方差公式。拟合仅使用第一快照，留出快照不重拟合；chi 在固定 n=32 的诊断中不可识别。95%区间只描述标准误，不是事后制定的通过门槛。
