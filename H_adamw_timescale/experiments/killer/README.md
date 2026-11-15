# W killer：运行与证据

模型和门槛唯一来源为 ../../PREREG_KILLER.md（sha256冻结）。先smoke后formal，正式完整结束再analysis/plot。所有源码为本题新写，不import D/E；仅import当前目录run.py、标准库与安装的numpy/matplotlib。

所有命令在 H_adamw_timescale 下执行，均使用同一限制前缀：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 120s /home/liuyibo/venvs/theory/bin/python experiments/killer/run.py smoke

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 3000s /home/liuyibo/venvs/theory/bin/python experiments/killer/run.py formal

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 180s /home/liuyibo/venvs/theory/bin/python experiments/killer/analyze.py

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 60s /home/liuyibo/venvs/theory/bin/python experiments/killer/plot.py

sha256sum -c PREREG_KILLER.sha256
sha256sum -c experiments/killer/ARTIFACTS.sha256
```

正式数据已存在；run.py会拒绝覆盖formal/，请保留原始证据。分析和作图可以对相同冻结数据重建。绘图cache限定在本目录.mplconfig/。环境中的 /home/liuyibo/... 与工具显示的 /mnt/c/Users/... 指向同一工作目录；本任务没有转入其他项目。

## 输出格式

- formal/*.npz：96个训练轨迹文件。risks形状为(4,42)单遍或(1,42)多遍，deltas形状(42,)。diagnostics为(检查点,3,42)：末步范数、上一步权重与Adam更新向量的余弦、一步旋转角（弧度）。这些是快照，不是稳态平均。
- formal/metadata.json：所有任务、随机种子、训练源码hash、冻结hash、UTC、资源峰值及wall_seconds。SeedSequence([seed,N,mode,ratio]).spawn(3)分别用于数据、初始化、洗牌。
- smoke.json/log：仅独立冒烟，不能用于正式均值。
- curves.csv：21×42=882行，每行包含均值、SE、八个种子的完整风险。
- summary.json：门槛、格点、近优集合、单种子最优、bootstrap及拟合。JSON中的null代表无穷时间尺度或无法判定，具体看对应字段。
- TABLES.md：所有21格，不删除边界或零衰减。*.svg/png为科学绘图，可独立导出。
- ARTIFACTS.sha256：源码、冻结文件、原始数据及主要输出的完整性清单；logs的scope回执可核对实际执行限制。

## 实现日志

2026-11-06：冻结后实现run.py，smoke一次通过，随后formal一次完整完成（42.17秒）。没有重跑正式或更改模型/网格/判据。

第一次分析出现无害的“divide by zero”警告：先对含零δ的完整数组求倒数再取正索引。修成只对已选正δ求倒数，同时新增文件inventory、总数和最小范数断言。原日志保留为analysis_initial.log；随后仅重跑汇总，结果G0–G3及所有选中格点不变。训练源码未修改。图中无穷点调整了显示高度，以免三个尺寸的符号重叠；无新增训练。

八种子bootstrap选中λ后，单遍/多遍完整斜率有效率均0，不产生合法置信区间；不能用正δ子网格诊断斜率当主结果。固定选中λ的配对改善区间存在选择偏差，不能作独立确认。
