# V3 直接在线杀手检验

范围和阈值见根目录 `PREREG_KILLER_V3.md`；预测模型为 `FORMULATION_V3.md`。`run.py` 无项目模块导入、无网络、无GPU；仅系统标准库、指定venv的numpy/scipy。`analyze.py` 从保存的完整检查点状态独立重算风险、偏差/方差和定位比，按冻结门槛判断并生成静态PNG/PDF。不是定理证明器。

所有命令从本题根目录执行。先生成SHA256冻结；以下scope名称本轮各用一次。外层shell重定向也只写本目录。`TMPDIR`、图缓存均在本目录。

```bash
mkdir -p experiments/killer_v3/tmp experiments/killer_v3/mplconfig
systemd-run --user --scope --unit=theory-d-killer-v3-smoke -p MemoryMax=2G -p MemorySwapMax=0 timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR="$PWD/experiments/killer_v3/tmp" KILLER_SCOPE=theory-d-killer-v3-smoke.scope /home/liuyibo/venvs/theory/bin/python -I -B experiments/killer_v3/run.py smoke --max-seconds 90 > experiments/killer_v3/smoke.log 2>&1
systemd-run --user --scope --unit=theory-d-killer-v3-smoke-audit -p MemoryMax=2G -p MemorySwapMax=0 timeout 60s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 KILLER_SCOPE=theory-d-killer-v3-smoke-audit.scope /home/liuyibo/venvs/theory/bin/python -I -B experiments/killer_v3/analyze.py smoke > experiments/killer_v3/smoke_analysis.log 2>&1
# Only start the formal experiment after successful smoke and its audit.
systemd-run --user --scope --unit=theory-d-killer-v3-formal -p MemoryMax=2G -p MemorySwapMax=0 timeout 3000s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR="$PWD/experiments/killer_v3/tmp" KILLER_SCOPE=theory-d-killer-v3-formal.scope /home/liuyibo/venvs/theory/bin/python -I -B experiments/killer_v3/run.py formal --max-seconds 2700 > experiments/killer_v3/formal.log 2>&1
systemd-run --user --scope --unit=theory-d-killer-v3-formal-audit -p MemoryMax=2G -p MemorySwapMax=0 timeout 180s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR="$PWD/experiments/killer_v3/tmp" MPLCONFIGDIR="$PWD/experiments/killer_v3/mplconfig" KILLER_SCOPE=theory-d-killer-v3-formal-audit.scope /home/liuyibo/venvs/theory/bin/python -I -B experiments/killer_v3/analyze.py formal > experiments/killer_v3/formal_analysis.log 2>&1
```

`-I -B`隔离用户Python路径并禁止写字节码。解释器与系统运行库是运行必需项；不加载其他项目文件。每个阶段保存实际scope属性和代码/预注册hash。计算预算包括冒烟、正式及分析，正式不成功也保留事件和incomplete manifest。相同输出目录已存在时脚本拒绝覆盖；重现实验应自行选择隔离的本目录副本，不能把新数据覆盖成本轮冻结结果。

正式训练样本记为每算法各自D；共同随机数只减少配对比较方差，不使两个算法都获得免费训练。`training_nD_per_algorithm`实际保存所有算法、所有重复的nD总和；名称表示采用每算法付费口径。kernel测量/积分、标量干预、静态图与审计均单列墙钟。`arrays/trajectory_*.npz`保存全部重复在全部检查点的误差状态，`control_*.npz`保存各重复的两臂终点；`events.jsonl`、`data.json`保存局部充分统计、种子命名空间、解析预测、MC SE、完整门槛及Gram消融。没有保存全部训练输入或全部局部batch，固定种子可复现。

五个引理是自足 `problem` LaTeX片段，均为待证任务；可在带amsmath/amssymb和problem环境的外壳中编译。本轮不触发外部QED或LLM。冻结文件完整性由 `FROZEN.sha256` 检查，结果完整性另列 `ARTIFACTS.sha256`。
