# review20 独立核验

审查结论见 `../PROOF_REVIEW20.md`。此目录的程序不导入、不调用、不执行 `aa2_checks/` 或其他被审项目代码，不调用 API，不联网。审计程序仅把已有 JSON 当作待核对的数据读取。

- `independent_check.py`：独立积分与有限维模拟。半正态矩使用对数变量积分；polar 均值用超几何表达，混合矩用未化简的两 resolvent 插入公式和 MP 角变量积分，避免复写被审程序的有理二维积分核。
- `theory.json`：交点、效率、预测召回、预测 MSR，以及受保护输入的前后 SHA-256。
- `main_simulation.json`：d=256、512、1024，各 24 个独立训练数据集；每数据集 8192 次独立噪声查询；polar 与 Adam e=.10、.12。完整训练候选全部保留，CE 梯度显式双边中心化。MSR 使用精确矩阵缩并，排除正确对角并减去各查询的错误候选均值。
- `main_run.log`：逐数据集进度和配对差。
- `summarize_and_audit.py`：小矩阵直接分数与精确缩并的独立比对、排名门槛检查、输入哈希复核、原 AA2 数字对比及独立模拟汇总。
- `audit_summary.json`：上述审计输出，包含原 AA2 六数据集结果的配对置信区间。
- `replay_saved_design.py`、`saved_design_replay.json`、`saved_design_replay.log`：从定义独立复算原保存随机设计的 16 条召回记录，全部完全一致。此项检查不导入项目代码，也不把同一种子的复算并入新的独立样本。
- `verify_final.py`、`final_validation.json`：最终语法、输入哈希、报告表格数值及产物清单检查；清单包含报告本身。

有限维比较采用相同训练集、查询索引及查询噪声。查询索引为均匀有放回抽样，每次重新抽独立噪声；它估计同一训练集的噪声平均召回，并非恰好对全部 n 个索引各执行一次。标准误与 95% Student t 区间以独立训练数据集为单位，不能把全部查询当作独立数据集。区间是 Monte Carlo 描述，不是有限维定理。辅助的普通带噪查询只使用每数据集最初 128 次抽样。

所有 Python 计算命令均使用：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python review20_checks/independent_check.py \
  --mode theory --output review20_checks/theory_NEW.json
```

主模拟将末两行改成：

```bash
  /home/liuyibo/venvs/theory/bin/python review20_checks/independent_check.py \
  --mode simulate --dims 256 512 1024 --seeds 24 --queries 8192 \
  --es .1 .12 --output review20_checks/main_simulation_NEW.json
```

审计使用同样的 scope、线程环境及解释器，脚本为 `review20_checks/summarize_and_audit.py`。该脚本只读取固定命名的已有结果，且拒绝覆盖 `audit_summary.json`。复跑审计需在允许的新目录或调整其新副本的输出名称。本次没有执行被审核验程序。运行串行，不并行启动多个计算进程。

保存设计复算与最终核对同样使用该 scope、线程环境和解释器，脚本依次为 `review20_checks/replay_saved_design.py`、`review20_checks/verify_final.py`；两者也拒绝覆盖输出。

数值积分与求根均为浮点诊断，没有声称区间认证；交点严格次序由审查报告中的解析比较判定。输出记录解释器、线程环境、实际 cgroup 的 `memory.max`、`memory.swap.max` 和 `memory.peak`。
