# EXP1 复现说明

所有命令从本题根目录执行，仅用本地 NumPy 和标准库。主脚本不导入其他项目文件，不访问网络或调用模型。冻结方案见根目录 `PREREG_EXP1.md`，结果见 `RESULTS_EXP1.md`。

本轮顺序：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python experiments/exp1/run.py moments
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python experiments/exp1/prepare_prereg.py
sha256sum -c PREREG_EXP1.sha256
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 310s /home/liuyibo/venvs/theory/bin/python experiments/exp1/run.py smoke
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout 3500s /home/liuyibo/venvs/theory/bin/python experiments/exp1/run.py formal
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python experiments/exp1/analyze.py
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python experiments/exp1/audit.py
sha256sum -c experiments/exp1/ARTIFACTS.sha256
```

这是执行记录，不应直接重跑前几条覆盖冻结积分文件。prepare_prereg.py 与新实验 run 均拒绝覆盖已有注册/run；moments 模式只用于注册前积分。重新计算已有原始数据的汇总可以运行 analyze.py 与 audit.py；原始数据复跑需要在本题目录下创建独立副本，并把脚本 OUT 指向新的子目录，不可覆盖本轮输出。

原始 JSONL 每行对应一个数据配置/种子/方法，含128查询的秩、正确分数、每查询33点错误候选CDF，以及相应预积分预测、超参数与真实学习率/epsilon。A/B重复的 d=128 数据属于实现一致性检查，统计汇总仅使用B一次。五个独立数据种子为SE单位。

- `all_gates.csv`：每格预注册门槛与通过/失败。
- `paired_reversal.csv`：配对两步减一步，含SE。
- `stability_by_dimension.csv`、`stability_trends.csv`：权重/对角得分误差及真实/代理曲线差。
- `score_cdf_diagnostics.csv`、`conditional_mean.csv`：方差、错误CDF、自作用条件均值诊断。
- `top1_exploratory.csv`：无有限维判决门槛的C阶段。
- `analysis_summary.json`：全部异常/非单调项。
- `integrity_audit.json`、`ARTIFACTS.sha256`：原始数据一致性及最终文件摘要。
- `qed_snapshot.txt`：只读QED核验状态快照；不调用QED或任何LLM。
