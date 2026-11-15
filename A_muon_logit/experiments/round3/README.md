# Round 3：只读旧训练轨迹的离线研究

所有路径在本项目内；`model.py` 只用于重建数据、前向及解析梯度。推进权重的唯一来源是原保存的方向，绝不调用 `Optimizer.direction` 或 `run_training`。Adam 一阶/二阶矩只是沿既有状态序列累计的诊断量。

先核验根目录 `PREREG_ROUND3.sha256`。实际执行命令（每条顺序执行）：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/replay.py --smoke
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/analyze.py --smoke
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/validate.py
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/calibrate.py --smoke
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/replay.py
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/analyze.py
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/calibrate.py
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 bash -c 'sha256sum experiments/round3/SUPPLEMENT_DENSE_CE.md > experiments/round3/SUPPLEMENT_DENSE_CE.sha256'
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/dense_ce_supplement.py
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/audit.py
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/write_report.py
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/round3/finalize.py
```

`smoke/`：1 seed、1 头的条件模型与 ACF 家族（原权重重放仍验证全矩阵），模拟每阶段仅 200 次，不参与正式结论。`full/`：18 条主运行、54 条对照；每个近似校准阶段严格 20000 次，整行置换 1999 次。日志保留 systemd scope 标识。

`full/arrays/` 保存每步 a,b、xi、真实全训练梯度投影 g、固定分块噪声 SD、原 batch 梯度投影、真实输入范数换算的 B，及冻结模型输出与创新；这是从旧轨迹重建的数据，不是新训练。`.npz` 均为普通数值数组，读取不需要 pickle。JSON 保存各运行结果、原始输入 SHA256、拟合参数、逐候选诊断、共同状态方向比较、密集 CE 和全部匹配/缺失记录。

`calibrate.py` 的高斯固定设计模拟只是预注册的附加校准。它重新拟合单步模型，但不模拟完整反馈状态，不能满足 §7.3 的主反馈 bootstrap，也不能支持 P-V2-5。整行置换只是一项交换性敏感性分析。任何主关不可检验都会在总报告中明示。

CE补充只补齐原稀疏目标穿越区间，不替换提前冻结的主 CE 表。若只是重现已冻结补充，先核验现有 SUPPLEMENT_DENSE_CE.sha256；不要覆盖原哈希来容纳规格变化。
