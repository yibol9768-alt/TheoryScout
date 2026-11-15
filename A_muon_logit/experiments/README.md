# 第1周离线 CPU 实验

从项目根目录执行（无外部 API）：

```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /home/liuyibo/venvs/theory/bin/python experiments/run_week1.py
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  /home/liuyibo/venvs/theory/bin/python experiments/summarize.py
```

`run_week1.py` 先存不可变的执行前登记与源码 SHA256，再做解析反传核验、含测量的50步基准、E0、72项E1和24项E2。已有结果会被同名文件覆盖；需要新一轮复现实验时请先保留本轮 `results/` 和 `figures/`。

`results/manifest.json` 记录解释器、版本、线程数、种子、预算、实际运行清单、RSS与计时。`preregistration.json` 保存运行前报告全文及哈希。

`e0.json` 为流式 Monte Carlo 的 ensemble 原始统计和独立重复尾段统计；`runs/*.json` 为完整测量、每步投影、窗口统计和前向预测；`directions/*.jsonl.gz` 每行一个 JSON 对象，含 step、D_Q、D_K，保存未除 k0 的 float32 方向矩阵。重现实际更新时用各运行 JSON 中的 eta、k0、alpha，QK-Clip 还须使用每步 `clip.scale`。训练实际使用 float64 方向。

图在 `figures/*.png`，中文报告在根目录 `RESULTS_WEEK1.md`。校准运行有独立 `calibration_` 前缀，不计入72个正式E1。
