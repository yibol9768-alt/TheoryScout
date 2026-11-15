# V3 A′ 旧轨迹重放

严格限定本项目文件；只导入本项目 `experiments/model.py` 的数据构造/前向及标准库、NumPy。权重推进的唯一来源是保存方向，未调用 `Optimizer` 或训练函数，无网络/API。

规格：根目录 `PREREG_V3_OFFLINE.md`，冒烟前冻结 SHA256 为 `fbba8417b81069153c4f969a8830f9589386b6fb38bd9d16d3432c9a9c037138`。复现时先核验现有哈希，不重新冻结。

实际按以下顺序执行（均在项目根目录）：

```bash
sha256sum -c PREREG_V3_OFFLINE.sha256
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/v3/run.py --smoke > experiments/v3/smoke.log 2>&1
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/v3/run.py > experiments/v3/full.log 2>&1
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/v3/audit.py > experiments/v3/audit.log 2>&1
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -u experiments/v3/write_report.py > experiments/v3/write_report.log 2>&1
```

- `smoke/`：1轨迹/1头；重放仍检查完整权重和全部方向谱范数；不并入正式结论。
- `full/`：18轨迹/72头。每运行 JSON 保存拟合与测试起点、冻结候选、T/T_exact、H、各gamma预测、拟合网格损失、门槛和在线扫描计划。`summary.json` 是关卡汇总。
- 同名 `.npz`：`training_steps=100..600`、`signed_logits`、`qnorm`、`knorm`（形状501×4×1024），以及 `xnorm`、`ynorm`。100..149只为首个拟合起点的历史50步特征，正式评价仍是登记区间。不需要pickle。
- `aggregate.json`：逐条件/eta保留11/23/47顺序的未舍入指标。
- `audit.json`：36个原输入哈希、1296个origin/head的独立标签/预测复核、32400个在线覆盖步复核，以及数值数学检查和LaTeX问题/花括号检查。检查通过不等于形式化证明。
- `ARTIFACTS.sha256`：最终文档、代码、JSON、NPZ和日志快照校验；不包含校验文件自身。原输入哈希保留在逐运行JSON中。

G2/G3的不可识别与失败原因按原门槛报告；没有加长窗口、降低.9、变更top4或补选头。成本比只计分数评估，异质证书保留O(N)锚点元数据，不宣称真实attention内核加速。
