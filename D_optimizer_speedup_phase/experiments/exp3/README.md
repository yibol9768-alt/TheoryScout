# EXP3 复现

零拟合输出 Gram 模型见根目录 BRIDGE_V3.md；所有门槛见 PREREG_EXP3.md。
不 import 旧实验或其他项目。CPU/安装库、单进程单线程，无网络/API。
从题目根目录运行；输出目录须不存在，更改scope名时同步更改--scope。

```bash
sha256sum -c PREREG_EXP3.sha256
sha256sum -c experiments/exp3/FROZEN.sha256
systemd-run --user --scope --unit=theoryscout-exp3-smoke -p MemoryMax=2G -p MemorySwapMax=0 timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp3/run.py --smoke --out experiments/exp3/smoke --scope theoryscout-exp3-smoke.scope
systemd-run --user --scope --unit=theoryscout-exp3-formal -p MemoryMax=2G -p MemorySwapMax=0 timeout 3000s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp3/run.py --out experiments/exp3/formal --scope theoryscout-exp3-formal.scope
systemd-run --user --scope --unit=theoryscout-exp3-analysis -p MemoryMax=2G -p MemorySwapMax=0 timeout 300s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp3/analyze.py theoryscout-exp3-analysis.scope
```

run.py核验冻结、记录scope/解释器/线程/峰值RSS；先写locked_model.json再打开留出。
无参数拟合阶段。每单元保存原始一二阶矩、16块均值、前后半统计、Gram矩；
analyze.py独立重构、按固定组判定，并导出analysis.json/TABLES.md和PNG/PDF。
实验原始batch在内存逐单元处理，不声称保存了全部随机样本。
最终ARTIFACTS.sha256覆盖报告和实验文件；cache不算科研产物。
