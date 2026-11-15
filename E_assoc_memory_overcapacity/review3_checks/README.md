# 第三轮检查记录

数学结论与逐段审查见 `../PROOF_REVIEW3.md`。本目录的有限数值检查不构成渐近证明。

- `check_review3.py`：独立反例搜索，不导入作者检查代码，不调用网络/API。
- `results.json`：种子、线程设置、输入散列、全部诊断结果、输入未变断言。
- `run.stdout`、`run.stderr`：独立检查输出、systemd scope 标识和 time 资源记录。
- `certificate.stdout`、`certificate.stderr`：审查代码后，只读重跑既有严格证书的输出和资源记录。

实际使用的两条命令（工作目录为项目根目录）：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 timeout 900s env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v /home/liuyibo/venvs/theory/bin/python -B review3_checks/check_review3.py > review3_checks/run.stdout 2> review3_checks/run.stderr

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 timeout 900s env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /usr/bin/time -v /home/liuyibo/venvs/theory/bin/python -B p04_checks/check_l1_l3_l4.py certificate > review3_checks/certificate.stdout 2> review3_checks/certificate.stderr
```

两次超时预算合计 1800 秒；实际墙钟合计约 5.56 秒。所有 Python 检查均在上述限制下执行。独立脚本用独占创建模式写 `results.json`，会拒绝覆盖已有结果；本次不删除或改写任何旧产物。
