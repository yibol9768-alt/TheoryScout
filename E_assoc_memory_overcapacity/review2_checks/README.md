# 第二轮审稿核对记录

审阅对象及原始 SHA-256 见 `input_sha256.txt`。脚本不导入项目代码、不访问网络、不调用 LLM API，输出只经 shell 重定向写入本目录。有限搜索不承担渐近证明责任。

最终可复现命令（在项目根目录运行）：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
      PYTHONDONTWRITEBYTECODE=1 \
  timeout --signal=TERM --kill-after=5s 480s \
  /home/liuyibo/venvs/theory/bin/python -B review2_checks/check_review2.py \
  > review2_checks/results_final.json 2> review2_checks/run_final.stderr
```

实际进行了两次数值运行，均退出 0，均采用上述 systemd 资源限制、单进程、各计算库线程设为 1。脚本内部总截止为 400 秒，每次外层截止为 480 秒；两次截止之和也小于 30 分钟。两次工具测得命令耗时合计约 1.26 秒，脚本记录的计算耗时合计约 0.624 秒，最大 RSS 约 80.43 MiB。

- `results.json` / `run.stderr`：第一次运行的原始输出。近零 GD 上界的双精度比值无数值解释价值，未将其视作数学反例。
- `results_final.json` / `run_final.stderr`：最终脚本的输出；只在分母大于 `1e-8` 时记录比值，同时保留所有近零分母计数与最大绝对超出量。另用 150 组精确有理运算核对 GD 的标量边界，避免以舍入误差判断这些边界。
- `check_review2.py`：最终独立核对脚本。包含 28,605 次浮点等式／不等式检查、150 次 GD 精确标量检查、附录 B 的 26 个严格及 2 个非严格有理比较，以及查询与平局随机数缺少联合独立性时的四状态精确反例。
- `input_sha256.txt`：本次读取的三个原有文件摘要；工作末执行 `sha256sum -c review2_checks/input_sha256.txt`，三项均一致。

反例位于最终 JSON 的 `missing_query_tie_independence_counterexample`，它针对 §10.3 的有限维恒等式所缺的条件。`violations: []` 仅表示其余所测数值不等式未超出脚本容差，不表示该反例不存在，也不表示定理经过形式化认证。
