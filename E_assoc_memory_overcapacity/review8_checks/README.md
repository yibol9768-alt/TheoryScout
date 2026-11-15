# 第八轮独立审稿计算

数学结论见根目录 `PROOF_REVIEW8.md`。没有调用 LLM API，没有导入、执行或复制 `w9_checks/` 的代码；审计程序仅只读检查其数据和源码散列。**最终采用 `strict_run/` 内结果。** 根目录同名数据保留了首轮探索运行，不作为满足总线程限制的最终证据。

- `independent_checks.py`：独立实现实际 GD/Sign/Adam；精确分段积分计算对角 Sign4 常数，一维自适应积分计算 Adam4 常数。包含非对称非高斯键、双非高斯 GD、精确三点分布四阶矩证书及高斯 Sign 平局反例。
- `strict_run/rows.jsonl`、`strict_run/summary.json`、`strict_run/SUMMARY.md`：144 个数据集、468 个算法端点的原始数据和按独立数据集种子汇总的结果。
- `strict_run/predictions.json`：独立于训练样本的预测。
- `strict_run/exact_certificates.json`：81 状态有理数恒等式核验；有限维高斯 Sign 平局反例。
- `artifact_audit.py`、`strict_run/artifact_audit.json`：独立重算原交付 600 行和 72 行的汇总；检查来源代码散列及输入未变。
- `strict_run/input_hashes_before.json`、`strict_run/input_hashes_after.json`：原审查输入（W9、M、w9_checks 全部已有文件）的 SHA-256。主体运行前后完全一致，后续审计再次核对。
- `strict_run/metadata.json`、`strict_computation.log`：解释器、线程配置、实际进程线程数、种子、维度、耗时及 RSS。首轮各库设 2，但诊断发现 NumPy/SciPy 两个线程池使进程有 3 个线程，故最后统一为 1 并全部重跑。初次启动另因缺少可选包 threadpoolctl 中止，其记录保留于 `run.log`；移除了该依赖，没有更改指定环境。
- `final_verify.py`、`strict_run/final_validation.json`：最终源码语法、实际总线程数、输入散列和端点完整性检查；包含报告和代码散列。

计算命令（项目根目录）：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
      REVIEW8_OUTPUT_DIR=review8_checks/repro_01 \
  /home/liuyibo/venvs/theory/bin/python review8_checks/independent_checks.py
```

审计使用相同资源前缀与输出目录，脚本换成 `review8_checks/artifact_audit.py`。输出采用排他创建；已有结果不能直接覆盖。复现时两个脚本均使用一个全新的子目录。`repro_01` 仅为复现预留，本轮没有创建该目录。

所有数据集的 κ=.5，d=64/128/256，各 12 个独立种子，每数据集 384 个无放回训练查询，候选为全部训练值。参数 η=.45、α=.55、β₁=.8、β₂=.93、e=.17、T=4。键谱为各占一半的 .2/1.8；非高斯基础变量为 −1/√2（概率 2/3）和 √2（概率 1/3）。非对角模型使用对称正定平方根。程序按合法均匀规则处理 ties，已测端点未观察到 ties。

区间是独立数据集种子的逐项 t₁₁ 95% 描述性区间。有限维偏差和多重比较没有被区间消除；没有进行等价性认证。逐查询 KS 仅在每数据集固定的前 16 个抽取查询上计算。模拟不能证明渐近定理或其收敛速率。
