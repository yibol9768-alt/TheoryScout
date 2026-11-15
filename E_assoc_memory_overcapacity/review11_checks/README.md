# Review11 独立核验

仅阅读本项目文件；不导入、不执行 `y1_checks/` 或项目的数学实现，不调用 LLM API。主审查在 `../PROOF_REVIEW11.md`。

执行命令（项目根目录）：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python -B review11_checks/independent_checks.py
```

脚本的结果文件使用独占创建，已有 `results.json` 时拒绝覆盖。核验方法：

1. 从原始高斯变量的全部 Isserlis 配对和自由坐标/样本指标计数，计算精确迹、训练查询均值及平方矩。没有采用 Y1 的“两中心块”迹公式作为核验前提。得到两任务的精确均值、正确方差和错误二阶矩的一阶系数。
2. 从精确的正确分数、输出模长表达式，以 `eps=d^(-1/2)` 重新展开，计算根位移、召回及秩的系数。没有从 Y1 的 `S,J,b_*,v_*` 输入开始做自洽性检查。
3. 构造 300 组实际高斯向量，逐路径验证联合表示的 `F,N,W,V` 四个恒等式，包含 `h=0`、`h=1` 和负信号。这是有限精度恒等式检查，不是余项概率证明。
4. 独立求出四步同数据损失匹配的无根例子。其最低损失比为 `0.7904020537292519 > 0.75`。
5. 在四个维度上计算精确条件阈值根与二阶根近似的差。这是确定配置的余项诊断，不认证随机模型的余项阶。

`results.json`：12 项符号残差严格为零；联合恒等式最大绝对残差约 `2.67e-15`；资源断言通过；18 个被审输入的执行前后 SHA256 相同（见 `input_hashes.json`）。运行约 15.07 秒，1 线程，`memory.max=3221225472`，`memory.swap.max=0`，cgroup 峰值 `90497024` 字节，进程 RSS 高水位 `352228` KiB；这两个峰值计量口径不同，均小于限额。

`final_audit.py` 只核对原输入散列、报告结构、输出散列及资源限制；不重新执行被审程序，也不把文档格式检查计为数学验证。`final_audit.json` 为初次快照；修正本 README 的输入文件计数并补充论文正文定位后，以同一资源命令运行 `final_audit.py final_audit_complete.json`，其独占创建的 `final_audit_complete.json` 为最终状态快照。
