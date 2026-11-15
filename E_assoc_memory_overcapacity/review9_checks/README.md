# 第九轮独立审稿检查

三个脚本均为本轮独立编写，不导入、不执行 `x2_checks/` 或其他项目内的被审数值模块。读取旧 JSON 仅用于对照数据。原有文件保持不变。

每条命令均从项目根目录单独运行；输出采用独占新建模式，已有 JSON 不会覆盖。

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python -B review9_checks/independent_checks.py

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python -B review9_checks/student_rank.py

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python -B review9_checks/final_audit.py
```

文件说明：

| 文件 | 作用 |
|---|---|
| `independent_checks.py` | 从原始高斯单项式独立枚举 Wick 配对；符号核对 FT1 全参数系数；独立展开 delta 漂移和扩散；从谱函数导数重算 FT4；自适应径向积分；三个配置各 60000 份实际原始训练数据检查 |
| `independent_ft_polynomials.json` | 独立生成的有限维整数 Laurent 多项式 |
| `results.json` | 系数、求积、原始数据检验、配对检验结果 |
| `student_rank.py` | 从正确与错误分数之差推导 Student-t 精确条件概率；对 GD1 作三变量确定性求积 |
| `student_rank_results.json` | 精确概率表达式的浮点求积结果、节点加密及独立条件模拟；不是区间证书 |
| `final_audit.py` | 从原 W4 原始 JSONL 重算残差；检查原排名求积误差；给出八格 Bonferroni-t 描述性区间 |
| `final_audit_results.json` | 上述统计核查结果和资源记录 |
| `input_hashes.json` | 被审主文、背景与全部已有 `x2_checks/` 工件的 SHA256；计算末尾再次核对 |
| `resources.json` | 主检查的 Python、线程、cgroup、时间和 RSS 记录 |

新代码是检错工件，不以 Monte Carlo 相容性证明恒等式。符号 Wick 系数比较的残差为精确零；浮点积分均未采用向外舍入。

统计限制：`chi2_4_reference_p` 只是四维度独立 seed 统计量的高斯近似诊断，SE 本身也由有限样本估计。Bonferroni-t 区间同样依赖 seed 均值的近似 t 推断；它不是分布无关置信保证。区间针对有限维有效系数，不能直接当作渐近系数的置信区间。两协议按同 seed 配对，未将它们合并成八个独立维度。
