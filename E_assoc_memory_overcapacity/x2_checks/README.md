# X2 复现

主文：`../X2_FINITE_D.md`。只读原项目输入；新增结果全部在本目录。

在项目根目录运行（默认新建 `x2_checks/replay_output/`，拒绝覆盖）：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python -B x2_checks/replay.py
```

也可追加一个位于 `x2_checks/` 内、尚不存在的输出目录路径。原始运行各阶段合计约一分钟；该入口顺序重放同一固定设计，不新增/筛选种子。

文件映射：

| 文件 | 内容 |
|---|---|
| `wick.py` | 有限维 FT1 Wick 精确求和，整数配对权重 |
| `checks.py`, `verified/` | 正式单线程解析系数、FT1 精确矩、Sign/Adam 精确积分、输入散列、原 W4 重算 |
| `scatter_checks.py`, `scatter_raw.jsonl`, `scatter_summary.json` | 精确 Wishart/Bartlett 散布样本，Haar 条件全查询矩，FT 固定/匹配配对 |
| `rank_and_small.py`, `rank_reduction.json`, `small_checks.json` | GD1 排名精确降维及独立原始小数据训练核验 |
| `ft_rank_reduction.py`, `ft_rank_reduction.json` | 独立任务 FT1 旧任务排名精确降维 |
| `summarize.py`, `report_tables.txt`, `w4_retrieval.json` | 报告数表、原 W4 检索重算、PNG/PDF |
| `audit.py`, `audit.json` | 输入未变、闭式系数与未化简积分一致、Hermite 48/80 节点核对 |
| `*_metadata.json`, `*.log` | 各阶段资源、用时、全部 stdout |
| `format_report.py` | 仅用于本轮新建报告的数学定界符排版，不参与科学计算/复现 |

顶层 `analytic_metadata.json` 是最初双池各 2 线程的记录，实际进程共 3 线程；保留这一偏差。正式结果采用 `verified/` 单线程完整重算。其余运行均实际进程 1 线程。所有 scope 的内存上限 3 GiB、swap=0；未安装包、未调 LLM API。

精度范围：矩条件积分是有限维精确恒等式的浮点计算；Gauss 求积加密不是区间证书。降维 rank 的球面尾概率误差为 O(n^-2)，降维 recall 使用连续球面阈值，与离散固定比例 recall 的差为 O(1/n)。scramble SE 不包括这两项确定性误差。没有新做同时等价性认证。
