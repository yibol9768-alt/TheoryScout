# Review 10 独立检查

本目录只存本轮新增代码、结果与文献快照。不导入或执行 `x3_checks/` 中的任何程序；读取其 JSON 仅用于取得待验端点和比较输出。

计算均使用指定解释器、2 GiB 上限、禁用 swap、最多两线程。两个主要脚本按顺序运行：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python review10_checks/independent_check.py

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python review10_checks/structural_checks.py
```

`independent_check.py`：

- 用 128、256、512、1024 阶 Gauss–Legendre 独立计算五个负载的均值、混合矩和方差，再用另一种标量积分求交点估计。
- 用 `N(p)/sqrt(d(p))` 递减、`pr/D(p,r)` 分别递增的性质构造独立矩形包围，网格 4096²。不是复制被审程序的自然区间代入。
- 均值使用平方根二项级数（不同于原程序的逆平方根级数），35 阶后加显式尾界。
- 对 30 个有理端点使用 55 位 mpmath 区间算术、28 次 Taylor、多面板 `r/R<=1/8`、Cauchy 余项及半正态尾项，独立计算效率包围。中心指数用 mpmath 的区间指数，不复用原程序的中心指数算法。
- 所有 15 个交点严格端点不等式均通过；检查本轮读取的被审文件前后哈希相同。

`structural_checks.py`：

- 从 MP 密度直接积分，检查 12 组两 resolvent 内积分和核，覆盖 γ=1.01、2、16、64。
- 用 Bartlett 分解独立抽样实 Wishart，d=512、1024，γ=16、64，每格三次。这里的 `nu_plugin` 只作尺度旁证。
- 检查实际 GD、Sign、Adam 四步轨迹的样本旋转不变性及分数矩阵共轭；Adam 特别取 β₂=0<β₁。
- 保存联网取得的 Barnfield v2 HTML 和 SHA-256。快照哈希与原比较稿的 source_manifest 相同。

输出为 `independent_results.json`、`structural_results.json`、`barnfield_v2.html`。

`grid_boundary_check.py` 给出 `n=2^27` 时初始网格变换不再精确的标量有理反例；`final_validate.py` 汇总 15 对符号、30 个效率区间的包含关系、全部 μ/k/ν 包围的包含关系、十六进制记录和输入哈希。复跑它们时同样使用上述完整的 scope/env/解释器前缀。

独立区间检查仍依赖注明的浮点原语及 mpmath 区间实现，不是整数形式化证明。Gauss–Legendre 阶数收敛和 Monte Carlo 不承担证书职责。

补充：曾在相同资源 scope 内尝试 `pdflatex`，系统没有该命令；没有因此写入任何已有文件。此次预览审查基于 wrapper、被引用片段和表格源码，没有把原有 PDF 的存在当成独立编译证据。
