# REVIEW14 独立核验

中文审稿报告见 `../PROOF_REVIEW14.md`。本目录只包含本轮新建代码和结果；没有导入或执行被审代码，没有调用 API。

计算命令（项目根目录）：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -B review14_checks/independent_checks.py
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -B review14_checks/algebra_checks.py
```

`independent_checks.py`：208,000 组原始高斯数据，T=4，d=2,3,4,6,8,12,16；每组同时测试 h=.1,.75,.95。通过谱权重凸函数的全局极小值判定全实线无根，独立统计固定区间规则失败和谱条件失败。64 次二分确定全局极小点，44 次二分计算通过规则的局部根。给出逐格 95% Clopper–Pearson 区间，不作多重覆盖声明。同一份数据的三个 h 结果相关。另有 4 组几何恒等式，以及每组 10,000 次完整联合行散布固定的 Haar 共轭检查。结果为 `results.json`，含随机种子、9 个原输入前后散列和实际 cgroup 控制值。

`algebra_checks.py`：从四步 GD 递推独立生成中心化非交换散布词，以同任务二次词的高斯迹收缩重算 1/d 系数；没有把 Y1B 的 Lo 或 L_Lambda 公式作为生成式。结果为 `algebra_results.json`。

脚本均用排他模式创建结果文件，直接再次运行会拒绝覆盖。重放时可在本目录另存脚本副本，仅把输出结果名改为新名字，保持项目 ROOT 不变；不要覆盖本轮结果。所有核验都只是解析审查的补充：零次失败不能证明零概率，频率相容不能证明指数速度，有限代数不能证明概率余项。Y1B 的原指数界在本轮小维度均为平凡界 1，报告另外给出仅适用于全实线无根的更强解析界。
