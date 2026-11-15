# P04：L1/L3/L4 核查

唯一执行脚本为 check_l1_l3_l4.py，不导入项目代码、不写文件、不调用网络或 LLM API。数值核查均使用 /home/liuyibo/venvs/theory/bin/python，外层由 systemd user scope 限制 MemoryMax=2G、MemorySwapMax=0，线程环境上限均为 2。两次完整执行的累计核查时间远低于 45 分钟；第二次输出附运行耗时与峰值 RSS。

最终完整运行退出码 0，耗时 2.56 秒，峰值 RSS 80,612 KiB；严格证书和所有有限数组核查均通过。

运行方式：

~~~bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  timeout 2400s env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -B \
  p04_checks/check_l1_l3_l4.py all
~~~

模式：

- certificate：只用标准库。384-bit dyadic 向外舍入、Machin 有理 π 界、分段 Taylor 加 Cauchy 余项、正态尾界，严格检查两个根端点的残差符号。
- diagnostics：普通 SciPy 积分求根；固定种子的小型有限数组恒等式；给定包络的若干角点核查；明确标作预测中心的噪声交叉数值。这部分不构成渐近定理证明。
- all：依次执行上述两部分。

文件：

- check_output.txt：首次完整执行原始输出。
- final_check_output.txt：最终脚本完整执行及资源摘要。
- source_sha256.txt：阅读的八个源文件指纹，用于固定本稿核对对象。只读记录，不代表本文对源文件作过任何修改。

严格结论为 0.050156130219 < e_* < 0.050156130220。证明与所有解析误差界见上级 PROOF_EXT_L1L3L4.md §3.2；有限数组核查不证明 EXT L2。源材料的历史待证状态没有由脚本自动改写。
