# Y1 核验

`Y1_verify.py` 只核验 `Y1_D3_PROOFS.md` 中的有限代数：块迹的一阶系数、两个 FT 能量/错误方差多项式、漂移和扩散、阈值根交叉矩、正态积分及 GD 化简、精确特殊候选及半径的二阶矩。没有导入已有项目源码，没有数值模拟或调用 LLM。

执行命令（在项目根目录）：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 BLIS_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python -B y1_checks/Y1_verify.py
```

输出以独占创建模式写入 `Y1_results.json` 和 `Y1_input_hashes.json`；已有输出时拒绝覆盖。如需重跑，应在 `y1_checks/` 内另存脚本并使用新的输出文件名。资源上限由脚本读取 cgroup 后断言核对，并记录实际进程线程数。输入散列在核验前后核对，不变只是本轮只读工作的核查，不是第三方时间认证。

这些检查不能代替概率余项证明。随机分母、指示函数、顺序统计量和坏事件的证明见主报告引理 1–11；四步全训练 MSR 与同数据损失匹配根未在本轮作完整认证，见主报告 §10。

本轮结果：15 项符号残差全部严格为零；七个主要输入的前后 SHA256 一致。实际计算约 0.62 秒、1 线程，cgroup `memory.max=3221225472`、`memory.swap.max=0`、`memory.peak=47460352` 字节；进程 RSS 高水位为 355096 KiB，两种峰值计量口径不同，均低于 3 GiB。完整数据在 `Y1_results.json`。

第一次执行在入口路径断言处退出，尚未开始代数计算或创建输出：工作目录是用户指定路径的符号链接。把断言两边统一为解析后的同一项目路径后，以同一资源限制执行成功。

`Y1_format_math.py` 只修复本轮新建报告的行内公式格式，转换清单在 `Y1_math_format.json`；其运行也使用相同的 systemd 内存限制和指定 Python，未访问或修改原项目文件。该一次性格式脚本不属于数学验证。

`Y1_final_audit.py` 在相同的 systemd 限制下只读检查最终报告的 107 组显示公式和行内公式定界符，并再次核对上述七个原始输入的散列；结果全部通过，记录在 `Y1_final_audit.json`。这是格式及只读核查，不是新增数学实验。
