# FT 审稿数值检查

全部为独立审稿新产物；原项目文件未被脚本导入或写入。解释器与系统数值库仅作为运行依赖使用。无网络、GPU 或 LLM API。审查结论见上一级 PROOF_REVIEW_FT.md。

## 运行命令

工作目录为用户指定项目根目录。每个脚本的结果文件用排他创建，复跑时须先在本目录的新子目录保存副本并调整输出位置，避免覆盖本次审稿证据。

```bash
systemd-run --user --scope --unit=proof-review-ft-checks-main \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  timeout 600 /home/liuyibo/venvs/theory/bin/python review_ft_checks/check_ft.py \
  > review_ft_checks/run_main.log 2>&1

systemd-run --user --scope --unit=proof-review-ft-boundaries \
  -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
  timeout 600 /home/liuyibo/venvs/theory/bin/python review_ft_checks/boundary_checks.py \
  > review_ft_checks/boundary_run.log 2>&1
```

两项顺序运行，数值库最多 2 线程。成功脚本内部时间合计约 14.93 秒；主脚本峰值 RSS 258.26 MiB。`scope_limits.txt` 是主脚本运行期间由 systemctl 抓取的真实资源上限。

## 产物

- `input_sha256.txt`：读取来源的十份输入版本指纹。
- `check_ft.py`、`check_results.json`、`run_main.log`：固定参数真实任务、独立训练与代理、首步积分、搜索精度和数据依赖测量边界；随机种子 84627613。
- `boundary_checks.py`、`boundary_results.json`、`boundary_run.log`：负旧均值的真实匹配反例、纯缩放匹配例子、独立标量积分和高精度搜索残差；随机种子 741523。
- `run.log`：首次尝试导入未安装的 threadpoolctl 后退出的原始记录；未运行数值实验。最终脚本使用 ctypes 查询已加载 OpenBLAS 的线程计数。
- `scope_limits.txt`：主 scope 的 MemoryMax、MemorySwapMax 实测快照。
- `input_hashcheck.log`：审查结束时的原输入指纹复核。
- `artifact_sha256.txt`：审稿报告及独立检查产物的最终指纹（不包含清单本身）。

这些检查用来尝试证伪和明确边界，不替代概率收敛证明，也不重估作者实验的确认性统计结论。
