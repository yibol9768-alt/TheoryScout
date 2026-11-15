# Review 23 独立核验记录

最终审查文件：[PROOF_REVIEW23.md](../PROOF_REVIEW23.md)。仅该文件和本目录为本轮新增范围。没有调用 LLM API，没有导入、执行或复制 `ac2_checks/`、`ac1_checks/`、`aa1_checks/` 的程序。读取这些程序只用于代码审查。

主程序固定新种子 `2306112026`，直接生成完整 X,Y，运行真实固定参数 GD3 / Sign2 / Adam2。所有 n 个原训练查询对全部 n 个候选直接排名，不使用根式、球面代理或提前停止。每维每方法数据集数为 d=64:64，128:48，256:32，512:16；共 480 份，1,541,600 个查询。方差的独立单位为完整数据集。

## 文件入口

| 文件 | 内容 |
|---|---|
| `independent_review.py` | 独立训练、完整排名、整数原始数据重算、Haar 两列耦合、共享极值模拟、能量方向和根式检查 |
| `design.json`、`moments.json` | 固定设计及独立一维标量效率积分 |
| `raw_reaggregation.json` | 作者 720 行完整数据集记录的独立汇总核验 |
| `fresh_datasets.jsonl`、`fresh_results.json` | 本轮 480 份新数据的整数成功数、召回和按数据集统计 |
| `haar_pairs.json` | 48,000 对真实正交列的 L¹ 耦合及联合矩形检查 |
| `joint_extremes.json` | 7200 对共享真实候选的两查询极值检查；有限维模拟不代替证明 |
| `transpose.json` | 独立完整数组的 Au / Aᵀu 方向及矩阵束同谱核验 |
| `edge_checks.py`、`edge_results.json` | Sign2 零端点的独立确定反例、leading 负载不足以定位窗口的演示、低秩反面机制 |
| `root_and_rates_corrected.json` | 最终采用的根式分支检查及保守速率诊断 |
| `final_checks.py`、`final_checks.json` | 全部新数据最终核验、72 个异常预算的独立复算、新旧均值和方差比较、末次输入完整性检查 |
| `inputs_before_sha256.json`、`integrity.json` | 630 个输入/被审代码/paper 文件的前后哈希，变化 `[]` |
| `resources.json` | 主程序资源实测：墙钟 309.29 秒、CPU 307.05 秒、峰值 RSS 365808 KiB、Threads=1、memory.max=3221225472、memory.swap.max=0 |
| `checks.log`、`edge_checks.log`、`final_checks_rerun.log` | 对应运行日志 |

`edge_results.json` 运行时主模拟尚未全部完成，故其中 `fresh_rows_so_far=446` 是当时快照；全部 480 行的最终核验以 `final_checks.json` 为准。

## 独立检查自身的修正记录

首次根式分区积分在“二次多项式没有实根、整条实线只有一个区间”时选取了无穷测试点。这使人工 H<0 例子的概率被误判为 1；正确结果为 0。最后汇总的显式一致性断言检出了差异，初次 `final_checks.log` 保留失败记录。随后将整个实线的测试点改为 0，生成 `root_and_rates_corrected.json`；原 `root_and_rates.json` 保留为被取代的初跑结果。其他根式例子及速率诊断不变。该错误没有进入完整数据的直接排名，也不影响被审程序异常分支返回 [0,1] 的合法性。

主模拟完成后，为独立排名函数增加了遇零端点即明确失败的断言。所有 480 行均非全召回，证明本次没有零端点；新断言不改变任何已采样结果。零端点正确平局规则另由 `edge_checks.py` 检查。没有重跑或合并同种子数据充当额外样本。

`report_before_format.md` 是本轮新建报告在最终排版及汇总填入前的工作快照，不是最终审查结论。

## 运行与复跑

主命令：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python -B -u review23_checks/independent_review.py
```

`edge_checks.py` 和 `final_checks.py` 也使用完全相同的 scope、解释器及线程环境运行。边界小检查与主模拟曾短暂并行，各自 1 线程，总线程不超过 2；末次汇总串行运行。

计算结果使用独占新建。本目录已有结果时不要直接复跑覆盖；应复制程序到本目录的新子目录，并相应明确项目根定位和新输出路径。最终汇总脚本还负责更新本轮新建的审稿报告，单独复跑需移除其报告排版步骤或使用新的报告目标。

数值均为 float64 交叉检查，不是向外舍入证书。理论 PASS 依据为最终审查文件的独立推导；随机检查主要用于搜索反例、尺度/转置错误和分组错误。
