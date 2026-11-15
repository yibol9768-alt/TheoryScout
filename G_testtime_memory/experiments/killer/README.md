# T killer 实验产物

从项目根目录阅读 `../../RESULTS_KILLER.md` 与 `../../PREREG_KILLER.md`。本次完整运行已结束，原始观测不应覆盖。

| 文件 | 用途 |
|---|---|
| design.py | 冻结的网格、种子、单参数谱滤波公式 |
| prepare.py / prepare_timing.json | 冻结前纯确定性数值预测表及耗时 |
| run.py / analyze.py / launch.sh | 原始冻结实现（环境缺 threadpoolctl，首次冒烟未生成数据） |
| run_v2.py / analyze_v2.py / launch_v2.sh | 仅修复资源检查依赖，实际执行版本 |
| IMPLEMENTATION_FIX.md / .sha256 | 修复说明与数据前冻结清单 |
| smoke.log / smoke_v2.log | 首次环境错误与成功冒烟证据 |
| raw.jsonl | 4224 条种子级方法记录，含年龄、top-10%、归一化秩和键范数诊断 |
| formal.log / formal_meta.json | 完整网格进度、运行时间、RSS、资源上限、冻结清单快照 |
| calibration.json | 唯一 c=0.835 的校准目标与完整 c 网格 |
| method_summary.csv | 各方法均值、SE、预测、绝对误差及辅助指标 |
| cell_summary.csv | 预测步长、配对优势、SE、步长 regret 与边界诊断 |
| summary.json / analysis.log | 逐条门槛及整体判定 |
| audit.py / audit.json / audit.log | 事后纯数值积分精度审计；没有新随机实验 |
| analysis_meta.json / smoke_meta.json | 其余阶段资源与时间记录 |

冻结验证命令（在项目根目录）：

```bash
sha256sum -c PREREG_KILLER.sha256
sha256sum -c experiments/killer/IMPLEMENTATION_FIX.sha256
```

实际执行顺序：

```bash
bash experiments/killer/launch_v2.sh smoke
bash experiments/killer/launch_v2.sh formal
bash experiments/killer/launch_v2.sh analyze
```

脚本使用指定 Python、systemd MemoryMax=2G / MemorySwapMax=0，默认每个线程环境变量为1；拒绝覆盖已有 smoke_meta/formal_meta/raw 数据。不需再次运行以阅读本次结论。若要新实验，应在本项目目录内用独立子目录、输出位置和新预注册，保留本次文件；不能删除失败格后继续用这份预注册宣布成功。

原始 JSONL 的每行以 `(split,d,load,alpha,seed,method)` 为唯一键。同格方法配对，不把 query 数目当成统计样本量；校准/验证种子不重叠。完整网格为96个校准序列与432个验证序列，各比较8个方法。

FORMULATION 的有界谱反馈 ODE 是未来待证/待测模型，本目录本次程序没有实现或检验它。本次没有神经网络训练、LLM 调用或外部项目 import。
