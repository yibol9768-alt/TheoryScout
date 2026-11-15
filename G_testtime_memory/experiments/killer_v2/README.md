# 第二版杀手检验产物

主模型和证明目标见 ../../FORMULATION_V2.md，冻结协议见 ../../PREREG_KILLER_V2.md，最终判断见 ../../RESULTS_KILLER_V2.md。第一版 experiments/killer/ 的文件不覆盖、不用于本轮验证。

| 文件 | 含义 |
|---|---|
| design.py | 固定网格、全新种子、有界/原始谱、反馈ODE和无拟合召回闭合 |
| common.py | 冻结校验、cgroup/线程检查、时间与资源元数据 |
| prepare.py | 在新随机数据前计算全部确定性预测；96阶主求积、64阶数值审计 |
| run.py | d=32工程冒烟；精确反向系数、全部候选读出、种子级正式数据 |
| analyze.py | 全网格完整性校验、配对统计、P1–P6逐条判定 |
| launch.sh | 指定Python和systemd资源限制的统一入口 |
| predictions.json | 1728个预测方法格，含年龄矩、召回和无反馈诊断 |
| raw.jsonl | 32256条种子/方法/值维度记录，全部候选排名与年龄召回 |
| moments.jsonl | 8064条不依赖值维度的精确信号、能量及离对角系数诊断 |
| method_summary.csv | 验证召回均值、种子SE、预测和绝对误差；含所有对照 |
| cell_summary.csv | β_pred、配对优势、regret、近最优倍数和P1单格判定 |
| moment_summary.csv | 校准与验证的四年龄桶m、Q均值/SE、相对误差及无反馈Q诊断 |
| summary.json | 冻结门槛的机器可读总判定、分层结果和最坏格 |
| describe.py / describe.sh / descriptive.json | 冻结分析后仅摘录已有CSV供报告引用；无新数据、拟合或门槛变更，明确为事后描述 |
| *_meta.json / *.log | 各阶段完成状态、资源、时间、进度与原始错误信息 |
| ARTIFACTS.sha256 | 完成后输出文件的校验清单，不替代数据前的预注册冻结 |

在题目根目录执行（已有数据时程序拒绝覆盖；阅读结果无需重跑）：

```bash
sha256sum -c PREREG_KILLER_V2.sha256
bash experiments/killer_v2/launch.sh prepare
bash experiments/killer_v2/launch.sh smoke
bash experiments/killer_v2/launch.sh formal
bash experiments/killer_v2/launch.sh analyze
```

所有数值入口均为 CPU、单进程、线程环境设1，且必须在 MemoryMax=2G、MemorySwapMax=0 的用户 systemd scope 内；使用 /home/liuyibo/venvs/theory/bin/python。没有 LLM API、外部项目 import、训练模型或子代理。Python系统包和系统资源接口属于执行环境，研究数据与代码均在本题目录。

矩记录的唯一键是(split,family,d,load,alpha,seed,method)，读出记录再加readout。readout间共享值矩阵列，不是独立重复。校准种子3001–3008只作诊断，没有拟合动作；β_pred只用数据前预测，验证种子4001–4016仅用于审计。原始强幂律和p=d对照不替换有界谱、三种τ的主域。
