# V2 实验与复核

模型、阈值、种子及确认性主臂见 ../../PREREG_KILLER_V2.md；冻结SHA-256为7ce6d0c24afe13f285f24f18f8be741b33f19fd4eb0720ce7c6465fc1b271337。A/B 独立判定，B主臂仅learned_adam_cold。

所有Python均用以下前缀，日志首行保留真实scope启动回执：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 120s /home/liuyibo/venvs/theory/bin/python experiments/killer_v2/run.py smoke
```

依次smoke、formal（timeout 2850s）、analyze.py（180s）、plot.py（90s）、audit.py（30s）。formal拒绝覆盖已有formal/；分析/绘图可对相同原始文件重建。不要重跑正式数据覆盖原证据。全部源码只import标准库、已安装numpy/matplotlib和本目录run.py，没有调用LLM API。所有文件写在本题目录，matplotlib缓存亦在本目录。

## 原始格式

- formal/model_p*_n*.npz：固定总体H、Cholesky、v、s²、特征值/特征向量；B额外保存A,C对角、θ和b。
- formal/p*_n*_s*_m*_r*.npz：全部候选。groups/kind/delta/h/warm/learned/amplitude描述每候选；A K=193，B K=386。risks为(检查点,K)，modes为(检查点,K,N)，initial_modes为(K,N)。
- diagnostics及monitor最后一轴8个量依次为：范数、幅值、总体切向梯度范数、实际信号方向投影、实际残差平方、原始径向交叉项、应用径向交叉项、旋转角。monitor_steps给实际步数，每64步及终点记录。
- probes最后一轴8个量：D、V、F、D的batch MC标准误、平均旋转角、原始径向交叉项均值、应用项均值、总体切向梯度范数。仅单遍r=64,256,1024，形状(3,K,8)。F在D≤0时以NaN存档，这表示预注册的桥接失败，不是训练数值异常。probe_mu另存(3,K,N)。
- 校准原始半径中位数、末两半窗口相对变化、各候选组名和生成成本保存在metadata.json；半径估计不被当作平衡已达到。所有正式样本和探针可由源码/种子重建，没有把校准样本计入任务学习。
- metadata.json含104正式任务、32校准任务、起止UTC、源码hash、冻结hash、环境、墙钟、峰值RSS。
- curves.csv完整八种子风险、均值/SE，probes.csv完整正δ条件量，TABLES.md全部225格，summary.json全部判定/区间/拟合。bootstrap条件于每N唯一固定投影，只描述训练种子不确定性。选中点配对区间有选择偏差。

## 实现记录

冻结后首次冒烟通过。正式前将冒烟轨迹延长至64步，使B也覆盖探针广播路径；初始方向分流明确使用规范单遍SeedSequence的init子流，让在线/回放共享初方向。再次冒烟通过后才启动正式。最初冒烟日志保留smoke_initial.log；无模型、网格、门槛变更。

源码run.py在正式启动后不再修改。固定h为u坐标长度，真实角步长arctan(h/√N)，不是数值h弧度。原始/投影AdamW使用的m,v均源于原梯度；投影只在应用更新前做。可学a使用无衰减SGD .01，与w同时按更新前参数计算。

shell工具显示的/mnt/c/Users/...路径和用户指定/home/liuyibo/...是同一目录映射。本任务未访问其他项目文件。sha256sum -c ../../PREREG_KILLER_V2.sha256应在题目根目录运行，具体为sha256sum -c PREREG_KILLER_V2.sha256；完整文件清单见ARTIFACTS.sha256。

## 解释边界

探针冻结的是整个Adam状态，每个batch仅贡献一个假想后继；所以其方差不含动量跨步累积和回放的跨epoch相关性。D≤0使派生F无定义，按冻结条款直接判桥接失败；G0核查原始状态、均值、方差等核心数值的有限性。探针F通过也不等于风险地板已被证出。

未学窗口采用预先写死的“模态风险仍≥初值50%”判据，混合了偏差与噪声，不能单独证明存在未学偏差的幂律带。每个N只有一个固定投影；8种子置信区间不反映跨投影波动。固定h对照没有δ，summary中τ为空且表中标为“—”，不能借h拟合恢复AdamW时间尺度律。

## 执行结束记录与偏离

正式104条轨迹和32次校准一次完整完成，耗时741.248秒，峰值RSS183.227 MiB；全部69480终点完整。两轮预注册hash均保持不变。

首次analysis与audit误并行约5秒，各自设置2线程，总上限可能4，未满足保守的全程总线程≤2要求。保留analysis_initial.log、audit_initial.log、summary_initial.json，不隐去偏离。随后串行重跑analysis和audit，除UTC/耗时外全部summary字段完全相同；integrity.json将full_resource_protocol_compliance明确标为false。正式训练从未并行运行其他数值任务。

绘图初版未自动扩展全为∞的回放r=256横轴；显式固定完整横轴范围后仅重画，无新训练/统计。原plot_initial.log保留。最后report_details.py为报告提取描述性统计，finalize.py检查串行结果一致、协议/源码hash和原始文件清单，保存ARTIFACTS.sha256；不把这些操作当成新增确认检验。

ARTIFACTS清单排除matplotlib缓存及finalize.log/checksums.log两个会在核验时写入的运行日志，避免自引用。其余交付源码、文档、原始数据和主要产物均进入清单。
