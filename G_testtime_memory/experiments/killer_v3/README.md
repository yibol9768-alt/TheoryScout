# V3 复核入口

决策、边界及阈值见根目录的 NOVELTY_V3.md、FORMULATION_V3.md、PREREG_KILLER_V3.md 和 RESULTS_KILLER_V3.md。核心代码不导入其他版本/项目；只使用本目录模块及 Python/NumPy/SciPy 系统运行库。

冻结查新文件须与根目录 NOVELTY_V3_ERRATA.md 一起阅读：后者更正一篇 NLMS 论文的作者，不修改实验方案或数据前冻结文件。

执行入口（冻结后、按顺序；输出使用独占创建，禁止覆盖既有正式数据）：

```bash
cd /home/liuyibo/auto-llm-driving/TheoryScout/G_testtime_memory
sha256sum -c PREREG_KILLER_V3.sha256
bash experiments/killer_v3/launch.sh prepare
bash experiments/killer_v3/launch.sh smoke
bash experiments/killer_v3/launch.sh formal
bash experiments/killer_v3/launch.sh analyze
```

launch.sh 为每个计算阶段启动独立的 user scope，固定 MemoryMax=2G、MemorySwapMax=0、所有线程环境为1，并使用指定 theory Python。meta 记录实际线程、cgroup、CPU/墙钟、RSS和冻结清单。主网格不运行 d⁴ 方差递推。

主要文件：

- `readout_tables.npz`、`numerical_audit.json`：冻结阶数/节点的独立候选读出表及抽查。
- `forecasts.json`：与验证无关的事前查询积分，固定选择每格 β。
- `unconditional_moments.json`：有限步长精确无条件年龄矩预测。
- `raw.jsonl`：27648条种子/方法/读出记录，含真实排名摘要和条件均值闭合预测。
- `moments.jsonl`：6912条种子/方法系数矩记录，含条件均值、信号条件方差及真实残差。
- `oracle.jsonl`：预定两格的384条读出诊断。
- `oracle_U_d*_seed*.npz`：预定 oracle 两格的真实 U 与全部查询索引，三种固定方法。
- `method_summary.csv`、`cell_summary.csv`、`moment_summary.csv`、`oracle_summary.csv`、`summary.json`：种子为重复单位的汇总与完整门槛。
- `smoke_checks.json`：d=2穷举求积与提升递推的 Q 条件方差核对。
- `*_meta.json`、`*.log`：原始执行证据。`ARTIFACTS.sha256`：最终产物清单。
- `report_extract.json`、`tail_audit.json`：只读报告摘录；后者是按最坏Q格进行的事后种子级审计，全部种子保留，不生成新随机数据，不改门槛。

主预测与两种 oracle 不可互换：条件均值预测只输入可见查询；真实 ρ 的独立读出输入真实系数；真实 w 的相关积分输入完整系数。仅最后一项是精确条件读出，其成功不能证明可迁移的谱召回律。

字段说明：`rho_sd_bins` 实际保存真实ρ与条件均值闭合ρ之差的**平方均值**，不是标准差；冻结实现中的字段名欠准确，报告须按MSE解释。它只作描述，不参与任何门槛。

相对误差分母沿用预注册的预测绝对值，.03 和5%均不使用SE放宽。所有对照与失败格保留。后续只读报告脚本如有新增，须与冻结核心区分，不能重算数据、选常数或修改门槛。
