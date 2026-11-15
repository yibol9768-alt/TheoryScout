# EXP2：状态标量桥接

根目录 `BRIDGE_V2.md` 给出模型/理论边界，`PREREG_EXP2.md` 给出所有冻结规则。
`model.py` 只实现本实验的高斯任务、残差生成、薄SVD、解析二阶矩和在线训练；
`run.py` 核验冻结，先校准锁定模型再打开留出；`analyze.py` 只应用冻结规则、审计和制图。
不 import EXP1 或其他项目。

从本题根目录复现。输出目录必须不存在；scope名称必须同步修改。

```bash
sha256sum -c PREREG_EXP2.sha256
sha256sum -c experiments/exp2/FROZEN.sha256
systemd-run --user --scope --unit=theoryscout-exp2-smoke -p MemoryMax=2G -p MemorySwapMax=0 timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp2/run.py --smoke --out experiments/exp2/smoke --scope theoryscout-exp2-smoke.scope
systemd-run --user --scope --unit=theoryscout-exp2-formal -p MemoryMax=2G -p MemorySwapMax=0 timeout 3300s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp2/run.py --out experiments/exp2/formal --scope theoryscout-exp2-formal.scope
systemd-run --user --scope --unit=theoryscout-exp2-analysis -p MemoryMax=2G -p MemorySwapMax=0 timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/liuyibo/venvs/theory/bin/python -I experiments/exp2/analyze.py
```

实际日志位于 `smoke.log`、`formal.log`、`analysis.log`；每阶段manifest含UTC时间、解释器、线程、RSS和哈希。
`scope_properties.txt` 为运行中的实际systemd内存属性。`calibration.json` 与 `locked_model.json` 在留出数据生成之前落盘。
`holdout.json` 含384单元的全/前半/后半指标及正对照，`.npz`保存16块均值、行协方差、谱响应、真实残差，六个代表单元保存完整协方差。
`training_calibration.json` 保存全部40候选与锁定η；`training.json` 保存96训练及所有记录时刻，不筛种子。
`analysis.json`、`TABLES.md`、PNG/PDF均从冻结指标直接汇总。问题文件未送外部证明系统。

`FROZEN.sha256` 覆盖预注册、桥接、四份引理和三个运行/分析源文件，冻结后不改。
最终 `ARTIFACTS.sha256` 覆盖本轮文档、日志和所有正式/冒烟数据（不含可再生字体缓存）；重跑应新建目录并保留原记录。
