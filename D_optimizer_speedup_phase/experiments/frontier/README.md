# V3 前沿复现

模型、种子与不可放宽的门槛见根目录 `PREREG_FRONTIER.md`；结论见 `RESULTS_FRONTIER.md`。`run.py`、`analyze.py`和预注册在首次冒烟前冻结；`audit.py`为冻结后静态审计，不增样本或改变判定。首次运行的输出目录拒绝覆盖；复现时应在本题目录内建立独立副本，不能覆盖本轮正式结果。

从本题根目录执行，解释器仅使用指定venv。所有缓存和临时文件留在本目录：

```bash
mkdir -p experiments/frontier/tmp experiments/frontier/mplconfig
sha256sum PREREG_FRONTIER.md > PREREG_FRONTIER.sha256
sha256sum PREREG_FRONTIER.md experiments/frontier/run.py experiments/frontier/analyze.py > experiments/frontier/FROZEN.sha256
systemd-run --user --scope --unit=theory-d-frontier-smoke -p MemoryMax=2G -p MemorySwapMax=0 timeout 120s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR="$PWD/experiments/frontier/tmp" FRONTIER_SCOPE=theory-d-frontier-smoke.scope /home/liuyibo/venvs/theory/bin/python -I -B experiments/frontier/run.py smoke --max-seconds 90 > experiments/frontier/smoke.log 2>&1
systemd-run --user --scope --unit=theory-d-frontier-smoke-audit -p MemoryMax=2G -p MemorySwapMax=0 timeout 60s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 FRONTIER_SCOPE=theory-d-frontier-smoke-audit.scope /home/liuyibo/venvs/theory/bin/python -I -B experiments/frontier/analyze.py smoke > experiments/frontier/smoke_analysis.log 2>&1
# Only after smoke and its audit succeed:
systemd-run --user --scope --unit=theory-d-frontier-formal -p MemoryMax=2G -p MemorySwapMax=0 timeout 3000s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR="$PWD/experiments/frontier/tmp" FRONTIER_SCOPE=theory-d-frontier-formal.scope /home/liuyibo/venvs/theory/bin/python -I -B experiments/frontier/run.py formal --max-seconds 2700 > experiments/frontier/formal.log 2>&1
systemd-run --user --scope --unit=theory-d-frontier-formal-audit -p MemoryMax=2G -p MemorySwapMax=0 timeout 240s env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR="$PWD/experiments/frontier/tmp" MPLCONFIGDIR="$PWD/experiments/frontier/mplconfig" FRONTIER_SCOPE=theory-d-frontier-formal-audit.scope bash -c '/home/liuyibo/venvs/theory/bin/python -I -B experiments/frontier/analyze.py formal && /home/liuyibo/venvs/theory/bin/python -I -B experiments/frontier/audit.py' > experiments/frontier/formal_analysis.log 2>&1
```

每个run/analysis阶段均检查冻结清单、查询实际scope属性、核对线程变量。正式保存228检查点、每检查点96个误差状态；同一训练的两个时间点相关。网格选择使用前48重复，评估使用后48重复。正式累计训练成本包括每个算法、每个重复、batch干预，不将搜索免费计入主路径。完整输入不保存，固定种子可重建。

资源说明：正式642.23秒、完整训练峰值RSS105.56MiB。所有数值和后置静态审计累计649.86秒，四段外层timeout总上限3420秒。分析脚本的RSS在绘图前采样；绘图受同一2GiB硬限制，但该RSS字段不是完整进程峰值，详见结果文档。`-I -B`隔离用户Python路径并禁止字节码写入，代码不import本地其他项目模块、不发网络请求、不使用GPU或LLM API。

`papers/`仅存公开论文HTML和解析文本（15个版本，13个不同arXiv编号）。初始v1/v2同时保留以审计查新版本更新。下载完整并不表示每个证明已逐行核实；实际比较范围与OpenReview访问失败见 `NOVELTY_V3.md`。本轮没有论文自动推理API调用。`static_audit.json`记录论文哈希、结构检查和TeX引擎缺失；新引理没有被当作自动证明PASS。

冻结核查：`sha256sum -c experiments/frontier/FROZEN.sha256`。最终全工件核查：`sha256sum -c experiments/frontier/ARTIFACTS.sha256`。后者也覆盖本轮理论文稿与四份新引理，不覆盖旧实验或其他项目。
