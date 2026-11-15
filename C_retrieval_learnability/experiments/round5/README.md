# 第五轮复现入口

从本项目根目录运行。Python使用标准安装的NumPy/SciPy；训练引擎复制自round4后在round5内修改，未import其他项目，不调用LLM API。文献核查使用公开原始论文网页，与数值实验分开。

预注册：[PREREG_ROUND5.md](../../PREREG_ROUND5.md)；校验：[PREREG_ROUND5.sha256](../../PREREG_ROUND5.sha256)。正式代码和最终冒烟输出冻结于manifest.json；原始复制来源哈希也在其中。禁止覆盖已交付结果；复现时先在本项目内复制到新目录，再修改输出路径。

所有计算包装为：

```sh
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 gcc -O3 -shared -fPIC experiments/round5/bias_solver.c -o experiments/round5/bias_solver.so -lm
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round5/smoke.py
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round5/freeze.py
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round5/run.py
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round5/validate.py
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round5/first_step_audit.py
systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round5/report.py
```

运行只创建本目录输出，BLAS/OMP/MKL/NUMEXPR每工作进程1线程，两个工作进程继承同一systemd scope；scope_properties.txt记录实际内存限制。MemoryMax=4294967296字节、MemorySwapMax=0。先冒烟，后冻结代码，再正式。freeze.py禁止覆盖已有manifest；run.py禁止覆盖main输出。重新编译二进制哈希可能因工具链改变，不能修改原manifest来掩盖差异。

A主网格：n16/k3、d6/d7、seed30–34、ZL/ZM/SL/SM、train温度，共40条。Z/S=零/逐行最优偏置初始化；L/M=第三轮式/参考有效位移匹配偏置lr。B主网格：同一数据/维度/种子，b/theta × original/stop80/cosine_rate，共60条。每条100000步。gate.json由正式前冻结的assess.py生成，只有预注册门控通过才执行n20/k3、d7、seed40–42的完整A四格和B的original/stop80 × b/theta，共24条。没有k外推。

main/raw/和条件holdout/raw/保留每条初态、校准、终态逐行gamma/zeta/shift、1000步历史、首次/持续/鲁棒时间、末段符号丢失、墙钟/CPU/RSS。weights/保存终态参数及固定温度分支80000步完整参数，供stop80/original前缀精确核验。初始只匹配原始U,V和同类init的有效分数；星号与零初始化本就不同。匹配偏置lr以零偏置参考点校准，不承诺最优初态真实首步达到0.001。真实第一步完整阈值位移使用beta(1)，已包含温度变化。

C根求解器复用round4，用于n16初始化；n20初始化使用已复制的Python参考求解器，根条件同为sum(sigmoid)=k。训练中不逐步解偏置。C中另有融合Adam更新，NumPy参考实现仍在experiment.py；冒烟对照20步。初始最优b不保证符号分离，这是有限温度目标性质而非求解器失败。

所有成功以完整一次运行的全部行计，不把查询当独立重复。终态优先；持续命中只覆盖已观测检测点，未命中用101000作删失描述。几何鲁棒间隔固定1e-4。失败、截断、未触发的留出都必须明确保留。预算以execution.json为准，CPU以逐运行之和另报；不能把训练时间当完整任务时间。

implementation_log.md记录正式前性能改动；smoke_attempt1/2及JSON是未通过预算门槛的冒烟，并非失败正式运行。交付时检查delivery_manifest.json及validation.json。report.py仅汇总，不再训练。

字段注意：calibration.moves是校准函数内试步，legacy格的moves.b在随后覆盖lr之前计算。选择lr后的真实阈值位移已正确保存在first_coordinate_move_rms及first_full_threshold_move_rms。first_step_audit.json给出实际各坐标首步，重算两个真实阈值字段最大差0。不要把试步b字段当实际值。
