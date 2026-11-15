import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
p=ROOT/'PREREG_EXP1.md'
assert not p.exists()
source=(ROOT/'FORMULATION_V2.md').read_text()
original=source[source.index('所有概率以独立数据种子'):source.index('## 6. CPU')]
mom=json.loads((ROOT/'experiments/exp1/moments.json').read_text())
text='''# EXP1 冻结预注册：真实高斯联想记忆的有限步召回

冻结时间：TIME。冻结发生在本轮新数据生成、冒烟及正式运行之前。已有 FORMULATION_V2 文末和 smoke_v2_results.json 的先导结果已公开于本目录；本轮不是对此信息盲法的首次注册。使用全新种子流，不把已有先导数据合并进统计量。不调用 LLM API、不访问网络、不导入其他项目代码。除用户指定的本题 QED 输出只读例外外，读写均在本题目录。原稿与 lemmas_v2 不修改。

## 假设、原文数字与不可放宽的门槛

IID 高斯 x_i,y_i~N(0,I_d/d)，全训练候选，真实 S=(d/n)XX^T、Z=(d/sqrt(n))YX^T，重复使用训练集；零初始化、全批量、无权重衰减、float64。主假设为主定理 M 的式 (7)–(10)，代理参数 (6) 预先积分且所有负载共用。以下逐字保留 FORMULATION_V2 第 5 节（不改数字）：

ORIGINAL

## 本轮网格与种子（执行前固定）

- 冒烟：仅 (d,kappa,seed)=(32,1,0)、(64,0.5,1)，运行 B 的完整方法网格，并作独立稠密小样例秩/CDF、自比较移除、正缩放不变性、Adam 原始 W 单位递推与重标度递推的一致性检查。通过这些实现检查才能正式运行。冒烟数据不进入正式统计。
- A：d={32,64,128} × kappa={0.5,1,2} × seeds={0,1,2,3,4}，每数据 128 个无放回均匀训练查询；GD(eta=1,T=1)、一步精确 SVD polar、一步 Sign(alpha=0.6)、两步 Sign(alpha=0.6)代理和真实、Adam(T=2,16,beta=(0.9,0.999),alpha=0.2,e=0.1)代理和真实。45 个数据配置。
- B：d={128,256} × 相同三个负载 × 五种子，GD/polar/Sign1；两步 Sign alpha={0.2,0.6,1.2,1.5}代理和真实；完整 P4 网格 T={1,2,8,16}、beta={(0,0),(0.9,0.999)}、e={0.1,1}，alpha=0.2，每项代理和真实。30 个数据配置，每配置 43 个方法。为避免缺失 d=256 的 P4 比较，完整参数网格也预设在 d=256 执行。
- A/B 同 (d,n,seed) 共享数据和查询，d=128 重复记录不作为额外种子；汇总优先 B。Numpy default_rng(SeedSequence([20261106,d,n,seed,1]))，n=round(kappa*d*d)。该流不同于旧 smoke_v2.py。冒烟同流但维度配置与正式 B 不重叠，分析时按 mode 隔离。
- C：d={64,128} × lambda=n log n/d² 的六个目标 {0.20,0.28,0.34,0.40,0.46,0.60} × 五种子，二分取使 n log n≥lambda*d² 的最小整数 n；SeedSequence([20261106,d,n,seed,2])。128 查询，只测 GD、Sign1、真实 Sign2(alpha=0.6)。阈值参照 lambda_c 分别 0.5、1/pi≈0.318310、0.404876；仅记录可见趋势，不设额外有限 d top-1 通过门槛。
- P5 模型外压力测试在本轮预先不执行（优先主模型网格），明确记未检验，不能写通过。不运行 MLP、Zipf、交叉熵、raw epsilon 压力或动量 Muon。

## 预先积分与数值预测

Sign 与 GD 常数采用原式，polar 采用已有参照 8/(3pi)（非 M 的坐标定理）。Adam 固定原始学习率 alpha/sqrt(n)、epsilon=e/sqrt(n)；完整偏差校正，状态全保留。积分在 [-9,9] 上 120001 与 240001 等距点的高斯加权求和；采用更细网格 q,a,c，列出两网格最大绝对差（只为积分稳定性诊断，非数学误差证书）。此积分在生成本轮数据之前完成。所有负载直接使用同一 c 的 Phi(-c/sqrt(2kappa)) 和 Phi(c/sqrt(kappa)-Phi^-1(1-q))，禁止对数据拟合 c。

| beta1,beta2 | e | T | q_T | c_T | 两积分差 |
|---|---:|---:|---:|---:|---:|
TABLE

积分完整状态及 a_T 在 experiments/exp1/moments.json，该文件 SHA256：MOMHASH。FORMULATION_V2.md SHA256：SOURCEHASH。

## 判决实现和测量

以每数据种子为独立重复单位；每个 d、负载、方法、指标的 SE=sd(五种子指标,ddof=1)/sqrt(5)，不把查询或候选作为独立重复。逐格输出秩均值、top-{1%,10%,50%} 与理论值的差、原文通过阈值 max(0.03,3SE) 及通过/失败；一项未过即该格未通过。若同方法、负载、指标在 d=128 和256 都有同向偏差且分别大于 max(0.05,4SE)，另标强失败/审计。两者之间不擅自解释为通过。零方差种子的 SE=0。

P1 主要看 Sign1、Sign2(alpha=0.6)代理/真实的全负载全 q；正确分数均值目标 c/sqrt(kappa)、方差目标1。P2 使用同种子配对比较：在两个主维度各负载，alpha=0.6 的两步减一步平均秩应负、各 top-q 差应正，alpha=1.5 相反；报告配对 SE，原稿未规定显著性门槛，不能另加或放宽。有限样本方向未满足则如实标失败并区别于严格反例。

P3 对固定方法和负载列出 d 增大时的 d^-2||E||F²、其除以 q_T 的归一化值、128 查询估计的训练对角 MSE（原始和除以 q_T）、真实减代理秩/top-q 差。以 d=256 相比128的种子均值是否缩小作趋势判读（不声称收敛率，0=0 的首步记精确重合）；原稿没有显著性或百分比门槛，不添加。非单调情况全部报告。抽样对角平均不能冒称全训练式(17)成立。

P4 逐格使用与 P1 相同门槛，代理和真实分别报；首步与 beta 无关另作数值一致性检查。P5、C 无原文数值判决标准，标未检验/探索，不硬造通过。

保存每查询正确分数/秩、错误 CDF 网格 [-4,4] 步长0.25、每查询网格最大偏差与聚合偏差、正确得分方差与错误得分一二阶矩，以及 Sign 的式(12)精确条件均值。网格偏差是完整 sup_s 的下界，不等价于完整 Kolmogorov 距离。方差/CDF 没有额外数值门槛，必须用种子级均值和 SE 展示系统偏差并讨论，不能用秩通过掩盖。所有方法用同数据同查询，显式移除正确候选；不归一化键值、不换 fresh batch。

## 硬预算、执行顺序与省略规则

所有 Python 数值工作均由 /home/liuyibo/venvs/theory/bin/python 执行，并外包 systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0；CPU 单进程，OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1（≤2），不使用 GPU。PYTHONDONTWRITEBYTECODE=1。

正式总墙钟截止3300秒，外部 timeout 3500秒，绝不超过3600秒。A 到全程600秒停止启动新配置，B 到1800秒停止启动新配置，C 到2700秒停止启动新配置，余600秒汇总/审计。冒烟另限300秒。每次矩阵乘法前用已校准最大耗时的两倍和至少2秒作为剩余预算保留；每配置写出并刷盘。虚拟地址硬限3,500,000,000 bytes；cgroup RSS/内存硬限2GiB，无交换；RSS达到1800MiB提前写停，比原稿3000MiB更严以适配用户2GiB要求。X,Y按配置串行生成，每查询32、候选1024分块，不生成 n×n 或 d²×d² 大矩阵。

按原稿预算优先级：已先删模型外压力；若 B 预算不够，记录未启动的 Adam 配置（代码按时间跳过整数据配置，不能按结果删种子），C 只在预算余量内执行。所有缺失/部分配置记录原因，缺少五种子的格不判通过。无基于结果的提前结束或追加种子。正式只运行一次，不改冻结门槛；实现修复另记审计且不得覆盖原始输出。

## QED 背景（不替代实验证据）

预注册前只读查看用户指定 out/E_assoc_memory_overcapacity__v2_L*：L1、L2、L3 的 decomposition/STATUS.md 均为 COMPLETED、Proof completed and verified successfully；L1/L2/L3 的结构核验 PASS。L5 结构核验 PASS 但状态仍在详细核验；L4 未见分解核验 PASS，仅有 proof.md，不能冒称全五条已 PASS。状态是本次读取时的快照，不把自动证明报告当作独立人工数学认证。实验仍按原 M 检验，不因 QED 通过放宽。
'''
text=text.replace('TIME',datetime.now(timezone.utc).isoformat()).replace('ORIGINAL',original)
table='\n'.join(f"| ({r['b1']:g},{r['b2']:g}) | {r['e']:g} | {r['T']} | {r['q']:.10f} | {r['c']:.10f} | {r['quadrature_difference']:.3g} |" for r in mom)
text=text.replace('TABLE',table).replace('MOMHASH',hashlib.sha256((ROOT/'experiments/exp1/moments.json').read_bytes()).hexdigest()).replace('SOURCEHASH',hashlib.sha256((ROOT/'FORMULATION_V2.md').read_bytes()).hexdigest())
p.write_text(text)
(ROOT/'PREREG_EXP1.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  PREREG_EXP1.md\n')
print((ROOT/'PREREG_EXP1.sha256').read_text())
