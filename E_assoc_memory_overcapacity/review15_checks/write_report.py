"""Render the Chinese audit from recorded, independently generated results."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from independent_checks import resources

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
assert Path.cwd()==ROOT
out=ROOT/'PROOF_REVIEW15.md'
assert not out.exists()
resources()
a=json.loads((HERE/'main.json').read_text())
b=json.loads((HERE/'analysis.json').read_text())
c=json.loads((HERE/'raw_datasets.json').read_text())
lines=[]
def put(s=''):
    lines.append(s)

put(r'''# 独立对抗审查 15：Z1 的矩展开证明

**总判定：需修正（假设表述，非系数或余项的实质反例）。** 在固定
$\kappa>0,r\in(-1,1),\alpha>0,e>0,q>0$、固定总体归一化和均匀训练查询的范围内，
逐引理检查未发现能推翻 $m_k,w_k$ 或期望余项 $O(d^{-3/2})$ 的缺口。
引理 3.4 应显式写“中心高斯、$A\sim N(0,I_s),B\sim N(0,I_t)$”；若只要求单位协方差，
该引理为假。应用中的块确实中心化，补明后主证明闭合。若作者原意中“标准化”已经包含零均值，
这就是消除歧义的修订，主定理的数学判定为 PASS。

本审查针对 `Z1_G1_PROOF.md` 和 `z1_checks/`，以
`paper/appendix/M_nonlinear_finite_d.tex` 的标签 `thm:nl-moments` 定位目标；
当前论文把它标为 derivation 的 Proposition，并仍明确写着余项只有提纲。
**独立证明文件的成立不等于论文已经接入完整证明，更不等于后续召回修正已经得到证明。**
未执行或导入被审脚本，未调用 LLM API；以下代码均为本轮新写。

## 1. 找到的修订点与排除的错误

1. **需修正：引理 3.4 的零均值条件应明确。** 令 $A=1+g,B=1+h$，
   $(g,h)$ 为标准中心二元高斯且相关系数 $\rho=1/2$，
   $F(x)=G(x)=x^2-2$。这两个函数中心化、偶、属于 $L^2$，块协方差都是 1。
   但 $\operatorname{Cov}(F(A),G(B))=4\rho+2\rho^2=2.5$，
   而所写右侧为 $\rho^2\sqrt{6}\sqrt{6}=1.5$。
   因此“单位协方差”本身不够。改写为中心标准高斯后 Hermite 论证成立；
   §6 所用的 $A_b^{-1/2}(z_b,t_b)$ 已经零均值，故没有把这个反例传入主定理。
2. **需修正（独立引理的维度措辞）：** 引理 3.1 写 $S\succ0$ 几乎处处时应注明
   $n\ge d$ 或“充分大 $d$”。例如 $\kappa=1/4,d=2,n=1$ 时 $S$ 必奇异。
   定理是 $d\to\infty$ 且 $n\asymp d^2$，最终总有 $n\ge d$，所以不影响渐近结论。
3. **需修正（证据范围）：** 原 `z1_checks` 的 `conditional_contraction` 是把回归向量有限和
   改写成迹的代数检查，未直接对随机标签积分；其条件方差检查断言的是普适方差分解不等式，
   没有据此检验维数一致的 $C/d$。这些结果可作查错诊断，不能单独认证对应概率引理。
   原 `Z1_RESULTS.md` 已披露大部分限制，因此这不是虚报定理证明。
4. **PASS：未检出常见的实质错误。** 没有将 $O_P$ 偷换为期望余项；没有把值散布当作与 $U$
   独立；没有对 Sign 使用经典导数/Poincaré；没有忽略非对角求和的维数；
   没有仅凭正确评分均值有界就删除正确候选。

## 2. 逐引理判定

行号针对本轮散列固定的 Z1 文件。

| 对象 | 判定 | 核查理由及适用条件 |
|---|---|---|
| §1 定义及定理（17–75） | PASS | $V$ 为未中心化的训练错误候选均方；$q$ 为固定总体量。与论文 75–87 行对象一致。没有随机归一化或召回结论。 |
| 2.1（83–108） | PASS | 网界实际为 $2\exp(Cd-cn\min(t^2,t))$；$n\asymp d^2$ 后给固定阈值坏事件指数小和所需全部固定阶矩。Wishart 三个矩正确。 |
| 2.2（110–148） | PASS | 坐标高斯矩、Minkowski、Hölder 足够；$U,C_{kk},S$ 不需独立。对坏事件使用原始量，避免了 $\beta$ 的逆方差尾部。粗界 $\|D\|_p=O(d)$ 只用于指数小坏事件，合法。 |
| 3.1（152–185） | PASS（渐近范围） | $J=rS-S^2$、$K=S^3-2rS^2+S$、$b=3-2r$ 及回归向量顺序均正确。正定性句须补 $n\ge d$。 |
| 3.2（187–205） | PASS | 固定物理 $e\sigma$，对密度求导；紧正方差区间内的多项式乘高斯包络可积。Sign 的不光滑性不妨碍任意阶方差微分。 |
| 3.3（207–240） | PASS | 奇性去掉所有偶阶；首阶系数为 $\sqrt v\beta(v)$。Parseval/Cauchy–Schwarz 给 $|\rho|^3$，包括 $\rho=\pm1$；不要求逐点收敛或函数导数。 |
| 3.4（242–262） | 需修正（明确零均值） | 在中心标准高斯条件下，$m$ 阶 chaos 的跨块算子范数至多 $\|R\|^m$，偶函数从二阶开始，证明有效；若只按单位协方差理解则有 §1 反例。 |
| 4.1（266–311） | PASS | $E_{bb}^2$ 的平均期望是 $2/n=O(d^{-2})$，不能替换成谱界。$\Delta_b^2$ 的平均为 $O(d^{-2})$；余下 $\varepsilon^3$ 为 $O(d^{-3/2})$。对所用响应，缩小 $c_0$ 即处于同一光滑邻域。 |
| 4.2（313–344） | PASS | 查询平均 $n^{-1}\sum_i x_ix_i^T=S/d$，无需声称单个查询与 $U$ 独立。$p_2=-r_{k0}$ 和 $m_k$ 正确；$n=\kappa d^2+O(1)$ 的变动只到 $O(d^{-2})$。 |
| 5.1（348–366） | PASS | 高斯回归给 $JB+BJ$，没有错误交换一般矩阵；$M_U$ 只依赖 $S$，查询平均合法。 |
| 5.2（368–408） | PASS | 完整维数求和后的 Hermite 高阶贡献至多 $C\varepsilon^4$，换响应和换 $K$ 分别至多 $C\varepsilon^3$；足以取绝对期望。 |
| 5.3（410–454） | PASS | 交叉多项式 $rS^2-S^3$ 的二阶系数为 $r-3$；非线性非对角项为 $2\sigma^2\beta^2$；对角 $Q$ 项只保留 $b\dot Q$。合并正好是 $\ell$。 |
| 6.1（458–504） | PASS（采用中心版 3.4） | 局部 $2\times2$ 块行列式 $1-r^2>0$；白化逆范数在好事件有界。平方是整个二维块的偶函数。全部跨块协方差平方和 $O(d)$，不是 $O(d^2)$。 |
| 6.2（506–532） | PASS | 正确结论是好事件上 $C(d^{-1}+\varepsilon^2)$，再积分得到 $C/d$；没有把后一式误写成每个矩阵都成立的界。坏事件原始二阶矩有多项式谱界。 |
| 6.3（534–548） | PASS | 给定 $X$，$\Gamma$ 仍为 $\chi_n^2/n$，条件均值 1、方差 $2/n$。条件协方差 Cauchy–Schwarz 后积分即 $n^{-1/2}d^{-1/2}$。不需要 $\Gamma,R$ 独立。 |
| 7.1（552–579） | PASS | 独立性在输出行之间；同一行的 $y$ 和 $u$ 保留相关。两行回归向量相乘后，$d/n$ 与 $\sum_jx_jx_j^T=(n/d)S$ 给精确 $1/n$。迹顺序正确。 |
| 7.2（581–630） | PASS | 跨行计数 $d(d-1)$ 正确；主项为 $a_k^2(d-1)/(nq)=\mu_k^2/d+O(d^{-2})$。坏事件非对角**条件期望**非负且被总条件能量支配；未把原始非对角和误认为逐样本非负。 |
| 8.1（637–689） | PASS | 共同右旋样本列保持 $S,Z,H,U$ 不变，对 Sign/Adam 无需特征旋转等变性。非对称 $T$ 的球面矩包含 $\|T\|_F^2+\operatorname{tr}T^2$；两条迹缩放正确，给 $ED^2=O(1)$。 |
| 8.2（691–709） | PASS | 对均匀查询先作样本平均才有精确能量恒等式；删除的 $ED^2/(n-1)$ 为 $O(d^{-2})$。固定查询在全期望下由交换性成立。 |
| §9 Sign 特例（720–745） | PASS | $\dot\beta=-\beta/(2\sigma^2),\dot Q=0$ 给 $m_0,m_1$ 正确；$r=-1/2$ 的旧任务系数消失。没有声称候选召回展开由此推出。 |

## 3. 三个重点环节的独立复核

### Sign 的 Hermite 可积余项

对中心标准高斯，条件期望算子是 $L^2$ 收缩；可先取有限 Hermite 和，再以 $L^2$
收敛传到乘积期望。$\operatorname{sign}$ 有界且为奇函数，所以 Parseval 可直接使用。
非光滑点 $0$ 是高斯零测集，证明既不求点态 Hermite 和，也不对 Sign 求导。
写 $E=S-I$、$\varepsilon=\|E\|_{op}$，关键求和独立复算为

\[
\frac1d\sum_{b\ne c}|E_{bc}||R_{bc}|
\le\frac C d\|E_{off}\|_F\|K_{off}\|_F\max_{b\ne c}|K_{bc}|^2
\le C\varepsilon^4.
\]

两次 Frobenius 因子各带 $\sqrt d$，恰被 $1/d$ 抵消；取期望为 $O(d^{-2})$。
响应替换及 $K_{off}\mapsto2\sigma^2E_{off}$ 的贡献是 $O(E\varepsilon^3)=O(d^{-3/2})$。
因此这里没有遗漏 Sign 导数的分布项，也没有未控制的维数倍数。

### 训练候选额外项与同一行相关

对 $a\ne c$，逐坐标独立积分给

\[
\frac dn\sum_j E[y_{kaj}u_a\mid X]^T S E[y_{kcj}u_c\mid X]
=\frac1n\operatorname{tr}(SC_kSC_k^T).
\]

该迹等于 $\|S^{1/2}C_kS^{1/2}\|_F^2\ge0$，但原始随机跨行项可以为负；
Z1 使用的是前者。$C_k$ 一般不对称，证明只作迹循环，没有依赖错误的交换性。
对角行相关只需
$|E[(\Gamma-1)R\mid X]|\le\sqrt{(2/n)\operatorname{Var}(R\mid X)}$，
积分条件方差界已经足够。

为独立数值积分，本轮另用高斯密度恒等式。令 $g_k\in\mathbb R^d$ 是标签在键行空间
的标准高斯投影，剩余 $n-d$ 个坐标独立。若 $F_k(t)=E[R]$ 表示把 $g_k$ 的协方差
改成 $tI$ 后的能量，则

\[
E[\Gamma R]=E[R]+\frac1nE[(\|g_k\|^2-d)R]
=F_k(1)+\frac2n F_k'(1).
\]

这给同一行修正的独立计算方式；后面再用原始标签 Monte Carlo 检查该条件积分。

### 删除正确候选

共同样本列旋转把评分对角元化为 $v^TTv$。非对称球面公式中的 $\operatorname{tr}T^2$
不能遗漏，也不能直接当作非负数；Z1 用 $|\operatorname{tr}T^2|\le\|T\|_F^2$ 是正确的。
$E(\operatorname{tr}Z_k^TU)^2=O(d^4)$ 与 $n\asymp d^2$ 恰好使此部分有界。
这避免了从粗界 $\|D\|_p=O(d)$ 得出不足以删除的二阶矩估计。

## 4. 余项阶与参数依赖

| 误差来源 | 取绝对期望后的包络 | $n=\kappa d^2$ 时 |
|---|---|---|
| 对角 Taylor 的平方项 | $C(n^{-1}+E\varepsilon^4)$ | $O(d^{-2})$ |
| 谱三阶、响应/协方差替换 | $CE\varepsilon^3$ | $O(d^{-3/2})$ |
| 纯 Hermite 三阶以上 | $CE\varepsilon^4$ | $O(d^{-2})$ |
| 同一行值范数相关 | $C n^{-1/2}d^{-1/2}$ | $O(d^{-3/2})$ |
| 跨行训练候选近似 | $C(d/n)E\varepsilon+C/n$ | $O(d^{-3/2})$ |
| 删除正确候选、改分母 | $C/n$ | $O(d^{-2})$ |
| 坏事件 | 多项式矩乘 $P(\mathcal G^c)^{1/2}$ | 指数小 |

好事件阈值可以依赖 $r$（以及响应局部方差区间）；块逆范数至少涉及 $(1-r^2)^{-1}$，
Sign 响应及其导数涉及 $\sigma$ 的负幂，归一化涉及 $q^{-1/2},q^{-1}$，
Wishart 缩放涉及 $\kappa$ 的负幂。Z1 明确允许固定参数依赖，因此这些不构成余项错误。
不应将结果扩展为随 $d$ 变化的 $r,e,\alpha,q,\kappa$，也不能自动覆盖 $r\to\pm1$、$q\to0$。
Adam 的积分和微分始终固定物理 $e\sigma$，未把它随局部方差 $v$ 改为 $e\sqrt v$。

## 5. 独立数值设计与交叉检查

`independent_checks.py` 未导入、执行或复制 `z1_checks/Z1_check.py`。
预测系数只在 `predicted()` 中作为比较目标计算；采样和条件积分不使用 $m_k,w_k$。
主实验取 $\kappa=1/4,\alpha=.6$；$r=0$ 检查 GD、Sign、Adam(.1)、Adam(1)、Adam(2)，
另取 $r=-.5$ 检查 Sign 的零系数边界。维度为
$16,24,32,48,64,96,128,192$，每维 4000 份独立 Wishart 键样本；各方法共享同一批键。

键由独立 Bartlett 三角阵产生，避免巨大 $d\times n$ 数组。
给定键，对输出行积分：Sign 使用 arcsin 精确协方差；Adam 在正半轴以 100 点
Gauss–Legendre 求积计算响应 Hermite 系数，自适应保留奇数阶至最多 63 阶。
用 Parseval 尾界控制未保留的协方差，并另做 180 点求积、方差方向有限差分和二元密度积分交叉检查。
删除项通过独立输出高斯行与球面四阶矩计算，只在该 $O(1/n)$ 项保留 Monte Carlo。
方差缩减只减去已知均值为零的 $\operatorname{tr}S/d-1$，系数由有限维条件积分差分得到，
没有强行把被检系数写入样本均值。

另有两条交叉路径：固定一份原始 $X$ 后，每方法产生 300000 对完整 $n$ 列标签行，
直接检验跨行收缩和全部候选能量；`raw_dataset_crosscheck.py` 在 $d=16,32,64,96$
各生成 1500 份原始 $X,Y_0,Y_E$，对正确评分逐列计算、对错误候选均方作有限维矩阵计算。
小维度还真正构造全部候选评分矩阵核对低秩计算。主采样种子 1511062026，原始数组路径种子 1511062027。
两条路径的代码仅共享本轮的标量响应/预测参数及条件积分器；原始统计量的产生不调用 Hermite 积分。
''')
put(f"主实验输入前后散列一致：**{a['inputs_unchanged']}**；资源记录为 "
    f"{a['resources']['threads']} 个线程，memory.max={a['resources']['memory_max']}，swap.max={a['resources']['swap_max']}。\n")
put('| 积分交叉检查 | 最大绝对差 |')
put('|---|---:|')
put(f"| 100 与 180 点的行能量 | {b['max_quadrature_difference']:.3g} |")
put(f"| 100 与 180 点的能量方向导数 | {b['max_derivative_quadrature_difference']:.3g} |")
put(f"| 解析方向导数与中心有限差分 | {b['max_finite_difference_error']:.3g} |")
put(f"| 正象限二元密度积分与 Hermite（最高相关 .85） | {max(x['difference'] for x in b['bivariate_checks']):.3g} |")
put(f"| 原始小维度全评分矩阵与低秩计算 | {max(x['direct_score_identity_max_error'] for x in c['rows']):.3g} |\n")
put('固定键的原始标签检查（z 为 Monte Carlo 差值除以其标准误；未作多重比较认证）：\n')
put('| 方法 | 任务 | 跨行收缩 z | 全候选能量 z |')
put('|---|---|---:|---:|')
for x in b['raw_checks']:
    label=x['method'] if x['method']!='Adam' else f"Adam({x['e']:g})"
    put(f"| {label} | {x['task']} | {x['contraction_z']:+.2f} | {x['all_energy_z']:+.2f} |")
rawz=[]
for x in c['rows']:
    for k in ('conditional_mean','conditional_all_energy'):
        rawz.append(abs(x[k+'_difference']/x[k+'_difference_se']))
put(f"\n原始全数据路径共 {len(c['rows'])*2} 个条件均值/能量差诊断，最大 $|z|={max(rawz):.2f}$。")
secs=[x['second_moment'] for x in c['rows']]
put(f"其正确评分二阶矩估计范围为 {min(secs):.3f}–{max(secs):.3f}，未见随维度发散；这仅是对解析界的诊断。\n")
tail_low=max(x['max_covariance_tail_bound'] for x in a['monte_carlo'] if x['d']==16)
tail_fit=max(x['max_covariance_tail_bound'] for x in a['monte_carlo'] if x['d']>=48)
put(f"Adam 协方差迹的最坏单样本截断包络在 $d=16$ 为 {tail_low:.3g}，在拟合用 $d\\ge48$ 为 {tail_fit:.3g}。")
put('低维度不用于渐近斜率拟合；方向导数的截断误差另外通过有限差分/求积一致性诊断，没有将上述协方差尾界冒称为导数的严格尾界。\n')
put(r'''## 6. $1/d$ 斜率与 $d^{-3/2}$ 残差

主拟合固定已知总体极限，对 $d\ge48$ 的五个维度作按 Monte Carlo 方差加权的
$c/d+b/d^2+c_3/d^3$ 拟合。表中“拟合”是 $1/d$ 系数及边际 95% 抽样区间，
不包括有限维拟合模型偏差；另一种模型允许 $d^{-3/2}$ 项，改拟合
$c/d+a/d^{3/2}+b/d^2$。二者都没有把理论 $1/d$ 系数固定。
有限维数据不能认证无限维大 O；若拟合区间与理论出现细小差别，应结合不同余项模型，
不能只按 z 分数把高阶项误判成系数错误。

| $r$ | 方法 | 量/任务 | 理论 $1/d$ 系数 | 拟合（95% 抽样半宽） | 允许 $d^{-3/2}$ 项的拟合（半宽） | 另一模型 $\chi^2$（2 自由度） |
|---:|---|---|---:|---:|---:|---:|''')
model_diagnostics=[]
for f in b['fits']:
    label=f['method'] if f['method']!='Adam' else f"Adam({f['e']:g})"
    kind='m' if f['kind']=='mean' else 'w'
    if f.get('exact_zero'):
        vals='数值上恒为 0 | 数值上恒为 0'
        chi='—'
    else:
        vals=f"{f['fitted_1d']:+.5f} ± {1.96*f['fitted_1d_se']:.5f} | {f['fitted_1d_allow_d15']:+.5f} ± {1.96*f['fitted_1d_allow_d15_se']:.5f}"
        rs=[x for x in a['monte_carlo'] if x['d']>=48 and all(x[k]==f[k] for k in ('r','method','e','task','kind'))]
        ds=np.array([x['d'] for x in rs],float)
        ys=np.array([x['d']*(x['mc']-x['limit']) for x in rs])
        se=np.array([x['d']*x['se'] for x in rs])
        design=np.array([np.ones_like(ds),np.sqrt(48/ds),48/ds]).T
        coeff=np.linalg.lstsq(design/se[:,None],ys/se,rcond=None)[0]
        chi2=float(np.sum(((ys-design@coeff)/se)**2))
        chi=f'{chi2:.1f}'
        model_diagnostics.append(dict(r=f['r'],method=f['method'],e=f['e'],task=f['task'],kind=f['kind'],
                                      alt_chi2=chi2,alt_coefficients=coeff.tolist()))
    put(f"| {f['r']:g} | {label} | {kind}{f['task']} | {f['target']:+.5f} | {vals} | {chi} |")
put(r'''
**数值判读：** 含 $d^{-2},d^{-3}$ 的主拟合中，23 个非恒零端点的理论系数全部在其拟合值
1.33 个抽样标准误以内，拟合残差 $\chi^2$ 最大 1.69（每项 2 自由度）；GD 新任务均值
在该参数点数值上恒等于总体值。第二种模型在若干端点的 $\chi^2$ 很大，尤其 GD 的错误均方，
因此其中超出抽样区间的“斜率偏差”不能当作系数反例；它是在有限维范围删掉 $d^{-3}$ 后的模型偏差。
没有因为这一结果而丢弃任何维度或隐藏替代拟合。
对 Sign/Adam 的描述性残差幂次及下表的缩放残差总体与 $O(d^{-2})$ 相容；
这比被检的 $O(d^{-3/2})$ 包络衰减更快，但本审查不据此升级定理余项。
''')
put(r'''
将**理论**系数减去后，查看
$R_d=d^{3/2}(\widehat E T-T_\infty-c_{theory}/d)$。
下表列出三个代表维度；括号内为 Monte Carlo 标准误。理论只要求它有界，不要求趋于非零常数。

| $r$ | 方法 | 量/任务 | $d=48$ | $d=96$ | $d=192$ |
|---:|---|---|---:|---:|---:|''')
for f in b['fits']:
    label=f['method'] if f['method']!='Adam' else f"Adam({f['e']:g})"
    kind='m' if f['kind']=='mean' else 'w'
    if f.get('exact_zero'):
        vals=['0','0','0']
    else:
        dmap={x['d']:x for x in f['scaled_residuals']}
        vals=[f"{dmap[d]['value']:+.3f} ({dmap[d]['se']:.3f})" for d in (48,96,192)]
    put(f"| {f['r']:g} | {label} | {kind}{f['task']} | "+' | '.join(vals)+' |')
put('\n完整 8 维度样本均值、标准误、经验系数和全部残差见 `review15_checks/main.json`，完整拟合与描述性残差幂次见 `analysis.json`。')
put('这些检查分别针对正确均值和错误均方，不把召回率残差拿来充当矩展开检验。\n')
put(r'''## 7. 与论文系数式和系数表比对

**PASS：** Z1 的 (1.4) 与论文 78–82 行的 $m_k,\ell,w_k$ 逐项相同，
包括 $r-3$、$2\sigma^2\beta^2$ 和训练候选额外 $\mu_k^2$。
论文 221–225 行系数表列的是候选召回系数 $B_R$，不是 $m_k,w_k$，故不能直接拿表值当作本次矩系数。
本轮从论文 H 的总体第一分支匹配方程重新计算幅度，并独立做半正态积分，再代入 M 的候选 $B_R$ 公式。
未读取训练召回结果作拟合，未使用旧输出文件中的幅度来生成该表。

以下为重新计算的 $B_R$（同论文精度，依次旧/新）：

| 方法 | $r=.5$ 旧/新 | $r=0$ 旧/新 | $r=-.5$ 旧/新 |
|---|---:|---:|---:|''')
table={}
for x in a['coefficient_table']:
    table.setdefault(x['method'],{})[x['r']]=x
for label,cols in table.items():
    vs=[' / '.join(f'{z:+.3f}' for z in cols[r]['BR']) for r in (.5,0.,-.5)]
    put('| '+label+' | '+' | '.join(vs)+' |')
put('\n30 个重算值全部落在论文三位小数的舍入区间（绝对差不超过 0.0005）。')
put('**该比对只说明公式与表格一致；召回的联合累积量闭合和非光滑指标转移仍未由 Z1 证明。**\n')
put(r'''## 8. `z1_checks/` 本身的审查

| 项目 | 判定 | 说明 |
|---|---|---|
| 符号与小维度恒等式 | PASS（诊断范围） | 六个符号残差和评分/球面/迹恒等式可查错；单个 $2\times2$ 路径不能取代一般随机 Wishart 证明。 |
| Sign arcsin 检查 | PASS（实例） | 可检出归一化错误，但有限相关系数网格不是 Hermite 引理的证明。解析 $L^2$ 证明已另给。 |
| 块偶函数检查 | PASS（实例） | 只检查平方范数，不覆盖一般非光滑函数；一般结论依赖中心标准高斯的 chaos 论证。 |
| 条件训练收缩的原数值检查 | 需修正（如作独立验证使用） | 原代码只对理论回归向量做有限和，没有直接对随机标签积分。本轮的原始标签路径补了该诊断。 |
| 条件方差的原数值检查 | 需修正（如作渐近认证使用） | 原断言检查普适方差分解；有限维实测不能证明统一 $C/d$。Z1 的 §6 才是解析依据。 |
| 多维 $1/d$ 和 $d^{-3/2}$ 检验 | 原材料缺失；本轮已补 | 原脚本没有本任务要求的多维系数/残差拟合。 |
| 输入完整性披露 | PASS（已披露范围） | 原记录明确披露额外读取的 `fine_tuning.tex` 在其运行期间散列变化，不能声称所有读过输入未变；没有据此推断修改者，也没有把这点当作概率反例。 |

## 9. 复现、文件范围和最终判断

所有 Python 均以以下前缀运行；两项同时计算时各为单线程，总计不超过两条计算线程：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python
```

在此前缀后依次使用：

```text
review15_checks/independent_checks.py --output main.json --reps 4000 --raw-samples 300000 --dims 16,24,32,48,64,96,128,192
review15_checks/raw_dataset_crosscheck.py
review15_checks/validate_and_fit.py
review15_checks/write_report.py
```

程序拒绝覆盖其结果；以上为新目录中首次执行的复现方式。`pilot.json` 是先行小样本诊断，未并入主拟合。
本轮只新建 `PROOF_REVIEW15.md` 和 `review15_checks/` 内的文件；没有对任何用户原有文件发出写操作。
只读取本项目、指定 Python 环境及系统资源元数据，没有读取其他项目或调用网络/LLM API。
主被审输入和 `z1_checks` 全部文件的前后 SHA-256 保存在 `main.json`，结果一致。

**最终：需修正引理 3.4 的显式零均值条件及少量维度/证据范围措辞；未找到主定理的实质 FAIL。**
按中心高斯和充分大维度的实际使用条件，行能量界、训练候选额外项、Sign 可积余项及删除二阶矩链条均通过。
数值结果是独立查错证据，不承担渐近定理的证明责任；论文中的候选召回修正仍保留其原有未证明状态。
''')
put('\n本轮审查对象 SHA-256：\n')
for name in ('Z1_G1_PROOF.md','paper/appendix/M_nonlinear_finite_d.tex','z1_checks/Z1_check.py'):
    put(f"- `{name}`：`{a['input_hashes_before'][name]}`")
with out.open('x') as f:
    f.write('\n'.join(lines)+'\n')
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
with (HERE/'alternative_fit_diagnostics.json').open('x') as f:
    json.dump(model_diagnostics,f,indent=2)
    f.write('\n')
with (HERE/'review_code_sha256.json').open('x') as f:
    json.dump(manifest,f,indent=2)
    f.write('\n')
print(str(out.relative_to(ROOT)))
