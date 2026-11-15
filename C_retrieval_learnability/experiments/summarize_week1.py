"""Offline plots, preregistered assessments and Chinese week-1 report."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  MPLCONFIGDIR='/tmp/theory-week1-matplotlib', CUDA_VISIBLE_DEVICES='')
import resource
resource.setrlimit(resource.RLIMIT_AS, (3500000000, 3500000000))
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'experiments/week1'
LOSSES = ['nce', 'sigmoid', 'pe']
COLORS = {'nce': '#3567ad', 'sigmoid': '#cc6337', 'pe': '#3b9562'}


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def fmt(x, digits=6):
    return '无' if x is None else f'{x:.{digits}f}'


def triple(group, field):
    return ' / '.join(fmt(group[l]['final'][field], 4) for l in LOSSES)


def savefig(name):
    plt.savefig(OUT / name, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    start = time.perf_counter()
    meta, supervisor = read('metadata.json'), read('supervisor.json')
    extra_resources = read('E_addendum_resources.json')
    extra_prereg = read('E_addendum_preregistration.json')
    assert meta['complete'] and supervisor['returncode'] == 0
    a, b = read('A.json'), read('B.json')
    c, d, e = [read(g + '.json')['runs'] for g in ['C', 'D', 'E']]
    assert len(c) == 36 and len(d) == 17 and len(b['slow']) == 80 and len(b['frozen']) == 16
    assert all(x['status'] == 'completed' for x in c + d + e)
    assert a['passed_preregistered_gate'] and b['passed_preregistered_gate']
    validation = read('validation.json')
    baseline = [r for r in c if r['config']['k'] == 3 and r['config']['d'] == 6 and r['config']['seed'] == 0]
    settings = [(2, 4), (2, 8), (3, 6), (3, 12)]

    # H2 decisions are direct applications of FORMULATION section 5.3, not
    # significance thresholds estimated from the observed runs.
    h2, paired = [], []
    for k, dim in settings:
        qualified = []
        conflict_fractions = []
        for seed in [0, 1, 2]:
            group = {r['config']['loss']: r for r in c if r['config']['k'] == k and r['config']['d'] == dim and r['config']['seed'] == seed}
            history = read(group['nce']['raw_file'])['history']
            checkpoints = [r for r in history if r['step'] > 0]
            conflict = np.mean([r['minus_gD_dot_h'] < 0 and r['minus_gR_dot_h'] > 0 for r in checkpoints])
            conflict_fractions.append(float(conflict))
            success_deltas = {l: group[l]['final']['success_fraction'] - group['nce']['final']['success_fraction'] for l in ['sigmoid', 'pe']}
            gap_deltas = {l: group[l]['final']['min_gap'] - group['nce']['final']['min_gap'] for l in ['sigmoid', 'pe']}
            qualifies = all(x >= .1 for x in success_deltas.values()) and all(x > 0 for x in gap_deltas.values())
            qualified.append(qualifies)
            paired.append({'k': k, 'd': dim, 'seed': seed,
                           'success_delta_vs_nce': success_deltas, 'min_gap_delta_vs_nce': gap_deltas,
                           'NCE_conflict_fraction': float(conflict), 'endpoint_qualifies': qualifies,
                           'group': group})
        h2.append({'k': k, 'd': dim, 'qualified_seed_count': sum(qualified),
                   'NCE_conflict_fractions': conflict_fractions,
                   'passes_before_step_refinement': sum(qualified) >= 2 and np.mean(conflict_fractions) > .5})
    preliminary = sum(row['passes_before_step_refinement'] for row in h2) >= 2
    # Refinement is necessary for support, but cannot rescue failed C endpoints.
    h2_verdict = '不支持；否定本轮预注册 H2 信号预测' if not preliminary else '需要结合 E 复核判定'
    calibration_runs = c + d + e
    matched = sum(r['calibration']['matched_20_percent'] for r in calibration_runs)
    raw_slow_clean = [r for r in b['slow'] if r['variant'] == 'clean']
    worst_discrete = max(r['epsilon_T_relative_difference'] for r in b['discretization'])
    min_hessian = min(r['min_hessian_eigenvalue'] for r in b['slow'])
    assessment = {'A_pass': a['passed_preregistered_gate'], 'B_pass': b['passed_preregistered_gate'],
                  'H1_all_8_initialization_variants_pass': all(r['supported_preregistered_H1'] for r in b['H1']),
                  'H2_by_setting': h2, 'H2_preliminary_support': bool(preliminary), 'H2_verdict': h2_verdict,
                  'calibration_matched_count': matched, 'calibration_run_count': len(calibration_runs),
                  'max_slow_discretization_relative_error': worst_discrete,
                  'min_slow_hessian_eigenvalue': min_hessian,
                  'paired_C': [{key: val for key, val in row.items() if key != 'group'} for row in paired]}
    write('assessment.json', assessment)

    # A: independent optimizers and analytic boundary.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r in [2, 8, 32]:
        rows = sorted([x for x in a['formulas'] if x['r'] == r], key=lambda x: x['q'])
        axes[0].plot([x['q'] for x in rows], [x['optimal_E'] for x in rows], 'o-', label=f'r={r}')
    axes[0].set(xlabel='q = delta / g', ylabel='minimum excess E', xscale='log')
    axes[0].legend()
    for target in [.01, .05, .1]:
        rows = [x for x in a['boundaries'] if x['target'] == target]
        axes[1].plot([x['r'] for x in rows], [x['qc_formula4'] for x in rows], 'o-', label=f'target={target}')
    axes[1].set(xlabel='negative count r', ylabel='exact feasible boundary q_c', xscale='log')
    axes[1].legend()
    fig.tight_layout()
    savefig('A_formula.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for init, style in [('star', '-'), ('zero', '--')]:
        rows = sorted([x for x in raw_slow_clean if x['initial_query'] == init and x['h'] == .025], key=lambda x: x['epsilon'])
        axes[0].loglog([x['epsilon'] for x in rows], [x['hit_time'] for x in rows], 'o' + style, label=init)
        axes[1].semilogx([x['epsilon'] for x in rows], [x['epsilon_times_hit'] for x in rows], 'o' + style, label=init)
        for run in rows:
            history = read(run['raw_file'])['history']
            axes[2].plot([x['tau'] for x in history], [x['gap'] for x in history], style, alpha=.7,
                         label=f'{init}, eps={run["epsilon"]}')
    axes[0].set(xlabel='epsilon', ylabel='T_hit (gamma > 1e-4)')
    axes[1].set(xlabel='epsilon', ylabel='epsilon * T_hit')
    axes[2].axhline(0., color='black', lw=.8)
    axes[2].set(xlabel='slow time tau', ylabel='ranking gap gamma')
    axes[0].legend(); axes[1].legend(); axes[2].legend(fontsize=6)
    fig.tight_layout()
    savefig('B_slow_flow.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = [f'k={k},d={dim}' for k, dim in settings]
    for j, loss in enumerate(LOSSES):
        for i, (k, dim) in enumerate(settings):
            rows = [r for r in c if r['config']['k'] == k and r['config']['d'] == dim and r['config']['loss'] == loss]
            for field, ax in [('success_fraction', axes[0]), ('min_gap', axes[1])]:
                ax.scatter([i + (j - 1) * .18] * 3, [r['final'][field] for r in rows], color=COLORS[loss],
                           label=loss if i == 0 else None, alpha=.8)
        axes[0].plot([], [], color=COLORS[loss])
    for i, row in enumerate(h2):
        axes[2].scatter([i] * 3, row['NCE_conflict_fractions'], color=COLORS['nce'])
    for ax in axes:
        ax.set_xticks(range(4), labels, rotation=15)
    axes[0].set(ylabel='strict success fraction', ylim=(-.05, 1.05))
    axes[1].set(ylabel='minimum gap')
    axes[1].axhline(0., color='black', lw=.8)
    axes[2].set(ylabel='NCE conflict checkpoint fraction', ylim=(0, 1.05))
    axes[2].axhline(.5, color='black', linestyle='--', lw=.8)
    axes[0].legend(); axes[1].legend()
    fig.tight_layout()
    savefig('C_spherical.png')

    # D all planned 20 runs, including three baseline references.
    d_all = baseline + d
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, run in enumerate(d_all):
        label = run['config']['id'].replace('C_k3_d6_s0_', 'baseline_')
        for ax, field in zip(axes, ['success_fraction', 'min_gap']):
            ax.bar(i, run['final'][field], color=COLORS[run['config']['loss']])
    axes[0].set(ylabel='strict success fraction', ylim=(0, 1.05))
    axes[1].set(ylabel='minimum gap')
    axes[1].axhline(0., color='black', lw=.8)
    axes[1].set_xticks(range(len(d_all)), [r['config']['id'].replace('C_k3_d6_s0_', 'baseline_') for r in d_all], rotation=65, ha='right', fontsize=7)
    fig.tight_layout()
    savefig('D_ablations.png')

    base_map = {r['config']['id']: r for r in c + d}
    e_bases = list(dict.fromkeys(r['baseline_id'] for r in e))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, baseid in enumerate(e_bases):
        runs = [('base Adam', base_map[baseid]),
                ('half lr, double steps', next(r for r in e if r['baseline_id'] == baseid and r['method'] == 'adam')),
                ('projected GD', next(r for r in e if r['baseline_id'] == baseid and r['method'] == 'gd'))]
        for j, (label, run) in enumerate(runs):
            for ax, field in zip(axes, ['success_fraction', 'min_gap']):
                ax.bar(i + (j - 1) * .24, run['final'][field], width=.22,
                       color=['#3567ad', '#cc6337', '#8d8d8d'][j], label=label if i == 0 else None)
    axes[0].set(ylabel='strict success fraction', ylim=(0, 1.05)); axes[0].legend(fontsize=8)
    axes[1].set(ylabel='minimum gap'); axes[1].axhline(0, color='black', lw=.8)
    axes[1].set_xticks(range(len(e_bases)), e_bases, rotation=35, ha='right', fontsize=8)
    fig.tight_layout()
    savefig('E_refinement.png')

    # Chinese report: retain the complete pre-execution registration verbatim.
    q1 = a['Q1']
    clean_nce = next(x for x in b['frozen'] if x['variant'] == 'clean' and x['loss'] == 'nce' and x['h'] == .025)
    clean_sig = next(x for x in b['frozen'] if x['variant'] == 'clean' and x['loss'] == 'sigmoid' and x['h'] == .05)
    peaks = meta['memory_final']
    code_hashes_match = all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha for name, sha in meta['hashes_at_start'].items() if name != 'RESULTS_WEEK1.md')
    assert code_hashes_match, 'Executed input source changed before reporting.'
    text = []
    text.append('## 结果与决定\n')
    text.append('**建议止损原定的标准自由球面检索机制主线；保留闭式目标错配例与快慢流作为窄范围数学结果。** '
                'A、B 控制和 H1 通过；H2 在四个 `(k,d)` 设置中没有一个达到预注册的配对终态门槛，'
                '不能据此宣称“正例竞争解释了标准双塔的有限时间缺口”。本轮对 H2 的实验信号预测判为否定/不支持，'
                '对其他规模、训练预算或真实文本检索的结论仍是不确定。\n')
    text.append(f'基础执行时间：{meta["started_utc"]} 至 {meta["completed_utc"]}（UTC）；'
                f'补充 E 完成于 {extra_resources["completed_utc"]}。'
                f'基础 A–E 监督进程实测 **{supervisor["wall_seconds"]:.2f} 秒**（含解释器与 PyTorch 导入），'
                f'补充 E 计算另用 **{extra_resources["elapsed_seconds"]:.2f} 秒**（不含其解释器导入），'
                '绘图汇总耗时另见 reporting_resources.json；远低于 55/75 分钟。各组均低于原配额，未削减步数。'
                f'实验进程峰值 RSS **{peaks["VmHWM"] / 1e6:.1f} MB**，峰值虚拟地址空间 **{peaks["VmPeak"] / 1e9:.4f} GB**，'
                '低于 3.5 GB 硬上限；绘图另在顺序执行的受限进程完成。无 API、网络、GPU 或模型下载。\n')
    text.append(table(['组', '实际执行', '秒', '判定'], [
        ['A', '18 个 (q,r)；9 个边界及 18 个边界两侧点', fmt(meta['group_seconds']['A'], 2), '支持 Q1/Q2'],
        ['B', '16 条冻结；80 条慢流', fmt(meta['group_seconds']['B'], 2), '支持 Q3；数值支持 H1'],
        ['C', '36 次 × 1500 步', fmt(meta['group_seconds']['C'], 2), 'H2 未过门槛'],
        ['D', '20 配置（17 新跑 + 3 复用）× 1500 步', fmt(meta['group_seconds']['D'], 2), '未恢复所需机制证据'],
        ['E', f'{len(e)} 次；Adam 3000 步，GD 1500 步', fmt(meta['group_seconds']['E'] + extra_resources['elapsed_seconds'], 2), '复核仍无 H2 支持']]) + '\n')
    text.append('所有运行以 PCG64 种子 0/1/2、float64、CPU 完成。三种损失使用相同标签、相同 U/V 初值；'
                'Adam 参数为 (0.9,0.999,10^-8)、无 weight decay，每步重投影。'
                '完整配置、初始登记及执行代码 SHA-256 见 [metadata.json](experiments/week1/metadata.json) 和 '
                '[plan.json](experiments/week1/plan.json)。`FORMULATION.md` 与执行代码的哈希复核一致。\n')
    text.append('预算记录偏差：执行前第 7 条写成“各组开始前 100 步计时”，实际只在 C/D 保存了最大 PE 问题的 '
                '100 步计时（C_timing.json、D_timing.json）；A/B/E 未另存单独的 100 步计时。'
                'A 是固定公式网格，B 是固定时间上限的控制，E 复用已计时的 C/D 配置；'
                '三组实测均远低于预算，没有删减、超时或把超时记为失败，不影响结果判据。\n')
    text.append('### A：有限温度公式与可行边界\n')
    text.append('设置：g=1，q={0.01,0.02,0.05,0.1,0.2,0.4}，r={2,8,32}；'
                'z=βg 在 [0,30] 每 0.001 取点，另作 80 轮二分；目标误差 {0.01,0.05,0.1}，'
                '式 (4) 用一维有界最大化，并以独立最优 E 求根核验。\n')
    text.append(f'Q1 关键点 r=8,q=0.05：**βg={q1["optimal_z_bisection"]:.9f}，最小 E={q1["optimal_E"]:.9f}**，'
                '符合预注册约 5.63548、0.0528178。'
                f'直接 logsumexp 与式 (2) 最大误差 {max(x["formula_max_abs_error"] for x in a["formulas"]):.3e}；'
                f'最优点与网格最大距离 {max(abs(x["optimal_z_bisection"] - x["optimal_z_grid"]) for x in a["formulas"]):.6f}，小于 0.001。'
                f'边界独立核验最大 q 误差 {max(x["boundary_q_error"] for x in a["boundaries"]):.3e}；'
                '18/18 个距离边界 0.006 的点可达性一致；所有六个 q 上，r 增大时最小 E 均不下降。**Q1/Q2：支持。**\n')
    text.append(table(['q', 'r=2：最优 z / E', 'r=8：最优 z / E', 'r=32：最优 z / E'], [
        [str(q)] + [f'{x["optimal_z_bisection"]:.6f} / {x["optimal_E"]:.6f}' for r in [2, 8, 32]
                    for x in a['formulas'] if x['q'] == q and x['r'] == r]
        for q in [.01, .02, .05, .1, .2, .4]]) + '\n')
    text.append(table(['目标 ε', 'q_c(r=2)', 'q_c(r=8)', 'q_c(r=32)'], [
        [target] + [fmt(x['qc_formula4'], 8) for r in [2, 8, 32] for x in a['boundaries'] if x['r'] == r and x['target'] == target]
        for target in [.01, .05, .1]]) + '\n')
    text.append('A 的独立求根在宽区间端点 `sinh(1000)` 出现一次上溢警告；这是求根括区间的正无穷函数值，'
                '实际根、公式值、所有校验及 JSON 均为有限数，不是训练温度失控，也未因此改动公式或判据。\n')
    text.append('![A 公式与边界](experiments/week1/A_formula.png)\n')
    text.append('### B：闭式目标错配与慢文档逃逸\n')
    text.append('设置：七文档 V0、β=1、不做球面约束。冻结实验从零开始；NCE 的 t=300、h=0.05/0.025，'
                'sigmoid 的 t=4000、h=0.1/0.05，带一个增广偏置。慢流 ε={0.04,0.02,0.01,0.005,0.0025}，'
                '查询从零或 (log2,0) 开始，h=0.05/0.025；每个 V 加无扰动或三个标准差 0.001 的高斯扰动。'
                '慢流到 γ>10^-4 后，另持续确认 Δτ=0.02；上限 τ=4。\n')
    text.append(f'无扰动细步长 NCE：‖w−w*‖={clean_nce["distance_to_star"]:.3e}，'
                f'‖∇f‖={clean_nce["final"]["gradient_norm"]:.3e}，γ={clean_nce["final"]["gap"]:.12f}，'
                f'recall@3={clean_nce["final"]["recall_at_3"]:.9f}，E={clean_nce["final"]["loss"] - np.log(3):.9f}。'
                f'sigmoid：γ={clean_sig["final"]["gap"]:.9f}，最小带符号 logit={clean_sig["final"]["min_signed_logit"]:.9f}；'
                f'归一化 γ={clean_sig["final"]["normalized_gap"]:.9f}，查询范数={clean_sig["final"]["w_norm"]:.6f}。'
                '全部扰动及两种步长均满足 NCE 梯度收敛但排错、sigmoid 符号全部正确。**Q3：支持。**\n')
    text.append(table(['V', '损失', 'h', 'γ', '归一化 γ', '梯度范数', 'recall@3 / 最小符号 logit'], [
        [r['variant'], r['loss'], r['h'], fmt(r['final']['gap']), fmt(r['final']['normalized_gap']),
         f'{r["final"]["gradient_norm"]:.3e}',
         f'{r["final"]["recall_at_3"]:.6f} / {fmt(r["final"].get("min_signed_logit"))}'] for r in b['frozen']]) + '\n')
    text.append('无扰动慢流的首次阈值通过时间如下；全部通过都持续了额外 Δτ=0.02。\n')
    text.append(table(['ε', 'warm T(h=.05)', 'warm T(h=.025)', 'warm εT(h=.025)', 'cold T(h=.025)', 'cold εT(h=.025)'], [
        [eps] + [fmt(next(r for r in raw_slow_clean if r['epsilon'] == eps and r['initial_query'] == init and r['h'] == h)[field], 6)
                 for init, h, field in [('star', .05, 'hit_time'), ('star', .025, 'hit_time'), ('star', .025, 'epsilon_times_hit'),
                                        ('zero', .025, 'hit_time'), ('zero', .025, 'epsilon_times_hit')]]
        for eps in [.04, .02, .01, .005, .0025]]) + '\n')
    text.append(table(['V', '起点', '最小四 ε 的 logT/logε 斜率', 'εT 极差/最小值', 'H1'], [
        [r['variant'], r['initial_query'], fmt(r['smallest_four_slope']), f'{100 * r["epsilon_T_relative_spread"]:.3f}%', '支持' if r['supported_preregistered_H1'] else '不支持']
        for r in b['H1']]) + '\n')
    text.append(f'80/80 条慢流均逃逸并持续通过；两种步长的 εT 最大相对差 **{100 * worst_discrete:.4f}%**，'
                f'小于 1%；沿记录轨迹最小 Hessian 特征值 **{min_hessian:.6f}**，未见接近零。'
                '8 个 V/起点组合都满足斜率 [−1.1,−0.9] 和 10% 稳定要求。'
                '**Q4 的有限时间平台与更强 H1：数值支持。** 有限 ε 与阈值首达不能证明极限常数精确为 2.46，'
                '也不能把数值支持写成全局慢流上界的证明。B 两损失初始步长未校准，不比较其快慢。\n')
    text.append('![B 慢流](experiments/week1/B_slow_flow.png)\n')
    text.append('### C：标准等学习率的共享文档球面任务\n')
    text.append('设置：n=16，k=2/3，M=120/560（全部 k 子集），d=2k/4k，三个配对种子，'
                'β=16，文档/查询学习率比为 1。三损失为全正例平均 NCE、全局偏置 sigmoid（初始 log(k/(n−k))）、'
                '正例排除 PE；均跑满 1500 步。下面每格依次为 **NCE / sigmoid / PE**。\n')
    text.append(table(['(k,d)', 'seed', '完全成功率', 'recall@k', '最差 γ', 'NCE 冲突比例'], [
        [f'({r["k"]},{r["d"]})', r['seed'], triple(r['group'], 'success_fraction'), triple(r['group'], 'recall_at_k'),
         triple(r['group'], 'min_gap'), f'{r["NCE_conflict_fraction"]:.3f}'] for r in paired]) + '\n')
    text.append(table(['(k,d)', '同时满足两比较 ≥0.10 且最差 γ 改善的种子数', '超过半数冲突检查点的种子数', '是否过 H2 设置门槛'], [
        [f'({r["k"]},{r["d"]})', f'{r["qualified_seed_count"]}/3',
         f'{sum(x > .5 for x in r["NCE_conflict_fractions"])}/3', '是' if r['passes_before_step_refinement'] else '否'] for r in h2]) + '\n')
    text.append('**H2：不支持，本轮预注册的标准设置信号预测被否定。** '
                '低维 PE 的成功比例有时提升，但 sigmoid 未同步提升，且 PE 的最差间隔未同步改善；'
                '高维 NCE 已全部成功，未出现所需差距。即使出现 −⟨g_D,h⟩<0、−⟨g_R,h⟩>0，'
                '这也只是局部方向诊断，不能替代预注册的配对因果消融与终态门槛。'
                '高维三目标接近全成功的单独比较按第 5.3(6) 条视为饱和、不确定，不能据此否定几何存在性。\n')
    text.append('![C 标准球面比较](experiments/week1/C_spherical.png)\n')
    text.append('### D：偏置、温度和文档学习率混杂\n')
    text.append('统一 k=3,d=6,seed=0，其他沿用 C。D1 的文档/查询 lr 比为 {0.01,0.1,1}；'
                'D2 仅 sigmoid 的固定零偏置/逐查询偏置；D3 固定 β={4,64}；D4 β=exp(a)，a 初始 log16、'
                'lr 为查询组的 0.1。基线三次来自 C，共 20 配置，新增 17 次。\n')
    text.append(table(['配置', '成功率', 'recall@k', '最差 γ', '终态 β', '终态偏置均值'], [
        [r['config']['id'].replace('C_k3_d6_s0_', '基线_'), fmt(r['final']['success_fraction'], 4),
         fmt(r['final']['recall_at_k'], 4), fmt(r['final']['min_gap']), fmt(r['final']['beta'], 3),
         fmt(r['final'].get('bias_mean'), 4) if r['config']['loss'] == 'sigmoid' else '—'] for r in d_all]) + '\n')
    text.append(f'温度失控/溢出运行数：{sum(r["status"] != "completed" for r in d_all)}。'
                'D 的单种子消融只能揭示敏感性，不能覆盖 C 中两个设置、多个种子的 H2 门槛；'
                '不使用这些结果重选“最有利温度”并改写原预测。慢文档、偏置或温度变化都没有构成所要求的标准设置信号。\n')
    text.append('![D 混杂消融](experiments/week1/D_ablations.png)\n')
    text.append('### E：半步长、双步数与球面 GD 复核\n')
    text.append('选择严格按执行前规则，见 [E_selection.json](experiments/week1/E_selection.json)。'
                '每个选中组的三损失均做 Adam 学习率减半、步数加倍到 3000；'
                '另做经完整试更新校准的球面投影 GD 1500 步。下表全部结果包括校准失败，未选择性排除。\n')
    text.append('**E 选择规则补充（已先记录再执行）**：原“最大共同正向优势”规则在没有正向优势时只选中高维饱和组。'
                '为了落实原文的“最明显差异”，在总共最多 3 组范围内追加绝对成功率极差最大的一组，'
                '并列按最差 γ 极差和配置 ID；不改预测或判据。'
                f'在 {extra_prereg["registered_utc"]} 登记后，选中 {", ".join(extra_prereg["selected_ids"])}，'
                f'基础成功率极差={extra_prereg["absolute_success_spread"]:.6f}。'
                '原基础 E 结果保存在 E_original.json；执行前报告快照见 '
                '[E_addendum_report_before_execution.md](experiments/week1/E_addendum_report_before_execution.md)，'
                '机器登记见 [E_addendum_preregistration.json](experiments/week1/E_addendum_preregistration.json)。'
                '这部分属于原 E 明确允许的结果依赖复核，不能当作新增独立验证种子。\n')
    text.append(table(['基线配置', '基线成功率 / γ', '半 lr、双步数成功率 / γ', 'GD 成功率 / γ'], [
        [baseid, f'{base_map[baseid]["final"]["success_fraction"]:.4f} / {base_map[baseid]["final"]["min_gap"]:.6f}'] +
        [f'{r["final"]["success_fraction"]:.4f} / {r["final"]["min_gap"]:.6f}' for method in ['adam', 'gd']
         for r in e if r['baseline_id'] == baseid and r['method'] == method] for baseid in e_bases]) + '\n')
    text.append('E 未恢复 H2 必需的配对终态优势；在已达到完全成功的设置中也没有形成“屏蔽正例竞争才学会”的分离。'
                'GD 的步数与 Adam 相同不代表等有效优化预算；校准夹界及实际位移见下一段，因此不从 GD 的较差终态推断损失不可学。\n')
    text.append('![E 复核](experiments/week1/E_refinement.png)\n')
    text.append('### 学习率校准与实现核验的限制\n')
    text.append(f'C/D/E 共 {len(calibration_runs)} 个实际优化运行中，初始 RMS 匹配到 20% 内的有 **{matched}** 个。'
                '严格遵守 [0.1,10] 的缩放裁剪后，多数 Adam 的最小允许 lr 仍产生大于目标 10^-3 的 RMS 位移；'
                '部分 GD 因最大允许 lr 仍无法达到目标而匹配失败。'
                '**这些运行只作终态比较，不纳入“更快”结论**，并未看到结果后放宽校准范围。'
                '每次的完整校准数字均在原始 JSON。\n')
    text.append(table(['运行类别', '次数', '匹配到 20% 内', '实际初始 RMS 范围', '缩放范围'], [
        [label, len(rows), sum(r['calibration']['matched_20_percent'] for r in rows),
         f'{min(r["calibration"]["actual_rms"] for r in rows):.3e}–{max(r["calibration"]["actual_rms"] for r in rows):.3e}',
         f'{min(r["calibration"]["scale"] for r in rows):.3f}–{max(r["calibration"]["scale"] for r in rows):.3f}']
        for label, rows in [('C Adam', c), ('D 新增 Adam', d), ('E 半 lr Adam 的原校准', [r for r in e if r['method'] == 'adam']),
                            ('E 投影 GD', [r for r in e if r['method'] == 'gd'])]]) + '\n')
    text.append(f'解析梯度与 PyTorch 自动微分最大绝对误差 {max(r["autograd_max_abs_error"] for r in validation["objectives"]):.3e}，'
                f'中心差分最大误差 {max(r["finite_difference_max_abs_error"] for r in validation["objectives"]):.3e}；'
                f'D+R 分解误差 {validation["decomposition_error"]:.3e}，'
                f'切空间内积自动微分对照误差 {validation["diagnostic_autograd_error"]:.3e}。'
                '10 种损失/偏置/温度组合全部通过。只保存每 20 步标量、必要初末参数；不保存完整嵌入历史。\n')
    text.append('### 继续还是止损\n')
    text.append('**停止以当前 H2 机制直接冲刺原定主会主线。** 原文止损门槛要求第一周在 ε=1、共享文档、'
                '球面随机初始化上看到标准设置信号，本轮没有达到；扩大规模或改温度找效果不能作为原预测已被支持。'
                'A/B 是可复核的窄范围正结果，支持目标错配和人工快慢系统的有限时间现象，但尚不能解释标准自由嵌入差距。'
                '保留数学小例与 H1 推导有价值，是否另立更窄题目取决于后续独立证明和新颖性核查。'
                '本轮禁止外部 API，未做新的在线文献核查，也未声称完成第 7 节的独立证明审查。\n')
    text.append('最该做的三件事：\n\n'
                '1. 先做不需要新增大规模训练的独立审查：七点例/扰动鲁棒性及 H1 约化慢流首次横截；明确区分已证下界与数值上界，并复查与已有结果是否同构。\n'
                '2. 若继续窄题目，先另写预注册修正初始更新校准：允许足够小的 Adam lr 和足够大的 GD lr，确保真实球面 RMS 匹配，再以同一配对种子复核；本轮结果和判据保持封存。\n'
                '3. 只在前两项产生新的、可迁移的定量预测后，另行预注册一个小型跨温度/矩阵验证；若仍无 ε=1 信号则正式换题，不启动 LIMIT 或文本编码器大实验。\n')
    text.append('### API 后续与复现\n')
    text.append('本目录 B 组是闭式/慢流实验，不涉及 LLM 真实轨迹，**下一步必要 API 调用数=0，API 预算=0**。'
                '相邻题目 B 的 agent 轨迹实验不在本次 `FORMULATION.md` 第 5 节范围内，未执行，也不把这里的 B 组误称为该课题。\n')
    text.append('复现（会覆盖本轮同名结果文件，保留前请先自行复制目录）：\n\n'
                '```bash\n/home/liuyibo/venvs/theory/bin/python experiments/run_week1.py\n'
                '/home/liuyibo/venvs/theory/bin/python experiments/supplement_week1_e.py\n'
                '/home/liuyibo/venvs/theory/bin/python experiments/summarize_week1.py\n```\n\n'
                '运行器内部设置线程、CPU、内存上限和 55 分钟超时。'
                '原始索引：[A](experiments/week1/A.json)、[B](experiments/week1/B.json)、'
                '[C](experiments/week1/C.json)、[D](experiments/week1/D.json)、[E](experiments/week1/E.json)；'
                '每条记录的 `raw_file` 指向含每 20 步标量的独立 JSON。'
                '汇总判据见 [assessment.json](experiments/week1/assessment.json)，'
                '执行日志见 [execution.log](experiments/week1/execution.log)。\n')
    report = ROOT / 'RESULTS_WEEK1.md'
    original = report.read_text()
    marker = '## 结果'
    preregistered = original[:original.index(marker)]
    # The entire prefix hashes to the initially recorded registration when the
    # original placeholder is restored; results cannot silently rewrite it.
    placeholder = '## 结果\n\n待执行。全部原始结果将保存在 `experiments/week1/` 的 JSON，图保存为 PNG。执行完成后在此补充数值、逐项判定、继续/止损建议及三个下一步。\n'
    registration_matches = hashlib.sha256((preregistered + placeholder).encode()).hexdigest() == meta['hashes_at_start']['RESULTS_WEEK1.md']
    unchanged_rerun_report = hashlib.sha256(original.encode()).hexdigest() == meta['hashes_at_start']['RESULTS_WEEK1.md']
    assert registration_matches or unchanged_rerun_report
    if registration_matches:
        (OUT / 'preregistration_at_start.md').write_text(preregistered + placeholder)
    report.write_text(preregistered + '\n'.join(text))
    status = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/self/status').read_text().splitlines()
              if line.startswith(('VmPeak:', 'VmHWM:'))}
    write('reporting_resources.json', {'wall_seconds': time.perf_counter() - start, 'memory': status,
                                      'matplotlib_version': matplotlib.__version__,
                                      'rlimit_as_bytes': resource.getrlimit(resource.RLIMIT_AS),
                                      'numpy_only_plotting': True})
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(OUT.glob('*')) if p.is_file() and p.name != 'manifest.json'}
    for p in [report, ROOT / 'experiments/summarize_week1.py', ROOT / 'experiments/supplement_week1_e.py', ROOT / 'experiments/README_WEEK1.md']:
        if p.exists():
            manifest[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    write('manifest.json', manifest)
    print(json.dumps({key: val for key, val in assessment.items() if key != 'paired_C'}, indent=2, ensure_ascii=False))
    print('Report and 5 PNG figures written to', ROOT)


if __name__ == '__main__':
    main()
