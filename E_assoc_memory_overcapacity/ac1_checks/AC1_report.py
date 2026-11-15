"""汇总正式单线程结果；输出新报告、独立图和完整性记录。"""
import os, json, math, hashlib, time, resource
from pathlib import Path
import numpy as np
from scipy.special import ndtr

root=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(root/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

wall,cpu=time.time(),time.process_time()
main=json.loads((root/'serial/AC1_simulation.json').read_text())
initial=json.loads((root/'AC1_simulation.json').read_text())
audit=json.loads((root/'serial/AC1_audit.json').read_text())
bridge=json.loads((root/'AC1_bridge_audit.json').read_text())
rows=main['rows']
max_replay=max(abs(np.array(a['p'])-b['p']).max() for a,b in zip(rows,initial['rows']))
max_smooth=max(abs(np.array(r['p'])-r['residual_smooth']).max() for r in rows)
lookup={(r['method'],r['d'],r['s']):r for r in rows}

plt.rcParams.update({'font.size':10,'figure.dpi':160})
fig,axes=plt.subplots(2,3,figsize=(12,6.4),sharex=True,sharey=True)
for j,method in enumerate(['GD3','Sign2','Adam2']):
    for k,L in enumerate([1,5]):
        ax=axes[k,j]
        for d,color in zip([64,128,256,512],['#c6d9ef','#8eb7dd','#468cc1','#174f88']):
            rr=[lookup[(method,d,s)] for s in [-.75,0.,.75]]
            ax.errorbar([r['s_eff'] for r in rr],[r['p'][k] for r in rr],
                        yerr=[1.96*r['p_SE'][k] for r in rr],label=f'd={d}',color=color,marker='o',capsize=2)
        xx=np.linspace(-1,1,160)
        ax.plot(xx,ndtr(xx),'k--',label='Phi(s)')
        ax.set_title(f'{method}, top-{L}')
        ax.grid(alpha=.2)
        if k==1:ax.set_xlabel('s (integer-load effective value)')
        if j==0:ax.set_ylabel('expected recall')
        if j==2 and k==1:ax.legend(fontsize=8,loc='lower right')
fig.tight_layout()
fig.savefig(root/'AC1_topL.png')
fig.savefig(root/'AC1_topL.pdf')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True)
for k,L in enumerate([1,5]):
    ax=axes[k]
    for method,marker in [('GD3','o'),('Sign2','s'),('Adam2','^')]:
        rr=[lookup[(method,d,0.)] for d in [64,128,256,512]]
        ax.errorbar([r['d'] for r in rr],[r['p'][k] for r in rr],
                    yerr=[1.96*r['p_SE'][k] for r in rr],marker=marker,label=method,capsize=2)
    r=[lookup[('GD3',d,0.)] for d in [64,128,256,512]]
    ax.plot([x['d'] for x in r],[x['sphere_gauss'][k] for x in r],'k:',label='GD3 spherical + Gaussian proxy')
    ax.axhline(.5,color='black',ls='--',label='limit 0.5')
    ax.set_title(f's=0, top-{L}');ax.set_xlabel('d');ax.grid(alpha=.2)
axes[0].set_ylabel('expected recall')
axes[1].legend(fontsize=7)
fig.tight_layout();fig.savefig(root/'AC1_s0.png');fig.savefig(root/'AC1_s0.pdf');plt.close(fig)

manifest=[]
for line in (root/'AC1_inputs.sha256').read_text().splitlines():
    old,name=line.split('  ',1)
    p=root.parent/name
    new=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    manifest.append({'path':name,'before':old,'after':new,'unchanged':old==new})
integrity={'baseline_count':len(manifest),'unchanged_count':sum(r['unchanged'] for r in manifest),
           'changed_inputs':[r['path'] for r in manifest if not r['unchanged']],
           'records':manifest,
           'scope':'本轮显式写入仅 AC1_ 新文件和 ac1_checks/。旧 paper 有外部变化；不声称整个项目静止。'}
with (root/'AC1_integrity.json').open('x') as f:json.dump(integrity,f,indent=2,ensure_ascii=False)

lines=['# AC1：top-L 临界窗口的结果与复现',
'', '2026-11-06（洛杉矶）。理论见 [AC1_TOPL_PROOF.md](AC1_TOPL_PROOF.md)，查新见 [AC1_NOVELTY.md](AC1_NOVELTY.md)。',
'', '## 1. 结论与证据等级', '',
'已完成一般完整散布端点桥和实际固定有限 T 的 GD、Sign、Adam 应用：在 `c_T d/√n−b_{n−1}→s` 时，每个固定 L 有 `P{rank≤L}→Φ(s)`。能量控制及 Sign 的 `√log n` 放大误差均已闭合。容量尺度 `d²/log d` 和样本旋转几何已有直接前作；新增贡献的边界在查新文件中逐一说明。',
'', '数值显示非退化检索及总体趋近趋势，但 d≤512 远没有消除有限维偏差，不能将这些维度称作渐近精度认证。top-1 与 top-5 在这些维度仍有显著差距。',
'', '## 2. 独立数值设计与估计量', '',
'程序 [AC1_simulate.py](ac1_checks/AC1_simulate.py) 独立编写，未导入或执行 AC0/AA1/Z2B 历史代码。正式结果为其完全相同的 [单线程副本](ac1_checks/serial/AC1_simulate.py)。设计在模拟循环之前写入 [AC1_design.json](ac1_checks/serial/AC1_design.json)。',
'', '| 方法 | 固定参数 | c_T |', '|---|---|---:|']
for method,desc in [('GD3','T=3, η=.7'),('Sign2','T=2, α=.6'),('Adam2','T=2, α=.3, e=1, β₁=.9, β₂=.99')]:
    lines.append(f"| {method} | {desc} | {main['design']['moments'][method]['c']:.10f} |")
lines += ['', '维度 `64,128,256,512`，s=`−.75,0,.75`，每方法单独解整数负载；所有表使用实际 `s_eff=c_Td/√n−b_{n−1}`。种子 2026110701。共 36 格，每格 128 个独立外层数据集充分统计量 `(S,Z)`，每组 64 个条件查询，共 4608 组、294912 次内层积分。',
'', 'S 用精确 Bartlett–Wishart 采样，Z 条件于 S 按其真实行 Gaussian 分布采样；每一步均保留真实 S、同一 Z 与 Adam 历史。查询按给定 S 的训练 Stiefel 列边际采样，既不是独立测试探针，也没有把训练更新换成 `S=I`。',
'', '排名期望用精确残差分解：令 `t=Ux`、`λ=√(d/n)(ZS⁻¹x)ᵀt/||t||`、`h_x=(d/n)xᵀS⁻¹x`、`P_x=||S⁻¹ᐟ² Zᵀt||²/||t||²`。保留 `(λ,h_x,P_x)` 的相关性，抽 `J~χ²_{n−d−1}` 和独立辅助球面阈值 T，再按 Z2B §3 的较大根积分掉最后一个 Gaussian G。阈值由完整 n−1 维正态向量的 top-1/top-5 与**同一向量半径**生成，不独立抽取半径。',
'', '这一表示给出原训练查询的正确无条件期望；并未声称固定完整 X、Z 后错误方向仍均匀。每个外层组只产生一个均值 W_b∈[0,1]。标准误统一为 `sd(W_b)/√128`，有效独立样本数是 128，不是 8192；表内 ± 为 1 个数据集标准误。它们是描述性 SE，不是同时覆盖证书，也不是浮点向外舍入区间。',
'', '对单格任取失败概率 η，独立有界组均值可另给严格统计公式 `|mean−p|≤√[log(2/η)/(2B)]`；若同时保护 72 个 top-L 单元格，99% Hoeffding 半宽为 `√[log(144/.01)/(2·128)]≈0.1934`，较宽。本轮不以较窄的正态 SE 冒充该有限样本保证。根式例外事件会触发失败而不会删样本；全部完成，未发生例外。',
'', '## 3. s≈0：共同极限 .5，有限维仍偏移', '',
'| 方法 | d | n | top-1 ± SE | top-5 ± SE |', '|---|---:|---:|---:|---:|']
for method in ['GD3','Sign2','Adam2']:
    for d in [64,128,256,512]:
        r=lookup[(method,d,0.)]
        lines.append(f"| {method} | {d} | {r['n']} | {r['p'][0]:.5f} ± {r['p_SE'][0]:.5f} | {r['p'][1]:.5f} ± {r['p_SE'][1]:.5f} |")
lines += ['', '![s=0](ac1_checks/AC1_s0.png)', '',
'## 4. 三个窗口位置的全部结果', '',
'理论目标 `Φ(−.75)=.22663, Φ(0)=.5, Φ(.75)=.77337`；整数 n 引起的实际目标略有变化，表中保留实际值。', '',
'| d | 方法 | s | n | Φ(s_eff) | top-1 ± SE | top-5 ± SE |', '|---:|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['d']} | {r['method']} | {r['s']:+.2f} | {r['n']} | {r['limit']:.5f} | {r['p'][0]:.5f} ± {r['p_SE'][0]:.5f} | {r['p'][1]:.5f} ± {r['p_SE'][1]:.5f} |")
errors=[]
for d in [64,128,256,512]:
    rr=[r for r in rows if r['d']==d]
    errors.append((d,float(np.mean([abs(r['p'][k]-r['limit']) for r in rr for k in [0,1]]))))
lines += ['', '18 个方法/s/top-L 单元格的平均绝对偏差（描述性）为：'+ '，'.join(f'd={d}: {e:.5f}' for d,e in errors)+'。没有拟合收敛指数，没有根据某格是否单调筛选结果。', '',
'![三种方法、两个短列表](ac1_checks/AC1_topL.png)', '',
'## 5. 有限维阈值位移与训练投影修正', '',
'球面顺序阈值的边缘极限给 `b(T_L−b)⇒−log Γ_L`。据此作解释性均值位移 `−ψ(L)/b`：top-1 约 `+.5772/b`，top-5 约 `−1.5061/b`。相应预测 `Φ(s_eff+ψ(L)/b)` 是可行修正项，但本轮没有证明它对实际算法的带符号余项。其 `1/√log n` 收敛很慢；论文固定 κ、固定比例的 `B_R/d` 展开不能直接代入。', '',
'下表为 d=512、s≈0：球面+Gaussian 列保留真实 T 的分布而令正确分数独立 N(μ₀,1)；它是辅助模型预测。平滑残差列保留实际 `(λ,h_x,P_x)`，以 J 的均值代替 J，并删去根式中的 O(1/n) 项；其公式为 `E Φ((λ−T√((P_x+n−d−1)/n))/√(1−h_x))`，不是对实际概率误差的严格有限维上界。', '',
'| 方法 | L | 实际精确积分 | 均值阈值位移预测 | 球面+Gaussian | 保留投影的平滑残差 |', '|---|---:|---:|---:|---:|---:|']
for method in ['GD3','Sign2','Adam2']:
    r=lookup[(method,512,0.)]
    for k,L in enumerate([1,5]):
        lines.append(f"| {method} | {L} | {r['p'][k]:.5f} | {r['shift_pred'][k]:.5f} | {r['sphere_gauss'][k]:.5f} | {r['residual_smooth'][k]:.5f} |")
lines += ['', f'全部 72 个单元格中，平滑残差与精确积分的最大绝对差为 {max_smooth:.6f}（同组描述性比较；未另给联合误差认证）。这说明保留训练投影量能解释仅修正顺序阈值后留下的偏差；并不证明一个通用的有限维余项。', '',
'## 6. 完整数组与直接球面能量的独立交叉检查', '',
'小维度另生成完整 X、Y，重新计算 S、Z 和实际训练，再直接比较全部候选排名；和独立残差积分比较。每方法 d=16,n=150，768 份完整数据，每份 12 个训练查询。配对差的 SE 按完整数据集分组。', '',
'| 方法 | top-1 直接−积分 ± SE | top-5 直接−积分 ± SE |', '|---|---:|---:|']
for method,r in audit['full_array'].items():
    lines.append(f"| {method} | {r['paired_difference'][0]:+.6f} ± {r['dataset_SE'][0]:.6f} | {r['paired_difference'][1]:+.6f} ± {r['dataset_SE'][1]:.6f} |")
lines += ['', '另有 3 个固定合法条件的根式/直接 Gaussian 事件比较，每例 200000 次；差/SE 分别为 '+', '.join(f"{r['difference']/r['SE']:.2f}" for r in audit['root_checks'])+'。以上用于排查公式/代码错误，不证明渐近速率。', '',
'[AC1_bridge_audit.py](ac1_checks/AC1_bridge_audit.py) 另按完整 2d 维联合散布采样，每格 12 份独立完整散布、256 个条件球面查询，直接计算定理中的 D、V，没有使用残差根公式。下表对 3 方法合并，只描述有限维量级；bδ、bR 不必单调，不能以四个维度确认其极限。', '',
'| d | bδ 均值 | bR 均值 | b(√V−1) RMS 均值 | (D−μ_C) 条件 SD 均值 |', '|---:|---:|---:|---:|---:|']
for d in [64,128,256,512]:
    rr=[r for r in bridge['rows'] if r['d']==d]
    vals=[np.mean([r[f] for r in rr]) for f in ['b_delta','b_R','energy_scaled_RMS','centered_D_SD']]
    lines.append(f"| {d} | "+' | '.join(f'{x:.4f}' for x in vals)+' |')
lines += ['', '## 7. 资源、文件约束与复现', '',
'所有 Python 均由指定解释器经 `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0` 运行。初跑每库设 2 线程，CPU/墙钟比超过 2，无法据此认证进程总线程≤2；为保守处理，正式主结果重跑设置全部数值库单线程。完整散布审计也单线程，并实测进程 Threads=1。初跑保留为诊断记录，未用它代替正式结果。', '',
f"两次主结果的最大绝对差为 {max_replay:.3g}。正式结果文件为 [AC1_simulation.json](ac1_checks/serial/AC1_simulation.json)，全部分组原始行为 [AC1_groups.jsonl](ac1_checks/serial/AC1_groups.jsonl)。", '',
'| 计算 | 墙钟秒 | CPU 秒 | 峰值 RSS KiB | cgroup memory.max / swap.max |', '|---|---:|---:|---:|---|']
for name,obj in [('初跑（每库2线程，诊断）',initial),('正式主计算（单线程）',main),('完整散布审计（单线程）',bridge)]:
    r=obj['resources']
    lines.append(f"| {name} | {r['wall_seconds']:.2f} | {r['cpu_seconds']:.2f} | {r['max_RSS_KiB']} | {r.get('memory.max','NA')} / {r.get('memory.swap.max','NA')} |")
lines += ['', '复现时程序使用 exclusive-create，不覆盖已存在输出；应复制脚本到 `ac1_checks/` 内新的空子目录后运行。主脚本将输出到其自身所在目录。', '',
'```bash', 'systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \\', '  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\', '  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\', '  /home/liuyibo/venvs/theory/bin/python -B -u ac1_checks/serial/AC1_simulate.py', '```', '',
f"完整性记录 [AC1_integrity.json](ac1_checks/AC1_integrity.json)：初始清单 {len(manifest)} 个既有输入，当前 {integrity['unchanged_count']} 个哈希未变；变化文件为 "+', '.join('`'+x+'`' for x in integrity['changed_inputs'])+'。本轮所有显式写操作仅涉及新建 AC1_ 文件与 ac1_checks/；没有运行论文构建，也没有改写这些旧文件。变化为运行期间外部并发写入的观察，不能声称全项目无变动。四份指定根目录材料保持不变，本轮证明的路径依赖以这些文档为准。', '',
'数值为普通 float64，没有对根式/求积做向外舍入认证。理论已证部分不依赖模拟通过；未完成部分见证明稿 §13，尤其没有临界窗口单数据集集中或全查询成功定理。']
out=root.parent/'AC1_RESULTS.md'
with out.open('x') as f:f.write('\n'.join(lines)+'\n')
summary={'max_replay_difference':max_replay,'max_smooth_difference':max_smooth,'mean_absolute_errors':errors,
         'resources':{'wall_seconds':time.time()-wall,'cpu_seconds':time.process_time()-cpu,
                      'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
with (root/'AC1_report_summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary))
