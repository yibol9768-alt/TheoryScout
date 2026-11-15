"""Render AC2 full-data experiments without modifying inputs."""
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
result=json.loads((here/'full/AC2_results.json').read_text())
audit=json.loads((here/'full/AC2_audit.json').read_text())
budget=json.loads((here/'AC2_root_budget.json').read_text())
integrity=json.loads((here/'full/AC2_integrity.json').read_text())
raw=[json.loads(x) for x in (here/'full/AC2_datasets.jsonl').read_text().splitlines()]
rows=result['rows']
assert len(raw)==720 and len(rows)==12
for row in rows:
    rr=[r for r in raw if (r['d'],r['method'])==(row['d'],row['method'])]
    assert len(rr)==row['datasets']
    assert len({r['rep'] for r in rr})==len(rr)
    for r in rr:
        assert np.allclose(np.array(r['counts'])/r['n'],r['recall'],rtol=0,atol=0)
    vals=np.array([r['recall'] for r in rr])
    assert np.allclose(vals.var(axis=0,ddof=1),row['dataset_variance'],rtol=1e-14)

with (here/'AC2_summary.csv').open('x',newline='') as f:
    w=csv.writer(f)
    w.writerow(['method','d','n','datasets','L','recall','dataset_variance','variance_bootstrap_low','variance_bootstrap_high','n_variance'])
    for r in rows:
        for k,L in enumerate((1,5)):
            w.writerow([r['method'],r['d'],r['n'],r['datasets'],L,r['mean'][k],r['dataset_variance'][k],
                        *r['bootstrap_variance_95'][k],r['n_times_variance'][k]])

fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for k,L in enumerate((1,5)):
    for method in ('GD3','Sign2','Adam2'):
        rr=[r for r in rows if r['method']==method]
        x=np.array([r['n'] for r in rr]); y=np.array([r['dataset_variance'][k] for r in rr])
        ci=np.array([r['bootstrap_variance_95'][k] for r in rr])
        axes[k].errorbar(x,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),marker='o',capsize=3,label=method)
    axes[k].set_xscale('log'); axes[k].set_yscale('log')
    axes[k].set_title(f'All training queries: top-{L}')
    axes[k].set_xlabel('Number of associations n')
    axes[k].grid(alpha=.2,which='both')
axes[0].set_ylabel('Variance across independent datasets')
axes[1].legend()
fig.tight_layout()
for ext in ('png','pdf'):
    path=here/f'AC2_dataset_variance.{ext}'
    assert not path.exists()
    fig.savefig(path,dpi=180)
plt.close(fig)

lines=['# AC2：独立完整数据集召回方差实验','',
       '2026-11-06。理论见 [AC2_CONCENTRATION.md](AC2_CONCENTRATION.md)。程序独立实现，不导入历史代码；每份训练集的全部 n 个查询都直接对真实候选排名。没有条件查询抽样噪声，没有根式例外筛选。','',
       '## 1. 设计、结果和证据等级','',
       '预先固定 s≈0、d=64–512、GD3（η=.7）、Sign2（α=.6）、Adam2（α=.3,e=1,β₁=.9,β₂=.99）。三个效率系数独立一维积分为 1、.8998627412、.9674342162。每格的数据集数 96、64、48、32。共 720 份独立完整训练集。分块只在超过正确分数的候选数已达 5 时停止该查询，因而 top-1/top-5 结果均精确（通常 float64 算术意义）。','',
       '| 方法 | d | n | 数据集数 | top-1 召回 ± SE | top-5 召回 ± SE | Var(top-1) | Var(top-5) |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['method']} | {r['d']} | {r['n']} | {r['datasets']} | {r['mean'][0]:.6f} ± {r['SE'][0]:.6f} | {r['mean'][1]:.6f} ± {r['SE'][1]:.6f} | {r['dataset_variance'][0]:.6g} | {r['dataset_variance'][1]:.6g} |")
lines+=['',f"方差均是不同完整数据集召回比例的样本方差，使用 B−1 分母；SE=sd/√B。各维度使用独立样本，未根据结果筛选数据集。累计直接排名 {sum(r['n'] for r in raw):,} 个训练查询，两个列表指标共用这些真实分数。",'',
        '![独立完整数据集方差](ac2_checks/AC2_dataset_variance.png)','',
        '## 2. 方差下降与 n 缩放','',
        '| 方法 | L | Var(d=64)/Var(d=512) | 各 d 的 n×Var | d=512 方差的描述性 bootstrap 95% 区间 |',
        '|---|---:|---:|---|---|']
ratios=[]
for method in ('GD3','Sign2','Adam2'):
    rr=[r for r in rows if r['method']==method]
    for k,L in enumerate((1,5)):
        ratio=rr[0]['dataset_variance'][k]/rr[-1]['dataset_variance'][k]
        ratios.append(ratio)
        nv=', '.join(f"{r['n_times_variance'][k]:.4f}" for r in rr)
        lo,hi=rr[-1]['bootstrap_variance_95'][k]
        lines.append(f'| {method} | {L} | {ratio:.2f} | {nv} | [{lo:.6g}, {hi:.6g}] |')
monotone=all(all(rr[i]['dataset_variance'][k]>rr[i+1]['dataset_variance'][k] for i in range(3))
    for k in range(2) for rr in [[r for r in rows if r['method']==m] for m in ('GD3','Sign2','Adam2')])
lines+=['',f'六条方差序列全部随 d 严格下降的检查结果为 {monotone}，d=64 到 512 的下降倍数介于 {min(ratios):.2f} 与 {max(ratios):.2f}。这是沿 s≈0 临界负载、d 与 n 共同增长的比较；没有固定 d 单独扫描 n，不能据此声称方差对任意负载都单调。',
        '', 'n×Var 只作描述性尺度诊断；这些数据与较快于理论保守上界的收缩相容，不证明 O(1/n)，也不证明查询独立。事实上以 Var(R)=p(1−p)/n+(n−1)Cov(I₁,I₂)/n 作插值会得到较小的负协方差估计；这是共同数据约束允许的现象。没有根据四个维度拟合并宣传渐近指数。',
        '', 'bootstrap 仅在各格完整数据集之间重采样 2000 次，是描述性百分位区间；未证明其有限样本覆盖率。每格 B≤96，方差估计仍有明显不确定性。',
        '', 'top-1 均值仍低于 .5、top-5 仍高于 .5。方差下降和均值的有限维偏移可以同时存在；这些维度不是 Φ(s) 的定量误差认证。','',
        '## 3. 更正后的球面能量审计','',
        '每方法每维 12 个独立完整散布、每散布 256 个条件球面查询。下表合并三个方法；使用正确的 Au 列能量，未用转置行能量冒充逐样本目标。','',
        '| d | 平均 bδ | 平均 bR | 平均 b(√V−1) RMS | 平均条件 SD(D−μ) |',
        '|---:|---:|---:|---:|---:|']
for d in (64,128,256,512):
    ee=[r for r in audit['corrected_energy_rows'] if r['d']==d]
    vals=[np.mean([r[key] for r in ee]) for key in ('b_delta','b_R','energy_RMS','centered_D_SD')]
    lines.append('| '+str(d)+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
ids=audit['pointwise_identities']
lines+=['','小维完整矩阵逐样本核对：','',
        '| 方法 | 正确 Au 能量最大绝对误差 | 转置 Aᵀu 的最大逐样本偏差 | 矩阵束同谱最大误差 |',
        '|---|---:|---:|---:|']
for r in ids:
    lines.append(f"| {r['method']} | {r['max_Au_error']:.3g} | {r['max_ATu_error']:.6f} | {max(r['pencil_errors']):.3g} |")
lines+=['','逐样本列能量恒等式通过；转置量逐样本确实不同。矩阵束谱检查只是对联合同分布证明的数值交叉检查，不是证明。',
        '',f"有效根式分支的独立直接 Gaussian 事件与积分差为 {audit['root_direct_difference']:.6g}，按独立样本 SE 为 {audit['root_direct_SE']:.6g}。负 T、H≤0 等异常分支均返回 [0,1]，未删样本。",
        '',f"全部 AC1 网格的 F5 异常概率浮点上界最大值为 {max(r['root_exception_union_bound'] for r in budget['rows']):.6g}；99% 同时 Hoeffding 半宽为 {budget['simultaneous_99_Hoeffding_halfwidth']:.6f}。前者没有机器向外舍入认证，严格统计处理采用 AC2_AC1_FIXES 的处处定义上下界估计量，不依赖这个极小数字。",'','## 4. 复核、资源与文件完整性','']
res=result['resources']
lines += [f"正式主计算（不含末次文件哈希遍历）墙钟 {res['wall_seconds']:.2f} 秒，CPU {res['cpu_seconds']:.2f} 秒，截至该时刻峰值 RSS {res['max_RSS_KiB']} KiB；实测 `{res['process_threads']}`，cgroup memory.max={res.get('memory.max')}，memory.swap.max={res.get('memory.swap.max')}。",'',
          f"主运行开始和结束核对六份输入及 paper/ 全部现存文件，共 {integrity['checked_files']} 个哈希；变化列表为 `{integrity['changed']}`。本轮的显式写入仅在 AC2_ 文件与 ac2_checks/。",'',
          '原始记录逐行复核 720 份数据集：组号无重复，整数成功数除 n 与比例完全一致，各格方差从原始行重新计算一致。程序自带小维完整分数矩阵与分块排名的交叉核对。', '',
          '初次审计运行因审计代码对 np.diag 视图作原地对角修改而主动失败，未生成主模拟行。已用 copy 修正本轮新代码，并在 full/ 新目录从头独立运行；失败日志保留。没有覆写旧实验。','',
          '正式运行命令：','', '```bash',
          'systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \\',
          '  env AC2_OUTPUT_DIR=ac2_checks/full OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\',
          '  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\',
          '  BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\',
          '  /home/liuyibo/venvs/theory/bin/python -B -u ac2_checks/AC2_independent.py',
          '```','',
          '程序输出使用 exclusive-create；复跑应指定新的空输出目录，不能直接覆盖 full/。', '',
          '入口：[设计](ac2_checks/full/AC2_design.json)、[逐数据集原始行](ac2_checks/full/AC2_datasets.jsonl)、[汇总 JSON](ac2_checks/full/AC2_results.json)、[CSV](ac2_checks/AC2_summary.csv)、[审计](ac2_checks/full/AC2_audit.json)、[完整性](ac2_checks/full/AC2_integrity.json)、[运行日志](ac2_checks/AC2_full_run.log)。']
with (here.parent/'AC2_NUMERICAL_RESULTS.md').open('x') as f:
    f.write('\n'.join(lines)+'\n')
with (here/'AC2_report_summary.json').open('x') as f:
    json.dump(dict(raw_datasets=len(raw),all_six_variance_series_decreasing=monotone,
                   variance_reduction_ratios=ratios,total_original_queries=sum(r['n'] for r in raw),
                   integrity_changed=integrity['changed']),f,indent=2)
print(json.dumps(dict(monotone=monotone,ratios=ratios,resources=res,integrity_changed=integrity['changed'])))
