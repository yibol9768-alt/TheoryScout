"""Recompute frozen criteria, draw PNGs, and generate the Chinese report."""
from core import *
os.environ.setdefault('MPLCONFIGDIR', '/tmp/theoryscout-round2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fmt(v):
    return f'{v:.4f}'


def main():
    begin = time.time()
    manifest = json.loads((OUT/'manifest.json').read_text())
    T = manifest['steps']
    records = [json.loads(p.read_text()) for p in sorted((OUT/'raw').glob('*.json'))]
    assert len(records)==90, f'Incomplete: {len(records)}/90'
    assert all(r['status']=='completed' and r['steps_completed']==manifest['steps'] for r in records)
    index = {(r['config']['n'],r['config']['k'],r['config']['d'],r['config']['seed'],r['config']['loss']):r
             for r in records}
    assert len(index)==90
    # Check primary endpoint aggregates directly against stored per-row observations.
    errors = []
    for r in records:
        f = r['final']
        gamma, zeta = np.array(f['gamma']), np.array(f['zeta'])
        errors += [abs((gamma>0).mean()-f['success_fraction']),
                   abs(zeta.min()-f['global_zero_sign_margin']),
                   float(np.max(abs(np.quantile(gamma,QUANTILES)-f['gamma_quantiles'])))]
        cal = r['calibration']['actual']
        assert all(abs(cal[key]['rms']/.001-1)<1e-6 for key in ('U','V'))
    assert max(errors)<1e-12
    pairing_ok = all(len({index[(n,k,d,s,l)]['initial_fingerprint'] for l in LOSSES})==1
                     for n,k,d in SETTINGS for s in range(10,15))
    assert pairing_ok
    hash_ok = {}
    for path, expected in manifest['hashes'].items():
        p = Path(path) if path.startswith('/') else ROOT/path
        hash_ok[path] = digest(p)==expected
    assert all(hash_ok.values())
    summaries = []
    for n,k,d in SETTINGS:
        group = {loss:[index[(n,k,d,s,loss)] for s in range(10,15)] for loss in LOSSES}
        finals = {loss:[r['final'] for r in group[loss]] for loss in LOSSES}
        delta = [p['success_fraction']-q['success_fraction'] for p,q in zip(finals['pe'],finals['nce'])]
        qdelta = [p['gamma_quantiles'][2]-q['gamma_quantiles'][2] for p,q in zip(finals['pe'],finals['nce'])]
        sdelta = [p['global_zero_sign_margin']-q['global_zero_sign_margin']
                  for p,q in zip(finals['sigmoid'],finals['nce'])]
        local = []
        for r in group['nce']:
            cfs = [cf for cf in r['counterfactuals'] if cf['step']>0]
            local.append(dict(seed=r['config']['seed'], checkpoints=len(cfs),
                              active_pair_positive_fraction=float(np.mean([cf['original_active_pair_effect']>1e-12 for cf in cfs])),
                              q10_positive_fraction=float(np.mean([cf['q10_effect']>1e-12 for cf in cfs])),
                              active_pair_effects=[cf['original_active_pair_effect'] for cf in cfs],
                              q10_effects=[cf['q10_effect'] for cf in cfs]))
        rank_pass = bool(np.mean(delta)>=.1 and sum(x>0 for x in delta)>=4
                         and np.mean(qdelta)>0 and sum(x>0 for x in qdelta)>=4)
        local_pass = sum(x['active_pair_positive_fraction']>.5 for x in local)>=4
        sign_pass = bool(np.mean(sdelta)>0 and sum(x>0 for x in sdelta)>=4)
        summaries.append(dict(n=n,k=k,d=d, success_deltas=delta,q10_deltas=qdelta,
                              sigmoid_sign_deltas=sdelta, rank_pass=rank_pass,
                              local_pass=local_pass, sign_pass=sign_pass,
                              local=local, means={loss:{key:float(np.mean([f[key] for f in fs])) for key in
                              ['success_fraction','recall_at_k','min_gap','global_zero_sign_margin',
                               'sign_row_success_fraction','sign_element_success_fraction','beta','positive_score_std']}
                              for loss,fs in finals.items()}))
    nce_cfs = [cf for r in records if r['config']['loss']=='nce'
               for cf in r['counterfactuals'] if cf['step']>0]
    positive_pair = sum(cf['original_active_pair_effect']>1e-12 for cf in nce_cfs)
    positive_q10 = sum(cf['q10_effect']>1e-12 for cf in nce_cfs)
    assessment = dict(settings=summaries, rank_prediction_pass=sum(x['rank_pass'] for x in summaries)>=2,
                      joint_rank_mechanism_pass=sum(x['rank_pass'] and x['local_pass'] for x in summaries)>=2,
                      independent_sigmoid_sign_prediction_pass=sum(x['sign_pass'] for x in summaries)>=2,
                      nce_noninitial_counterfactual_checkpoints=len(nce_cfs),
                      positive_active_pair_effect_count=positive_pair,
                      positive_q10_effect_count=positive_q10,
                      all_query_success_runs={loss:sum(r['final']['success_fraction']==1 for r in records
                                                      if r['config']['loss']==loss) for loss in LOSSES},
                      completed=len(records), pairing_ok=pairing_ok, hash_checks=hash_ok,
                      raw_metric_recompute_max_error=max(errors))
    save(OUT/'assessment.json', assessment)
    # Fixed setting panels, paired terminal comparisons.
    colors = dict(nce='#225ea8', pe='#238b45', sigmoid='#cb181d')
    fig, axes = plt.subplots(2,3,figsize=(13,7),sharey=True)
    for ax, summary in zip(axes.flat,summaries):
        n,k,d = (summary[x] for x in ('n','k','d'))
        for loss in LOSSES:
            values = [index[(n,k,d,s,loss)]['final']['success_fraction'] for s in range(10,15)]
            ax.plot(range(10,15),values,'o-',color=colors[loss],label=loss)
        ax.set(title=f'n={n}, k={k}, d={d}',xlabel='Paired seed',ylim=(-.02,1.03))
        ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Strict top-k success fraction')
    axes[1,0].set_ylabel('Strict top-k success fraction')
    axes[0,0].legend()
    fig.tight_layout()
    fig.savefig(OUT/'success.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,summary in zip(axes.flat,summaries):
        n,k,d=(summary[x] for x in ('n','k','d'))
        for loss in LOSSES:
            vals = np.array([index[(n,k,d,s,loss)]['final']['gamma_quantiles'] for s in range(10,15)])
            ax.plot(QUANTILES,vals.mean(0),color=colors[loss],label=loss)
        ax.axhline(0,color='black',lw=.6)
        ax.set(title=f'n={n}, k={k}, d={d}',xlabel='Quantile',ylabel='Row ranking margin')
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.tight_layout()
    fig.savefig(OUT/'margin_quantiles.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,summary in zip(axes.flat,summaries):
        n,k,d=(summary[x] for x in ('n','k','d'))
        for loss in LOSSES:
            histories=[index[(n,k,d,s,loss)]['history'] for s in range(10,15)]
            steps=[x['step'] for x in histories[0]]
            ys=np.array([[x['success_fraction'] for x in h] for h in histories])
            ax.plot(steps,ys.mean(0),color=colors[loss],label=loss)
            ax.fill_between(steps,ys.min(0),ys.max(0),color=colors[loss],alpha=.12)
        ax.set(title=f'n={n}, k={k}, d={d}',xlabel='Step',ylabel='Strict top-k success',ylim=(-.02,1.03))
        ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.tight_layout()
    fig.savefig(OUT/'trajectories.png',dpi=170)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=[f'{x[0]}/{x[1]}/{x[2]}' for x in SETTINGS]
    for j,summary in enumerate(summaries):
        for local in summary['local']:
            axes[0].scatter(j+(local['seed']-12)*.04,local['active_pair_positive_fraction'],color=colors['nce'])
        for loss in LOSSES:
            n,k,d=(summary[x] for x in ('n','k','d'))
            values=[index[(n,k,d,s,loss)]['final']['global_zero_sign_margin'] for s in range(10,15)]
            axes[1].scatter(np.full(5,j)+{'nce':-.18,'pe':0,'sigmoid':.18}[loss],values,
                            color=colors[loss],label=loss if j==0 else None,s=18)
    axes[0].axhline(.5,color='black',ls='--',lw=.7)
    axes[1].axhline(0,color='black',lw=.7)
    for ax in axes:
        ax.set_xticks(range(6),labels,rotation=20)
        ax.set_xlabel('n / k / d')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Positive no-D effect\n(fraction of checkpoints)')
    axes[1].set_ylabel('Global zero-threshold\ngeometric sign margin')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT/'counterfactual_and_sign.png',dpi=170)
    plt.close(fig)

    execution=json.loads((OUT/'execution.json').read_text())
    decision=json.loads((OUT/'budget_decision.json').read_text())
    checks=json.loads((OUT/'validation.json').read_text())
    lines=['# 第二轮结果：对齐更新后的多正例检索（中文）','',
           f'本轮完成 **90/90 次 × {manifest["steps"]:,} 步**，六个设置、配对新种子10–14、三种损失。'
           f'排序预测通过 {sum(x["rank_pass"] for x in summaries)}/6 个设置；'
           f'局部机制单独通过 {sum(x["local_pass"] for x in summaries)}/6；'
           f'排序与局部机制共同通过 {sum(x["rank_pass"] and x["local_pass"] for x in summaries)}/6；'
           f'sigmoid 独立符号预测通过 {sum(x["sign_pass"] for x in summaries)}/6。', '',
           '## 设计、预算与有效性','',
           '先写 [PREREG_ROUND2.md](PREREG_ROUND2.md)，然后只做计时与实现核验，最后冻结哈希启动主实验。'
           '三个损失均用可训练 beta=exp(a)、初始beta=16；sigmoid偏置固定为0；Adam后重投影、余弦学习率。'
           '查询与文档分别求根，使初始每坐标RMS位移都为0.001，无学习率缩放夹界。'
           '这对齐两组初始位移，不保证后续每步位移或温度轨迹相同，也不是后续动力学逐步匹配。','',
           f'18段1000步计时总计 {sum(x["seconds"] for x in decision["timing"]):.3f} 秒；'
           f'100000步保守单进程估计 {decision["conservative_serial_estimate_100k_seconds"]:.2f} 秒，'
           f'按预先6000秒主训练配额统一冻结为{T}步（{T/100000:.0%}）。此规则不预支并行加速，并为核验与汇总预留时间。'
           '这是预算缩减版，不声称执行了原论文100000步完整复现。','',
           f'监督进程加两个工作进程，主训练墙钟 {execution["wall_seconds"]:.2f} 秒，'
           f'90次运行耗时合计 {sum(r["elapsed_seconds"] for r in records):.2f} 秒；'
           f'实验工作进程峰值RSS {max(r["memory"]["VmHWM"] for r in records)/1e6:.2f} MB，'
           f'峰值虚拟地址 {max(r["memory"]["VmPeak"] for r in records)/1e6:.2f} MB，'
           '每进程地址空间硬限3500000000字节。OMP_NUM_THREADS=3；NumPy BLAS单线程；无外部API、网络或GPU。'
           f'汇总进程峰值RSS {memory()["VmHWM"]/1e6:.2f} MB。'
           '总执行资源见 [reporting_resources.json](experiments/round2/reporting_resources.json)，时间范围含计时、主训练、汇总。','',
           f'90/90均完成，无预算截断和温度异常；90/90查询与文档校准均满足相对误差1e-6。'
           f'三损失配对初值指纹一致；主实验前登记的所有文件哈希仍一致。'
           f'解析梯度有限差分最大绝对误差 {max(checks["gradient_max_abs_errors"].values()):.3g}，'
           f'D梯度误差 {checks["D_gradient_max_abs_error"]:.3g}，分解残差 {checks["decomposition_error"]:.3g}。'
           '单步反事实与真实Adam更新一致，且不污染主轨迹；原始逐行间隔可重算全部终态汇总。','',
           '## 主要终态结果','',
           '每格为五个配对种子的均值；成功定义为行内间隔严格大于0。几何间隔均取未缩放单位向量内积。','',
           '| n,k,d | NCE成功 | PE成功 | sigmoid成功 | PE−NCE均值 | 同向种子 | q10差均值 | 排序通过 |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for s in summaries:
        lines.append(f'| {s["n"]},{s["k"]},{s["d"]} | '+
                     ' | '.join(fmt(s['means'][l]['success_fraction']) for l in LOSSES)+
                     f' | {np.mean(s["success_deltas"]):+.4f} | {sum(x>0 for x in s["success_deltas"])}/5 | '
                     f'{np.mean(s["q10_deltas"]):+.6f} | {"是" if s["rank_pass"] else "否"} |')
    lines += ['', '审计门槛按预注册细化为：配对成功比例差均值≥0.10、至少4/5种子同向、q10均值改善且至少4/5同向。'
              '未要求sigmoid同时胜出，也未要求最差行必然改善。下表保留全部配对差值，避免均值隐藏异质性。','',
              '| n,k,d | seed10 | seed11 | seed12 | seed13 | seed14 | 达到+0.10的种子数 | q10同向种子 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f'| {s["n"]},{s["k"]},{s["d"]} | '+' | '.join(f'{v:+.6f}' for v in s['success_deltas'])+
                     f' | {sum(v>=.1 for v in s["success_deltas"])}/5 | {sum(v>0 for v in s["q10_deltas"])}/5 |')
    lines += ['', '![配对成功比例](experiments/round2/success.png)', '',
              '## 行内间隔与零阈值符号间隔','',
              '两种间隔分别评价排序与统一零阈值分类。每个设置的最差间隔先在单次运行内取min，再对五种子取均值；'
              '它不是合并所有种子的共同最小值。详细分位数及每行数据在raw JSON。','',
              '| n,k,d | 损失 | 最差行内γ | q10 γ均值 | 全局零阈值符号间隔 | 符号全对行比例 | 元素符号正确率 | beta均值 | recall@k |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        n,k,d=(s[x] for x in ('n','k','d'))
        for loss in LOSSES:
            m=s['means'][loss]
            q10=np.mean([index[(n,k,d,seed,loss)]['final']['gamma_quantiles'][2] for seed in range(10,15)])
            lines.append(f'| {n},{k},{d} | {loss} | {m["min_gap"]:.6f} | {q10:.6f} | '
                         f'{m["global_zero_sign_margin"]:.6f} | {m["sign_row_success_fraction"]:.4f} | '
                         f'{m["sign_element_success_fraction"]:.4f} | {m["beta"]:.2f} | {m["recall_at_k"]:.4f} |')
    lines += ['', 'sigmoid符号优势独立判定：'+
              '；'.join(f'{s["n"]}/{s["k"]}/{s["d"]}：均值差{np.mean(s["sigmoid_sign_deltas"]):+.6f}，'
                       f'{sum(x>0 for x in s["sigmoid_sign_deltas"])}/5同向，{"通过" if s["sign_pass"] else "未通过"}'
                       for s in summaries)+'。相对符号优势允许两种间隔都为负，不能解释成所有标签已经零阈值分离。','',
              '![间隔分位数](experiments/round2/margin_quantiles.png)','',
              '## 同检查点的真实Adam单步反事实','',
              '每次分叉复制相同U,V,a、Adam一二阶矩及步号，分别输入∇NCE与∇NCE−∇D，按当时余弦lr更新并重投影。'
              '包含温度梯度移除；这是保留既有Adam历史的瞬时干预，不是从头训练无D的轨迹，也不等同于PE损失。'
              '主要机制统计只使用NCE轨迹的非初始、非终态检查点。方向改善阈值为无D减有D>1e-12。','',
              '| n,k,d | 每种子活跃最差对改善检查点比例（10–14） | 超过一半的种子数 | q10改善比例均值 | 局部通过 |',
              '|---|---|---:|---:|---|']
    for s in summaries:
        lines.append(f'| {s["n"]},{s["k"]},{s["d"]} | '+
                     ' / '.join(f'{v["active_pair_positive_fraction"]:.3f}' for v in s['local'])+
                     f' | {sum(v["active_pair_positive_fraction"]>.5 for v in s["local"])}/5 | '
                     f'{np.mean([v["q10_positive_fraction"] for v in s["local"]]):.3f} | '
                     f'{"是" if s["local_pass"] else "否"} |')
    compare_steps = [1000, min(10000,T), round(T*.5), T]
    lines += ['', f'汇总全部{len(nce_cfs)}个NCE非初始检查点，移除D改善原活跃最差对的有'
              f'{positive_pair}/{len(nce_cfs)}（{positive_pair/len(nce_cfs):.1%}），改善q10的有'
              f'{positive_q10}/{len(nce_cfs)}（{positive_q10/len(nce_cfs):.1%}）。这些检查点不是独立种子。','',
              '![反事实与符号间隔](experiments/round2/counterfactual_and_sign.png)','',
              '## 训练延长、追平与判据对照','',
              '以下仅用预先固定检查点，不选择最有利时点。每格为PE−NCE配对成功比例差均值。','',
              '| n,k,d | '+' | '.join(f'{step}步' for step in compare_steps)+' |','|---|---:|---:|---:|---:|']
    for s in summaries:
        n,k,d=(s[x] for x in ('n','k','d'))
        values=[]
        for step in compare_steps:
            def get(seed,loss):
                return next(h['success_fraction'] for h in index[(n,k,d,seed,loss)]['history'] if h['step']==step)
            values.append(np.mean([get(seed,'pe')-get(seed,'nce') for seed in range(10,15)]))
        lines.append(f'| {n},{k},{d} | '+' | '.join(f'{v:+.6f}' for v in values)+' |')
    lines += ['', '![训练轨迹](experiments/round2/trajectories.png)','',
              '## 判断与下一轮建议','',
              ((OUT/'interpretation.md').read_text() if (OUT/'interpretation.md').exists() else
               '<!-- INTERPRETATION: add evidence-based discussion here after reading frozen results. -->'),'',
              '## 产物与复现','',
              '- 预注册：[PREREG_ROUND2.md](PREREG_ROUND2.md)。',
              '- 原始结果：[raw/](experiments/round2/raw/)（90个JSON，含初末逐行间隔、检查点指标、逐行反事实差值）。',
              '- 冻结与资源：[manifest.json](experiments/round2/manifest.json)、[budget_decision.json](experiments/round2/budget_decision.json)、[execution.json](experiments/round2/execution.json)。',
              '- 独立重算判据：[assessment.json](experiments/round2/assessment.json)；数值核验：[validation.json](experiments/round2/validation.json)。','',
              '复现需在新目录保留旧结果，依次使用指定Python运行 experiments/round2/prepare.py、run.py、summarize.py；'
              'prepare会按本机1000步实测重新冻结步数并追加预注册，因此跨机器步数可能变化。'
              '若仅重算本轮已冻结结果，直接运行summarize.py。主训练前清单会核对数据与核心代码哈希。','']
    (ROOT/'RESULTS_ROUND2.md').write_text('\n'.join(lines))
    import scipy
    import platform
    save(OUT/'reporting_resources.json',dict(report_seconds=time.time()-begin,
          memory=memory(), total_since_timing_seconds=time.time()-decision['preparation_start_epoch'],
          python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
          matplotlib=matplotlib.__version__, platform=platform.platform(),
          report_code_sha256=digest(OUT/'summarize.py')))
    print(json.dumps(dict(completed=len(records), rank_pass=assessment['rank_prediction_pass'],
                         joint_pass=assessment['joint_rank_mechanism_pass'],
                         sign_pass=assessment['independent_sigmoid_sign_prediction_pass'])),flush=True)


if __name__=='__main__':
    main()
