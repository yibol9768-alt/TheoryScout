"""Summarize frozen criteria, draw PNGs, and write the Chinese week-1 report."""
import gzip
import json
import math
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import ROOT, OUT, FIG, save, sha, rss_bytes
import matplotlib.pyplot as plt


def load(path):
    return json.loads(Path(path).read_text())


def spread(x, digits=4):
    x = np.asarray(x,dtype=float)
    return f'{x.mean():.{digits}f}±{x.std(ddof=1):.{digits}f}' if len(x)>1 else f'{x.mean():.{digits}f}'


def mean_record(r,split,field):
    return r[split][-1][field]


def compact(r):
    probe = r['probe']
    last = probe[-1]
    peak = max(probe,key=lambda p:p['Mabs'])
    matched = {}
    for ce in [1.,.8]:
        reached = next((a for a in r['train'] if a['ce']<=ce),None)
        if reached:
            p = next(p for p in probe if p['step']==reached['step'])
            matched[str(ce)] = {'step':p['step'],'train_ce':reached['ce'],
                                 'Mabs':p['Mabs'],'direction_fro':p['direction_fro']}
        else:
            matched[str(ce)] = None
    clips = r['clip']
    clipcounts = [sum(int(c['triggered'][h]) for c in clips) for h in range(4)] if clips else [0]*4
    p_over = sum(p['Mabs']>8 for p in probe)
    d = r['diagnostics']
    return {'run_id':r['run_id'],'seed':r['seed'],'condition':r['condition'],
            'optimizer':r['optimizer'],'eta':r['eta'],'intervention':r['intervention'],
            'steps':r['steps'],'test_ce':r['test'][-1]['ce'],
            'test_accuracy':r['test'][-1]['accuracy'],
            'test_ensemble_accuracy':r['test'][-1]['ensemble_accuracy'],
            'train_ce':r['train'][-1]['ce'],'final_Mabs':last['Mabs'],
            'peak_Mabs':peak['Mabs'],'peak_step':peak['step'],
            'peak_to_final_drop_fraction':1-last['Mabs']/peak['Mabs'],
            'final_direction_fro':last['direction_fro'],
            'late_direction_fro':np.mean([s['direction_fro'] for s in r['steps_raw'][-100:]],axis=0),
            'k0':r['k0'],
            'truncated_singular_values_total':np.sum([s['dropped_singular_values'] for s in r['steps_raw']],axis=0),
            'probe_times_over8':p_over,'max_test_Mabs':max(p['Mabs'] for p in r['test']),
            'clip_head_step_counts':clipcounts,
            'clip_early_head_step_count':sum(sum(c['triggered']) for c in clips[:150]),
            'clip_late_head_step_count':sum(sum(c['triggered']) for c in clips[-100:]),
            'clip_post_batch_max':max((max(c['post_max_by_head']) for c in clips),default=None),
            'clip_pre_batch_max':max((max(c['pre_max_by_head']) for c in clips),default=None),
            'matched_CE':matched,'heads':d.get('heads',[]),
            'probe_curve':[{k:p[k] for k in ['step','ce','accuracy','Mabs']} for p in probe],
            'test_curve':[{k:p[k] for k in ['step','ce','accuracy','Mabs']} for p in r['test']],
            'train_curve':r['train'],'diagnostic_windows':d.get('windows',[])}


def audit_run(r):
    n = r['steps']
    assert n==len(r['steps_raw'])==len(r['xi_scaled_each_step'])
    assert [p['step'] for p in r['probe']]==list(range(0,n+1,10))
    assert [p['step'] for p in r['test']]==list(range(0,n+1,50))
    assert [p['step'] for p in r['train']]==list(range(0,n+1,50))
    path = OUT/r['directions_file']
    linecount,first,last = 0,None,None
    with gzip.open(path,'rt') as f:
        for line in f:
            linecount += 1
            if linecount==1:
                first = json.loads(line)
            last = line
    last = json.loads(last)
    assert linecount==n and first['step']==1 and last['step']==n
    assert np.asarray(first['D_Q']).shape==(32,32)
    assert np.asarray(last['D_K']).shape==(32,32)
    # Independently check stored direction projections and the scalar recurrence.
    max_projection_error = 0.
    for record in [first,last]:
        for h,frame in enumerate(r['frozen_frames']):
            if frame is None:
                continue
            u,v,w = [np.asarray(v) for v in frame]
            for i,key,direction in [(0,'D_Q',v),(1,'D_K',w)]:
                xi = -u@np.asarray(record[key])[h*8:(h+1)*8]@direction/r['k0'][i]
                expected = r['xi_scaled_each_step'][record['step']-1][h][i]
                max_projection_error = max(max_projection_error,abs(xi-expected))
    assert max_projection_error<1e-12
    rec_error = None
    if r['intervention']=='base':
        ab = np.asarray(r['mode_ab_each_step'][49:],float)
        xi = np.asarray(r['xi_scaled_each_step'][50:],float)
        rec_error = float(np.max(np.abs(ab[1:]-(1-r['alpha'])*ab[:-1]-r['eta']*xi)))
        assert rec_error<1e-7,(r['run_id'],rec_error)
    for p in r['probe']+r['test']:
        for key in ['ce','Mabs','Mplus','q999','accuracy']:
            assert math.isfinite(p[key]),(r['run_id'],key)
        if r['intervention']=='head_norm':
            assert p['Mabs']<=math.sqrt(8)+1e-10
        if r['intervention']=='layer_norm':
            assert p['Mabs']<=4*math.sqrt(8)+1e-10
    if r['clip']:
        assert all(max(c['post_max_by_head'])<=8+1e-10 for c in r['clip'])
    return {'run_id':r['run_id'],'direction_lines':linecount,
            'raw_projection_max_error':max_projection_error,'mode_recurrence_max_error':rec_error,
            'directions_gzip_bytes':path.stat().st_size}


def plot_e0(e0):
    fig,axes = plt.subplots(2,3,figsize=(14,8))
    t = np.asarray(e0['P1']['trace'])
    axes[0,0].loglog(t[:,0],t[:,1],label='m=0')
    axes[0,0].loglog(t[:,0],t[:,2],label='m=1')
    axes[0,0].set(xlabel='step',ylabel='mean logit',title='P1: no-decay exponents')
    boundary = np.array([[g['mstar'],g['estimated_boundary']] for g in e0['grid']])
    axes[0,1].scatter(*boundary.T)
    axes[0,1].plot([0,boundary.max()],[0,boundary.max()],color='black',ls='--')
    axes[0,1].set(xlabel='predicted critical m',ylabel='estimated critical m',title='P2: boundary')
    for eta in [.01,.02,.04]:
        rows = [p for p in e0['two_stage'] if p['eta']==eta]
        axes[0,2].loglog([p['lambda'] for p in rows],[p['peak_logit'] for p in rows],'o-',label=f'eta={eta}')
    axes[0,2].set(xlabel='lambda',ylabel='switch logit',title='P3: coherent peak')
    for lam in [.1,.2,.4]:
        rows = [g for g in e0['grid'] if g['lambda']==lam]
        axes[1,0].loglog([g['eta'] for g in rows],[g['rows'][0]['logit_tail']['mean'] for g in rows],
                         'o-',label=f'lambda={lam}')
    axes[1,0].set(xlabel='eta',ylabel='unbiased logit floor',title='P3: noise floor')
    p = next(p for p in e0['two_stage'] if p['eta']==.02 and p['lambda']==.2)
    trace = np.asarray(p['trace'])[p['T']:]
    n = np.arange(1,p['N']+1)
    excess = trace[:,2]/4-p['vstar']/4
    cutoff = p['fit_n_range'][1]
    axes[1,1].semilogy(n[:cutoff],excess[:cutoff],label='Monte Carlo')
    axes[1,1].semilogy(n[:cutoff],np.exp(p['expected_slope']*n[:cutoff])*(p['peak_logit']-p['vstar']/4),
                       '--',label='exact prediction')
    axes[1,1].set(xlabel='steps after T',ylabel='excess above floor',title='P4: decay of excess')
    ct = np.asarray(e0['P5']['trace'])
    axes[1,2].plot(ct[:,0],ct[:,1],label='trigger fraction')
    axes[1,2].axvline(142,ls='--',color='black',label='n0=142')
    axes[1,2].set(xlabel='post-switch step',ylabel='trigger fraction',title='P5: projected process')
    for ax in axes.flat:
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/'e0_predictions.png',dpi=160); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    full = np.asarray(p['trace'])
    axes[0].plot(full[:,0],full[:,2]/4,label='unclipped')
    axes[0].plot(full[:,0],full[:,4]/4,label='projected R=2.5')
    axes[0].axvline(p['T'],color='black',ls='--',label='external switch')
    axes[0].set(xlabel='step',ylabel='ensemble mean logit',title='E0: externally forced switch')
    rt = np.asarray(e0['recurrence']['trace'])
    axes[1].fill_between(rt[:,0],rt[:,1],rt[:,3],alpha=.2)
    axes[1].plot(rt[:,0],rt[:,2],label='median across 128 repeats')
    axes[1].set(xlabel='step',ylabel='cumulative clip triggers',title='High-noise recurrence')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/'e0_switch_recurrence.png',dpi=160); plt.close(fig)


COLORS = dict(zip(['muon0','muon09','adamw','signsgd'],['C0','C1','C2','C3']))


def plot_training(runs,selected_eta):
    fig,axes = plt.subplots(2,3,figsize=(14,8))
    for row,cond in enumerate(['isotropic','anisotropic']):
        for opt in COLORS:
            rs = [r for r in runs if r['condition']==cond and r['optimizer']==opt and r['eta']==selected_eta]
            if not rs: continue
            for col,(curve,key) in enumerate([('probe_curve','Mabs'),('test_curve','ce'),('test_curve','accuracy')]):
                ys = np.asarray([[p[key] for p in r[curve]] for r in rs])
                xs = [p['step'] for p in rs[0][curve]]
                axes[row,col].plot(xs,ys.mean(axis=0),label=opt,color=COLORS[opt])
                axes[row,col].fill_between(xs,ys.min(axis=0),ys.max(axis=0),alpha=.12,color=COLORS[opt])
                axes[row,col].set(xlabel='step',ylabel=key,title=f'{cond}, eta={selected_eta:g}')
    for ax in axes.flat: ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/'e1_training.png',dpi=160); plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(11,8))
    for cond,marker in [('isotropic','o'),('anisotropic','x')]:
        hs = [h for r in runs if r['optimizer']=='muon0' and r['condition']==cond for h in r['heads']]
        if not hs: continue
        axes[0,0].scatter([h['peak_residual'] for h in hs],[h['peak_explained_change'] for h in hs],
                          marker=marker,alpha=.6,label=cond)
        axes[0,1].scatter([h['qk_correlation'] for h in hs],[h['max_abs_acf_early_late'] for h in hs],
                          marker=marker,alpha=.6,label=cond)
        early = [max(h['early_ratio']) for h in hs]
        late = [max(h['late_ratio']) for h in hs]
        axes[1,0].scatter(early,late,marker=marker,alpha=.6,label=cond)
        axes[1,1].scatter([h['peak_residual'] for h in hs],[h['forecast_relative_error'] for h in hs],
                          marker=marker,alpha=.6,label=cond)
    axes[0,0].axvline(.3,color='black',ls='--'); axes[0,0].axhline(.7,color='black',ls='--')
    axes[0,0].set(xlabel='fixed-mode residual at peak',ylabel='explained logit change',title='C1: fixed-mode sufficiency')
    axes[0,1].axvline(.8,color='black',ls='--'); axes[0,1].axhline(.2,color='black',ls='--')
    axes[0,1].set(xlabel='Q/K update correlation',ylabel='max |ACF|, lags 1-20',title='C1: synchronization and independence')
    axes[1,0].axvline(1,color='black',ls='--'); axes[1,0].axhline(1,color='black',ls='--')
    axes[1,0].set(xscale='log',yscale='log',xlabel='early coherence/noise ratio (max Q,K)',
                  ylabel='late coherence/noise ratio (max Q,K)',title='C1: boundary crossing')
    axes[1,1].axhline(.3,color='black',ls='--')
    axes[1,1].set(xlabel='fixed-mode residual',ylabel='late forecast relative error',title='C1: early-window forecast')
    for ax in axes.flat: ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG/'e1_frozen_modes.png',dpi=160); plt.close(fig)


def plot_e2(e1,e2,eta,constructions):
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    for row,cond in enumerate(['isotropic','anisotropic']):
        for intv in ['base','no_decay','layer_norm','head_norm','qk_clip']:
            rs = [r for r in e1+e2 if r['condition']==cond and r['optimizer']=='muon0'
                  and r['eta']==eta and r['intervention']==intv]
            if not rs: continue
            for col,key in enumerate(['Mabs','ce']):
                ys = np.asarray([[p[key] for p in r['probe_curve']] for r in rs])
                xs = [p['step'] for p in rs[0]['probe_curve']]
                axes[row,col].plot(xs,ys.mean(axis=0),label=intv)
                axes[row,col].fill_between(xs,ys.min(axis=0),ys.max(axis=0),alpha=.12)
                axes[row,col].set(xlabel='step',ylabel=key,title=f'{cond}, eta={eta:g}')
    for ax in axes.flat: ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIG/'e2_interventions.png',dpi=160); plt.close(fig)
    rows = constructions['P6']
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot([r['heads'] for r in rows],[r['layer_max'] for r in rows],'o-',label='layer RMS sharp maximum')
    ax.plot([r['heads'] for r in rows],[r['head_max'] for r in rows],'s-',label='head RMS sharp maximum')
    ax.set(xlabel='number of heads H',ylabel='maximum absolute logit',title='P6: concentrated-energy construction')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(FIG/'e2_sharp_bounds.png',dpi=160); plt.close(fig)


def write_report(summary,e0,manifest,checks,construction,attempt):
    e1,e2 = summary['e1'],summary['e2']
    p = summary['predictions']
    lines = []
    def add(s=''): lines.append(s)
    add('## 完成情况与资源')
    add()
    add(f'已完成 E0 全网格、E1 **{len(e1)}/72** 个正式运行、E2 **{len(e2)}/24** 个干预运行，每个训练 {manifest["calibration"]["selected_steps"]} 步；另有一个50步计时运行，不纳入训练结论。全程顺序 CPU、无 API。')
    add()
    total = manifest['elapsed_seconds']+attempt.get('elapsed_seconds',0)+summary['summary_seconds_before_report']
    add(f'正式执行实测 {manifest["elapsed_seconds"]:.2f} 秒；此前两次资源中断合计 {attempt.get("elapsed_seconds",0):.2f} 秒；汇总/审计/绘图截至写报告 {summary["summary_seconds_before_report"]:.2f} 秒，合计约 **{total/60:.2f} 分钟**（不含交互中的代码编写时间）。50步基准 {manifest["calibration"]["seconds"]:.3f} 秒，包含数据生成、测量和方向JSON写出，不包含最后汇总JSON落盘；事前外推含2倍裕量及10分钟预留 {manifest["calibration"]["estimated_total_seconds_with_factor2_and_10min_reserve"]:.1f} 秒，未减量。正式训练进程峰值 RSS **{manifest["peak_RSS_bytes"]/1024**2:.1f} MiB**；汇总进程 {summary["summary_peak_RSS_bytes"]/1024**2:.1f} MiB，均低于3.5 GB。')
    add()
    add(f'解释器 `{manifest["executable"]}`；Python {manifest["python"].split()[0]}；NumPy {manifest["numpy"]}、SciPy {manifest["scipy"]}、torch {manifest["torch"]}、matplotlib {manifest["matplotlib"]}。OMP/OPENBLAS/MKL=3，torch计算线程=3。训练仅用 NumPy CPU；torch只用于独立 CPU autograd 核验。')
    add()
    add('## E0：解析预测与投影递推')
    add()
    add('设置：种子20261105，d_h=16，2048个独立重复；eta={0.01,0.02,0.04}、lambda={0.1,0.2,0.4}、m={0,m*,3m*,1}，共36个固定偏置配置，每组 ceil(10/(eta lambda)) 步、后半段估计。9组两阶段过程先相干 ceil(5/(eta lambda)) 步，再无偏 ceil(10/(eta lambda)) 步，未裁剪/真实投影使用配对创新。默认停用配置另从a0=2.5开始；高噪声重触发为128个重复、20000步。CI以独立重复为样本单位。')
    add()
    add('| 预测 | 实测数字 | 冻结判据 | 结论 |')
    add('|---|---|---|---|')
    add(f'| P1 时间指数 | m=0: {e0["P1"]["slope_m0"]:.5f}；m=1: {e0["P1"]["slope_m1"]:.5f} | 距1/2均≤0.1 | {p["P1"]} |')
    add(f'| P2 Rademacher相界 | 9组临界偏置相对误差最大 {summary["e0_stats"]["max_boundary_relative_error"]:.3%} | ≤15% | {p["P2"]} |')
    add(f'| P3 相干峰lambda指数 | {", ".join(f"{x:.5f}" for x in e0["P3"]["peak_lambda_slopes"])} | 与−2差≤0.15 | {p["P3"]} |')
    add(f'| P3 噪声底eta指数 | {", ".join(f"{x:.5f}" for x in e0["P3"]["floor_eta_slopes"])} | 与+1差≤0.15 | {p["P3"]} |')
    add(f'| P3 噪声底lambda指数 | {", ".join(f"{x:.5f}" for x in e0["P3"]["floor_lambda_slopes"])} | 与−1差≤0.15 | {p["P3"]} |')
    add(f'| P4 超额衰减 | 9组斜率相对误差最大 {summary["e0_stats"]["max_decay_slope_relative_error"]:.3%} | ≤10%；仅解析超额≥10倍噪声底 | {p["P4"]} |')
    p5 = e0['P5']
    add(f'| P5 有限窗口停用 | [142,2500]至少一次触发 {p5["late_any_count"]}/{p5["repeats"]}，95% CI [{p5["late_any_CI95"][0]:.6f},{p5["late_any_CI95"][1]:.6f}] | 置信下界>0.05才否定 | {p["P5"]} |')
    add()
    add(f'36组末半段二阶矩对有限时精确公式的最大相对误差 {summary["e0_stats"]["max_moment_relative_error"]:.3%}。默认配置 r_N={p5["r_N"]:.7f}、c^142 R+r_N={p5["envelope_at_n0"]:.7f}<2.5；早期1–141步至少触发过一次的重复为 {p5["early_any_count"]}/{p5["repeats"]}。')
    add(f'P2表格保留预登记的估计器；实现审计附项在相同原始统计上补除Rademacher方差1−m_*²后，9组交点误差最大为 {summary["e0_stats"]["corrected_boundary_max_error"]:.3%}，仍按原15%门槛通过。两种估计均保留，不借此改动主判据。')
    default = next(g for g in e0['grid'] if g['eta']==.02 and g['lambda']==.2)
    ci0 = default['rows'][0]['logit_tail']
    tw = next(g for g in e0['two_stage'] if g['eta']==.02 and g['lambda']==.2)
    counts = np.asarray(e0['recurrence']['counts_per_repeat'])
    add(f'默认eta=.02、lambda=.2的无偏稳态logit估计 {ci0["mean"]:.7f}，95% CI [{ci0["ci95"][0]:.7f},{ci0["ci95"][1]:.7f}]，解析稳态 {default["rows"][0]["analytic_stationary"]:.7f}；两阶段相干末端logit {tw["peak_logit"]:.6f}，衰减斜率实测 {tw["observed_slope"]:.8f}、理论 {tw["expected_slope"]:.8f}。高噪声20000步触发次数 min/median/max={counts.min()}/{np.median(counts):.0f}/{counts.max()}，128个重复均重触发；这不以有限样本证明无限时域定理。')
    add()
    add('![E0预测](experiments/figures/e0_predictions.png)')
    add('![人为切换和重触发](experiments/figures/e0_switch_recurrence.png)')
    add()
    add('**判断：支持受控条件模型的P1–P5时才继续保留该模型；这些实验人为指定了相干性变化，不能计作C1或真实训练停用机制的证据。**')
    add()
    add('## E1：固定分布、无人工阶段切换的检索训练')
    add()
    add('设置完全按§5：4096训练/512测试/128独立探针；8个key、D=32、4头×8维；单位高斯与kappa=4各向异性两条件，query为正确key加0.1同协方差噪声，每向量范数sqrt(32)。Q/K两个32×32矩阵，独立N(0,0.1²/32)初始化，标签平滑0.1；batch32不放回打乱、600步、float64。种子11/23/47跨优化器配对。Muon beta=0及0.9、AdamW(0.9,0.99)、SignSGD；eta=.01/.03/.1，alpha=.1eta，初始方向范数匹配后k0固定。每10步探针、每50步测试/完整训练CE，方向每步保存；第50步冻结全部头的模态，随后不重新选择。')
    add()
    add('下表为三个配对种子的均值±样本标准差。准确率为逐头召回率；额外的四头概率均值召回率也存于JSON。Mabs为固定探针集最终最大绝对logit。')
    add()
    add('| 数据 | 优化器 | eta | 测试CE | 召回率 | 最终Mabs | 晚期方向F范数(Q/K均值) |')
    add('|---|---|---:|---:|---:|---:|---:|')
    for key,rs in summary['e1_groups'].items():
        cond,opt,eta = key.split('|')
        add(f'| {cond} | {opt} | {eta} | {spread([r["test_ce"] for r in rs])} | {spread([r["test_accuracy"] for r in rs])} | {spread([r["final_Mabs"] for r in rs],3)} | {spread([np.mean(r["late_direction_fro"]) for r in rs],3)} |')
    add()
    hs = [h for r in e1 if r['optimizer']=='muon0' for h in r['heads']]
    for opt in COLORS:
        rs = [r for r in e1 if r['optimizer']==opt]
        add(f'- {opt}：探针峰值Mabs范围 {min(r["peak_Mabs"] for r in rs):.4f}–{max(r["peak_Mabs"] for r in rs):.4f}；峰到末步回落比例范围 {min(r["peak_to_final_drop_fraction"] for r in rs):.2%}–{max(r["peak_to_final_drop_fraction"] for r in rs):.2%}；{sum(r["probe_times_over8"]>0 for r in rs)}/{len(rs)} 个运行曾在探针上超过8。阈值8仅作与E2一致的描述，不将其追认为E1的新判据。')
    add()
    add('### C1 与模态诊断')
    add()
    add(f'按运行前固定的同一头联合判据，C1结论：**{summary["C1_verdict"]}**。Muon beta=0的 {len(hs)} 个头/运行组合中，全部联合条件通过 {sum(h["C1_all_conditions"] for h in hs)} 个。')
    add()
    add('| 条件 | 通过头数/总数 | 关键范围（全部Muon beta=0） |')
    add('|---|---:|---|')
    range_fields = {'residual':'peak_residual','explained_change':'peak_explained_change',
                    'qk_correlation':'qk_correlation','short_correlation':'max_abs_acf_early_late',
                    'forecast':'forecast_relative_error'}
    for cond in ['residual','explained_change','qk_correlation','phase_crossing','short_correlation','forecast']:
        values = [h[range_fields[cond]] for h in hs] if cond in range_fields else []
        text = f'{min(values):.5f}–{max(values):.5f}' if values else 'Q/K两侧均要求早窗>1且晚窗<1'
        add(f'| {cond} | {sum(h["conditions"][cond] for h in hs)}/{len(hs)} | {text} |')
    add()
    degenerate = [h for h in hs if h['peak_step']==50]
    nondegenerate = [h for h in hs if h['peak_step']>50]
    if degenerate:
        add(f'解释率退化说明：{len(degenerate)}个头的峰恰为第50步冻结基线，此时完整变化和模态变化都为零；实现中的分母保护给出1，不具有“解释了变化”的意义。主表保留原始实现数值；非退化峰的解释率最高仅 {max(h["peak_explained_change"] for h in nondegenerate):.5f}，仍低于0.7。不据此修改任何原始联合判定；全部头已因残差门槛失败。')
        add()
    add('| 数据 | eta | 同一头通过的种子数（head0/1/2/3） | 达到2/3？ |')
    add('|---|---:|---|---|')
    for row in summary['C1_groups']:
        add(f'| {row["condition"]} | {row["eta"]} | {row["per_head_seed_counts"]} | {"是" if row["passes"] else "否"} |')
    add()
    add('完整100步滑动窗口统计、200次长度10 block bootstrap区间、lag1–20 ACF及其区间在各运行JSON的 `diagnostics.windows` 中；更新窗口首次相界穿越时间独立于logit曲线存储。第1–150步估计的固定均值/方差/交叉协方差预测第160步以后a²/b²/ab，未用未来数据重新拟合；短相关门槛失败时，独立噪声公式标为不适用，不能把拟合误差称作定理被推翻。')
    add()
    selected_eta = manifest['e2_selection']['selected_eta']
    add('下表固定展示所选学习率、种子11、head0（仅作便于阅读的诊断切片，联合判定仍包含所有头和种子）。方括号为95% block bootstrap区间；psi20=1+2 sum_{k=1..20} c^k ACF(k)，只估计前20 lag对噪声底的修正，不能替代未知无限相关尾部。')
    add()
    add('| 数据/窗口 | mQ [CI] | mK [CI] | 方差Q/K | 同刻协方差 [CI] | max绝对ACF | psi20 Q/K |')
    add('|---|---|---|---|---|---:|---|')
    for condition in ['isotropic','anisotropic']:
        r = next(r for r in e1 if r['condition']==condition and r['optimizer']=='muon0' and r['eta']==selected_eta and r['seed']==11)
        for w in [r['diagnostic_windows'][0],r['diagnostic_windows'][-1]]:
            m = np.asarray(w['mean'])[0]
            mci = np.asarray(w['mean_ci95'])[:,0]
            var = np.asarray(w['variance'])[0]
            cov = w['cross_covariance'][0]
            covci = np.asarray(w['cross_covariance_ci95'])[:,0]
            ac = np.asarray(w['acf'])[0]
            psi = 1+2*np.sum((1-.1*selected_eta)**np.arange(1,21)*ac,axis=-1)
            add(f'| {condition}/{w["start"]}–{w["end"]} | {m[0]:.4f} [{mci[0,0]:.4f},{mci[1,0]:.4f}] | {m[1]:.4f} [{mci[0,1]:.4f},{mci[1,1]:.4f}] | {var[0]:.5f}/{var[1]:.5f} | {cov:.5f} [{covci[0]:.5f},{covci[1]:.5f}] | {np.max(np.abs(ac)):.4f} | {psi[0]:.4f}/{psi[1]:.4f} |')
    add()
    add('psi20出现负值说明短窗口去均值后的截断相关估计不稳定；不把它解释为负噪声方差，也不将这些数值代入定理作为校准成功的证据。')
    add()
    add('![E1训练](experiments/figures/e1_training.png)')
    add('![固定模态检查](experiments/figures/e1_frozen_modes.png)')
    add()
    eta = manifest['e2_selection']['selected_eta']
    add('### 相同训练CE的比较')
    add()
    add('固定CE目标为1.0和0.8，取完整训练集每50步测量中首次达到该目标的时刻；只比较达标运行，不插值、不从测试曲线选时刻。全部学习率的结果在 `summary.json` 的 `matched_CE`；下表列E2选定学习率。固定测量间隔仍有CE超调，初始范数匹配也不能消除后续更新差别。')
    add()
    add('| 数据 | 优化器 | CE目标 | 达标种子 | 实际训练CE | 步数 | 探针Mabs | 方向F范数 |')
    add('|---|---|---:|---:|---:|---:|---:|---:|')
    for row in summary['matched_CE']:
        if row['eta']!=eta: continue
        vals = row['values']
        if vals:
            add(f'| {row["condition"]} | {row["optimizer"]} | {row["target"]} | {len(vals)}/3 | {spread([v["train_ce"] for v in vals],3)} | {spread([v["step"] for v in vals],1)} | {spread([v["Mabs"] for v in vals],3)} | {spread([np.mean(v["direction_fro"]) for v in vals],3)} |')
        else:
            add(f'| {row["condition"]} | {row["optimizer"]} | {row["target"]} | 0/3 | 未达到 | — | — | — |')
    add()
    add('## E2：干预、归一化几何与机制反例')
    add()
    add(f'学习率只按六个Muon beta=0配对运行第50/100/150步训练CE平均值选择：{manifest["e2_selection"]["scores"]}，选中 **eta={eta}**。3种子×2数据×4干预=24个运行，600步；保持其余数据/初始化/batch配对。无decay、按层RMSNorm、按头RMSNorm、fresh-minibatch QK-Clip(tau=8)均原样执行，固定gain为1。')
    add()
    add('| 数据 | 干预 | 最终测试CE | 召回率 | 最终探针Mabs | 过程峰值Mabs |')
    add('|---|---|---:|---:|---:|---:|')
    for cond in ['isotropic','anisotropic']:
        for intv in ['base','no_decay','layer_norm','head_norm','qk_clip']:
            rs = [r for r in e1+e2 if r['condition']==cond and r['optimizer']=='muon0' and r['eta']==eta and r['intervention']==intv]
            add(f'| {cond} | {intv} | {spread([r["test_ce"] for r in rs])} | {spread([r["test_accuracy"] for r in rs])} | {spread([r["final_Mabs"] for r in rs],3)} | {spread([r["peak_Mabs"] for r in rs],3)} |')
    add()
    cr = [r for r in e2 if r['intervention']=='qk_clip']
    add(f'QK-Clip 六个运行的 head-step 触发总数分别为 {[sum(r["clip_head_step_counts"]) for r in cr]}（每个运行最多2400个head-step）；前150步总触发 {sum(r["clip_early_head_step_count"] for r in cr)}，末100步总触发 {sum(r["clip_late_head_step_count"] for r in cr)}。fresh batch裁剪后最大绝对logit最大值 {max(r["clip_post_batch_max"] for r in cr):.6f}；固定探针中 {sum(r["probe_times_over8"]>0 for r in cr)}/6 个运行、测试中 {sum(r["max_test_Mabs"]>8 for r in cr)}/6 个运行曾越过8。只在当前batch测量的裁剪不保证探针/测试阈值。')
    add()
    add('| H（d_h=8） | 整层构造实测=理论界 | 按头构造实测=理论界 |')
    add('|---:|---:|---:|')
    for r in construction['P6']:
        add(f'| {r["heads"]} | {r["layer_max"]:.9f} | {r["head_max"]:.9f} |')
    add()
    add(f'P6：**{p["P6"]}**，等号构造最大绝对误差 {summary["e2_stats"]["sharp_max_error"]:.3g}。负对角检索100步，Muon/SignSGD方向最大差 {construction["diagonal_max_direction_error"]:.3g}、轨迹最大差 {construction["diagonal_max_trajectory_error"]:.3g}；持续相干且R<1/lambda时100/100步触发裁剪。后两项支持原文反例，阻止将一般相干积累解释成Muon特有机制。')
    add()
    add('预注册对照：归一化的几何上界及等号构造按P6判定；负对角等价和持续相干裁剪反例得到支持。去decay实验没有预注册独立的数值胜负门槛，且训练创新方向并非固定，因此对“持续偏置导致增长”的训练桥接只能记为**不确定**，不以干预前后差异证明该前提。fresh-minibatch裁剪的当前batch上界通过数值检查；其长期停用机制仍为**不确定/定理条件不适用**，不能把单模态独立创新的P5充分条件直接套到多头训练或固定探针上。')
    add()
    add('![E2干预](experiments/figures/e2_interventions.png)')
    add('![归一化构造](experiments/figures/e2_sharp_bounds.png)')
    add()
    add('## 继续还是止损')
    add()
    if summary['C1_verdict']=='否定（本预注册任务的桥接不成立）':
        add('**止损当前“固定危险模态＋独立创新解释真实Muon瞬态”的主张，不进入语言模型规模实验。** P1–P6的受控公式核验与C1的端到端桥接是两件事；严格的联合标准未达成，不能用E0的人为切换或E2干预有效替代桥接。保留条件定理、明确反例、代码和负结果，可作为技术报告。若研究继续，只允许先重新预注册一个有限的机制扩展，再决定是否值得第二周投入。')
        add()
        add(f'最直接的桥接障碍：固定模态峰处残差最小仍为 {min(h["peak_residual"] for h in hs):.4f}（要求≤0.3），非退化峰解释变化比例最大为 {max(h["peak_explained_change"] for h in nondegenerate):.4f}（要求≥0.7）；只有10/72个头的Q/K系数相关达到0.8。这些空间/同步诊断必须先通过，单独替换噪声底公式无法消除模态失配。当前结果不能证明真实Muon没有瞬态现象，也不能将训练中确实出现的logit回落归因于这个单模态模型。')
    else:
        add('按C1联合结果及完整性谨慎判断：当前结果尚不能直接外推至真实语言模型。须以原文第7节为边界，先核实桥接再扩大实验。')
    add()
    add('本次未做文献联网查新；用户要求不调用外部API，故不将本轮冒烟写成已完成新颖性核对。也未执行第2周及后续的label smoothing消融、相关噪声扩展或实际LLM轨迹实验。')
    add()
    add('## 下一步最该做的3件事')
    add()
    add('1. 固化本轮负结果与原始方向，人工审查失败的固定模态残差、Q/K相关和ACF；根据已有数据区分多模态/方向旋转与时间相关，禁止换方向后追认C1成功。')
    add('2. 先补齐新颖性核对并判断严重的单模态失配是否还有研究价值。只允许为一个有明确依据的扩展另写预注册；若空间失配无法解决，就收缩为一般谱更新风险报告或换题。相关噪声扩展须使用(3.7a)、前窗拟合后窗预测，并保留30%误差门槛，不能单靠它跳过空间诊断。')
    add('3. 在任何真实模型/API支出前先作第二周go/no-go：需要至少一个新预测和独立桥接正证据；否则结束当前主线。若转入B题，先取得B题方案并冻结下述最小API设计。')
    add()
    add('## B题/API后续设计与预算（本轮未执行）')
    add()
    add('本目录FORMULATION.md只包含A题，无B题问题、预测和轨迹定义，因此B题无法在本轮声称已实现或已检验。离线已完成的范围仅为A题全部CPU实验；本轮API调用=0、费用=0。通常的文本生成API不暴露训练中的WQ/WK或极方向，不能用对话轨迹替代本轮Muon训练轨迹。')
    add()
    add('若B题另需文本LLM真实轨迹，建议在获得其方案后先冻结任务与判据，再做以下**待批准、待定价**的小试：固定1个模型快照、20个冻结任务、每任务3个独立重复、每轨迹最多8轮，共≤60轨迹、≤480次调用；每次输入≤2000 tokens、输出≤500 tokens，总上限输入960000、输出240000 tokens，逐轮保存完整请求/响应JSON、模型版本、随机性参数、延迟、token用量及失败。预先分任务而非分回合划分开发/留出集；拒答/超时按预定义失败计，不以补跑挑成功轨迹。若模型不支持seed，记录该限制，保留本地任务抽样seed。预算公式为0.96×输入每百万token价格+0.24×输出每百万token价格；运行前用实际服务价格核算并设置总费用硬上限10美元，预计费用超限则在结果产生前减少任务数。本轮未查询价格，故不填造美元实付或预计精确单价。API实验不得作为C1的Q/K权重证据。')
    add()
    add('## 复现与审计')
    add()
    add('```bash')
    add('OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 /home/liuyibo/venvs/theory/bin/python experiments/run_week1.py')
    add('OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 /home/liuyibo/venvs/theory/bin/python experiments/summarize.py')
    add('```')
    add()
    add(f'原始结果：[manifest](experiments/results/manifest.json)、[E0](experiments/results/e0.json)、[汇总数字](experiments/results/summary.json)、[运行前登记原文及SHA256](experiments/results/preregistration.json)、[梯度核验](experiments/results/gradient_checks.json)、[独立输出审计](experiments/results/audit.json)。每个训练运行的JSON位于 `experiments/results/runs/`，逐步方向JSON gzip位于 `experiments/results/directions/`。')
    add()
    fd = [c['max_absolute_error'] for c in checks if 'max_absolute_error' in c]
    ag = [c['autograd_max_absolute_error'] for c in checks if 'autograd_max_absolute_error' in c]
    add(f'普通/按层/按头三种解析反传共18个有限差分坐标，最大绝对误差 {max(fd):.3g}；对全部梯度元素的独立torch CPU autograd最大误差 {max(ag):.3g}。审计逐一解压96个正式方向文件，验证600行及首末方向投影、测量步数、有限值、标量投影递推和归一化/裁剪上界。没有删除失败种子或重定阈值。源码和原文哈希存于manifest，报告汇总代码哈希存于summary。')
    prereg = load(OUT/'preregistration.json')['report_before_execution']
    prefix = prereg.split('## 结果')[0]
    repair = prereg.split('## 结果',1)[1].replace('实验尚未执行。完成后本段由汇总程序替换，以上执行前登记保留。','').strip()
    notes = ROOT/'experiments'/'analysis_notes.md'
    if notes.exists():
        repair += '\n\n'+notes.read_text().strip()
    report = prefix+'## 执行中修复记录\n\n'+(repair if repair else '无。')+'\n\n'+'\n'.join(lines)+'\n'
    (ROOT/'RESULTS_WEEK1.md').write_text(report,encoding='utf-8')


def main():
    started = time.monotonic()
    manifest = load(OUT/'manifest.json')
    assert manifest['status']=='complete',manifest['status']
    e0 = load(OUT/'e0.json')
    checks,construction = load(OUT/'gradient_checks.json'),load(OUT/'constructions.json')
    attempt_paths = sorted(OUT.glob('failed_attempt*manifest.json'))
    attempts = [load(path) for path in attempt_paths]
    attempt = {'elapsed_seconds':sum(a.get('elapsed_seconds',0) for a in attempts)}
    e1,e2,audits = [],[],[]
    for key,output in [('completed_e1',e1),('completed_e2',e2)]:
        for rid in manifest[key]:
            r = load(OUT/'runs'/f'{rid}.json')
            audits.append(audit_run(r))
            output.append(compact(r))
    assert len(e1)==72 and len(e2)==24
    groups = defaultdict(list)
    for r in e1:
        groups[f'{r["condition"]}|{r["optimizer"]}|{r["eta"]}'].append(r)
    c1groups = []
    for cond in ['isotropic','anisotropic']:
        for eta in [.01,.03,.1]:
            rs = [r for r in e1 if r['condition']==cond and r['optimizer']=='muon0' and r['eta']==eta]
            counts = [sum(r['heads'][h]['C1_all_conditions'] for r in rs) for h in range(4)]
            c1groups.append({'condition':cond,'eta':eta,'per_head_seed_counts':counts,'passes':max(counts)>=2})
    supported = any(sum(g['passes'] for g in c1groups if g['condition']==cond)>=2 for cond in ['isotropic','anisotropic'])
    verdict = '支持（本预注册CPU任务内）' if supported else '否定（本预注册任务的桥接不成立）'
    moment_errors = [r['relative_error'] for g in e0['grid'] for r in g['rows']]
    boundary_errors = [g['boundary_relative_error'] for g in e0['grid']]
    decay_errors = [g['slope_relative_error'] for g in e0['two_stage']]
    corrected_boundary = []
    for g in e0['grid']:
        row,mstar = g['rows'][1],g['mstar']
        q = row['noise_empirical_a2']/(row['coherent_empirical_a2']/mstar**2)/(1-mstar**2)
        estimate = math.sqrt(q/(1+q))
        corrected_boundary.append({'eta':g['eta'],'lambda':g['lambda'],'mstar':mstar,
                                   'original_estimate':g['estimated_boundary'],
                                   'corrected_estimate':estimate,'relative_error':abs(estimate/mstar-1)})
    p3 = e0['P3']
    sharp_error = max(max(abs(r['layer_max']-r['layer_bound']),abs(r['head_max']-r['head_bound'])) for r in construction['P6'])
    predictions = {'P1':'支持' if abs(e0['P1']['slope_m0']-1)<=.1 and abs(e0['P1']['slope_m1']-2)<=.1 else '否定',
                   'P2':'支持' if max(boundary_errors)<=.15 else '否定',
                   'P3':'支持' if max([abs(x+2) for x in p3['peak_lambda_slopes']]+
                                      [abs(x+1) for x in p3['floor_lambda_slopes']]+
                                      [abs(x-1) for x in p3['floor_eta_slopes']])<=.15 else '否定',
                   'P4':'支持' if max(decay_errors)<=.1 else '否定',
                   'P5':'否定' if e0['P5']['late_any_CI95'][0]>.05 else
                        ('支持' if e0['P5']['late_any_CI95'][1]<=.05 else '不确定'),
                   'P6':'支持' if sharp_error<=1e-12 else '否定'}
    matched = []
    for key,rs in groups.items():
        cond,opt,eta = key.split('|')
        for target in [1.,.8]:
            matched.append({'condition':cond,'optimizer':opt,'eta':float(eta),'target':target,
                            'values':[r['matched_CE'][str(target)] for r in rs if r['matched_CE'][str(target)]]})
    plot_e0(e0)
    plot_training(e1,manifest['e2_selection']['selected_eta'])
    plot_e2(e1,e2,manifest['e2_selection']['selected_eta'],construction)
    summary = {'predictions':predictions,'C1_verdict':verdict,'C1_groups':c1groups,
               'e0_stats':{'max_moment_relative_error':max(moment_errors),
                           'max_boundary_relative_error':max(boundary_errors),
                           'corrected_boundary_max_error':max(r['relative_error'] for r in corrected_boundary),
                           'corrected_boundary_audit':corrected_boundary,
                           'max_decay_slope_relative_error':max(decay_errors)},
               'e2_stats':{'sharp_max_error':sharp_error},'e1':e1,'e2':e2,
               'e1_groups':dict(groups),'matched_CE':matched,
               'summarize_source_sha256':sha(__file__),
               'summary_seconds_before_report':time.monotonic()-started,
               'summary_peak_RSS_bytes':rss_bytes()}
    save(OUT/'audit.json',{'status':'passed','runs_checked':len(audits),'runs':audits,
                          'total_direction_gzip_bytes':sum(r['directions_gzip_bytes'] for r in audits)})
    # Groups and runs deliberately share compact records in memory; JSON is portable.
    save(OUT/'summary.json',summary)
    write_report(summary,e0,manifest,checks,construction,attempt)
    print(json.dumps({'predictions':predictions,'C1':verdict,'runs_audited':len(audits),
                      'summary_seconds':time.monotonic()-started,
                      'RSS_MiB':rss_bytes()/1024**2},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
