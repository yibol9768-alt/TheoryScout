"""Post-training descriptive tables/plots; does not change frozen assessments."""
import experiment as c
from experiment import np,json,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    assessment=json.loads((c.OUT/'assessment.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((c.OUT/'raw').glob('*.json'))]
    replay=[json.loads(p.read_text()) for p in sorted((c.OUT/'replay').glob('*.json'))]
    complete=[r for r in rows if r['status']=='completed']
    lookup={(r['config']['d'],r['config']['seed'],r['config']['mode']):r for r in complete}
    lines=['|基础d（提升实际d+1）|分支|全排序/5|全有效符号/5|排序成功均值|recall@3|最差gamma均值|有效符号间隔均值|beta均值|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in assessment['table']:
        lines.append(f'|{x["d"]}|{x["mode"]}|{x["rank_full"]}|{x["sign_full"]}|{x["success"]:.6f}|{x["recall"]:.6f}|{x["min_gap"]:.6g}|{x["sign_margin"]:.6g}|{x["beta"]:.2f}|')
    lines += ['','|d|分支|seed15|seed16|seed17|seed18|seed19|','|---|---|---:|---:|---:|---:|---:|']
    for d in (6,7,8):
        for mode in c.MODES:
            cells=[]
            for s in range(15,20):
                r=lookup.get((d,s,mode))
                cells.append('缺失' if r is None else f'{r["final"]["success_fraction"]:.6f} / {r["final"]["effective_sign_margin"]:.6g}')
            lines.append(f'|{d}|{mode}|'+'|'.join(cells)+'|')
    lines += ['','每格为终态排序成功比例 / 最小有效符号间隔。','',
              '|d|零偏置首次全符号对（每1000步及固定检查点检测）|终态全符号数|','|---|---|---:|']
    for d in (6,7,8):
        rr=[lookup[(d,s,'sigmoid_zero')] for s in range(15,20) if (d,s,'sigmoid_zero') in lookup]
        cells=['>T' if r['first_all_sign_step'] is None else str(r['first_all_sign_step']) for r in rr]
        lines.append(f'|{d}|'+', '.join(cells)+f'|{sum(r["final"]["all_sign"] for r in rr)}|')
    (c.OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
    for col,d in enumerate((6,7,8)):
        for mode in c.MODES:
            rr=[lookup[(d,s,mode)] for s in range(15,20) if (d,s,mode) in lookup]
            if not rr: continue
            for row,key in enumerate(('success_fraction','effective_sign_row_fraction')):
                steps=[h['step'] for h in rr[0]['history']]
                vals=np.array([[h[key] for h in r['history']] for r in rr])
                axes[row,col].plot(steps,vals.mean(0),label=mode)
                axes[row,col].fill_between(steps,vals.min(0),vals.max(0),alpha=.08)
                axes[row,col].set_ylim(-.02,1.02); axes[row,col].grid(alpha=.2)
        axes[0,col].set_title(f'base d={d}; lift uses d={d+1}')
        axes[1,col].set_xlabel('Updates')
    axes[0,0].set_ylabel('Rank success fraction')
    axes[1,0].set_ylabel('Effective sign row success')
    axes[0,0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(c.OUT/'trajectories.png',dpi=170); plt.close(fig)
    resources=[r['memory'] for r in complete+replay]
    for name in ('manifest','validation','execution'):
        resources.append(json.loads((c.OUT/f'{name}.json').read_text())['memory'])
    decision=json.loads((c.OUT/'budget_decision.json').read_text())
    execution=json.loads((c.OUT/'execution.json').read_text())
    report=dict(total_seconds_from_timing_start=time.time()-decision['preparation_start_epoch'],
                run_wall_seconds=execution['wall_seconds'],sum_training_seconds=sum(r['elapsed_seconds'] for r in complete),
                sum_replay_seconds=sum(r['elapsed_seconds'] for r in replay),max_worker_RSS_bytes=max(m['VmHWM'] for m in resources),
                max_worker_AS_bytes=max(m['VmPeak'] for m in resources),reporting_memory=c.r2.memory(),
                replay_max_abs_error=max(r['old_replay_max_abs_error'] for r in replay),
                deterministic_max_abs_error=max(max(r['deterministic_check']['errors'].values()) for r in replay),
                calibration_max_relative_error=max(abs(r['calibration']['actual'][k]['rms']/.001-1) for r in complete+replay for k in ('U','V')),
                paired_initial_fingerprints_match=all(len({lookup[(d,s,m)]['initial_fingerprint'] for m in c.MODES[:3]})==1 for d in (6,7,8) for s in range(15,20)),
                strict_robust_all_sign_counts_match=all(x['sign_full']==x['robust_full'] for x in assessment['table']))
    c.save(c.OUT/'reporting_resources.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
