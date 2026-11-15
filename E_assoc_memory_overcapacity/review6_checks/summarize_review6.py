"""Deterministic summaries; datasets, not queries, are the replication units."""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from check_review6 import OUT, resources, dump_csv


def load(name):
    with (OUT/name).open() as stream:
        return list(csv.DictReader(stream))


def avg(rows, fn):
    return float(np.mean([fn(row) for row in rows]))


def f(row, key):
    return float(row[key])


def main():
    resources()
    data, joint = load('per_dataset.csv'), load('joint_scores.csv')
    samples = np.load(OUT/'query_samples.npz')
    pooled, summary = [], []
    for d in [64, 128]:
        dimension = [x for x in data if int(x['d']) == d]
        for config in range(6):
            for name in ['GD', 'Sign', 'Adam_.2', 'Adam_2']:
                subset = [x for x in dimension if int(x['config']) == config and x['name'] == name]
                cross = [x for x in joint if int(x['d']) == d and int(x['config']) == config and x['name'] == name]
                key = f'{d}_{config}_{name}'
                correct = samples[key+'_correct'].transpose(1, 0, 2).reshape(2, -1)
                ranks = samples[key+'_ranks'].transpose(1, 0, 2).reshape(2, -1)
                for task in [0, 1]:
                    ss = [x for x in subset if int(x['task']) == task]
                    mu = f(ss[0], 'mu_theory')
                    observed = np.quantile(ranks[task], [.1, .5, .9])
                    theoretical = ndtr(-mu+ndtri([.1, .5, .9]))
                    row = {'d': d, 'config': config, 'rx': f(ss[0], 'rx'), 'ry': f(ss[0], 'ry'),
                           'name': name, 'task': task, 'mu_theory': mu,
                           'pooled_mean': np.mean(correct[task]),
                           'pooled_variance': np.var(correct[task], ddof=1),
                           'pooled_corr': np.corrcoef(correct)[0, 1],
                           'r_theory': f(ss[0], 'rx')*f(ss[0], 'ry'),
                           'mean_error': abs(np.mean(correct[task])-mu),
                           'variance_error': abs(np.var(correct[task], ddof=1)-1),
                           'corr_error': abs(np.corrcoef(correct)[0, 1]-f(ss[0], 'rx')*f(ss[0], 'ry')),
                           'wrong_ks': avg(ss, lambda x: f(x, 'wrong_ks')),
                           'cdf_maxerr': np.max(np.abs(np.mean(ranks[task, :, None] <= np.array([.01,.05,.1,.25,.5,.75,.9]), axis=0)-ndtr(mu+ndtri([.01,.05,.1,.25,.5,.75,.9])))),
                           'quantile10': observed[0], 'quantile10_theory': theoretical[0],
                           'quantile50': observed[1], 'quantile50_theory': theoretical[1],
                           'quantile90': observed[2], 'quantile90_theory': theoretical[2],
                           'quantile_maxerr': np.max(np.abs(observed-theoretical)),
                           'recall_mean': avg(ss, lambda x: f(x, 'recall10')),
                           'recall_dataset_sd': np.std([f(x, 'recall10') for x in ss], ddof=1)}
                    pooled.append(row)
        pp = [x for x in pooled if x['d'] == d]
        summary.append({'d': d, 'mean_abs_error': avg(pp, lambda x: x['mean_error']),
                        'variance_abs_error': avg(pp, lambda x: x['variance_error']),
                        'corr_abs_error': avg(pp, lambda x: x['corr_error']),
                        'wrong_ks_mean': avg(pp, lambda x: x['wrong_ks']),
                        'rank_cdf_maxerr_mean': avg(pp, lambda x: x['cdf_maxerr']),
                        'rank_quantile_maxerr_mean': avg(pp, lambda x: x['quantile_maxerr']),
                        'worst_rank_quantile_error': max(x['quantile_maxerr'] for x in pp)})
    dump_csv('pooled.csv', pooled)
    dump_csv('summary.csv', summary)
    concentration = []
    full = load('full_dataset.csv')
    for d in [64, 128]:
        for ry in [0., -.5]:
            for task in [0, 1]:
                ss = [x for x in full if int(x['d']) == d and f(x, 'ry') == ry and int(x['task']) == task]
                concentration.append({'d': d, 'ry': ry, 'task': task,
                                      'full_recall_mean': avg(ss, lambda x: f(x, 'full_recall10')),
                                      'theory': f(ss[0], 'theory'),
                                      'dataset_sd': np.std([f(x, 'full_recall10') for x in ss], ddof=1),
                                      'full_rank_mean': avg(ss, lambda x: f(x, 'full_rank_mean')),
                                      'rank_mean_theory': f(ss[0], 'rank_mean_theory')})
    dump_csv('concentration_summary.csv', concentration)
    stability = load('stability.csv')
    stable = []
    for name, steps in sorted(set((x['name'], int(x['steps'])) for x in stability)):
        for d in [64, 128]:
            ss = [x for x in stability if x['name'] == name and int(x['steps']) == steps and int(x['d']) == d]
            stable.append({'d': d, 'name': name, 'steps': steps,
                           'fro_mse': avg(ss, lambda x: f(x, 'fro_mse')),
                           'correct_mse': avg(ss, lambda x: f(x, 'correct_mse'))})
    dump_csv('stability_summary.csv', stable)
    def table(rows, fields):
        lines = ['| '+' | '.join(fields)+' |', '| '+' | '.join(['---']*len(fields))+' |']
        for x in rows:
            lines.append('| '+' | '.join(f'{x[k]:.6f}' if isinstance(x[k], (float, np.floating)) else str(x[k]) for k in fields)+' |')
        return '\n'.join(lines)
    text = '# Review 6 独立数值结果\n\n'
    text += '每格 16 个独立数据集；每数据集 160 个不同查询；不把查询当作独立复制。均值/方差/相关系数在同格内合并后计算；CDF KS 在每数据集前 8 个查询上计算并平均。下面的误差是诊断量，不是渐近定理的证明。\n\n'
    text += table(summary, list(summary[0]))+'\n\n'
    text += '## 共享键的完整数据集 recall\n\n每格 12 个独立数据集，实际一步 Sign，逐一计算全部 n 个查询的训练 rank。\n\n'
    text += table(concentration, list(concentration[0]))+'\n\n'
    text += '## 实际与代理轨迹误差\n\n'+table(stable, list(stable[0]))+'\n\n'
    text += '## 各格合并统计\n\n'+table(pooled, ['d','config','name','task','mu_theory','pooled_mean','pooled_variance','pooled_corr','wrong_ks','quantile10','quantile10_theory','quantile50','quantile50_theory','quantile90','quantile90_theory'])+'\n'
    (OUT/'results.md').write_text(text)
    print(table(summary, list(summary[0])))
    print(table(concentration, list(concentration[0])))
    print(table(stable, list(stable[0])))


if __name__ == '__main__':
    main()
