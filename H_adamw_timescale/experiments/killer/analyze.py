"""Analyze the fixed grid without selecting new hyperparameters or rerunning training."""
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
from run import OUT, ROOT, DELTAS, SEEDS, B, digest, frozen, utc, write_json


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def line(x, y):
    if not np.isfinite(y).all():
        return None
    coef = np.polyfit(np.log(x), np.log(y), 1)
    residual = np.log(y) - np.polyval(coef, np.log(x))
    sst = np.sum((np.log(y) - np.mean(np.log(y))) ** 2)
    return dict(slope=float(coef[0]), r2=float(1 - np.sum(residual ** 2) / sst) if sst else 1.,
        endpoint_ratio=float(y[-1] / y[0]), adjacent_slopes=(np.diff(np.log(y)) / np.diff(np.log(x))).tolist())


def interval(samples):
    samples = np.asarray(samples, float)
    ok = np.isfinite(samples)
    return dict(valid_fraction=float(ok.mean()), ci95=np.quantile(samples[ok], [.025, .975]).tolist()
        if ok.mean() >= .9 else None)


def main():
    start = time.monotonic()
    prereg = frozen()
    meta = json.loads((OUT / 'formal/metadata.json').read_text())
    assert meta['complete'] and len(meta['tasks']) == 96
    assert meta['code_sha256'] == digest(OUT / 'run.py')
    cells, curve_rows, raw = [], [], []
    for mode in [0, 1]:
        for n in [32, 64, 128]:
            ratios = [16, 64, 256, 1024] if mode == 0 else [16, 64, 256]
            for j, ratio in enumerate(ratios):
                arrays = [np.load(OUT / 'formal' / f'm{mode}_n{n}_r{ratio if mode else 0}_s{s}.npz')
                    for s in SEEDS]
                risks = np.array([a['risks'][j if mode == 0 else 0] for a in arrays])
                diagnostics = np.array([a['diagnostics'][j if mode == 0 else 0] for a in arrays])
                assert risks.shape == (8, 42) and np.isfinite(risks).all()
                mean, se = risks.mean(0), risks.std(0, ddof=1) / np.sqrt(8)
                best = int(mean.argmin())
                pos = 1 + int(mean[1:].argmin())
                near = np.flatnonzero(mean <= 1.01 * mean[best])
                steps = ratio * n // B
                tau = 1 / (DELTAS[best] * steps) if best else np.inf
                pos_tau = 1 / (DELTAS[pos] * steps)
                identifiable = (best not in [0, 1, 41] and not any(k in near for k in [0, 1, 41])
                    and DELTAS[near[-1]] / DELTAS[near[0]] <= 2.5)
                cell = dict(mode='stream' if mode == 0 else 'replay', N=n, ratio=ratio,
                    best_index=best, best_delta=float(DELTAS[best]), best_lambda=float(DELTAS[best] / .01),
                    tau=finite_or_none(tau), positive_best_index=pos, positive_tau=pos_tau,
                    risk=float(mean[best]), risk_se=float(se[best]), no_decay_risk=float(mean[0]),
                    improvement=float(1 - mean[best] / mean[0]), interior=best not in [0, 1, 41],
                    identifiable=bool(identifiable), near_indices=near.tolist(),
                    near_delta_ratio=float(DELTAS[near[-1]] / DELTAS[near[0]]) if near[0] else None,
                    seed_best_indices=risks.argmin(1).tolist(),
                    best_norm_mean=float(diagnostics[:, 0, best].mean()),
                    best_radial_cos_mean=float(diagnostics[:, 1, best].mean()),
                    best_angle_mean=float(diagnostics[:, 2, best].mean()))
                cells.append(cell)
                raw.append(risks)
                for k, delta in enumerate(DELTAS):
                    curve_rows.append(dict(mode=cell['mode'], N=n, ratio=ratio, delta=delta,
                        lam=delta / .01, mean_risk=mean[k], se_risk=se[k],
                        **{f'seed_{s}': risks[q, k] for q, s in enumerate(SEEDS)}))
    raw = np.array(raw)
    rng = np.random.default_rng(20261106)
    indices = rng.integers(0, 8, size=(1000, 8))
    boot_means = raw[:, indices, :].mean(axis=2)  # cells, bootstrap, grid
    boot_best = boot_means.argmin(axis=2)
    with np.errstate(divide='ignore'):
        for j, cell in enumerate(cells):
            steps = cell['ratio'] * cell['N'] // B
            bt = 1 / (DELTAS[boot_best[j]] * steps)
            cell['tau_bootstrap'] = interval(bt)
            cell['bootstrap_interior_fraction'] = float(np.mean(~np.isin(boot_best[j], [0, 1, 41])))
            paired = (raw[j, :, 0] - raw[j, :, cell['best_index']]) / raw[j, :, 0]
            cell['fixed_selected_relative_gain_ci95'] = interval(paired[indices].mean(axis=1))
    fits = []
    for mode in ['stream', 'replay']:
        for n in [32, 64, 128]:
            ii = [i for i, c in enumerate(cells) if c['mode'] == mode and c['N'] == n]
            rr = np.array([cells[i]['ratio'] for i in ii])
            tau = np.array([cells[i]['tau'] if cells[i]['tau'] is not None else np.inf for i in ii])
            pos = np.array([cells[i]['positive_tau'] for i in ii])
            slopes = []
            for q in range(1000):
                kk = boot_best[ii, q]
                if np.any(kk == 0):
                    slopes.append(np.nan)
                else:
                    tt = 1 / (DELTAS[kk] * (rr * n / B))
                    slopes.append(float(np.polyfit(np.log(rr), np.log(tt), 1)[0]))
            fits.append(dict(mode=mode, N=n, fit=line(rr, tau), positive_diagnostic=line(rr, pos),
                tau_max_min=finite_or_none(np.max(tau) / np.min(tau)),
                slope_bootstrap=interval(slopes)))
    stream, replay = cells[:12], cells[12:]
    sf, rf = fits[:3], fits[3:]
    valid_sf = all(f['fit'] is not None for f in sf)
    slope_range = max(f['fit']['slope'] for f in sf) - min(f['fit']['slope'] for f in sf) if valid_sf else None
    g1 = all(c['identifiable'] for c in stream) and sum(c['improvement'] >= .02 for c in stream) >= 9
    numeric_g2 = bool(valid_sf and all(-.65 <= f['fit']['slope'] <= -.35 and f['fit']['r2'] >= .95
        and .067 <= f['fit']['endpoint_ratio'] <= .168 for f in sf) and slope_range < .1)
    replay_tau = [c['tau'] for c in replay]
    all_replay_tau = all(t is not None for t in replay_tau)
    global_replay_ratio = max(replay_tau) / min(replay_tau) if all_replay_tau else None
    g3 = bool(all(c['identifiable'] for c in replay) and all(f['fit'] is not None
        and abs(f['fit']['slope']) <= .1 and f['tau_max_min'] <= 1.5 for f in rf)
        and global_replay_ratio is not None and global_replay_ratio <= 2)
    # Independently verify the expected checkpoint inventory and numerical domain.
    expected_files = {f'm0_n{n}_r0_s{s}.npz' for n in [32, 64, 128] for s in SEEDS}
    expected_files |= {f'm1_n{n}_r{r}_s{s}.npz' for n in [32, 64, 128]
        for r in [16, 64, 256] for s in SEEDS}
    assert {t['file'] for t in meta['tasks']} == expected_files
    assert all(t['min_norm'] > 1e-12 for t in meta['tasks'])
    assert raw.size == 21 * 8 * 42 and np.isfinite(raw).all()
    summary = dict(prereg_sha256=prereg, analysis_utc=utc(), formal=meta,
        analysis_code_sha256=digest(Path(__file__)), terminal_values=int(raw.size),
        G0=True, G1=bool(g1), G2_numeric_only=numeric_g2, G2_evaluable=bool(g1),
        G2=bool(g1 and numeric_g2), G3=g3, continue_original=bool(g1 and numeric_g2 and g3),
        stream_interior=sum(c['interior'] for c in stream),
        stream_identifiable=sum(c['identifiable'] for c in stream),
        stream_gain_2pct=sum(c['improvement'] >= .02 for c in stream),
        replay_interior=sum(c['interior'] for c in replay),
        replay_identifiable=sum(c['identifiable'] for c in replay),
        stream_slope_range=slope_range, replay_global_tau_ratio=global_replay_ratio,
        fits=fits, cells=cells, analysis_wall_seconds=time.monotonic() - start)
    write_json(OUT / 'summary.json', summary)
    with (OUT / 'curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    lines = ['# 冻结网格完整汇总', '', '|制度|N|比例|δ*|τ*|风险±SE|相对零衰减改善|内部|1%近优可识别|',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for c in cells:
        tau = f"{c['tau']:.5g}" if c['tau'] is not None else '∞'
        lines.append(f"|{c['mode']}|{c['N']}|{c['ratio']}|{c['best_delta']:.5g}|{tau}|"
            f"{c['risk']:.6g}±{c['risk_se']:.2g}|{c['improvement']:.2%}|{c['interior']}|{c['identifiable']}|")
    lines += ['', '## 四点/三点 OLS（全部点，未删格）', '', '|制度|N|斜率|R²|末/首τ|斜率bootstrap95%|',
        '|---|---:|---:|---:|---:|---|']
    for f in fits:
        a = f['fit']
        lines.append(f"|{f['mode']}|{f['N']}|{a['slope']:.5f}|{a['r2']:.5f}|{a['endpoint_ratio']:.5g}|"
            f"{f['slope_bootstrap']}|" if a else f"|{f['mode']}|{f['N']}|无法拟合|—|—|{f['slope_bootstrap']}|")
    lines += ['', 'summary.json 另含每种子最优、近优集合、正δ诊断、配对改善、bootstrap有效率和径向快照。',
        '所有bootstrap区间仅描述此网格的不确定性；选中λ后的改善区间存在选择偏差，不是独立验证。']
    (OUT / 'TABLES.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['formal', 'cells']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
