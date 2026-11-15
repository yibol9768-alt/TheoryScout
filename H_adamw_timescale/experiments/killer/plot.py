"""Export scientific plots from already frozen results; no new experiments."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = json.loads((HERE / 'summary.json').read_text())
    colors = ['#0072b2', '#009e73', '#e69f00', '#cc79a7']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), constrained_layout=True)
    for row, mode in enumerate(['stream', 'replay']):
        for col, n in enumerate([32, 64, 128]):
            ax = axes[row, col]
            cc = [c for c in data['cells'] if c['mode'] == mode and c['N'] == n]
            for c, color in zip(cc, colors):
                risks = np.array([np.load(HERE / 'formal' /
                    f"m{row}_n{n}_r{c['ratio'] if row else 0}_s{s}.npz")['risks'][
                    0 if row else [16,64,256,1024].index(c['ratio'])] for s in range(1101,1109)])
                mean = risks.mean(0)
                delta = np.load(HERE / 'formal' /
                    f"m{row}_n{n}_r{c['ratio'] if row else 0}_s1101.npz")['deltas']
                zero = mean[0] / mean.min()
                ax.plot(delta[1:], mean[1:] / mean.min(), color=color,
                    label=f"r={c['ratio']}; zero WD={zero:.2f}")
                ax.axhline(zero, ls=':', lw=.9, color=color, alpha=.8)
                if c['best_index']:
                    ax.plot(c['best_delta'], 1, 'o', color=color, ms=5)
            ax.axhline(1.01, color='black', ls='--', lw=.7)
            ax.set(xscale='log', yscale='log', xlabel=r'$\delta=\eta\lambda$',
                ylabel='Mean excess risk / grid minimum', title=f'{mode}, N={n}')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle('Frozen killer test: 8 training seeds per curve; dotted = zero decay; dashed = 1% band', fontsize=11)
    fig.savefig(HERE / 'risk_curves.svg')
    fig.savefig(HERE / 'risk_curves.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for ax, mode in zip(axes, ['stream', 'replay']):
        for color_index, (n, color) in enumerate(zip([32, 64, 128], colors)):
            cells = [c for c in data['cells'] if c['mode'] == mode and c['N'] == n]
            x = np.array([c['ratio'] for c in cells])
            y = np.array([c['tau'] if c['tau'] is not None else np.nan for c in cells])
            ax.plot(x, y, 'o-', color=color, label=f'N={n}')
            for xx in x[np.isnan(y)]:
                ax.scatter(xx, 65 + 15 * color_index, marker='^', color=color, s=28)
            first = next((c for c in cells if c['tau'] is not None), None)
            if first:
                ref = first['tau'] * (x / first['ratio']) ** (-.5 if mode == 'stream' else 0.)
                ax.plot(x, ref, ':', color=color, lw=.9, alpha=.7)
        ax.text(.02, .97, 'Triangles: tau = infinity (vertical placement is arbitrary)',
            transform=ax.transAxes, va='top', fontsize=7)
        ax.set(xscale='log', yscale='log', ylim=(.02,120), xlabel='D/N' if mode == 'stream' else 'M/N',
            ylabel='tau_data' if mode == 'stream' else 'tau_epoch', title=mode)
        ax.grid(alpha=.2)
        ax.legend(loc='lower left', fontsize=8)
    fig.suptitle('Grid optima; dotted guides show preregistered exponents -0.5 and 0 (no fitted intercept claim)',fontsize=10)
    fig.savefig(HERE / 'timescales.svg')
    fig.savefig(HERE / 'timescales.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
