"""Integrity check and closed artifact manifest, without any new fitting or training."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from run import ROOT, OUT, DELTAS, SEEDS, digest, frozen, write_json, utc


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    metadata = json.loads((OUT / 'formal/metadata.json').read_text())
    assert frozen() == summary['prereg_sha256'] == metadata['prereg_sha256']
    assert digest(OUT / 'run.py') == metadata['code_sha256']
    assert digest(OUT / 'analyze.py') == summary['analysis_code_sha256']
    assert metadata['complete'] and len(metadata['tasks']) == 96
    rows = list(csv.DictReader((OUT / 'curves.csv').open()))
    assert len(rows) == 882
    index = {(r['mode'], int(r['N']), int(r['ratio']), float(r['delta'])): r for r in rows}
    assert len(index) == 882
    for cell in summary['cells']:
        mode = int(cell['mode'] == 'replay')
        n, ratio = cell['N'], cell['ratio']
        checkpoint = 0 if mode else [16,64,256,1024].index(ratio)
        risks = []
        for seed in SEEDS:
            with np.load(OUT / 'formal' / f'm{mode}_n{n}_r{ratio if mode else 0}_s{seed}.npz') as d:
                assert np.array_equal(d['deltas'], DELTAS)
                assert d['risks'].shape == ((1 if mode else 4), 42)
                assert np.isfinite(d['risks']).all() and np.isfinite(d['diagnostics']).all()
                risks.append(d['risks'][checkpoint])
                for k, delta in enumerate(DELTAS):
                    row = index[(cell['mode'], n, ratio, float(delta))]
                    assert float(row[f'seed_{seed}']) == d['risks'][checkpoint,k]
        means = np.mean(risks, axis=0)
        assert int(np.argmin(means)) == cell['best_index']
        assert means.min() == cell['risk']
    assert summary['stream_interior'] == summary['stream_identifiable'] == 7
    assert summary['replay_interior'] == summary['replay_identifiable'] == 4
    for tex in sorted((ROOT / 'lemmas').glob('L*.tex')):
        contents = tex.read_text()
        assert contents.count('\\begin{problem}') == contents.count('\\end{problem}') == 1
        assert '\\documentclass' in contents and '\\end{document}' in contents
    assert len(list((ROOT / 'lemmas').glob('L*.tex'))) == 5
    for name in ['risk_curves.svg', 'risk_curves.png', 'timescales.svg', 'timescales.png']:
        assert (OUT / name).stat().st_size > 1000
    write_json(OUT / 'integrity.json', dict(utc=utc(), passed=True, prereg_sha256=frozen(),
        raw_trajectories=96, terminal_risks=7056, curve_rows=882, latex_problems=5,
        checked='All raw seed values match CSV and argmins; source hashes and protocol match; '
        'standalone LaTeX structure only, no proof or TeX compilation certification.'))
    docs = [ROOT / name for name in ['FORMULATION.md', 'PREREG_KILLER.md',
        'PREREG_KILLER.sha256', 'RESULTS_KILLER.md']]
    paths = docs + list((ROOT / 'lemmas').glob('L*.tex'))
    paths += [p for p in OUT.iterdir() if p.is_file() and p.suffix in
        ['.py', '.md', '.json', '.csv', '.png', '.svg', '.log']]
    paths += [p for p in (OUT / 'formal').iterdir() if p.is_file()]
    manifest = ''.join(f'{digest(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in sorted(paths))
    (OUT / 'ARTIFACTS.sha256').write_text(manifest)
    print(json.dumps({'passed': True, 'manifest_files': len(paths), 'prereg_sha256': frozen()}))


if __name__ == '__main__':
    main()
