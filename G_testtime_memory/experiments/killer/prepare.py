"""Deterministic numerical predictions only; run once before freeze."""
from pathlib import Path
from design import prediction, BETAS

p = Path('PREREG_KILLER.md')
text = p.read_text()
assert '<!-- PREDICTIONS -->' in text
rows = ['| d | a | kappa | Hebb预测 | 最佳delta预测 | beta预测 | 预测优势 |',
        '|---|---|---|---|---|---|---|']
for d in (64, 128, 256):
    for a in (1.0, 1.5):
        for k in (2, 4, 8):
            h = prediction(d,k,a,'hebb')
            vals = [prediction(d,k,a,f'delta_{b:g}') for b in BETAS]
            j = max(range(len(vals)), key=lambda j: vals[j])
            rows.append(f'|{d}|{a}|{k}|{h:.4f}|{vals[j]:.4f}|{BETAS[j]:g}|{vals[j]-h:+.4f}|')
p.write_text(text.replace('<!-- PREDICTIONS -->', '\n'.join(rows)))
print('\n'.join(rows))
