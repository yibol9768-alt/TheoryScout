"""Final read-only consistency check; writes one exclusive review artifact."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
out = HERE/'final_validation.json'
if out.exists():
    raise FileExistsError(out)
simulation = json.loads((HERE/'main_simulation.json').read_text())
audit = json.loads((HERE/'audit_summary.json').read_text())
replay = json.loads((HERE/'saved_design_replay.json').read_text())
report = (ROOT/'PROOF_REVIEW20.md').read_text()
assert '<!--' not in report
assert 'TODO' not in report
assert len(simulation['records']) == 216
assert len(simulation['paired']) == 144
assert len(replay['rows']) == 16 and replay['max_recall_difference'] == 0
assert audit['independent_root_max_absolute_difference'] < 1e-10
for p, h in simulation['input_sha256_before'].items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h, p
for row in simulation['summary']:
    e = '.10' if row['method']=='adam_0.1' else '.12'
    r, m = row['recall_difference'], row['msr_difference']
    # Check the complete independently computed numerical table against prose.
    # Main table intentionally drops zero before decimal points.
    def f(value):
        return f'{value:.6f}'.replace('-0.', '−.').replace('0.', '.', 1)
    expected = (f"|{row['d']}|{e}|{f(r['mean'])}|[{f(r['ci95'][0])}, {f(r['ci95'][1])}]"
                f"|{f(m['mean'])}|[{f(m['ci95'][0])}, {f(m['ci95'][1])}]"
                f"|{row['positive_recall_datasets']}/24|")
    assert expected in report, expected
    assert r['ci95'][0] > 0 and m['ci95'][1] < 0
parsed = []
for path in sorted(HERE.glob('*.py')):
    ast.parse(path.read_text(), filename=str(path))
    parsed.append(path.name)
paths = [ROOT/'PROOF_REVIEW20.md', *sorted(HERE.iterdir())]
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}
with out.open('x') as f:
    json.dump(dict(inputs_unchanged=True, report_table_matches=True,
                   all_six_pointwise_recall_intervals_positive=True,
                   all_six_pointwise_msr_intervals_negative=True,
                   saved_design_records_reproduced=16, syntax_parsed=parsed,
                   artifacts_sha256=manifest,
                   note='Integrity and arithmetic checks, not a formal proof or strict statistical coverage certificate.'), f, indent=2)
print(json.dumps(dict(inputs_unchanged=True, report_table_matches=True,
                      saved_records_reproduced=16, artifact_count=len(manifest))))
