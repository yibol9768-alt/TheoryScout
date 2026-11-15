"""Integrity and accounting, no training or hyperparameter changes."""
import hashlib
import json
from pathlib import Path
import time
from run import OUT,ROOT,digest,dump,frozen,utc
START=time.monotonic(); prereg=frozen()
s=json.loads((OUT/'summary.json').read_text()); initial=json.loads((OUT/'summary_initial.json').read_text())
for key in ['analysis_utc','analysis_wall_seconds']:
    s.pop(key); initial.pop(key)
assert s==initial,'Serial analysis changed substantive outputs'
audit=json.loads((OUT/'audit.json').read_text()); assert audit['passed']
meta=json.loads((OUT/'formal/metadata.json').read_text()); assert meta['complete']
assert meta['code_sha256']==digest(OUT/'run.py')
assert s['analysis_code_sha256']==digest(OUT/'analyze.py')
assert len(list((ROOT/'lemmas_v2').glob('L*.tex')))==5
old=(ROOT/'PREREG_KILLER.sha256').read_text().split()[0]
assert digest(ROOT/'PREREG_KILLER.md')==old
# These two initial plotting/analysis runtimes were observed before runtime.json
# was replaced by serial rebuilding; retained here explicitly, not estimated.
account=dict(formal=meta['wall_seconds'],smoke=json.loads((OUT/'smoke.json').read_text())['wall_seconds'],
    smoke_initial=json.loads((OUT/'smoke_initial.log').read_text().splitlines()[-1])['wall_seconds'],
    analysis_initial=6.953370177987381,
    analysis_serial=json.loads((OUT/'analysis_runtime.json').read_text())['wall_seconds'],
    audit_initial=json.loads((OUT/'audit_initial.log').read_text().split('\n',1)[1])['wall_seconds'],
    audit_serial=audit['wall_seconds'],plot_initial=6.546511826993083,
    plot_final=json.loads((OUT/'plot_runtime.json').read_text())['wall_seconds'],
    report_details=json.loads((OUT/'report_details.json').read_text())['wall_seconds'])
assert sum(account.values())<800
integrity=dict(utc=utc(),prereg_sha256=prereg,serial_analysis_identical=True,
    numerical_integrity=True,full_resource_protocol_compliance=False,
    execution_deviation='Initial analysis/audit overlapped for about 5 seconds, each with a 2-thread cap; aggregate cap may be 4. Formal training and serial verification were each <=2.',
    numeric_wall_seconds_sum_excluding_finalizer=sum(account.values()),accounting=account,
    finalizer_wall_seconds=time.monotonic()-START)
dump(OUT/'integrity.json',integrity)
files=[ROOT/'FORMULATION_V2.md',ROOT/'PREREG_KILLER_V2.md',ROOT/'PREREG_KILLER_V2.sha256',ROOT/'RESULTS_KILLER_V2.md']
files+=list((ROOT/'lemmas_v2').glob('L*.tex'))
files+=[p for p in OUT.rglob('*') if p.is_file() and '.mplconfig' not in p.parts and p.name not in ['ARTIFACTS.sha256','finalize.log','checksums.log']]
(OUT/'ARTIFACTS.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT)}\n' for p in sorted(set(files))))
print(json.dumps(integrity,ensure_ascii=False,indent=2))
