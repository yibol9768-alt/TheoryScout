"""Delivery integrity, after manuscript and numeric validation are complete."""
import experiment as c
from datetime import datetime,timezone
from zoneinfo import ZoneInfo

def main():
    manifest=c.json.loads((c.OUT/'manifest.json').read_text())
    assert all(c.digest(c.ROOT/p)==h for p,h in manifest['hashes'].items())
    assert all(c.digest(c.ROOT/p)==h for p,h in manifest['copied_sources'].items())
    assert c.json.loads((c.OUT/'validation.json').read_text())['pass']
    expected=(c.ROOT/'PREREG_ROUND5.sha256').read_text().split()[0]
    assert c.digest(c.ROOT/'PREREG_ROUND5.md')==expected
    required=['RESULTS_ROUND5.md','PREREG_ROUND5.md','PREREG_ROUND5.sha256']
    required += ['experiments/round5/'+p for p in ('execution.json','validation.json','assessment.json','summary.json','tables.md','tables.csv','README.md','manifest.json','late_margins.png','factorial.png')]
    assert all((c.ROOT/p).exists() for p in required)
    manuscript=(c.ROOT/'RESULTS_ROUND5.md').read_text()
    assert not any(x in manuscript for x in ('__HOLDOUT','__EXECUTION__','__VALIDATION__','__THRESHOLD'))
    now=c.time.time(); execution=c.json.loads((c.OUT/'execution.json').read_text())
    completion=dict(completed_epoch=now,completed_utc=datetime.fromtimestamp(now,timezone.utc).isoformat(),completed_local=datetime.fromtimestamp(now,ZoneInfo('America/Los_Angeles')).isoformat(),prereg_sha256=expected,prereg_write_epoch=(c.ROOT/'PREREG_ROUND5.md').stat().st_mtime,code_freeze_epoch=manifest['frozen_epoch'],formal_start_epoch=execution['start_epoch'],formal_end_epoch=execution['end_epoch'],formal_wall_seconds=execution['wall_seconds'],formal_training_cpu_seconds=execution['cpu_seconds'],elapsed_since_prereg_write_seconds=now-(c.ROOT/'PREREG_ROUND5.md').stat().st_mtime,frozen_files_unchanged=True,round4_source_files_unchanged=True,required_files=required)
    c.save(c.OUT/'completion.json',completion)
    paths=[c.ROOT/p for p in ('RESULTS_ROUND5.md','PREREG_ROUND5.md','PREREG_ROUND5.sha256')]
    paths += [p for p in c.OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json' and '__pycache__' not in p.parts and not any(part in ('mplconfigdir','xdg_cache_home','tmpdir') for part in p.parts)]
    c.save(c.OUT/'delivery_manifest.json',dict(created_epoch=now,hashes={str(p.relative_to(c.ROOT)):c.digest(p) for p in sorted(paths)}))
    print(c.json.dumps(completion),flush=True)
if __name__=='__main__': main()
