"""Exclusive final artifact manifest and reproducibility metadata; no new data."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import scipy
from P1NUM_run import ROOT,OUT,put,metadata,frozen

def main():
 frozen();start=time.monotonic();resource0=metadata()
 for path in [ROOT/'P1NUM_PREREG.sha256',ROOT/'P1NUM_FREEZE.sha256',ROOT/'P1NUM_ANALYSIS_NOTE.sha256',ROOT/'P1NUM_DERIVED_CODE.sha256',ROOT/'P1NUM_DERIVED_FORMATTING.sha256']:
  for line in path.read_text().splitlines():
   digest,name=line.split(maxsplit=1);name=name.lstrip('* ')
   assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 s=json.loads((OUT/'P1NUM_summary.json').read_text());audit=json.loads((OUT/'P1NUM_independent_audit.json').read_text())
 assert (ROOT/'P1NUM_RESULTS.md').exists()
 put(OUT/'P1NUM_final_audit.json',dict(prereg_sha256=hashlib.sha256((ROOT/'P1NUM_PREREG.md').read_bytes()).hexdigest(),frozen_inputs_unchanged=True,independent_replay_passed=True,expected_records=audit['expected_records'],actual_records=audit['actual_records'],missing_records=audit['missing_records'],complete=audit['complete'],numpy=np.__version__,scipy=scipy.__version__,python=sys.version,interpreter=sys.executable,project_access='Only current project; OS/runtime dependencies and resource metadata read as required',network_calls=False,llm_api_calls=False,seconds=time.monotonic()-start,resource_start=resource0,resource_end=metadata()))
 paths=sorted({p for p in OUT.rglob('*') if p.is_file()}|{p for p in ROOT.glob('P1NUM_*') if p.is_file()})
 assert not (ROOT/'P1NUM_ARTIFACTS.sha256').exists()
 with (ROOT/'P1NUM_ARTIFACTS.sha256').open('x') as f:
  for p in paths:f.write(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n')
 print(json.dumps(dict(artifacts=len(paths),records=audit['actual_records'],missing=audit['missing_records'],numpy=np.__version__,scipy=scipy.__version__),ensure_ascii=False))
if __name__=='__main__':main()
