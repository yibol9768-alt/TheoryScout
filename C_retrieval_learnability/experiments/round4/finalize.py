"""Delivery integrity manifest; does not modify frozen preregistration or results."""
import experiment as c
from experiment import json,time

def main():
    m=json.loads((c.OUT/'manifest.json').read_text())
    assert all(c.digest(c.ROOT/p)==h for p,h in m['hashes'].items())
    expected=(c.ROOT/'PREREG_ROUND4.sha256').read_text().split()[0]
    assert c.digest(c.ROOT/'PREREG_ROUND4.md')==expected
    assessment=json.loads((c.OUT/'assessment.json').read_text())
    assert assessment['valid'] and assessment['total_present']==100 and assessment['completed']==100
    assert json.loads((c.OUT/'gradient_audit.json').read_text())['pass_validation']
    audit=json.loads((c.OUT/'first_step_audit.json').read_text()); assert len(audit)==100
    ex=json.loads((c.OUT/'execution.json').read_text())
    done=dict(completed_epoch=time.time(),formal_wall_seconds=ex['wall_seconds'],
              seconds_since_prereg_write=time.time()-(c.ROOT/'PREREG_ROUND4.md').stat().st_mtime,
              overall_wall_budget_seconds=10800,
              overall_wall_budget_met=(time.time()-(c.ROOT/'PREREG_ROUND4.md').stat().st_mtime<=10800),
              prereg_sha256=expected,frozen_inputs_unchanged=True,completed_runs=100,
              scope_properties=dict(MemoryMax='6G',MemorySwapMax=0),device='CPU',LLM_API_calls=0,
              decisions=assessment['decisions'])
    c.save(c.OUT/'completion.json',done)
    paths=[p for p in c.OUT.rglob('*') if p.is_file() and not any(x in p.parts for x in ('__pycache__','mplconfig')) and p.name!='delivery_manifest.json']
    paths += [c.ROOT/'RESULTS_ROUND4.md',c.ROOT/'PREREG_ROUND4.md',c.ROOT/'PREREG_ROUND4.sha256']
    c.save(c.OUT/'delivery_manifest.json',dict(hashes={str(p.relative_to(c.ROOT)):c.digest(p) for p in sorted(paths)},created_epoch=time.time()))
    print(json.dumps(done,ensure_ascii=False))
if __name__=='__main__': main()
