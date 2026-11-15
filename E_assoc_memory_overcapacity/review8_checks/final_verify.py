"""Final integrity and resource assertions; writes only a new review artifact."""
import ast
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parent
R = P/"strict_run"
for name in ("independent_checks.py","artifact_audit.py","final_verify.py"):
    ast.parse((P/name).read_text(),filename=name)
metadata = json.loads((R/"metadata.json").read_text())
audit = json.loads((R/"artifact_audit.json").read_text())
assert metadata["observed_process_threads_max"] <= 2
assert audit["observed_process_threads"] <= 2
assert metadata["datasets"]==144 and metadata["rows"]==468
assert metadata["maxrss_kib"] < 2*1024*1024
initial = json.loads((R/"input_hashes_before.json").read_text())
final = json.loads((R/"input_hashes_after.json").read_text())
assert initial == final
assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value
           for name,value in initial.items())
rows = [json.loads(s) for s in (R/"rows.jsonl").read_text().splitlines()]
assert len(rows)==468
assert len({(r["case"],r["d"],r["seed"],r["method"]) for r in rows})==468
assert all(r["tie_total"]==0 for r in rows)
assert len(list(Path("/proc/self/task").iterdir())) <= 2
report = (ROOT/"PROOF_REVIEW8.md").read_text()
assert "FAIL" in report and "需修正" in report and "论文可以采用的准确陈述" in report
manifest = {
    str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [ROOT/"PROOF_REVIEW8.md",P/"independent_checks.py",P/"artifact_audit.py",P/"final_verify.py"]
}
result = dict(syntax_valid=True,existing_review_inputs_unchanged=True,
    final_rows=468,final_datasets=144,observed_process_threads_max=metadata["observed_process_threads_max"],
    audit_process_threads=audit["observed_process_threads"],resource_assertions_pass=True,
    source_and_report_hashes=manifest)
with (R/"final_validation.json").open("x") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
    f.write("\n")
print(json.dumps(result,ensure_ascii=False))
