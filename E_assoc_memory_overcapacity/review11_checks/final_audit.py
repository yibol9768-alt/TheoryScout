"""Read-only audit of reviewed inputs and newly created review artifacts."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    assert 1 <= int(os.environ[key]) <= 2
from pathlib import Path
import hashlib
import json
import re
import sys

root = Path.cwd().resolve()
assert root == Path("/home/liuyibo/auto-llm-driving/TheoryScout/E_assoc_memory_overcapacity").resolve()
out = root / "review11_checks"
original = json.loads((out / "input_hashes.json").read_text())
for name, digest in original.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
report = (root / "PROOF_REVIEW11.md").read_text()
assert len(re.findall(r"^### 引理 \d+：", report, re.M)) == 11
assert len(re.findall(r"^### 引理 \d+：.*——PASS", report, re.M)) == 10
assert "### 引理 2：复用训练查询的双线性误差——需修正" in report
assert "最终总判定：需修正" in report
assert report.count(r"\[") == report.count(r"\]")
assert not re.search("[\x00-\x08\x0b\x0c\x0e-\x1f]", report)
inside = False
for no,line in enumerate(report.splitlines(),1):
    if line.strip() == r"\[":
        assert not inside, no
        inside = True
    elif line.strip() == r"\]":
        assert inside, no
        inside = False
    elif not inside:
        assert line.count("$") % 2 == 0, no
assert not inside
cg_rel = next(line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
cg = Path("/sys/fs/cgroup") / cg_rel.lstrip("/")
controls = {name:(cg/name).read_text().strip() for name in ("memory.max", "memory.swap.max")}
assert controls["memory.max"] == str(3*1024**3)
assert controls["memory.swap.max"] == "0"
threads = len(list(Path("/proc/self/task").iterdir()))
assert threads <= 2
artifacts = [root/"PROOF_REVIEW11.md", *sorted(p for p in out.iterdir() if p.is_file())]
result = {"original_input_count":len(original), "original_inputs_unchanged":True,
          "lemma_count":11,"lemma_primary_PASS":10,"lemma_needs_correction":1,
          "resource_controls":controls,"threads":threads,"python":sys.executable,
          "artifact_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}}
target = out / (sys.argv[1] if len(sys.argv) > 1 else "final_audit.json")
assert target.parent == out and target.suffix == ".json"
with target.open("x") as f:
    json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
