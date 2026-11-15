"""冻结本次审计证据入口；NPZ哈希已由各夹具账本/最终凭据绑定。"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_y3c_checks'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
state=json.loads((OUT/'unchanged.json').read_text())
boundary=json.loads((OUT/'file_boundary.json').read_text())
assert set(state['changed'])=={'codex_ay3c.log'} and not state['unexpected_new'],state
assert boundary['original_bytes_preserved'] and boundary['unchanged_original_files']==50805
assert not state['formal_y3c_exists'] and not state['formal_y3b_exists']
paths=[ROOT/'AUDIT_Y3C.md']
paths += sorted(p for p in OUT.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.csv','.log','.diff'}
                and p.name!='seal_evidence.log')
with (OUT/'EVIDENCE.sha256').open('x') as f:
    for p in paths:f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
print(json.dumps(dict(sealed_files=len(paths),npz_hashes='bound by per-fixture events/data_hashes/accepted credentials',
                     original_files_checked=state['original_files'],unchanged_original_files=50805,
                     session_log_append_only='codex_ay3c.log',unexpected_new=0)))
