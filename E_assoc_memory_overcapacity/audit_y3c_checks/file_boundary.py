"""单列运行环境的会话日志追加，不掩盖既有文件哈希变化。"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'audit_y3c_checks'
state=json.loads((OUT/'unchanged.json').read_text())
assert set(state['changed'])=={'codex_ay3c.log'},state
assert not state['unexpected_new']
before=state['changed']['codex_ay3c.log']['before']
with (ROOT/'codex_ay3c.log').open('rb') as f:prefix=f.read(before['size'])
prefix_hash=hashlib.sha256(prefix).hexdigest()
assert prefix_hash==before['sha256']
result=dict(original_files=state['original_files'],unchanged_original_files=state['original_files']-1,
    changed_only='codex_ay3c.log',original_bytes_preserved=True,
    original_prefix_sha256=prefix_hash,original_size=before['size'],
    observed_after_size=state['changed']['codex_ay3c.log']['after']['size'],
    observed_appended_bytes=state['changed']['codex_ay3c.log']['after']['size']-before['size'],
    unexpected_new=[],audit_scripts_do_not_write_session_log=True,
    formal_y3c_exists=state['formal_y3c_exists'],formal_y3b_exists=state['formal_y3b_exists'])
with (OUT/'file_boundary.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
print(json.dumps(result))
