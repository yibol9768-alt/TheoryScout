"""只迁移执行命名空间；统计源码不变。"""
from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'y3c'
# 原有文件的只读基线；日志可能由会话记录器追加，单独排除。
baseline={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(ROOT.rglob('*')) if p.is_file() and not p.is_symlink()
          and not p.is_relative_to(OUT) and not p.name.startswith('Y3C_') and p.suffix!='.log'}
with (OUT/'original_baseline.json').open('x') as f:json.dump(baseline,f,indent=2)
for name in ['common.py','run.py','analyze.py','launch.py','test_synthetic.py']:
    p=OUT/name
    p.write_text(p.read_text().replace('Y3B','Y3C').replace('y3b','y3c'))
