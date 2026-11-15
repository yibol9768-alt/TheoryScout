import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
import json,hashlib,re
from pathlib import Path

out=Path(__file__).resolve().parent
root=out.parent
entry=next(v[3:] for v in Path('/proc/self/cgroup').read_text().splitlines() if v.startswith('0::'))
cg=Path('/sys/fs/cgroup')/entry.lstrip('/')
assert (cg/'memory.max').read_text().strip()=='2147483648'
assert (cg/'memory.swap.max').read_text().strip()=='0'
report=root/'PROOF_REVIEW7.md'
body=report.read_text()
repaired=body.count(chr(8))
body=body.replace(chr(8),'')
# Restore inline math delimiters lost while constructing the new report string.
# Work on the new report only; preserve numbered equation references and prose.
body=body.replace('\\)',')')
formatted=[]
display=False
fenced=False
for line in body.splitlines(keepends=True):
    marker=line.lstrip('> ').strip()
    if marker.startswith('```'):
        fenced=not fenced
    if marker=='\\[':
        display=True
    if display or fenced or '$' in line or marker.startswith('```'):
        formatted.append(line)
    else:
        spans=[]
        depth=0
        for pos,char in enumerate(line):
            if char=='(':
                if depth==0: start=pos
                depth+=1
            elif char==')' and depth:
                depth-=1
                if depth==0:
                    inside=line[start+1:pos]
                    is_math=(re.search('[A-Za-z_\\\\^=<>]',inside) is not None
                             and re.search('[\u4e00-\u9fff]',inside) is None
                             and re.fullmatch('[abc][0-9]?',inside) is None)
                    if is_math: spans.append((start,pos,inside))
        for start,end,inside in reversed(spans):
            line=line[:start]+'$'+inside+'$'+line[end+1:]
        formatted.append(line)
    if marker=='\\]':
        display=False
body=''.join(formatted)
report.write_text(body)
assert not any(ord(c)<32 and c not in '\n\t\r' for c in body)
assert body.count('\\[')==body.count('\\]')
for path,expected in json.loads((out/'input_hashes.json').read_text()).items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
for path in out.glob('*.json'):
    json.loads(path.read_text())
source=out/'independent_checks.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==json.loads((out/'metadata.json').read_text())['source_sha256']
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [report,*sorted(out.glob('*.py')),*sorted(out.glob('*.json'))]
        if p.name!='final_audit.json'}
with (out/'final_audit.json').open('w') as f:
    json.dump(dict(inputs_unchanged=True,json_valid=True,display_math_balanced=True,
                   repaired_new_report_control_characters=repaired,sha256=hashes),f,indent=2)
print('Original inputs unchanged; output JSON valid; report control characters and display math checked.')
