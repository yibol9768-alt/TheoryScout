"""Final presentation/link checks on the report newly created in this review."""
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
path=ROOT/'PROOF_REVIEW23.md'
text=path.read_text()
text=text.replace(chr(92)+'$', '$')
text=text.replace(r'\beta_{1,2}\in[0,1$)',r'\beta_{1,2}\in[0,1)$')
text=text.replace('Var$d=64$/Var$d=512$','Var(d=64)/Var(d=512)')
text=text.replace('样本量不能辨识 (1/n) 级协方差','样本量不能辨识 $1/n$ 级协方差')
text=text.replace('控制为 (O(R))','控制为 $O(R)$')
text=text.replace('超过 (b+O(1/b))','超过 $b+O(1/b)$')
text=text.replace('半径误差 (a/[4(b+1)])','半径误差 $a/[4(b+1)]$')
text=text.replace('宽度 (2h)','宽度 $2h$')
missing=[]
for target in re.findall(r'\]\(([^)]+)\)',text):
    if target.startswith(('https://','http://')): continue
    local=target.rsplit(':',1)[0] if re.search(r':\d+$',target) else target
    if not (ROOT/local).exists(): missing.append(target)
assert not missing, missing
assert text.count(r'\[')==text.count(r'\]')
assert text.count('$')%2==0
assert chr(92)+'$' not in text
assert '本节最终统计' not in text
assert 'root_and_rates_corrected.json' in text
path.write_text(text)
summary=dict(local_links_checked=len(re.findall(r'\]\(([^)]+)\)',text)),
             missing=missing,lines=len(text.splitlines()),
             display_equation_pairs=text.count(r'\['),dollar_delimiters=text.count('$'))
with (HERE/'report_validation.json').open('x') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary))
