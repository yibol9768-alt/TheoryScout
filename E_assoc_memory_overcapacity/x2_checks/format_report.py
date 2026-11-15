"""Normalize math delimiters in the new X2 report only."""
from pathlib import Path
import re
p=Path(__file__).resolve().parent.parent/'X2_FINITE_D.md'
s=p.read_text()
s=s.replace(r'\operatorname{tr}(Sp)^2',r'\operatorname{tr}[(Sp)^2]')
out=[];display=False;code=False
def normalize(t):
    t=t.replace(r'\(', '(').replace(r'\)',')')
    ans=[];i=0
    while i<len(t):
        if t[i]!='(':
            ans.append(t[i]);i+=1;continue
        depth=1;j=i+1
        while j<len(t) and depth:
            if t[j]=='(':depth+=1
            if t[j]==')':depth-=1
            j+=1
        if depth:
            ans.append(t[i:]);break
        body=t[i+1:j-1]
        math=bool(re.search(r'[A-Za-z\\_=^/]',body)) and not re.search(r'[\u4e00-\u9fff]',body)
        link=i>0 and t[i-1]==']'
        if math and not link:
            if body.startswith('(') and body.endswith(')'):
                ans.append('($'+body[1:-1]+'$)')
            else:ans.append('$'+body+'$')
        else:ans.append(t[i:j])
        i=j
    return ''.join(ans)
for line in s.splitlines():
    if line.startswith('```'):code=not code;out.append(line);continue
    if code:out.append(line);continue
    if line.strip()==r'\[':display=True;out.append('$$');continue
    if line.strip()==r'\]':display=False;out.append('$$');continue
    if display:out.append(line);continue
    parts=line.split('`')
    for j in range(0,len(parts),2):parts[j]=normalize(parts[j])
    out.append('`'.join(parts))
p.write_text('\n'.join(out)+'\n')
