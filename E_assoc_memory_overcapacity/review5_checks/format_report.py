"""Restore inline math delimiters in this newly created review document."""
from pathlib import Path
p=Path('PROOF_REVIEW5.md')
lines=p.read_text().splitlines(keepends=True)
display=False
fence=False
out=[]
for line in lines:
    if line.startswith(chr(96)*3):
        fence=not fence
        out.append(line)
        continue
    if line.strip()==r'\[':
        display=True
    if display or fence:
        out.append(line)
        if line.strip()==r'\]':
            display=False
        continue
    chars=list(line)
    inline_code=False
    stack=[]
    for i,ch in enumerate(chars):
        if ch==chr(96):
            inline_code=not inline_code
        elif not inline_code:
            if ch=='(':
                stack.append(i)
            elif ch==')' and stack:
                start=stack.pop()
                if not stack:
                    chars[start]='$'
                    chars[i]='$'
    assert not stack,line
    out.append(''.join(chars))
p.write_text(''.join(out))
