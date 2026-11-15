"""Repair inline math delimiters in this turn's newly created report only."""
from pathlib import Path
import json
import re

target = Path('Y1_D3_PROOFS.md')
assert target.parent == Path('.') and target.name.startswith('Y1_')
source = target.read_text()
changes = []
display = False
out = []
for line_number, line in enumerate(source.splitlines(keepends=True), 1):
    if line.strip() == r'\[':
        assert not display
        display = True
        out.append(line)
        continue
    if line.strip() == r'\]':
        assert display
        display = False
        out.append(line)
        continue
    if display:
        out.append(line)
        continue
    result = []
    pos = 0
    in_code = False
    while pos < len(line):
        if line[pos] == chr(96):
            in_code = not in_code
            result.append(line[pos])
            pos += 1
            continue
        escaped_open = line.startswith(r'\(', pos)
        if not in_code and (escaped_open or line[pos] == '('):
            start = pos + (2 if escaped_open else 1)
            end = start
            depth = 1
            while end < len(line) and depth:
                if line[end] == '(':
                    depth += 1
                elif line[end] == ')':
                    depth -= 1
                if depth:
                    end += 1
            if depth == 0:
                escaped_close = line[end-1] == '\\'
                body = line[start:end-(1 if escaped_close else 0)]
                mathematical = (
                    escaped_open or escaped_close or
                    any(c in body for c in ['\\', '_', '^', '=', '<', '>']) or
                    bool(re.fullmatch(r"[a-zA-Z](?:[a-zA-Z0-9 ,']*)", body))
                )
                if mathematical:
                    replacement = '$' + body + '$'
                    changes.append({'line': line_number,
                                    'old': line[pos:end+1],
                                    'new': replacement})
                    result.append(replacement)
                    pos = end+1
                    continue
        result.append(line[pos])
        pos += 1
    out.append(''.join(result))

assert not display
fixed = ''.join(out)
fixed = fixed.replace('对有界 $C^3$ 函数 $f$ 及任意固定紧区间内的阈值 $z_0$，',
                      '对有界且前三阶导数有界的 $C^3$ 函数 $f$ 及任意固定紧区间内的阈值 $z_0$，')
assert r'\(' not in fixed
assert r'\)' not in fixed
with Path('y1_checks/Y1_math_format.json').open('x') as f:
    json.dump({'purpose': 'formatting only, no original project file touched',
               'changes': changes}, f, ensure_ascii=False, indent=2)
target.write_text(fixed)
print(json.dumps({'converted_inline_math': len(changes)}, ensure_ascii=False))
