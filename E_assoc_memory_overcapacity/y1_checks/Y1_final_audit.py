"""Read-only final checks of report formatting and original input hashes."""
from pathlib import Path
import hashlib
import json
import re

report = Path('Y1_D3_PROOFS.md').read_text()
inside = False
display_count = 0
leftover = []
for number, line in enumerate(report.splitlines(), 1):
    if line.strip() == r'\[':
        assert not inside, number
        inside = True
        display_count += 1
        continue
    if line.strip() == r'\]':
        assert inside, number
        inside = False
        continue
    if inside:
        continue
    assert line.count('$') % 2 == 0, (number, line)
    plain = re.sub(r'\$[^$]*\$', '', line)
    plain = re.sub(chr(96)+r'[^'+chr(96)+r']*'+chr(96), '', plain)
    for found in re.findall(r'\([^()]*[A-Za-z][^()]*\)', plain):
        leftover.append({'line': number, 'text': found})
assert not inside
assert not leftover, leftover
assert not re.search('[\x00-\x08\x0b\x0c\x0e-\x1f]', report)
original = json.loads(Path('y1_checks/Y1_input_hashes.json').read_text())
current = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
           for name in original}
assert original == current
result = {'balanced_display_blocks': display_count,
          'balanced_inline_math': True,
          'unformatted_letter_parentheses': leftover,
          'primary_original_input_hashes_unchanged': original == current}
with Path('y1_checks/Y1_final_audit.json').open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False))
