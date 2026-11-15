from pathlib import Path
import json
base=Path('supplement')
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ['resources','seconds','elapsed_seconds']}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
for name in ['stabilizer_ordering','matched_chain','moment_history','history_reversal','polar_variance']:
    folder='polar_certificate' if name=='polar_variance' else 'two_step_adam_certificate'
    assert clean(json.loads((base/f'{folder}/reference/{name}.json').read_text()))==clean(json.loads((base/f'final_runs/from_zero/{name}.json').read_text())),name
print('PASS: all five regenerated certificates have identical mathematical data')
