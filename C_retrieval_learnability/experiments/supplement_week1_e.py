"""Complete E's largest-absolute-difference check; see report's prior addendum."""
import os
os.environ.update(OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  MPLCONFIGDIR='/tmp/theory-week1-matplotlib', CUDA_VISIBLE_DEVICES='')
import resource
resource.setrlimit(resource.RLIMIT_AS, (3500000000, 3500000000))
import datetime
import hashlib
import json
from pathlib import Path
import time

import torch
torch.set_num_threads(3)
torch.set_default_dtype(torch.float64)
from week1_common import OUT, save, train, memory


def main():
    def read(name):
        return json.loads((OUT / name).read_text())
    start = time.perf_counter()
    c, d, e = [read(g + '.json') for g in ['C', 'D', 'E']]
    groups = {}
    for run in c['runs'] + d['runs']:
        cfg = run['config']
        key = (cfg['k'], cfg['d'], cfg['seed'], cfg['ratio'], str(cfg['beta']), cfg['bias'])
        groups.setdefault(key, {})[cfg['loss']] = run
    candidates = [g for g in groups.values() if set(g) == {'nce', 'sigmoid', 'pe'}]
    def spread(group, field):
        vals = [r['final'][field] for r in group.values()]
        return max(vals) - min(vals)
    candidates.sort(key=lambda g: (-spread(g, 'success_fraction'), -spread(g, 'min_gap'), g['nce']['config']['id']))
    selected = candidates[0]
    prior = {r['baseline_id'] for r in e['runs']}
    assert selected['nce']['config']['id'] not in prior
    prereg = {'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'rule': 'max absolute success fraction spread; tie: max min-gap spread, then NCE config id',
              'selected_ids': [selected[l]['config']['id'] for l in ['nce', 'sigmoid', 'pe']],
              'absolute_success_spread': spread(selected, 'success_fraction'),
              'min_gap_spread': spread(selected, 'min_gap'),
              'report_sha256_before_execution': hashlib.sha256((OUT.parent.parent / 'RESULTS_WEEK1.md').read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'predictions_or_decision_criteria_changed': False}
    save('E_addendum_preregistration.json', prereg)
    save('E_original.json', e)
    print('Registered E addendum:', prereg, flush=True)
    for loss in ['nce', 'sigmoid', 'pe']:
        base = selected[loss]
        for method in ['adam', 'gd']:
            cfg = dict(base['config'])
            cfg['id'] = 'E_extra_' + method + '_' + cfg['id']
            result = train(cfg, steps=2 * base['steps_requested'] if method == 'adam' else base['steps_requested'],
                           method=method, half_lr=method == 'adam')
            result['baseline_id'] = base['config']['id']
            result['selection_addendum'] = True
            save(cfg['id'] + '.json', result)
            compact = {key: val for key, val in result.items() if key != 'history'}
            compact['raw_file'] = cfg['id'] + '.json'
            e['runs'].append(compact)
            print(cfg['id'], result['final']['success_fraction'], result['final']['min_gap'], flush=True)
    e['selection_addendum'] = prereg
    save('E.json', e)
    save('E_addendum_resources.json', {'elapsed_seconds': time.perf_counter() - start,
                                     'memory': memory(), 'torch_threads': torch.get_num_threads(),
                                     'rlimit_as_bytes': resource.getrlimit(resource.RLIMIT_AS),
                                     'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
