"""Final read-only consistency checks; writes only the local check record."""
import csv
import hashlib
import json

import numpy as np

from check_review6 import OUT, ROOT, INPUTS, resources


def main():
    limits = resources()
    metadata = json.loads((OUT/'metadata.json').read_text())
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS}
    assert metadata['input_sha256'] == metadata['input_sha256_after']
    changes = {p: {'initial': metadata['input_sha256'][p], 'final': hashes[p]}
               for p in INPUTS if hashes[p] != metadata['input_sha256'][p]}
    # A concurrent external edit to this background file was detected and
    # manually re-read. Its model, theorem and scope are unchanged. Never
    # rewrite it or silently describe all original inputs as unchanged.
    assert set(changes) <= {'paper/sections/fine_tuning.tex'}
    if changes:
        assert hashes['paper/sections/fine_tuning.tex'] == '9abd26fbe9e9f25ed0edd124e44028452797a0074e3f3393bc5d8633e31c025b'
    counts = {}
    expected = {'per_dataset.csv': 1536, 'joint_scores.csv': 768,
                'pooled.csv': 96, 'full_dataset.csv': 96,
                'branch_grid.csv': 120, 'regression.csv': 6}
    for filename, count in expected.items():
        with (OUT/filename).open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == count, (filename, len(rows), count)
        counts[filename] = len(rows)
    with np.load(OUT/'query_samples.npz') as arrays:
        assert len(arrays.files) == 96
        for key in arrays.files:
            a = arrays[key]
            assert a.shape == (16, 2, 160)
            assert np.isfinite(a).all()
            if key.endswith('_ranks'):
                assert np.all((a >= 0)&(a <= 1))
    for filename in ['check_review6.py', 'check_edges.py',
                     'summarize_review6.py', 'verify_artifacts.py']:
        compile((OUT/filename).read_text(), filename, 'exec')
    edges = json.loads((OUT/'edges_metadata.json').read_text())['branches']
    assert edges['tangent_example']['t_peak'] > 0
    assert edges['tangent_example']['old_at_peak'] < 0
    assert edges['later_branch_counterexample']['Adam_e2_later_old'] < edges['later_branch_counterexample']['Sign_first_old']
    assert edges['max_formula_error'] < 1e-10
    assert edges['minimum_adjacent_old_signal_gap'] > 0
    assert (ROOT/'PROOF_REVIEW6.md').stat().st_size > 10000
    assert not list(OUT.rglob('__pycache__'))
    result = {'status': 'PASS_WITH_RECORDED_EXTERNAL_BACKGROUND_EDIT',
              'final_input_sha256': hashes, 'external_input_changes': changes,
              'csv_rows': counts, 'resource_check': limits}
    (OUT/'verification.json').write_text(json.dumps(result, indent=2))
    print('PASS: core input hashes, output dimensions, finite samples, branches, script syntax, resource limits.')
    print('Recorded external background edit:', json.dumps(changes))


if __name__ == '__main__':
    main()
