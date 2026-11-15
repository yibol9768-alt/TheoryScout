"""Validate the completed continuation and independently reduce all queries."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import time

root = Path.cwd()
base = root/'supplement'
folder = base/'full_experiments_continuation'
dest = folder/'independent_validation'
dest.mkdir(exist_ok=False)
records = []


def run(job, name):
    start = time.monotonic()
    with (dest/(name+'.log')).open('x') as handle:
        proc = subprocess.run([sys.executable, '-B', '-u', *job], cwd=base, stdout=handle, stderr=subprocess.STDOUT)
    row = dict(command='python '+' '.join(job), seconds=time.monotonic()-start, exit_code=proc.returncode)
    records.append(row)
    (dest/'timings.json').write_text(json.dumps(records, indent=2))
    assert proc.returncode == 0, (dest/(name+'.log')).read_text()


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def key(row):
    return (int(row['d']), int(row['ir']), int(row['seed']), row['method'], row['task'])


result = {}
for design, count in [('matched', 1152), ('confirmation', 288)]:
    summary = json.loads((folder/design/'result.json').read_text())
    assert summary['status'] == 'PASS' and summary['datasets'] == count
    assert summary['reused_datasets']+summary['generated_datasets'] == count
    assert not summary['rejected_archives']
    combined = []
    for label, sub in [('before_restart', 'full_experiments_recomputation'), ('after_restart', 'full_experiments_continuation')]:
        output = Path('full_experiments_continuation/independent_validation')/(design+'_'+label+'.csv')
        run(['experiments_replay/measurements_from_queries.py', '--input', str(Path(sub)/design), '--output', str(output)], design+'_'+label)
        combined.extend(rows(base/output))
    actual = {key(row): row for row in rows(folder/design/'measurements.csv')}
    independent = {key(row): row for row in combined}
    assert len(independent) == len(combined) == len(actual) == count*10
    assert independent.keys() == actual.keys()
    fields = ['recall', 'D', 'V', 'var_D', 'var_V', 'cov_DV']
    maxerr = max(abs(float(row[field])-float(actual[k][field])) for k, row in independent.items() for field in fields)
    assert maxerr == 0
    if design == 'confirmation':
        old = {key(row): row for row in rows(base/'experiments_replay/finite_dimension_measurements.csv')}
        original_error = max(abs(float(row[field])-float(old[k][field])) for k, row in independent.items() for field in fields)
        assert original_error < 1e-12
    else:
        old = {}
        for line in (base/'experiments_replay/matched_retrieval_measurements.jsonl').read_text().splitlines():
            row = json.loads(line)
            for task in ['old', 'new']:
                old[int(row['d']), int(row['ir']), int(row['seed']), row['method'], task] = row[task]
        original_error = max(abs(float(row['recall'])-old[k]) for k, row in independent.items())
        assert original_error == 0
    output = dest/(design+'_measurements.csv')
    with output.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    result[design] = dict(**summary, independently_extracted_measurements=len(combined), independent_extraction_max_error=maxerr, original_measurement_max_error=original_error)
run(['experiments_replay/recompute_tables.py', '--matched-measurements', 'full_experiments_continuation/matched/measurements.csv',
    '--confirmation-measurements', 'full_experiments_continuation/confirmation/measurements.csv', '--output', 'full_experiments_continuation/tables'], 'full_tables')
run(['experiments_replay/recompute_tables.py', '--matched-measurements', 'full_experiments_continuation/independent_validation/matched_measurements.csv',
    '--confirmation-measurements', 'full_experiments_continuation/independent_validation/confirmation_measurements.csv', '--output', 'full_experiments_continuation/independent_validation/tables'], 'independent_full_tables')
table = json.loads((folder/'tables/result.json').read_text())
independent_table = json.loads((dest/'tables/result.json').read_text())
assert table == independent_table and table['manuscript_tables_reproduced']
for path in (folder/'tables').iterdir():
    peer = dest/'tables'/path.name
    assert peer.read_bytes() == path.read_bytes(), path.name
result.update(status='PASS', query_archives=1440, measurements=14400, tables=table,
    original_evidence_preserved=True, independent_table_files_identical=True,
    resources=json.loads((root/'PKG_continuation_resources.json').read_text()), commands=records)
(root/'PKG_continuation_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
