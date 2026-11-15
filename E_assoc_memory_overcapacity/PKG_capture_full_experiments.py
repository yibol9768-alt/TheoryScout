from pathlib import Path
import csv,hashlib,json,tarfile
root=Path.cwd();base=root/'supplement';folder=base/'full_experiments_recomputation'
for design,expected in [('matched',1152),('confirmation',288)]:
    r=json.loads((folder/design/'result.json').read_text());assert r['status']=='PASS' and r['full'] and r['datasets']==expected and r['max_recall_error']==0
    with (folder/design/'measurements.csv').open() as f:actual=list(csv.DictReader(f))
    assert len(actual)==expected*10
    if design=='confirmation':
        with (base/'experiments_replay/finite_dimension_measurements.csv').open() as f:old=list(csv.DictReader(f))
        expected={(int(r['d']),int(r['ir']),int(r['seed']),r['method'],r['task']):r for r in old}
        error=max(abs(float(r[k])-float(expected[int(r['d']),int(r['ir']),int(r['seed']),r['method'],r['task']][k])) for r in actual for k in ['recall','D','V','var_D','var_V','cov_DV'])
        assert error<1e-12,error
result=json.loads((folder/'tables/result.json').read_text());assert result['manuscript_tables_reproduced']
resources=json.loads((root/'PKG_full_resources.json').read_text());assert resources['max_threads_per_process']<=2
assert 'oom 0\n' in resources['resource_limits']['memory.events']+'\n' and 'oom_kill 0\n' in resources['resource_limits']['memory.events']+'\n'
artifact=root/'PKG_full_experiment_queries.tar';assert not artifact.exists()
with tarfile.open(artifact,'w') as f:f.add(folder,arcname='full_experiments')
meta=dict(status='PASS',file=artifact.name,bytes=artifact.stat().st_size,sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),matched_datasets=1152,confirmation_datasets=288,query_archives=1440,all_recall_errors=0.,max_confirmation_measurement_error=error,manuscript_tables_reproduced=True,resources=resources)
(root/'PKG_full_replay_verification.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
