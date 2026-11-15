from pathlib import Path
import json,csv,math,time,hashlib
import numpy as np
root=Path.cwd();base=root/'supplement/experiments_replay';start=time.monotonic();hashes={};checked=0;worst=0.
rows=[json.loads(line) for line in (base/'matched_retrieval_measurements.jsonl').read_text().splitlines()]
mr={}
for r in rows:mr.setdefault((r['d'],r['ir'],r['seed']),[]).append(r)
for (d,ir,seed),rr in mr.items():
    path=root/'experiments/x1'/f'X1_formal_d{d}_r{ir}_s{seed}.npz'
    hashes[f'matched_retrieval_d{d}_r{ir}_seed{seed}']=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as a:
        assert len(a['ids'])==2048
        for r in rr:
            for task in ['old','new']:
                ranks=a[r['method']+'_'+task+'_ranks'];success=a[r['method']+'_'+task+'_success']
                assert np.array_equal(success,ranks<=math.ceil(.1*(d*d//4)))
                assert float(success.mean())==r[task]
    checked+=1
with (base/'finite_dimension_measurements.csv').open() as f:rows=list(csv.DictReader(f))
fd={}
for r in rows:fd.setdefault(tuple(int(r[k]) for k in ['d','ir','seed']),[]).append(r)
for (d,ir,seed),rr in fd.items():
    path=root/'y3d/formal'/f'd{d}_r{ir}_s{seed}.npz'
    hashes[f'finite_dimension_d{d}_r{ir}_seed{seed}']=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as a:
        assert len(a['ids'])==(8192 if (d,ir)==(512,2) else 512)
        for r in rr:
            prefix=r['method']+'_'+r['task']+'_';rank=a[prefix+'ranks'];ok=a[prefix+'success'];D=a[prefix+'D'];V=a[prefix+'V']
            assert np.array_equal(ok,rank<=math.ceil(.1*(d*d//4)))
            observed=dict(recall=float(ok.mean()),D=float(D.mean()),V=float(V.mean()),var_D=float(D.var(ddof=1)),var_V=float(V.var(ddof=1)),cov_DV=float(np.cov(D,V,ddof=1)[0,1]))
            for key,v in observed.items():worst=max(worst,abs(v-float(r[key])));assert abs(v-float(r[key]))<1e-12
    checked+=1
result=dict(status='PASS',matched_datasets=len(mr),confirmation_datasets=len(fd),query_level_archives_checked=checked,max_export_error=worst,seconds=time.monotonic()-start)
(root/'PKG_measurement_validation.json').write_text(json.dumps(dict(**result,original_archives_sha256=hashes),indent=2))
(root/'supplement/experiments_replay/measurement_provenance.json').write_text(json.dumps(dict(**result,original_archives_sha256=hashes),indent=2))
print(json.dumps(result),flush=True)
