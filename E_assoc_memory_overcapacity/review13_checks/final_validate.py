"""Final read-only input and prediction checks; exclusive review outputs."""
import csv,ctypes,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CPU,(40,45))
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:assert os.environ[k]=='2'
cg=next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
CP=Path('/sys/fs/cgroup')/cg.lstrip('/')
assert (CP/'memory.max').read_text().strip()=='3221225472' and (CP/'memory.swap.max').read_text().strip()=='0'
threads=[]
for path in {s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in s and '.so' in s}:
    lib=ctypes.CDLL(path)
    for symbol in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads64_','openblas_get_num_threads']:
        if hasattr(lib,symbol):
            fn=getattr(lib,symbol);fn.restype=ctypes.c_int;threads.append(fn());break
assert threads and max(threads)<=2
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
ind={(int(r['ir']),r['method'],r['task']):r for r in read(OUT/'scalar_and_block_checks.csv')}
pred=read(ROOT/'y3_checks/Y3_frozen_predictions.csv')
assert len(pred)==90
assert len({(r['d'],r['ir'],r['method'],r['task']) for r in pred})==90
err=0.
for r in pred:
    d=int(r['d']);assert d in [128,256,512]
    c=ind[int(r['ir']),r['method'],r['task']]
    expected=float(c['recall_limit'])+float(c['B_recall'])/d
    err=max(err,abs(expected-float(r['recall_first_order'])))
    if r['task']=='new':err=max(err,abs(float(c['B_recall'])/d-float(r['new_matching_bias'])))
assert err<1e-10
manifest={}
for line in (ROOT/'Y3_FROZEN.sha256').read_text().splitlines():
    h,p=line.split('  ',1);manifest[p]=sha(ROOT/p)==h
assert all(manifest.values())
old=json.loads((OUT/'existing_nonlog_sha256.json').read_text())
changed=[p for p,h in old.items() if sha(ROOT/p)!=h]
assert not changed,changed
allowed=[];unexpected=[]
for path in ROOT.rglob('*'):
    if not path.is_file():continue
    rel=str(path.relative_to(ROOT))
    if rel in old or path.suffix=='.log':continue
    if path==ROOT/'PROOF_REVIEW13.md' or OUT in path.parents:allowed.append(rel)
    else:unexpected.append(rel)
report=(ROOT/'PROOF_REVIEW13.md').read_text()
assert all(s in report for s in ['需修正','FAIL','PASS','候选','18.3%','20.1%','2299'])
result={'frozen_hashes':manifest,'prediction_rows':90,'max_independent_prediction_difference':err,
    'existing_nonlog_files_rechecked':len(old),'existing_nonlog_files_changed':changed,
    'new_review_paths':sorted(allowed),'other_new_nonlog_paths_observed':unexpected,
    'note':'Other new paths, if any, are external concurrency observations, not modifications by this review.',
    'new_Y3_training_data_generated_by_review':False,'threads':threads,
    'memory_max':(CP/'memory.max').read_text().strip(),'memory_swap_max':(CP/'memory.swap.max').read_text().strip(),
    'cpu_seconds':time.process_time(),'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
with (OUT/'final_validation.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
files=sorted([ROOT/'PROOF_REVIEW13.md']+[p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.log' and p.name!='ARTIFACTS.sha256'])
with (OUT/'ARTIFACTS.sha256').open('x') as f:
    for path in files:f.write(sha(path)+'  '+str(path.relative_to(ROOT))+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['new_review_paths','frozen_hashes']},ensure_ascii=False))
