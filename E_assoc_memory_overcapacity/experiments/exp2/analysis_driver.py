"""Run the unchanged frozen analysis and record its own resource peak."""
import json,os,resource,sys,time
from design import OUT
import analyze
from run import resource_meta
start=time.monotonic();before=resource_meta()
analyze.main()
data=dict(phase='analysis',elapsed_seconds=time.monotonic()-start,max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    python=sys.executable,threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
    resource_start=before,resource_end=resource_meta())
(OUT/'analysis_resource.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data))
