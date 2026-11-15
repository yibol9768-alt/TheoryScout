"""Reconstruct and export every recorded DP mode vector from frozen inputs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import resource,json,time
from pathlib import Path
import numpy as np
from core import coefficients,value
from e0_checks import budget_dp
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
root=Path(__file__).resolve().parent; start=time.monotonic(); count=0
existing=set()
if (root/'dp_policies.jsonl').exists():
    for line in (root/'dp_policies.jsonl').read_text().splitlines():
        old=json.loads(line);existing.add((old['id'],old['budget']))
with (root/'dp_policies.jsonl').open('a') as out, (root/'formal/results.jsonl').open() as f:
    for line in f:
        try:d=json.loads(line)
        except json.JSONDecodeError:break
        par=d['calibration']['homogeneous']; lam=par['lambda_hat'];ws=par['ws_hat'];p=np.array(d['p'] if d['dist']==0 else d['p_hat']);vs=d['v_s_hat'];vr=d['v_r_hat'];c,w,u=coefficients(lam,ws,vs,vr)
        for row in d['rows']:
            if row['policy']!='dp':continue
            if (d['id'],row['budget']) in existing:continue
            val,z=budget_dp(p,c,u,w,row['budget'],lam,tolerance=1e-11)
            assert int((z==2).sum())==row['n'] and int((z==1).sum())==row['s']
            assert abs(.5*(1+val)-row['predicted_accuracy'])<1e-12
            out.write(json.dumps({'id':d['id'],'budget':row['budget'],'modes_0_s_r':z.tolist(),'predicted_accuracy':.5*(1+val)})+'\n');count+=1
report={'status':'PASS','verified_dp_vectors':count+len(existing),'newly_exported_vectors':count,'elapsed_seconds':time.monotonic()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
(root/'dp_policy_reconstruction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
