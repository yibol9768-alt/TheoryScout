import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import resource,json
from pathlib import Path
import numpy as np
from core import calibration_index
resource.setrlimit(resource.RLIMIT_AS,(1900000000,1900000000))
cases=0
for g in (16,32,64):
    for target in (0,g-1):
        for field in (0,3):
            for typ in (0,1):
                for m in (1,8,16,32,64,128,256):
                    if m>5*g:continue
                    for fraction in (0.,.5,1.):
                        for variant in (0,1):
                            ids,ns,nr=calibration_index(0,g,target,field,typ,m,fraction,variant)
                            assert ns+nr==len(ids) and len(np.unique(ids))==len(ids)
                            sg=ids[ids<g]; rg=(ids[ids>=g]-g)//4
                            assert not np.intersect1d(sg,np.unique(rg)).size
                            assert ids[0]==(target if typ==0 else g+4*target+field)
                            if m>1:
                                assert all((rg==group).sum()==4 for group in np.unique(rg))
                                assert len(ids)<=4*g and len(sg)+len(np.unique(rg))<=g
                            cases+=1
out={'status':'PASS','episode_exclusive_calibration_indices':cases,'full_raw_episodes_except_single_evidence_anchor':True}
Path(__file__).with_name('calibration_audit_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
