"""Independently reduce saved query-level arrays to dataset measurements."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
METHODS=['GD','Adam2','Adam1','Adam.1','Sign']
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=[];files=sorted(a.input.glob('*.npz'));assert files
    for path in files:
        match=re.fullmatch(r'd(\d+)_r(\d+)_s(\d+)\.npz',path.name);assert match
        d,ir,seed=map(int,match.groups());n=d*d//4
        with np.load(path,allow_pickle=False) as arrays:
            ids=arrays['ids'];assert len(np.unique(ids))==len(ids) and ids.min()>=0 and ids.max()<n
            for method in METHODS:
                for task in ['old','new']:
                    prefix=method+'_'+task+'_';rank,success,D,V=[arrays[prefix+k] for k in ['ranks','success','D','V']]
                    assert all(x.shape==ids.shape for x in [rank,success,D,V])
                    assert rank.min()>=1 and rank.max()<=n and np.array_equal(success,rank<=math.ceil(.1*n))
                    assert np.isfinite(D).all() and np.isfinite(V).all() and (V>=0).all()
                    result.append(dict(d=d,ir=ir,seed=seed,method=method,task=task,recall=float(success.mean()),D=float(D.mean()),V=float(V.mean()),
                        var_D=float(D.var(ddof=1)),var_V=float(V.var(ddof=1)),cov_DV=float(np.cov(D,V,ddof=1)[0,1])))
    with a.output.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print(json.dumps(dict(status='PASS',query_archives=len(files),dataset_method_task_measurements=len(result))))
if __name__=='__main__':main()
