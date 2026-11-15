"""Synthetic deterministic dense-kernel timing, no MR data/streams/recall.

The shapes match the draft, but operands are periodic trigonometric test tensors;
no dataset is generated and no scientific endpoint is computed. Not a Y3 pilot.
"""
import ctypes,json,math,os,resource,sys,time
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(160,170))
OUT=Path(__file__).resolve().parent
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:assert os.environ[key]=='2'
cg=next(s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
CP=Path('/sys/fs/cgroup')/cg.lstrip('/')
assert (CP/'memory.max').read_text().strip()=='3221225472'
assert (CP/'memory.swap.max').read_text().strip()=='0'
threads=[]
for path in {s.split()[-1] for s in Path('/proc/self/maps').read_text().splitlines() if 'openblas' in s and '.so' in s}:
    lib=ctypes.CDLL(path)
    for symbol in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads64_','openblas_get_num_threads']:
        if hasattr(lib,symbol):
            fn=getattr(lib,symbol);fn.restype=ctypes.c_int;threads.append(fn());break
assert threads and max(threads)<=2
rows=[];begin=time.process_time()
for d in [128,256,512]:
    n=d*d//4;Q=512
    base=np.arange(d*2048,dtype=float).reshape(d,2048)
    base=np.sin(.0031*base)+np.cos(.017*base)
    x=np.tile(base,(1,n//2048));y=np.tile(np.roll(base,17,axis=0),(1,n//2048))
    v=np.sin(np.arange(d*Q,dtype=float).reshape(d,Q)*.047)
    # Warm BLAS once, then time the three covariance/training-shaped GEMMs.
    warm=x[:,:2048]@y[:,:2048].T
    st=time.process_time();sw=time.monotonic()
    for _ in range(3):dummy=y@x.T
    train_cpu=time.process_time()-st;train_wall=time.monotonic()-sw
    st=time.process_time();sw=time.monotonic()
    # Ten method/task passes, full candidates, threshold comparison and squared sums.
    for _ in range(10):
        counts=np.zeros(Q,dtype=np.int64);sums=np.zeros(Q)
        for lo in range(0,n,2048):
            score=y[:,lo:lo+2048].T@v
            counts+=np.count_nonzero(score>v[0],axis=0)
            sums+=np.einsum('ij,ij->j',score,score)
    rank_cpu=time.process_time()-st;rank_wall=time.monotonic()-sw
    flops_train=6*n*d*d;flops_score=20*n*d*Q
    rows.append(dict(d=d,n=n,Q=Q,training_3_GEMM_cpu=train_cpu,training_wall=train_wall,
        candidate_10passes_cpu=rank_cpu,candidate_wall=rank_wall,core_cpu=train_cpu+rank_cpu,
        flops_train=flops_train,flops_score=flops_score,core_Gflops_per_CPU_second=(flops_train+flops_score)/(train_cpu+rank_cpu)/1e9))
    print(json.dumps(rows[-1]),flush=True)
    del x,y,v,base,score
block=3*sum(r['core_cpu'] for r in rows)
result=dict(rows=rows,projected_core_CPU_per_complete_block=block,
    projected_core_CPU_24blocks=block*24,projected_core_CPU_64blocks=block*64,
    core_only_blocks_at_5300=5300/block,
    omitted_costs=['Gaussian generation','Y1 mixing','5 update constructions','query projection','normalization and self-candidate removal','atomic persistence','final analysis'],
    warning='Hardware/shape-specific timing, no throughput guarantee; implementation and load can change it. These tensors are not MR samples.',
    peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
    CPU_seconds=time.process_time()-begin,threads=threads,
    memory_max=(CP/'memory.max').read_text().strip(),memory_swap_max=(CP/'memory.swap.max').read_text().strip())
with (OUT/'kernel_budget.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
