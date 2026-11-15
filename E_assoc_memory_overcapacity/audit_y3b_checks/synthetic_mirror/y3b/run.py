"""正式运行器。只有人工审阅后经 launch.py 调用；本轮不调用。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU, (21100, 21340))
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
from common import ROOT, OUT, METHODS, TASKS, RS, config, entropy, jobs, verify, dump, sha, usage
from predict import load

STOP = None
def soft_stop(signum, frame):
    global STOP
    STOP = 'cpu_soft' if signum == signal.SIGXCPU else 'wall_soft' if signum == signal.SIGALRM else 'signal_' + str(signum)

class StopRun(Exception): pass
def checkpoint():
    if STOP: raise StopRun(STOP)

def rng(d, ir, seed, stream):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy(d, ir, seed, stream))))

def dataset(d, ir, seed, Q):
    n = d*d//4
    # 每流只调用一次standard_normal，float64，C顺序(d,n)；就地缩放。
    x = rng(d, ir, seed, 0).standard_normal((d, n), dtype=np.float64)
    x /= math.sqrt(d); checkpoint()
    y0 = rng(d, ir, seed, 1).standard_normal((d, n), dtype=np.float64)
    y0 /= math.sqrt(d); checkpoint()
    y1 = rng(d, ir, seed, 2).standard_normal((d, n), dtype=np.float64)
    y1 /= math.sqrt(d)
    y1 *= math.sqrt(1 - RS[ir]**2)
    y1 += RS[ir] * y0; checkpoint()
    # choice返回原顺序；不排序，不追加重抽。
    ids = rng(d, ir, seed, 3).choice(n, size=Q, replace=False, shuffle=True)
    checkpoint()
    s = (d/n) * (x @ x.T); checkpoint()
    z0 = (d/math.sqrt(n)) * (y0 @ x.T); checkpoint()
    z1 = (d/math.sqrt(n)) * (y1 @ x.T); checkpoint()
    h = z1 - z0 @ s
    return x, y0, y1, ids, z0, h

def direction(h, m, r):
    if m == 'GD': return h
    if m == 'Sign': return np.sign(h)
    e = {'Adam2': 2., 'Adam1': 1., 'Adam.1': .1}[m]
    return h / (np.abs(h) + e * math.sqrt(2*(1-r)))

def ranks(u, x, y, ids, q, query_chunk=256, candidate_chunk=2048):
    """所有训练候选；正确项einsum，候选GEMM；显式排除自比较。"""
    n = x.shape[1]
    ranks_out = np.empty(len(ids), dtype=np.int64)
    ds = np.empty(len(ids)); vs = np.empty(len(ids))
    for a in range(0, len(ids), query_chunk):
        checkpoint()
        ix = ids[a:a+query_chunk]
        uv = u @ x[:, ix]
        truth = np.einsum('ij,ij->j', y[:, ix], uv)
        counts = np.zeros(len(ix), dtype=np.int64)
        squares = np.zeros(len(ix))
        for lo in range(0, n, candidate_chunk):
            hi = min(lo+candidate_chunk, n)
            score = y[:, lo:hi].T @ uv
            mask = (ix >= lo) & (ix < hi)
            score[ix[mask]-lo, np.flatnonzero(mask)] = 0.
            squares += np.einsum('ij,ij->j', score, score)
            gt = (score > truth) | ((score == truth) & (np.arange(lo, hi)[:, None] < ix))
            gt[ix[mask]-lo, np.flatnonzero(mask)] = False
            counts += np.count_nonzero(gt, axis=0)
            checkpoint()
        ranks_out[a:a+len(ix)] = counts+1
        ds[a:a+len(ix)] = truth / math.sqrt(q)
        vs[a:a+len(ix)] = squares / ((n-1)*q)
    return ranks_out, ranks_out <= math.ceil(.1*n), ds, vs

def cell(d, ir, seed, cfg, params):
    Q = cfg['query_target'] if (d, ir) == (512, 2) else cfg['query_default']
    x, y0, y1, ids, z0, h = dataset(d, ir, seed, Q)
    arrays = dict(ids=ids)
    for m in METHODS:
        checkpoint()
        p = params[ir, m]
        u = z0 + p['amplitude'] * direction(h, m, RS[ir])
        for task, y in [('old', y0), ('new', y1)]:
            rank, ok, D, V = ranks(u, x, y, ids, p['q'], cfg['query_chunk'], cfg['candidate_chunk'])
            for suffix, v in [('ranks', rank), ('success', ok), ('D', D), ('V', V)]:
                arrays[m+'_'+task+'_'+suffix] = v
    return arrays

def commit(path, arrays):
    # link排他发布完整NPZ；绝不覆盖任何已存在的最终文件。
    partial = path.with_suffix('.partial')
    with partial.open('xb') as f:
        np.savez(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    os.link(partial, path)
    partial.unlink()
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def main():
    assert sys.argv[1:] == ['--formal-via-launcher']
    assert os.environ.get('Y3B_LAUNCHER') == '1'
    mh = verify()  # 任何Generator/训练/查询抽样之前
    cfg = config(); pred, params = load()
    output = OUT / 'formal'
    assert output.is_dir() and (output / 'launch.json').is_file()
    for sig in [signal.SIGXCPU, signal.SIGALRM, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, soft_stop)
    signal.alarm(cfg['runner_wall_soft'])
    start = time.monotonic()
    dump(output / 'start.json', dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         manifest_sha256=mh, streams_sha256=sha(OUT/'streams.json'),
         program_sha256=sha(Path(__file__)), cfg=cfg, usage=usage(), no_draw_before_this_record=True))
    completed = []; current = None; reason = 'planned_complete'; error = None
    with (output / 'events.jsonl').open('x') as f:
        def event(obj):
            f.write(json.dumps(obj, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        try:
            for d, ir, seed in jobs(cfg['blocks']):
                checkpoint()
                current = [d, ir, seed]
                event(dict(state='started', cell=current, usage=usage()))
                arrays = cell(d, ir, seed, cfg, params)
                arrays['metadata'] = np.array(json.dumps(dict(d=d, ir=ir, seed=seed,
                    manifest_sha256=mh, schema=1)))
                path = output / f'd{d}_r{ir}_s{seed}.npz'
                commit(path, arrays)
                completed.append(current)
                event(dict(state='complete', cell=current, sha256=sha(path), usage=usage()))
                print(json.dumps(dict(completed=len(completed), cell=current, cpu=usage()['cpu_seconds'])), flush=True)
                current = None
                del arrays
        except StopRun as exc:
            reason = str(exc)
        except BaseException as exc:
            reason = 'exception'; error = repr(exc)
        event(dict(state='stopped', reason=reason, incomplete_cell=current, error=error, usage=usage()))
    signal.alarm(0)
    dump(output / 'runner_status.json', dict(reason=reason, error=error, completed=completed,
         incomplete_cell=current, usage=usage(), wall_seconds=time.monotonic()-start))
    return 0 if reason != 'exception' else 1

if __name__ == '__main__':
    sys.exit(main())
