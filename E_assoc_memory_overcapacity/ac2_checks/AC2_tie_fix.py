"""Replay AC2's 720 datasets with uniform-tie-integrated top-L recall.

Only this new script and its exclusively created Markdown report are written.
Historical code is read/imported for the unchanged training/design functions;
its ranking implementation is replaced here. No API or new scientific seeds.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import importlib.util
import itertools
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('historical_ac2', HERE/'AC2_independent.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def uniform_recall(X, Y, U, lists=(1, 5), query_block=256, candidate_block=2048):
    """Integrate independent uniform priorities; exact equality in float64.

    With A strictly larger and E equal wrong scores the success probability is
    clip((L-A)/(E+1),0,1). Stop only when A >= max(lists); then all fail.
    Includes U=0, Ux_i=0 and partial ties, explicitly excluding self matches.
    Returns query probabilities, capped strict counts, relevant equality counts.
    """
    n = X.shape[1]
    assert all(1 <= L <= n for L in lists)
    W = U @ X
    correct = np.sum(Y*W, axis=0)
    if not (np.isfinite(W).all() and np.isfinite(Y).all()):
        raise ValueError('nonfinite score input')
    strict, equal = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    cap = max(lists)
    for start in range(0, n, query_block):
        ids = np.arange(start, min(start+query_block, n))
        a, e = np.zeros(len(ids), dtype=np.int64), np.zeros(len(ids), dtype=np.int64)
        for j in range(0, n, candidate_block):
            active = np.flatnonzero(a < cap)
            if not len(active):
                break
            qids = ids[active]
            scores = Y[:, j:j+candidate_block].T @ W[:, qids]
            overlap = (qids >= j) & (qids < min(j+candidate_block, n))
            scores[qids[overlap]-j, np.flatnonzero(overlap)] = -np.inf
            a[active] += np.count_nonzero(scores > correct[qids], axis=0)
            e[active] += np.count_nonzero(scores == correct[qids], axis=0)
        strict[ids], equal[ids] = np.minimum(a, cap), e
    probs = np.stack([np.clip((L-strict)/(equal+1), 0., 1.) for L in lists], axis=1)
    return probs, strict, equal


def fixtures():
    # Actual positive-probability Sign2 zero endpoint, not merely hand-set U=0.
    d, n = 3, 17
    X = np.zeros((d, n)); X[:, :d] = np.sqrt(n/d)*np.eye(d)
    Z = np.array([[.2, -.3, .25], [-.15, .35, -.2], [.3, .1, -.4]])
    Y = np.zeros((d, n)); Y[:, :d] = Z/np.sqrt(d)
    S, Zactual = d/n*(X@X.T), d/np.sqrt(n)*(Y@X.T)
    U = old.train(S, Zactual, 'Sign2')
    assert np.array_equal(U, np.zeros_like(U))
    p, a, e = uniform_recall(X, Y, U, query_block=4, candidate_block=5)
    assert np.allclose(p, np.array([1/n, 5/n]), atol=0, rtol=1e-15)
    assert np.all(a == 0) and np.all(e == n-1)
    # A deterministic interior perturbation: all keys/values are nonzero,
    # scatter remains in the review's open neighborhood, and U2 is still zero.
    coords = np.arange(d*n, dtype=float).reshape(d,n)+1
    Xp = X+1e-4*np.sin(coords)
    Yp = Y+1e-4*np.cos(coords)
    Sp, Zp = d/n*(Xp@Xp.T), d/np.sqrt(n)*(Yp@Xp.T)
    assert np.linalg.norm(Sp-np.eye(d),2) < 1/(12*np.sqrt(d))
    assert np.min(np.abs(Zp)) > .05 and np.max(np.abs(Zp)) < .45
    Up = old.train(Sp,Zp,'Sign2')
    assert np.array_equal(Up,np.zeros_like(Up))
    pp, _, _ = uniform_recall(Xp,Yp,Up,query_block=4,candidate_block=5)
    assert np.allclose(pp,np.array([1/n,5/n]),atol=0,rtol=1e-15)
    # Partial ties and a zero query: enumerate every legal candidate priority.
    X = np.array([[0., 1., 2., -1., 1.]])
    Y = np.array([[1., 1., 2., -1., 1.]])
    scores = Y.T @ X
    expected = np.zeros((5, 3))
    for order in itertools.permutations(range(5)):
        priority = np.argsort(order)
        for i in range(5):
            rank = 1+sum(scores[j,i] > scores[i,i] or
                         (scores[j,i] == scores[i,i] and priority[j] < priority[i])
                         for j in range(5) if j != i)
            expected[i] += np.array([rank <= L for L in (1, 3, 5)])/120
    for qb, cb in ((1, 1), (2, 3), (8, 8)):
        p, _, _ = uniform_recall(X, Y, np.ones((1,1)), (1,3,5), qb, cb)
        assert np.allclose(p, expected, atol=3e-15, rtol=0)
    return 'Sign2 d=3,n=17 中心及开邻域扰动: top-1=1/17, top-5=5/17；部分平局/零查询的 120 个排列及三种分块全部通过。'


def advance_audit_rng(rng, moments):
    """Replay only historical audit RNG calls, in their exact original order."""
    for _ in old.METHODS:
        rng.normal(size=(2,10,100))
    for d in old.DESIGN:
        for method in old.METHODS:
            n, _ = old.load(d, moments[method]['c'])
            for _ in range(12):
                rng.normal(size=(2,d,n))
                rng.normal(size=(2*d,256))
                rng.chisquare(n-2*d, size=256)
    rng.normal(size=200000)
    rng.chisquare(983, size=200000)


def main():
    if (HERE/'AC2_tie_fix_report.md').exists():
        raise FileExistsError('Report already exists; refusing replacement before replay. '
                              'Use --check-fixtures for the deterministic regression checks.')
    wall, cpu = time.time(), time.process_time()
    paths = [HERE/'AC2_independent.py', *sorted((HERE/'full').glob('AC2_*'))]
    before = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if p.is_file()}
    check = fixtures()
    original = [json.loads(s) for s in (HERE/'full/AC2_datasets.jsonl').read_text().splitlines()]
    summary = json.loads((HERE/'full/AC2_results.json').read_text())['rows']
    moments = {m: old.efficiency(m) for m in old.METHODS}
    rng = np.random.default_rng(old.SEED)
    advance_audit_rng(rng, moments)
    index, queries, tied, zero_queries, zero_endpoints = 0, 0, 0, 0, 0
    rows = []
    for d, reps in old.DESIGN.items():
        for method in old.METHODS:
            n, _ = old.load(d, moments[method]['c'])
            values = []
            for rep in range(reps):
                X, Y = rng.normal(size=(2,d,n))/np.sqrt(d)
                S, Z = d/n*(X@X.T), d/np.sqrt(n)*(Y@X.T)
                U = old.train(S, Z, method)
                p, a, e = uniform_recall(X,Y,U)
                record = original[index]
                assert (record['d'],record['method'],record['rep'],record['n']) == (d,method,rep,n)
                counts = [int(np.sum(a == 0)), int(np.sum(a < 5))]
                assert counts == record['counts'], ('replay mismatch', index, counts)
                val = p.mean(axis=0)
                assert np.array_equal(val, np.array(record['recall'])), ('tie changed result',index)
                values.append(val)
                queries += n
                tied += int(np.count_nonzero((a < 5) & (e > 0)))
                zero_queries += int(np.count_nonzero(np.all(U@X == 0, axis=0)))
                zero_endpoints += int(not np.any(U))
                index += 1
            values = np.array(values)
            # Preserve the original RNG stream between cells (bootstrap follows data).
            draws = rng.integers(0,reps,size=(2000,reps))
            boot = values[draws].var(axis=1,ddof=1)
            current = dict(mean=values.mean(axis=0), SE=values.std(axis=0,ddof=1)/np.sqrt(reps),
                           dataset_variance=values.var(axis=0,ddof=1),
                           bootstrap_variance_95=np.quantile(boot,[.025,.975],axis=0).T)
            prior = summary[len(rows)]
            maxdiff = max(float(np.max(np.abs(v-np.array(prior[k])))) for k,v in current.items())
            assert maxdiff == 0., (d,method,maxdiff)
            rows.append((method,d,n,reps,*current['mean'],*current['dataset_variance'],maxdiff))
            print(f'{method} d={d}: {reps} datasets replayed; max table change={maxdiff}', flush=True)
    assert index == len(original) == 720
    assert before == {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in before}
    cg = Path('/proc/self/cgroup').read_text().strip().split(':')[-1]
    cgp = Path('/sys/fs/cgroup')/cg.lstrip('/')
    resources = {key:(cgp/key).read_text().strip() for key in ('memory.max','memory.swap.max','memory.peak')}
    threads = next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('Threads:'))
    report = ['# AC2 均匀平局修正与原表复算', '',
              '修正：严格超过数 A、相等错误候选数 E，top-L 成功概率为 clip((L−A)/(E+1),0,1)。'
              '对平局随机性积分；零端点和零查询自动给 L/n。不覆盖或修改原程序、原数据。', '',
              check, '',
              f'按原种子 {old.SEED} 重放审计 RNG 调用和各格 bootstrap 调用，'
              f'逐份重建全部 {index} 份完整训练集、{queries:,} 个原训练查询；'
              '每份严格计数均与历史 counts 精确相同。', '',
              f'影响 top-1/top-5 的平局查询 {tied}；零查询 {zero_queries}；零端点 {zero_endpoints}。'
              '所有均值、SE、数据集方差及 bootstrap 区间与 full/AC2_results.json 的最大绝对差均为 0。'
              'AC2_NUMERICAL_RESULTS.md 已报告的主表及方差趋势不变。', '',
              '这里复算的是积分平局后的数据集召回。一般情况下，其跨数据集方差不等于额外抽取随机平局排列后的方差；'
              '本批没有相关平局，故两种规则在本批观测上相同。未观察到平局不能证明其事件概率为零。', '',
              '|方法|d|n|B|top-1|top-5|Var1|Var5|表格最大变化|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    report += ['|'+'|'.join(str(x) if isinstance(x,(str,int)) else f'{x:.12g}' for x in row)+'|' for row in rows]
    report += ['', '相等判断使用 float64 精确相等，不将邻近浮点分数人为合并；'
               '非有限输入拒绝。独立穷举夹具核对部分平局，所有数据均保留。', '',
               f'资源：墙钟 {time.time()-wall:.2f}s；CPU {time.process_time()-cpu:.2f}s；'
               f'峰值 RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KiB；{threads}；{resources}。', '',
               '原文件读前/读后 SHA-256 一致：', '']
    report += [f'- `{name}`: `{digest}`' for name,digest in before.items()]
    with (HERE/'AC2_tie_fix_report.md').open('x') as out:
        out.write('\n'.join(report)+'\n')


if __name__ == '__main__':
    if sys.argv[1:] == ['--check-fixtures']:
        print(fixtures())
    else:
        main()
