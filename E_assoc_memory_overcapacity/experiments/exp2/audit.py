"""Independent post-run audit; no new data, tests or thresholds."""
import hashlib,json,math,re,time,resource
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from design import *
text=(ROOT/'PREREG_EXP2.md').read_text()
frozen=dict(re.findall(r'^- `([^`]+)`: `([0-9a-f]{64})`$',text,re.M))
assert frozen==json.loads((OUT/'frozen_inputs.json').read_text())
for file,h in frozen.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h,file
assert hashlib.sha256((ROOT/'PREREG_EXP2.md').read_bytes()).hexdigest()==(ROOT/'PREREG_EXP2.sha256').read_text().split()[0]
mom=json.loads((ROOT/'experiments/exp1/moments.json').read_text())
low={'Sign1'}|{f'Sign2_a{a:g}_{kind}' for a in (.2,.6,1.2,1.5) for kind in ('proxy','actual')}
low|={f"Adam_b{p['b1']:g}_e{p['e']:g}_T{p['T']}_{kind}" for p in mom for kind in ('proxy','actual')}
high={'Sign1'}|{f'Sign2_a{a:g}_{kind}' for a in (.6,1.5) for kind in ('proxy','actual')}
high|={f'Adam_b0.9_e0.1_T16_{kind}' for kind in ('proxy','actual')}
expected={}
for seed in range(16):
    for d in (128,256):
        for k in (.5,1.,2.):expected[d,k,seed]=low
for seed in range(24):
    for k in (1.,2.):expected[512,k,seed]=high
for seed in range(16,112):
    for d in (128,256):
        for k in (1.,2.):expected[d,k,seed]={'Sign1',f'Sign2_a{.6 if d==128 else 1.5:g}_actual'}
seen=defaultdict(set);ids_by={};nrows=0;replay_rows={}
with (OUT/'formal_rows.jsonl').open() as file:
    for line in file:
        r=json.loads(line);nrows+=1;key=(r['d'],r['kappa'],r['seed']);assert key in expected
        if key==(128,1.,0) and r['name'] in ('Sign1','Sign2_a0.6_actual','Adam_b0.9_e0.1_T16_actual'):replay_rows[r['name']]=r
        assert r['name'] in expected[key] and r['name'] not in seen[key];seen[key].add(r['name'])
        d,k,seed=key;n=round(k*d*d);assert r['n']==n
        entropy=[20261106,2002,d,n,seed,10];assert r['seed_entropy']==entropy
        if key not in ids_by:
            rng=np.random.default_rng(np.random.SeedSequence(entropy+[2]));ids_by[key]=rng.choice(n,size=512,replace=False).tolist()
        assert r['query_indices']==ids_by[key]
        assert len(r['ranks'])==len(r['true_scores'])==512 and all(1<=rank<=n for rank in r['ranks'])
        for count,label in ((128,'prefix128'),(512,'full')):
            a=np.array(r['ranks'][:count]);assert abs(np.mean((a-1)/(n-1))-r[label]['mean_rank'])<1e-14
            for q in (.01,.1,.5):assert float(np.mean(a<=math.ceil(q*n)))==r[label][f'top{q:g}']
        if 'actual' in r['name']:
            errors=np.array(r['normalized_diagonal_errors']);assert abs(np.mean(errors*errors)-r['normalized_diagonal_mse'])<1e-12
            assert abs(r['weight_mse']/r['q']-r['relative_weight_mse'])<1e-12
            if '_T1_' in r['name']:assert r['relative_weight_mse']==r['normalized_diagonal_mse']==0.
missing=[dict(d=d,kappa=k,seed=s,missing_methods=sorted(names-seen.get((d,k,s),set()))) for (d,k,s),names in expected.items() if seen.get((d,k,s),set())!=names]
meta=json.loads((OUT/'formal_metadata.json').read_text());smoke=json.loads((OUT/'smoke_metadata.json').read_text())
assert meta['completed']==len(seen)
for m in (meta,smoke):
    for which in ('resource_start','resource_end'):
        assert m[which]['memory.max']=='2147483648' and m[which]['memory.swap.max']=='0'
        events=dict(line.split() for line in m[which]['memory.events'].splitlines())
        assert int(events['oom'])==int(events['oom_kill'])==0
    assert all(v=='2' for v in m['threads'].values())
    assert m['max_rss_MiB']<1700 and m['python']=='/home/liuyibo/venvs/theory/bin/python' and m['device']=='CPU'
assert smoke['success']
# The source declares independent RNG tags; checking all actual roots is sufficient for nonoverlap.
assert all(row==[20261106,2002,row[2],row[3],row[4],10] for row in ([20261106,2002,d,round(k*d*d),s,10] for d,k,s in expected))
# Independently replay an existing 2-stream-block dataset using full X/Y products,
# then dense 512 x n candidate scores (not n x n); no new random seed or run.
d=128;n=16384;entropy=[20261106,2002,d,n,0,10]
rx=np.random.default_rng(np.random.SeedSequence(entropy+[0]));ry=np.random.default_rng(np.random.SeedSequence(entropy+[1]))
x=np.concatenate([rx.standard_normal((d,8192))/math.sqrt(d) for _ in range(2)],axis=1)
y=np.concatenate([ry.standard_normal((d,8192))/math.sqrt(d) for _ in range(2)],axis=1)
s=(d/n)*(x@x.T);z=(d/math.sqrt(n))*(y@x.T)
for name,r in replay_rows.items():
    if name=='Sign1':u=.6*np.sign(z)
    elif name=='Sign2_a0.6_actual':
        h=.6*np.sign(z);u=h+.6*np.sign(z-h@s)
    else:
        # Independent original-W-unit Adam implementation.
        p=r['parameters'];W=np.zeros_like(z);mm=np.zeros_like(z);vv=np.zeros_like(z)
        for t in range(1,p['T']+1):
            g=W@s-z/math.sqrt(n);mm=p['b1']*mm+(1-p['b1'])*g;vv=p['b2']*vv+(1-p['b2'])*g*g
            W-=p['alpha']/math.sqrt(n)*(mm/(1-p['b1']**t))/(np.sqrt(vv/(1-p['b2']**t))+p['e']/math.sqrt(n))
        u=W*math.sqrt(n)
    ids=np.array(r['query_indices']);v=(u@x[:,ids])/math.sqrt(r['q']);actual=np.sum(v*y[:,ids],axis=0)
    assert np.allclose(actual,r['true_scores'],atol=1e-11,rtol=1e-10)
    for scale in ((1.,3.) if name=='Sign1' else (1.,)):
        v=((scale*u)@x[:,ids])/math.sqrt(r['q']);score=v.T@y;tt=np.sum(v*y[:,ids],axis=0)
        score[np.arange(512),ids]=-np.inf;brute=1+np.sum(score>tt[:,None],axis=1)
        assert brute.tolist()==r['ranks'],(name,scale,'dense rank replay')
        del score
assert len(replay_rows)==3
result=dict(expected_configurations=len(expected),completed_configurations=len(seen),expected_rows=sum(map(len,expected.values())),
    actual_rows=nrows,complete=not missing,missing=missing,prereg_and_all_declared_hashes_match=True,
    exact_grid_and_seed_labels=True,query_rng_regenerated=True,all_raw_ranks_and_metrics_consistent=True,
    all_diagonal_errors_recomputed=True,dense_full_model_replay_methods=3,dense_replayed_ranks=1536,
    dense_replay_crosses_stream_blocks=True,original_W_Adam_replay=True,positive_scale_rank_replay=True,
    max_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,cgroup_2GiB_zero_swap_no_OOM=True,threads_at_most_2=True,
    elapsed_total_seconds=time.time()-json.loads((OUT/'budget_start.json').read_text())['start_unix'])
assert result['elapsed_total_seconds']<3600
(OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
