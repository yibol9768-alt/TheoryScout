"""独立 Z4 审计：仅离线推导、确定性小矩阵、合成凭据；不创建科学 RNG。"""
import os
import sys
import json
import math
import hashlib
import resource
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'audit_z4_checks'
sys.path.insert(0, str(ROOT / 'z4'))
import numpy as np
from scipy.stats import norm, t
import common_v2 as C
import model as M
import run_v2 as R
import analyze_v2 as A
import protocol_v2 as P

def save(name, value):
    with (DEST / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def hashes():
    entries = [line.split('  ', 1) for line in (ROOT/'Z4_FROZEN.sha256').read_text().splitlines()]
    bad = [rel for h, rel in entries if C.sha(ROOT/rel) != h]
    inputs = {}
    for name in ['input_hashes.json', 'additional_inputs.json']:
        obj = C.read(ROOT/'z4'/name)
        for rel, h in obj.items():
            if isinstance(h, str) and len(h) == 64:
                inputs[rel] = C.sha(ROOT/rel) == h
    result = dict(manifest_sha256=C.sha(ROOT/'Z4_FROZEN.sha256'), entries=len(entries),
                  mismatches=bad, input_checks=inputs, formal_absent=not (ROOT/'z4/formal').exists(),
                  environment=C.environment())
    save('freeze_before.json', result)
    return result

def independent_model():
    # 无随机生成：固定解析数组，独立从 W 坐标平方损失梯度构造端点。
    d, n = 7, 13
    ix = np.arange(d*n, dtype=float).reshape(d,n)
    x = (np.sin(ix*1.137+.13)+np.cos(ix*.371+.4))/math.sqrt(2*d)
    y0 = (np.cos(ix*.743+.2)+np.sin(ix*1.313+.7))/math.sqrt(2*d)
    y1 = (np.sin(ix*.917+.9)+np.cos(ix*1.721+.5))/math.sqrt(2*d)
    w0 = d/n*(y0@x.T)
    grad = d/n*((w0@x-y1)@x.T)
    z0 = d/math.sqrt(n)*(y0@x.T)
    S = d/n*(x@x.T)
    H = d/math.sqrt(n)*(y1@x.T)-z0@S
    ids = np.array([12, 1, 7, 4], dtype=np.int64)
    rows=[]
    for method in C.METHODS:
        for alpha in [0., .19, .71]:
            if method == 'GD':
                w = w0-alpha*grad
            elif method == 'Sign':
                w = w0-alpha/math.sqrt(n)*np.sign(grad)
            else:
                e = {'Adam2':2.,'Adam.1':.1}[method]*math.sqrt(2)/math.sqrt(n)
                # 重置且偏差校正首步：mhat=grad，vhat=grad²。
                w = w0-alpha/math.sqrt(n)*grad/(np.sqrt(grad**2)+e)
            u = z0+alpha*M.direction(H,method)
            assert np.allclose(math.sqrt(n)*w,u,rtol=1e-12,atol=1e-12)
            for task,y in [('old',y0),('new',y1)]:
                scores = y.T@(math.sqrt(n)*w@x[:,ids])
                direct = np.array([list(np.lexsort((np.arange(n),-scores[:,j]))).index(k)+1
                                   for j,k in enumerate(ids)])
                actual = M.ranks_many(z0,M.direction(H,method),[alpha],x,y,ids,qc=3,cc=5)[0]
                assert np.array_equal(actual,direct)
                rows.append(dict(method=method,alpha=alpha,task=task,ranks=direct.tolist(),
                                 max_endpoint_error=float(np.max(np.abs(math.sqrt(n)*w-u)))))
    tied=M.ranks_many(z0*0,H*0,[0.,1.],x,y0,ids,qc=3,cc=5)
    assert np.array_equal(tied,np.tile(ids+1,(2,1)))
    save('independent_model.json',dict(passed=True,rows=rows,ties_passed=True,
        note='确定性数组测试代数与排名，不以伪数据验证高斯分布定理'))

def seeds():
    cfg=C.config()
    expected=[]
    for d in cfg['dims']:
        for m in range(4):
            for s in range(cfg['n_cal']):
                for k in range(4): expected.append([20261106,4097162831,4,0,d,m+1,s,k])
    for d in cfg['dims']:
        for s in range(cfg['n_eval']):
            for k in range(4): expected.append([20261106,4097162831,4,1,d,0,s,k])
    stored=C.read(ROOT/'z4/streams.json')
    assert stored==expected
    def state(e): return tuple(int(v) for v in np.random.SeedSequence(e).generate_state(8))
    new={state(e) for e in expected}
    old=C.read(ROOT/'z4/old_entropies.json')
    overlap=sum(state(e) in new for e in old)
    assert len(new)==len(expected) and overlap==0
    save('seeds.json',dict(planned=len(expected),old=len(old),internal_collisions=len(expected)-len(new),
         recorded_old_collisions=overlap,training_or_query_draws=0,
         limitation='只核对已存旧熵集合；不证明未记录历史或独立性'))

def gate_power(ncal, neval, family, bias=0.):
    crit=float(t.ppf(1-.025/(2*family),neval-1))
    def calc(sdcal,sdev):
        half=crit*sdev/math.sqrt(neval)
        sd=math.sqrt(sdcal**2/ncal+sdev**2/neval)
        margin=.01-half
        power=max(0.,float(norm.cdf((margin-bias)/sd)-norm.cdf((-margin-bias)/sd))) if margin>0 else 0.
        return dict(halfwidth=half,total_sd=sd,power=power)
    return dict(critical=crit,absolute=calc(.030,.025),pair=calc(math.sqrt(2)*.030,.030))

def cost(dims,ncal,neval,targets):
    kernel=C.read(ROOT/'review13_checks/kernel_budget.json')['rows']
    rows=[]
    for row in kernel:
        if row['d'] not in dims: continue
        tr=row['training_3_GEMM_cpu']; scan=row['candidate_10passes_cpu']/10
        gen=.035*row['d']*row['n']/1e6
        cal=ncal*4*(2*(tr+scan*17*2)+gen+1)
        ev=neval*(2*(tr+scan*2*targets*4*2*2*2)+gen+1)
        rows.append(dict(d=row['d'],cal=cal,evaluation=ev))
    return dict(rows=rows,cpu_seconds=sum(r['cal']+r['evaluation'] for r in rows)+500)

def power():
    original=gate_power(128,224,42)
    # 全族成功为任意一个必需门控成功的子集，无需假设门控之间独立。
    original['all18_upper_bound']=original['pair']['power']
    revised=[]
    for b in [0.,.00125,.0025,.005]:
        p=gate_power(1024,1024,7,b)
        p.update(bias_per_absolute=b,pair_bias_bound=b,
                 all7_union_lower=max(0.,1-4*(1-p['absolute']['power'])-3*(1-p['pair']['power'])))
        revised.append(p)
    # 重新分配方案的完整预算仍使用原规划的保守外推公式。
    costs=cost([128],1024,1024,1)
    # 在解析有限维曲线上评估固定17点网格插值误差，作为风险诊断，不是实测界。
    interpolation=[]
    for d in [128,256]:
        for method in C.METHODS:
            grid=M.alpha_grid(method,17)
            raw=np.array([M.predict(method,a,d,'new')['finite'] for a in grid])
            for target in [.35,.425,.5]:
                s=M.select(grid,raw,target)
                a=s['alpha']
                interpolation.append(dict(d=d,method=method,target=target,alpha=a,
                    predicted_target_error=M.predict(method,a,d,'new')['finite']-target if a else None))
    gap={}
    for a,b in C.PAIRS:
        aa=M.analytic_alpha(a,128,.425);ab=M.analytic_alpha(b,128,.425)
        gap[a+'-'+b]=M.predict(a,aa,128,'old')['finite']-M.predict(b,ab,128,'old')['finite']
    save('power_and_repair.json',dict(original=original,repair_bias_sensitivity=revised,
        repair=dict(dims=[128],targets=[.425],ncal_per_method=1024,neval=1024,qcal=512,qeval=1024,
                    family_match=7,family_old=3,cost=costs,gaps=gap),
        original_cost=cost([128,256],128,224,3),grid_model_diagnostic=interpolation,
        caveat='所有功效仍以原SD假设和正态规划为条件；偏差不是已知实测上界。'))

def integrity():
    # 重放已存阳性夹具，所有注入均发生在新建的审计子目录。
    from shutil import copytree
    src=ROOT/'z4/synthetic_final/stable_control'
    data=DEST/'stable_control'
    copytree(src,data)
    cfg=C.read(data/'launch.json')['config'];mh=C.read(data/'launch.json')['manifest']
    accepted=P.verify_accepted(data,mh,cfg)
    tests=[dict(name='frozen_positive_replay',passed=accepted['confirmation_valid'])]
    for name,mutate in [
        ('observer_missing',lambda p:(p/'EXIT_OBSERVED.json').unlink()),
        ('observer_killed',lambda p:change(p/'EXIT_OBSERVED.json',dict(returncode=-9,killed=True))),
        ('runner_abnormal_exit',lambda p:change(p/'launch_status.json',dict(stages=[]))),
        ('report_bytes',lambda p:(p/'analysis/old_pairs.csv').open('a').write('tamper\n')),
        ('partial_remains',lambda p:(p/'extra.partial').write_bytes(b'x')),
        ('extra_npz',lambda p:(p/'extra.npz').write_bytes(b'x')),
        ('event_truncated',lambda p:(p/'events.jsonl').write_bytes((p/'events.jsonl').read_bytes()[:-17])),
    ]:
        p=DEST/name;copytree(src,p);mutate(p)
        try: P.verify_accepted(p,mh,cfg)
        except Exception as e: tests.append(dict(name=name,passed=True,rejection=repr(e)))
        else: tests.append(dict(name=name,passed=False))
    assert all(v['passed'] for v in tests)
    save('integrity.json',dict(checks=tests,note='重放与注入验证；没有启动正式 worker，没有新造科学数据。'))

def change(path, values):
    obj=C.read(path);obj.update(values);path.write_text(json.dumps(obj))

def main():
    start=time.process_time();initial=hashes()
    independent_model();seeds();power();integrity()
    mismatches=[]
    for line in (ROOT/'Z4_FROZEN.sha256').read_text().splitlines():
        h,rel=line.split('  ',1)
        if C.sha(ROOT/rel)!=h:mismatches.append(rel)
    u=resource.getrusage(resource.RUSAGE_SELF)
    save('summary.json',dict(completed=True,final_frozen_mismatches=mismatches,
         manifest_unchanged=C.sha(ROOT/'Z4_FROZEN.sha256')==initial['manifest_sha256'],
         formal_absent=not (ROOT/'z4/formal').exists(),cpu_seconds=time.process_time()-start,
         total_cpu=u.ru_utime+u.ru_stime,peak_rss_mib=u.ru_maxrss/1024,formal_sampling=False,
         llm_api_calls=0))
    print(json.dumps(C.read(DEST/'summary.json'),ensure_ascii=False))

if __name__=='__main__':main()
