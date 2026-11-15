"""额外审计：实际排名算子计时、运行器流接线、修订功效的最坏偏差。"""
import json
import math
import sys
import time
import resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'audit_z4_checks'))
from checks import C,M,R,DEST,save,gate_power,cost,np

def stream_wiring():
    # 用假生成器提供确定性数列，绝不构造真实 PCG64/Generator。
    original={k:getattr(np.random,k) for k in ['SeedSequence','PCG64','Generator']}
    calls=[]
    class Fake:
        def __init__(self,e):self.e=e
        def standard_normal(self,shape,dtype):
            calls.append(dict(entropy=self.e,call='normal',shape=shape,dtype=str(dtype)))
            return np.sin(np.arange(math.prod(shape)).reshape(shape)+self.e[-1]+.5)
        def choice(self,n,Q,replace,shuffle):
            calls.append(dict(entropy=self.e,call='choice',n=n,Q=Q,replace=replace,shuffle=shuffle))
            return np.arange(Q,dtype=np.int64)[::-1]
    try:
        np.random.SeedSequence=lambda e:e
        np.random.PCG64=lambda e:e
        np.random.Generator=Fake
        result=R.dataset(1,16,-1,3,7)
    finally:
        for k,v in original.items():setattr(np.random,k,v)
    x,y0,y1,ids,z0,H=result
    d,n=x.shape
    assert np.array_equal(ids,np.arange(7)[::-1])
    assert np.allclose(x,np.sin(np.arange(d*n).reshape(d,n)+.5)/math.sqrt(d))
    assert np.allclose(H,d/math.sqrt(n)*(y1@x.T)-z0@(d/n*(x@x.T)))
    assert [v['entropy'][-1] for v in calls]==[0,1,2,3]
    save('stream_wiring.json',dict(passed=True,calls=calls,scientific_random_draws=0))

def repair():
    rows=[]
    for b in [0.,.00125,.0025,.005]:
        absolute=gate_power(1024,1280,7,b)['absolute']
        pair=gate_power(1024,1280,7,2*b)['pair']
        rows.append(dict(per_method_absolute_bias_bound=b,pair_bias_bound=2*b,
                         absolute=absolute,pair=pair,
                         all7_union_lower=max(0.,1-4*(1-absolute['power'])-3*(1-pair['power']))))
    base=cost([128],1024,1280,1)
    side=cost([128],1024,64,3)['rows'][0]['evaluation']-cost([128],1024,64,1)['rows'][0]['evaluation']
    save('repair_refined.json',dict(design=dict(dims=[128],primary_target=.425,
         n_cal_per_method=1024,n_eval=1280,q_cal=512,q_eval=1024,primary_match_F=7,primary_old_F=3,
         side_targets=[.35,.50],side_evaluation_first_fixed_seeds=64,
         side_targets_confirmatory=False),bias_sensitivity=rows,base_cost=base,
         side_curve_extra_cpu=side,total_cpu_forecast=base['cpu_seconds']+side,
         runner_soft=20200,runner_hard=20400,preparation_reserve=600,
         note='绝对方法偏差 b 的配对最坏偏差为 2b；此表优先于初版探索表。仅预算方案，未实施。'))

def timing():
    rows=[]
    for d in [128,256]:
        n=d*d//4
        t0=time.process_time();w0=time.monotonic()
        # 数组数值确定，避免训练/查询随机抽样。形状与正式程序相同。
        ix=np.arange(d*n,dtype=float).reshape(d,n)
        x=np.sin(ix*.719+.2)/math.sqrt(d)
        y0=np.cos(ix*1.137+.8)/math.sqrt(d)
        y1=np.sin(ix*1.731+.3)/math.sqrt(d)
        S=d/n*(x@x.T);z0=d/math.sqrt(n)*(y0@x.T);z1=d/math.sqrt(n)*(y1@x.T)
        H=z1-z0@S
        gen_cpu=time.process_time()-t0
        ids=np.arange(1024,dtype=np.int64)
        cal0=time.process_time();calw=time.monotonic()
        M.ranks_many(z0,M.direction(H,'GD'),M.alpha_grid('GD',17),x,y1,ids[:512])
        cal_cpu=time.process_time()-cal0;cal_wall=time.monotonic()-calw
        ev0=time.process_time();evw=time.monotonic()
        for method in C.METHODS:
            aa=[M.analytic_alpha(method,d,target) for target in [.35,.425,.5]]
            # 两臂各三个幅度；微扰只是避免重复值，不影响算子成本。
            aa=aa+[a*1.001 for a in aa]
            h=M.direction(H,method)
            for y in [y0,y1]:M.ranks_many(z0,h,aa,x,y,ids)
        rows.append(dict(d=d,n=n,deterministic_matrix_cpu=gen_cpu,
                    calibration_one_method_cpu=cal_cpu,calibration_wall=cal_wall,
                    evaluation_four_methods_two_arms_three_targets_cpu=time.process_time()-ev0,
                    evaluation_wall=time.monotonic()-evw))
    forecast=sum(4*128*(r['calibration_one_method_cpu']+r['deterministic_matrix_cpu'])+
                 224*(r['evaluation_four_methods_two_arms_three_targets_cpu']+r['deterministic_matrix_cpu'])
                 for r in rows)+500
    save('kernel_timing.json',dict(rows=rows,linear_forecast_cpu=forecast,
         peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
         limitation='每维单次、确定性非高斯数值；只验证成本量级，不是方差pilot或运行保证。未计真实高斯生成/全部磁盘fsync。'))

if __name__=='__main__':
    start=time.process_time()
    stream_wiring();repair();timing()
    save('followup_summary.json',dict(cpu_seconds=time.process_time()-start,
         peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,formal_sampling=False))
    print(json.dumps(C.read(DEST/'followup_summary.json')))
