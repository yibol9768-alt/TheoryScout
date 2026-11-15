"""离线流核验、完整确认功效和成本复算；不建立 Generator。"""
import ast
import itertools
import json
import math
import resource
import subprocess
from pathlib import Path
import numpy as np
from scipy.stats import norm,t,nct
from common_v2 import ROOT,OUT,read,dump,sha,config,entropy,environment,usage,METHODS,PAIRS
from model import analytic_alpha,predict

def main():
    start=usage();cfg=config();inputs={}
    def source(rel):
        p=ROOT/rel
        assert p.resolve().is_relative_to(ROOT) and not p.is_symlink()
        inputs[rel]=sha(p);return p
    for name in ['input_hashes.json','additional_inputs.json']:
        for rel,h in read(source('z4/'+name)).items(): assert sha(source(rel))==h,rel
    for rel in ['AUDIT_Z4.md','audit_z4_checks/repair_refined.json','audit_z4_checks/complete_power_scenario.json',
                'Z4_PREREG.md','Z4_FROZEN.sha256','z4/streams.json','z4/old_entropies.json','z4/rng_inventory.json']:
        source(rel)
    old={tuple(e) for e in read(ROOT/'z4/old_entropies.json')}
    old.update(tuple(e) for e in read(ROOT/'z4/streams.json'))
    # 此前审计之后新增的 RNG 源重新登记；静态文字不执行。
    paths=subprocess.check_output(['rg','--files','--hidden','-g','*.py','-g','!z4b/**','-g','!Z4B_*','-g','!.git/**'],cwd=ROOT,text=True).splitlines()
    known={x['file'] for x in read(ROOT/'z4/rng_inventory.json')}
    inventory=[];new_sources=[]
    tokens=['SeedSequence','default_rng','RandomState','random.seed','manual_seed','Sobol(']
    for rel in paths:
        p=ROOT/rel
        if p.is_symlink():continue
        txt=p.read_text(errors='replace')
        hits=[dict(file=rel,line=i,text=line.strip()) for i,line in enumerate(txt.splitlines(),1) if any(x in line for x in tokens)]
        if not hits:continue
        source(rel);inventory.extend(hits)
        if rel not in known and not rel.startswith('z4/'):
            new_sources.append(rel)
        tree=ast.parse(txt)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['SeedSequence','default_rng','RandomState'] and node.args:
                try:v=ast.literal_eval(node.args[0])
                except (ValueError,TypeError):continue
                if type(v) is int and v>=0:old.add((v,))
                elif isinstance(v,(list,tuple)) and v and all(type(x) is int and x>=0 for x in v):old.add(tuple(v))
    # 新增源只允许已审计的 Z4 只派生状态/假 RNG 接线代码。
    allowed={'audit_z4_checks/checks.py','audit_z4_checks/followup.py','audit_z4_checks/volume_check.py',
             'aa1_checks/AA1_verify.py','aa2_checks/retrieval_check.py','aa2_checks/cue_check.py'}
    assert set(new_sources)<=allowed,('unreviewed RNG source',new_sources)
    # AA1 固定两整数种子；AA2 是单整数域。保守覆盖 d=1..4096、s=0..255
    # 两个相邻基址（含默认 [64,128,256]×6 及可见配置），不读取科学终点选择设计。
    old.update([(202611060100,),(202611060101,)])
    for d in range(1,4097):
        for s in range(257):old.add((202611060000+10000*d+s,))
    for rel in ['aa2_checks/cue_results.json','aa2_checks/cue_large_results.json','aa2_checks/retrieval_results.json']:
        params=read(source(rel))['parameters']
        for d in params['dims']:
            for s in range(params['seeds']+1):old.add((202611060000+10000*d+s,))
    # 新增/已有 JSON 中显式 entropy 记录；不导入旧实验，不运行其代码。
    record_paths=subprocess.check_output(['rg','-l','--hidden','-g','*.json','-g','*.jsonl','-g','!z4b/**','-g','!Z4B_*','-g','!.git/**','"[^"]*entropy[^"]*"'],cwd=ROOT,text=True).splitlines()
    def add(v):
        if type(v) is int and v>=0:old.add((v,))
        elif isinstance(v,list) and v and all(type(x) is int and x>=0 for x in v):
            old.add(tuple(v))
            for k in range(6):old.add(tuple(v+[k]))
    def walk(o):
        if isinstance(o,dict):
            for k,v in o.items():
                if 'entropy' in k.lower():add(v)
                walk(v)
        elif isinstance(o,list):
            for v in o:
                if isinstance(v,(dict,list)):walk(v)
    for rel in record_paths:
        p=source(rel)
        if p.suffix=='.json':walk(read(p))
        else:
            with p.open() as f:
                for line in f:
                    if 'entropy' in line:walk(json.loads(line))
    streams=[entropy(0,d,m,s,k) for d,m,s,k in itertools.product(cfg['dims'],range(4),range(cfg['n_cal']),range(4))]
    streams += [entropy(1,d,-1,s,k) for d,s,k in itertools.product(cfg['dims'],range(cfg['n_eval']),range(4))]
    def state(e):return tuple(map(int,np.random.SeedSequence(e).generate_state(8)))
    new={state(e) for e in streams};collision=sum(state(e) in new for e in old)
    assert len(new)==len(streams) and collision==0
    dump(OUT/'streams.json',streams);dump(OUT/'old_entropies.json',sorted(map(list,old)))
    dump(OUT/'rng_inventory.json',inventory)
    dump(OUT/'seed_audit.json',dict(passed=True,new_streams=len(streams),old_unique=len(old),internal_collisions=0,
         old_collisions=collision,state_bits=256,new_sources_reviewed=new_sources,record_files=len(record_paths),
         training_or_query_draws=0,limitation='可见记录、静态及既有保守动态域；不能核验未记录历史，不把不同状态称为数学独立性'))
    ne=cfg['n_eval'];nc=cfg['n_cal'];c=float(t.ppf(1-.025/14,ne-1));co=float(t.ppf(1-.025/6,ne-1))
    def scenario(b,scale=1.):
        def gate(se,sc,bias):
            half=c*se*scale/math.sqrt(ne);sd=math.sqrt((sc*scale)**2/nc+(se*scale)**2/ne);room=.01-half
            return dict(halfwidth=half,total_sd=sd,power=max(0.,float(norm.cdf((room-bias)/sd)-norm.cdf((-room-bias)/sd))) if room>0 else 0.)
        a=gate(.025,.03,b);p=gate(.03,math.sqrt(2)*.03,2*b)
        calfail=min(1.,68*2*float(norm.sf(.006*math.sqrt(nc)/(.03*scale))))
        oldfail=3*float(nct.cdf(co,ne-1,.03*math.sqrt(ne)/(.03*scale)))
        lower=max(0.,1-4*(1-a['power'])-3*(1-p['power'])-calfail-oldfail)
        return dict(per_method_bias_bound=b,pair_bias_bound=2*b,SD_multiplier=scale,absolute=a,pair=p,
                    calibration_failure_union_upper=calfail,old_direction_failure_union_upper=oldfail,
                    complete_primary_direction_power_planning_lower=lower)
    cases=[scenario(b) for b in [0.,.00125,.0025,.005]]+[scenario(.0025,k) for k in [1.1,1.25,1.5]]
    maincase=cases[2]
    predictions=[]
    for a,b in PAIRS:
        aa=analytic_alpha(a,128,.425);ab=analytic_alpha(b,128,.425)
        pa=predict(a,aa,128,'old');pb=predict(b,ab,128,'old')
        la=analytic_alpha(a,128,.425,False);lb=analytic_alpha(b,128,.425,False)
        pop=predict(a,la,128,'old')['limit']-predict(b,lb,128,'old')['limit']
        predictions.append(dict(pair=a+'-'+b,finite_gap=pa['finite']-pb['finite'],
              population_matched_limit_gap=pop,finite_matched_total_shift=pa['finite']-pb['finite']-pop,
              fixed_alpha_B_shift=(pa['B']-pb['B'])/128))
    assert maincase['complete_primary_direction_power_planning_lower']>=.8
    dump(OUT/'power.json',dict(design_qualified=True,qualification='完整主方向终点在写明假设的正态规划近似下>=.8；不构成未知真实分布的保证',
         primary_endpoint='primary_all3_directions',primary_matching_F=7,primary_old_F=3,
         assumptions=dict(sd_cal_grid_max=.030,sd_new_eval_max=.025,sd_new_pair_eval_max=.030,sd_old_pair_eval_max=.030,
             per_method_systematic_target_bias_max=.0025,pair_systematic_bias_max=.005,
             population_PAVA_discrepancy_max=.003,target_endpoint_margin_min=.03,
             all_68_grid_mean_errors_sufficient_bound=.006,true_old_pair_effect_min=.03,
             locked_model_gap_positive_given_successful_calibration=True,
             normal_planning_approximation=True,no_empirical_variance_or_bias_certificate=True,
             calibration_check_stage=False),scenarios=cases,primary_scenario=maincase,predictions=predictions,
         finite_agreement_power_not_primary_qualification=True,coefficient_confirmation=False,
         failure_interpretation='失败不否定保留排序；不增样、不调目标/带宽、不删除格、不替换经验臂'))
    row=next(r for r in read(ROOT/'review13_checks/kernel_budget.json')['rows'] if r['d']==128)
    train=row['training_3_GEMM_cpu'];scan=row['candidate_10passes_cpu']/10;gen=.035*128*4096/1e6
    cal=nc*4*(2*(train+scan*17*2*512/512)+gen+1)
    ev=ne*(2*(train+scan*2*1*4*2*2*1024/512)+gen+1)
    side=min(64,ne)*2*(scan*2*2*4*2*2*1024/512)
    total=cal+ev+side+500
    assert abs(total-read(ROOT/'audit_z4_checks/repair_refined.json')['total_cpu_forecast'])<1e-6
    dump(OUT/'resources.json',dict(calibration_cpu=cal,primary_evaluation_cpu=ev,side_extra_cpu=side,
         runner_forecast_overhead=500,runner_forecast_cpu=total,forecast_cpu_hours=total/3600,
         runner_forecast_wall_at_cpu_per_wall_1_8=total/1.8,
         hard_allocation=dict(preparation=600,runner=20400,analysis=500,supervisor=60,observer=40,total=21600),
         timer_allocation=dict(supervisor=59,observer=39),wall_hard=12600,memory_bytes=3221225472,swap_bytes=0,threads_max=2,
         preparation_start=start,preparation_end=usage(),historical_usage='跨重启与未记账历史不认证；预留不是实测上界'))
    dump(OUT/'input_hashes.json',inputs);dump(OUT/'environment.json',environment())
    print(dict(streams=len(streams),old=len(old),power=maincase['complete_primary_direction_power_planning_lower'],cpu=total,usage=usage()))

if __name__=='__main__':main()
