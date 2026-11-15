"""修订设计专属确定性测试：计划、流、缺失掩码及真正执行接线。"""
import copy
import json
import numpy as np
import analyze_v2 as A
import run_v2 as R
import test_resume as T
from common_v2 import OUT,config,plan,scheduled,expected_n,dump,read,sha,entropy,usage

checks=[]
def check(name,fn):
    fn();checks.append(dict(name=name,passed=True))

def planning():
    c=config();jobs=plan(c)
    assert len(jobs)==5377 and jobs[4096]==['lock']
    assert all(j[0]=='cal' for j in jobs[:4096])
    assert jobs[-1]==['eval',128,1279]
    assert [s for s in range(1280) if scheduled(c,s,.35)]==list(range(64))
    assert expected_n(c,.425)==1280 and expected_n(c,.5)==64
    es=[entropy(0,128,m,s,k) for m in range(4) for s in range(1024) for k in range(4)]
    es += [entropy(1,128,-1,s,k) for s in range(1280) for k in range(4)]
    assert read(OUT/'streams.json')==es and len(es)==21504
check('5377_jobs_lock_order_64_side_seeds_21504_streams',planning)

T.BASE=OUT/'synthetic_revision';T.BASE.mkdir()
c=config();c.update(n_cal=3,n_eval=5,q_cal=40,q_eval=40,side_n_eval=2)
p,c,m=T.fixture('mixed_plan',c)
report=A.make_report(p,m,c)
def coverage():
    assert report['data_errors']==[]
    assert report['decisions']['primary_all3_directions']
    assert report['decisions']['primary_all7_matching']
    assert report['decisions']['primary_pair_count']==3
    assert report['decisions']['primary_match_count']==7
    assert all('all18' not in k for k in report['decisions'])
    assert all(r['n']==(5 if r['target']==.425 else 2) for r in report['old_pairs'])
    assert sum(r['scheduled'] for r in report['all_completed'])==2*4*(5+2*2)
    assert all(not r['valid'] and r['old'] is None and r['new'] is None for r in report['all_completed'] if not r['scheduled'])
    for row in report['old_pairs']:
        assert (row['role']=='primary')==(row['arm']=='empirical' and row['target']==.425)
check('primary_fields_descriptive_roles_exact_N_and_unscheduled_nulls',coverage)

def execution():
    # 实际 evaluation 函数；数据与排名接口替换成固定数组，禁止调用科学 RNG。
    calls=[];orig=R.dataset;rank=R.ranks_many
    def data(*args):
        return np.ones((2,40)),np.ones((2,40)),np.ones((2,40)),np.arange(40),np.eye(2),np.eye(2)
    def ranks(z,h,aa,x,y,ids,checkpoint):
        calls.append(len(aa));return np.ones((len(aa),len(ids)),dtype=np.int64)
    R.dataset=data;R.ranks_many=ranks
    try:
        late=R.evaluation(128,2,c,m,read(p/'lock.json'),sha(p/'lock.json'))
        assert calls==[2]*8 and late['scheduled'].tolist()==[False,True,False]
        assert not late['valid'][:,[0,2]].any() and not late['ranks'][:,[0,2]].any()
        calls.clear()
        early=R.evaluation(128,1,c,m,read(p/'lock.json'),sha(p/'lock.json'))
        assert calls==[6]*8 and early['scheduled'].all()
    finally:R.dataset=orig;R.ranks_many=rank
check('actual_evaluation_computes_2_vs_6_alphas_no_side_work_after_prefix',execution)

def mutate_array(name,func):
    path=p/name
    with np.load(path,allow_pickle=False) as f:arr={k:f[k] for k in f.files}
    original={k:v.copy() for k,v in arr.items()};func(arr)
    with path.open('wb') as f:np.savez(f,**arr)
    try:assert A.make_report(p,m,c)['data_errors']
    finally:
        with path.open('wb') as f:np.savez(f,**original)

check('late_side_schedule_forgery_rejected',lambda:mutate_array('eval_128_4.npz',lambda a:a['scheduled'].__setitem__(0,True)))
check('late_side_zero_sentinel_required',lambda:mutate_array('eval_128_4.npz',lambda a:a['ranks'].__setitem__((0,0,0,0,0),1)))
check('late_side_alpha_must_be_zero',lambda:mutate_array('eval_128_4.npz',lambda a:a['alphas'].__setitem__((0,0,0),1.)))

def missing():
    path=p/'eval_128_4.npz';backup=path.read_bytes();path.unlink()
    try:
        r=A.make_report(p,m,c)
        assert not r['decisions']['primary_plan_complete'] and not r['decisions']['primary_all3_directions']
        assert all(x['n']==2 for x in r['old_pairs'] if x['target']!=.425)
        assert all(x['n']==4 for x in r['old_pairs'] if x['target']==.425)
    finally:path.write_bytes(backup)
check('missing_late_primary_cannot_be_replaced_by_side_complete_N',missing)

def empty():
    q=OUT/'synthetic_revision/empty';q.mkdir();r=A.make_report(q,m,c)
    assert not any(r['decisions'][k] for k in ['primary_plan_complete','primary_all7_matching','primary_all3_directions','primary_all3_finite_agreement'])
check('empty_plan_never_vacuously_confirms',empty)

def power():
    power=read(OUT/'power.json');s=power['primary_scenario']
    assert s['pair_bias_bound']==2*s['per_method_bias_bound']
    assert .84<s['complete_primary_direction_power_planning_lower']<.85
    assert power['scenarios'][3]['complete_primary_direction_power_planning_lower']==0
    assert not config()['independent_calibration_check']
check('complete_power_includes_failure_and_2b_bias_sensitivity',power)

dump(OUT/'revision_tests.json',dict(passed=True,checks=checks,training_or_query_draws=0,usage=usage()))
print(dict(passed=True,checks=len(checks),usage=usage()))
