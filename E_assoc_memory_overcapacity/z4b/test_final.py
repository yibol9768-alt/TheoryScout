"""最后一次源代码绑定检查；额外保存不会随后被故障注入改变的阳性夹具。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(25,30))
import json
import sys
from common_v2 import OUT,dump,read,sha,usage
import analyze_v2 as analysis
import test_resume as T
from protocol_v2 import publish,verify_accepted,inspect,digest
from resume_extra import observer_receipt

T.BASE=OUT/'synthetic_final';T.BASE.mkdir();checks=[]
def check(name,fn):
    fn();checks.append(dict(name=name,passed=True))

data,cfg,mh=T.fixture('stable_control')
def control():
    assert publish(data,mh,cfg,T.GOOD,2.,1.,1.)['confirmation_valid']
    observer_receipt(data,mh)
    assert verify_accepted(data,mh,cfg)['confirmation_valid']
check('stable_full_positive_control_and_exit_observation',control)

def opposite():
    original_predict=analysis.predict;original_interval=analysis.interval
    def prediction(method,alpha,d,task):
        v=original_predict(method,alpha,d,task)
        # Deliberately reverse the model's GD-Sign prediction in this artificial call.
        if task=='old':v.update(finite=0. if method=='GD' else 1.)
        return v
    def interval(values,family,alpha=.025):
        v=original_interval(values,family,alpha)
        if family in (3,9) and v['lower'] is not None:v.update(mean=-.1,lower=-.11,upper=-.09)
        return v
    analysis.predict=prediction;analysis.interval=interval
    try:
        r=analysis.make_report(data,mh,cfg)
        gd=[x for x in r['old_pairs'] if x['pair']=='GD-Sign']
        assert gd and all(x['Delta_finite']<0 and x['upper']<0 and not x['direction_supported'] for x in gd)
        assert not r['decisions']['primary_all3_directions']
    finally:analysis.predict=original_predict;analysis.interval=original_interval
check('predicted_and_observed_reversal_cannot_pass_preregistered_positive_direction',opposite)

def pending_cpu():
    p,c,m=T.fixture('pending_cpu')
    T.overwrite_json(p/'analysis/PENDING.json',lambda o:o['usage'].update(cpu=c['analysis_cpu_hard']+1))
    status=publish(p,m,c,T.GOOD,2.,1.,1.)
    assert not status['confirmation_valid'] and 'analysis CPU' in status['errors']
check('analysis_CPU_resource_tamper_rejected',pending_cpu)

def schema():
    p,c,m=T.fixture('rehashed_unknown_event');events=[];previous=m
    for i,line in enumerate((p/'events.jsonl').read_text().splitlines()):
        o=json.loads(line);o.pop('hash');o['prev']=previous
        if i==0:o['unknown']='rehashed'
        previous=digest(o);events.append(dict(o,hash=previous))
    (p/'events.jsonl').write_text(''.join(json.dumps(o)+'\n' for o in events))
    T.overwrite_json(p/'runner_status.json',lambda o:o.update(event_tail=previous))
    audit=inspect(p,m,c)
    assert 'started schema' in audit['errors'] and 'event hash/order' not in audit['errors']
    assert not publish(p,m,c,T.GOOD,2.,1.,1.)['confirmation_valid']
check('correctly_rehashed_unknown_event_still_rejected_by_schema',schema)

def stable():
    assert verify_accepted(data,mh,cfg)['confirmation_valid']
    assert not (OUT/'formal').exists()
check('positive_evidence_still_replays_after_all_fault_tests',stable)

runtime=['common_v2.py','model.py','run_v2.py','analyze_v2.py','protocol_v2.py','launch_v2.py','entry.py']
dump(OUT/'final_tests.json',dict(passed=True,checks=checks,source_sha256={n:sha(OUT/n) for n in runtime},
    training_or_query_draws=0,formal_sampling=False,usage=usage()))
print(dict(passed=True,checks=len(checks),usage=usage()))
