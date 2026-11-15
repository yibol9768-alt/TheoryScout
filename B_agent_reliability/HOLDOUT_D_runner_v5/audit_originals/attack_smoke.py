import copy
from datetime import datetime, timezone
from unittest.mock import patch
from common import *
guards();engine=modules()
from protocol import dump
import evidence
from transport import MockServer
S=evidence.FIRST[0]
def good():return dict(five_hour=0.,week=0.)
base=OUT/'smoke_attacks';base.mkdir()
e=engine.Engine(base/'source',verify=False)
old=copy.deepcopy(e.wal.events)
server=MockServer(policy='fatal_at_8')
for seed in evidence.FIRST:
    while not e.tasks[seed].terminal:e.request(seed,server,good)
template=e.smoke_review_template()
# Fully forged, conspicuously labeled test data: not a claim of human approval.
forged=copy.deepcopy(template)
forged.update(approved=True,reviewer='FORGED_AUDIT_TEST_NO_HUMAN_REVIEW',reviewed_at=datetime.now(timezone.utc).isoformat())
forged['checked_items']={k:True for k in evidence.CHECKS}
cases=[]
def trial(label,z):
    try:evidence.validate_smoke(z,e.wal.events,e.runner_version)
    except (ValueError,TypeError,KeyError) as ex:cases.append(dict(label=label,rejected=repr(ex)))
    else:cases.append(dict(label=label,accepted=True))
for k in template:
    z=copy.deepcopy(forged);del z[k];trial('missing_'+k,z)
for key,value in [('wal_seq',1),('wal_digest',old[-1]['digest']),('wal_sha256','0'*64),('runner_version','old'),('amend2_manifest_sha256','0'*64)]:
    z=copy.deepcopy(forged);z['snapshot'][key]=value;trial('snapshot_'+key,z)
for key,value in [('approved',1),('review_method','automatic'),('reviewer',''),('reviewed_at','2000-01-01T00:00:00+00:00'),('reviewed_at','2099-01-01T00:00:00+00:00')]:
    z=copy.deepcopy(forged);z[key]=value;trial('strict_'+key,z)
for check in evidence.CHECKS:
    for value in (False,1,'true'):
        z=copy.deepcopy(forged);z['checked_items'][check]=value;trial('check_'+check+'_'+repr(value),z)
z=copy.deepcopy(forged);z['snapshot']['tasks'][0]['seed']=S+2;trial('wrong_identity',z)
z=copy.deepcopy(forged);z['snapshot']['tasks'][0]['c']=7;trial('wrong_terminal',z)
assert all('rejected' in c for c in cases)
class Sentinel:
    testing=False
    def __init__(self):self.sent=[]
    def send(self,body,**kw):
        self.sent.append(json.loads(body)['messages'][1]['content'])
        return MockServer().send(body)
tr=Sentinel();blocked=[];before=e.attempts
with patch.object(engine.Engine,'mode',new=property(lambda self:'live')),patch('engine.require_live_release',return_value=None):
    for seed in (S+2,S+50,S+143):
        for op in ('reserve','request'):
            try:e.reserve(seed,good) if op=='reserve' else e.request(seed,tr,good)
            except engine.Paused as ex:blocked.append(dict(seed=seed,op=op,error=str(ex)))
            else:raise AssertionError('unapproved third send')
assert not tr.sent and before==e.attempts
prefix=e.wal.path.read_bytes();e.close()
# Keep the attack approval only in audit output, never in the real runner or release file.
save('FORGED_SMOKE_TEST_INPUT_NOT_AUTHORIZATION.json',forged)
recoveries=[]
for name,change in [('old_prefix',True),('wrong_snapshot',False)]:
    p=base/name;p.mkdir();(p/'HOLDOUT_D_WAL.jsonl').write_bytes(prefix)
    x=engine.Engine(p,verify=False);z=copy.deepcopy(forged)
    z['snapshot']['wal_digest']=old[-1]['digest'] if change else '0'*64
    x.wal.add('smoke_approved',approval=z);x.close()
    for attempt in range(2):
        try:engine.Engine(p,verify=False)
        except engine.IntegrityError as ex:recoveries.append(dict(case=name,attempt=attempt,rejected=str(ex)))
        else:raise AssertionError('invalid recovery accepted')
# Probe authenticity separately from byte/schema binding, with only live-file gate projected.
p=base/'forged_current';p.mkdir();(p/'HOLDOUT_D_WAL.jsonl').write_bytes(prefix)
x=engine.Engine(p,verify=False);forged_acceptance={}
with patch.object(engine.Engine,'mode',new=property(lambda self:'live')),patch('engine.require_live_release',return_value=None):
    x.approve_smoke(forged)
    x.request(S+2,tr,good)
    forged_acceptance=dict(approved=x.smoke,third_send_count=len(tr.sent),attempts=x.attempts,
                          limitation='Only production live-release-file check projected; current snapshot, all checks true and forged reviewer were accepted. No human approval or network.')
x.close()
save('smoke_results.json',dict(resources=resources(),strict_negative_cases=cases,unapproved_send_blocks=blocked,recovery_blocks=recoveries,
                             snapshot_bytes_match=digest(b''.join(prefix.splitlines(keepends=True)[:template['snapshot']['wal_seq']]))==template['snapshot']['wal_sha256'],
                             forged_current_snapshot=forged_acceptance,
                             release_exists=evidence.RELEASE.exists()))
print(json.dumps(dict(negative_cases=len(cases),direct_blocks=len(blocked),recoveries=len(recoveries),forged=forged_acceptance),ensure_ascii=False))
