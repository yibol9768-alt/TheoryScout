"""审计自建故障 mock。所有响应在内存生成，未构造 HTTP 客户端。"""
import json
import threading
from common import ROOT, OUT, dump_file, install_guards, resource_info, runner_modules

install_guards()
engine=runner_modules()
from run import execute
Engine, Paused, IntegrityError=engine.Engine,engine.Paused,engine.IntegrityError
S=116064001
BASE=OUT/'own_faults'
BASE.mkdir()
def good():return dict(five_hour=0.,week=0.)
class Clock:
    def __init__(self):self.t=2000000000.
    def __call__(self):return self.t
    def advance(self,dt):self.t+=dt
def expect_pause(fn):
    try:fn()
    except Paused as ex:return str(ex)
    raise AssertionError('expected pause')

def response(body, fatal=False):
    req=json.loads(body);payload=json.loads(req['messages'][1]['content']);state=payload['checkpoint'].copy()
    for ins in payload['operation']['instructions']:
        a,b,c=[f'r{ins[x]}' for x in ('a','b','c')]
        state[a]=(state[a]+ins['m']*state[b]-state[c]+ins['d'])%997
        state[c]=(state[c]+state[a])%997
    state['F0']=(state['r0']+2*state['r1']-state['r2'])%997
    state['F1']=(3*state['F0']+state['r3']-state['r4'])%997
    state['F2']=(state['F1']+2*state['r5'])%997
    content='not-json' if fatal else json.dumps(dict(writes=state,detect=len(req['messages'])==4),separators=(',',':'))
    return json.dumps(dict(model='gpt-6-luna',usage=dict(prompt_tokens=1,completion_tokens=1,total_tokens=2),
                           choices=[dict(message=dict(content=content),finish_reason='stop')]),separators=(',',':')).encode()
class LocalMock:
    def __init__(self, fatal=False):self.bodies=[];self.fatal=fatal
    def send(self,body):self.bodies.append(body);return 200,response(body,self.fatal),None

results={}

# 1. 两并发真实走 send/receive 同时 408；9000 预留边界附近，原文重发。
c=Clock();e=Engine(BASE/'double_408',verify=False,clock=c,fixture=dict(success=8998))
barrier=threading.Barrier(2)
class Double408:
    def __init__(self):self.bodies=[]
    def send(self,body):self.bodies.append(body);barrier.wait(timeout=10);return 408,b'{}','mock_408'
server=Double408();thread_errors=[]
def request_first():
    try:e.request(S,server,good)
    except BaseException as ex:thread_errors.append(repr(ex))
t=threading.Thread(target=request_first);t.start();e.request(S+1,server,good);t.join(10)
assert not t.is_alive() and not thread_errors
assert (e.success,e.attempts,len(e.pending))==(8998,9000,0)
for seed in (S,S+1):expect_pause(lambda:e.reserve(seed,good))
c.advance(299.999);expect_pause(lambda:e.reserve(S,good));c.advance(.001)
retry=LocalMock();e.request(S,retry,good);e.request(S+1,retry,good)
assert sorted(retry.bodies)==sorted(server.bodies)
assert e.success==9000 and e.attempts==9002 and e.success+len(e.pending)<=9000
expect_pause(lambda:e.request(S+2,retry,good));assert len(retry.bodies)==2
results['double_concurrent_408']=dict(passed=True,successful=e.success,attempts=e.attempts,extra_sends_during_wait=0,
                                    identical_retry_bodies=True,next_attempts=[e.requests[i]['attempt'] for i in (9001,9002)])
e.close()

# 2. 第100个返回后 quota 超时；重启不能复用旧值/跨过查询屏障。
c=Clock();e=Engine(BASE/'quota_timeout',verify=False,clock=c,fixture=dict(success=99))
server=LocalMock();e.request(S,server,good)
def timeout():raise TimeoutError('independent quota timeout')
expect_pause(lambda:e.request(S+1,server,timeout));assert (e.attempts,len(server.bodies))==(100,1);e.close()
e=Engine(BASE/'quota_timeout',verify=False,clock=c)
c.advance(299.999);expect_pause(lambda:e.request(S+1,server,good));assert e.attempts==100
c.advance(.001);e.request(S+1,server,good);assert e.attempts==101 and len(server.bodies)==2
results['quota_timeout_at_100_restart']=dict(passed=True,attempts=e.attempts,requests_before_fresh_quota=1,requests_after_fresh_quota=2)
e.close()

# 3. 有一已消费、一未决时尾部撕裂：连续恢复拒绝，原证据字节不变。
path=BASE/'torn_tail';e=Engine(path,verify=False);server=LocalMock();e.request(S,server,good)
e.reserve(S+1,good);e.close();p=path/'HOLDOUT_D_WAL.jsonl';valid=p.read_bytes()
torn=valid+b'{"seq":9,"kind":"response"';p.write_bytes(torn);failures=[]
for _ in range(2):
    try:Engine(path,verify=False)
    except IntegrityError as ex:failures.append(str(ex))
    else:raise AssertionError('torn WAL accepted')
    assert p.read_bytes()==torn
assert len(server.bodies)==1 and len(failures)==2
results['torn_tail_two_recoveries']=dict(passed=True,fail_closed=failures,preserved_bytes=True,extra_sends=0,
                                       known_successful=1,unknown_reservation_preserved_in_valid_prefix=1,
                                       automatic_recovery=False)

# 4. 完整冻结的 live 模式引擎 + 纯内存 mock：预先生成的旧前缀凭据能否放行。
path=BASE/'smoke_stale_prefix';e=Engine(path,mode='live')
early_prefix=e.wal.head
preapproval=dict(approved=True,reviewer='PRECOMPUTED MOCK; NO HUMAN REVIEW',checked_items=['arbitrary'],wal_prefix=early_prefix)
dump_file('precomputed_smoke_approval.json',preapproval)
server=LocalMock(fatal=True);z=execute(e,server,good,mock=False)
assert z['terminal']==2 and z['successful']==6 and not e.smoke
e.close();e=Engine(path,mode='live')
e.approve_smoke(preapproval)
class StopAtThird:
    def __init__(self):self.bodies=[]
    def send(self,body):self.bodies.append(body);return 200,b'bad_api_envelope','audit_stop'
third=StopAtThird();execute(e,third,good,mock=False)
assert e.smoke and third.bodies and e.success>=7
results['stale_pre_smoke_approval']=dict(safety_requirement_passed=False,accepted=True,human_review=False,
        prefix_before_first_request=early_prefix,third_wave_sends=len(third.bodies),successful=e.success,
        halt=e.halt,mode='live ledger semantics with local mock only')
e.close()

# 5. 公共发送入口没有首两题门；与 CLI 调度器保护范围区分。
e=Engine(BASE/'direct_third_before_smoke',mode='live');server=LocalMock()
e.request(S+2,server,good)
results['direct_engine_third_seed']=dict(safety_requirement_passed=False,sends=len(server.bodies),
    first_two_terminal=[e.tasks[s].terminal for s in (S,S+1)],third_seed_completed=e.tasks[S+2].step)
assert len(server.bodies)==1 and e.tasks[S+2].step==1;e.close()

# 6. 未完成复查的前缀：题保持 U，不把原始动作错误算成致命。
e=Engine(BASE/'unfinished_review',verify=False,fixture=dict(success=8999));server=LocalMock(fatal=True)
e.request(S,server,good);expect_pause(lambda:e.request(S,server,good));summary=e.summary()
assert not e.tasks[S].terminal and e.tasks[S].review==1
assert summary['prefix_bounds']['8']==dict(K=0,F=0,U=144,success_count_bound=[0,144])
results['cap_mid_review_prefix']=dict(passed=True,prefix8=summary['prefix_bounds']['8'],
    raw_task_counts=e.tasks[S].counts,note='Task.counts 暴露在未决复查时已加1；最终机制分析必须剔除未决推进。')
e.close()

dump_file('fault_mocks.json',dict(resources=resource_info(),cases=results,no_network=True,no_llm=True))
print(json.dumps(results,ensure_ascii=False))
