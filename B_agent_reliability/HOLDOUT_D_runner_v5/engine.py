"""fsync WAL 为唯一账本。默认不具备网络能力。"""
import base64, fcntl, json, os, threading, time
from pathlib import Path
from protocol import ROOT, Task, dump, sha, guard, GuardError
from evidence import version, require_live_release, smoke_snapshot, validate_smoke

class Paused(Exception): pass
class DispatchPaused(Paused): pass
class IntegrityError(Exception): pass

def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def allowed_dir(out):
    out=Path(out).resolve();root=ROOT/'HOLDOUT_D_runner_v5'
    if not out.is_relative_to(root) or out==root:raise ValueError('output_must_be_under_HOLDOUT_D_runner_v5')
    return out

def verify_manifest(path):
    for line in path.read_text().splitlines():
        digest,name=line.split('  ',1);p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or sha(p.read_bytes())!=digest:raise IntegrityError('freeze_mismatch:'+name)

def verify_frozen():
    verify_manifest(ROOT/'HOLDOUT_D_PREREG.sha256')
    verify_manifest(ROOT/'HOLDOUT_D_AMEND1.sha256')
    verify_manifest(ROOT/'HOLDOUT_D_AMEND2.sha256')
    verify_manifest(ROOT/'HOLDOUT_D_AMEND3.sha256')
    verify_manifest(ROOT/'HOLDOUT_D_AMEND4.sha256')
    verify_manifest(ROOT/'HOLDOUT_D_AMEND5.sha256')

class WAL:
    def __init__(self,out):
        self.out=allowed_dir(out);self.out.mkdir(parents=True,exist_ok=True)
        self.lockfile=(self.out/'HOLDOUT_D_RUN.lock').open('a')
        try:fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.lockfile.close();raise
        self.path=self.out/'HOLDOUT_D_WAL.jsonl'
        try:
            self.events=[];self.head='0'*64
            if self.path.exists():
                raw=self.path.read_bytes()
                if raw and not raw.endswith(b'\n'):raise IntegrityError('torn_WAL_tail_pause_preserve_evidence')
                for line in raw.splitlines():
                    ev=json.loads(line);digest=ev.pop('digest')
                    if ev['seq']!=len(self.events)+1 or ev['prev']!=self.head or sha(dump(ev).encode())!=digest:raise IntegrityError('WAL_chain_mismatch')
                    ev['digest']=digest
                    if dump(ev).encode()!=line:raise IntegrityError('noncanonical_WAL_bytes')
                    self.events.append(ev);self.head=digest
            self.f=self.path.open('ab',buffering=0);fsync_dir(self.out)
        except BaseException:self.lockfile.close();raise
    def add(self,kind,**fields):
        ev=dict(seq=len(self.events)+1,prev=self.head,kind=kind,utc=time.time(),**fields)
        ev['digest']=sha(dump(ev).encode());raw=(dump(ev)+'\n').encode()
        view=memoryview(raw)
        while view:
            written=self.f.write(view)
            if not written:raise OSError('WAL_short_write')
            view=view[written:]
        os.fsync(self.f.fileno())
        self.events.append(ev);self.head=ev['digest'];return ev
    def close(self):
        self.f.close();self.lockfile.close()

class Engine:
    def __init__(self,out,clock=time.time,verify=True,fixture=None,mode='mock',live=False):
        if mode not in ('mock','live') or (mode=='live' and (not verify or fixture)):
            raise ValueError('live_requires_frozen_nonfixture_state')
        if mode=='live':require_live_release(live)
        if mode=='live' and Path(out).resolve()!=ROOT/'HOLDOUT_D_runner_v5/HOLDOUT_D_live':
            raise ValueError('live_ledger_and_flock_path_is_fixed_no_budget_reset')
        self._mode=mode;self._explicit_live=live;self.runner_version=version()
        if verify:verify_frozen()
        self.wal=WAL(out);self.lock=threading.RLock();self.clock=clock
        self.tasks={s:Task(s) for s in range(116064001,116064145)}
        self.attempts=0;self.success=0;self.failed=0;self.usage=dict(prompt_tokens=0,completion_tokens=0,total_tokens=0)
        self.unknown_usage=0;self.pending={};self.active=set();self.attempt_by_rid={};self.requests={};self.responses={};self.applied=set();self.dispatched=set()
        self.pause_until=0.;self.need_probe=True;self.quota_bucket=-1;self.probe_retry=0.;self.halt=None;self.smoke=False
        if not self.wal.events:
            # Test-only initial counter state is explicit evidence, never accepted by live CLI.
            self.wal.add('init',mode='fixture' if fixture else mode,fixture=fixture,freeze_verified=verify,
                         amend3_manifest_sha256=sha((ROOT/'HOLDOUT_D_AMEND5.sha256').read_bytes()) if verify else None,
                         runner_version=self.runner_version,protocol_source_sha256=sha((ROOT/'experiments/real/reliability_core.py').read_bytes()))
        if verify and self.wal.events[0].get('mode')!=mode:
            self.wal.close();raise IntegrityError('cannot_mix_mock_fixture_live_history')
        archive=None
        try:
            if len(self.wal.events)>1:
                raw=self.wal.path.read_bytes()
                name=f'HOLDOUT_D_RECOVERY_{len(self.wal.events)}_{self.wal.head[:16]}.jsonl'
                p=self.wal.out/name
                if p.exists():
                    if p.read_bytes()!=raw:raise IntegrityError('recovery_archive_mismatch')
                else:
                    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                    fsync_dir(self.wal.out)
                archive=dict(path=name,sha256=sha(raw),chain_head=self.wal.head)
            self._replay()
        except BaseException:self.wal.close();raise
        if self.pending and self.halt is None:self.halt='unknown_inflight'
        self.wal.add('recovery',prefix_hash=self.wal.head,attempts=self.attempts,successful=self.success,
                     unresolved=list(self.pending),archive=archive,
                     positions={str(s):[t.step,t.review,t.status] for s,t in self.tasks.items() if t.step or t.review or t.terminal},
                     next_requests={str(s):dict(rid=t.rid,body_sha256=sha(t.body()),next_attempt=self.attempt_by_rid.get(t.rid,0)+1,
                                               blocked_unknown=any(r['seed']==s for r in self.pending.values()))
                                    for s,t in self.tasks.items() if not t.terminal and any(r['seed']==s for r in self.requests.values())})
        self.summary()
    def _consume_usage(self,raw):
        try:
            z=guard(raw)
            for k in self.usage:self.usage[k]+=z['usage'][k]
        except GuardError:self.unknown_usage+=1
    def _replay(self):
        # Reconstruct responses and actions sequentially. Missing action is rebuilt before next WAL event.
        missing=[]
        for ev in list(self.wal.events):
            k=ev['kind']
            if k=='init':
                if ev.get('runner_version')!=self.runner_version:raise IntegrityError('runner_version_mismatch')
                if ev['protocol_source_sha256']!=sha((ROOT/'experiments/real/reliability_core.py').read_bytes()):raise IntegrityError('protocol_source_mismatch')
                if ev['fixture']:
                    self.success=ev['fixture'].get('success',0);self.attempts=ev['fixture'].get('attempts',self.success)
            elif k=='reserve':
                if self.mode=='live':self._send_gate(ev['seed'])
                call=ev['call'];rid=ev['rid'];body=base64.b64decode(ev['body'],validate=True)
                if ev['body_sha256']!=sha(body) or self.tasks[ev['seed']].rid!=rid or self.tasks[ev['seed']].body()!=body:raise IntegrityError('recovery_prompt_bytes_mismatch')
                if call!=self.attempts+1 or ev['attempt']!=self.attempt_by_rid.get(rid,0)+1:raise IntegrityError('attempt_sequence_mismatch')
                self.attempts=call;self.attempt_by_rid[rid]=ev['attempt'];self.requests[call]=ev;self.pending[call]=ev
                if self.success+len(self.pending)>9000 or self.attempts>12000:raise IntegrityError('budget_WAL_violation')
            elif k=='dispatch_started':
                call=ev['call'];req=self.pending.get(call)
                if (not req or call in self.dispatched or ev['rid']!=req['rid']
                        or ev['body_sha256']!=req['body_sha256']):raise IntegrityError('invalid_dispatch_started')
                self.dispatched.add(call)
            elif k=='response':
                call=ev['call']
                if call not in self.pending:raise IntegrityError('response_without_reservation')
                if self.mode=='live' and call not in self.dispatched:raise IntegrityError('response_without_dispatch')
                del self.pending[call];self.responses[call]=ev
                raw=base64.b64decode(ev['raw'],validate=True)
                if sha(raw)!=ev['raw_sha256']:raise IntegrityError('raw_response_hash')
                if 200<=ev['status']<300:
                    self.success+=1;self._consume_usage(raw)
                    try:
                        response=guard(raw);t=self.tasks[self.requests[call]['seed']]
                        record=t.apply(response);missing.append((call,record))
                    except GuardError:self.halt='envelope_model_usage_token_guard'
                elif ev['status']==408 or 500<=ev['status']<=599 or ev['status']==0:
                    self.failed+=1;self.pause_until=max(self.pause_until,ev['finished']+300);self.need_probe=True
                else:self.failed+=1;self.halt='other_http'
            elif k=='action':
                call=ev['call'];entry=next((x for x in missing if x[0]==call),None)
                if not entry or entry[1]!=ev['record']:raise IntegrityError('deterministic_action_mismatch')
                missing.remove(entry);self.applied.add(call)
            elif k=='halt':self.halt=ev['reason']
            elif k=='smoke_approved':
                try:validate_smoke(ev['approval'],self.wal.events[:ev['seq']-1],self.runner_version)
                except (ValueError,KeyError,TypeError) as ex:raise IntegrityError('invalid_recovered_smoke:'+str(ex)) from ex
                if self.mode!='live':raise IntegrityError('mock_cannot_record_human_approval')
                self.smoke=True;self.smoke_approval=ev['approval']
            elif k=='reconcile':
                if self.halt=='unknown_inflight':self.halt=None
            elif k=='quota':
                # Preserve a failed/threshold query's 300s wait across restarts.
                # A previously successful query still cannot authorize the new process.
                self.probe_retry=0. if ev['ok'] else max(self.probe_retry,ev['checked']+300)
            elif k not in ('recovery','pause','launch','smoke_wait'):raise IntegrityError('unknown_WAL_event')
        for call,record in missing:
            self.wal.add('action',call=call,record=record,recovered=True);self.applied.add(call)
        # Every process restart requires a NEW quota probe, never trust previous successful query.
        self.need_probe=True
    def summary(self):
        # Snapshot is expendable; never read during recovery.
        with self.lock:
            z=dict(successful=self.success,attempts=self.attempts,service_or_http_failed=self.failed,
                   unresolved=len(self.pending),active=len(self.active),reserved_total=self.success+len(self.pending),
                   known_usage=self.usage,unknown_usage_returns=self.unknown_usage,
                   service_unknown_token_attempts=self.failed,pause_until=self.pause_until,halt=self.halt,
                   terminal=sum(t.terminal for t in self.tasks.values()),prefix_bounds={})
            for L in (8,16,32,64):
                K=sum(t.step>=L for t in self.tasks.values());F=sum(t.terminal and t.step<L for t in self.tasks.values());U=144-K-F
                z['prefix_bounds'][str(L)]=dict(K=K,F=F,U=U,success_count_bound=[K,K+U])
            tmp=self.wal.out/'HOLDOUT_D_SUMMARY.tmp';p=self.wal.out/'HOLDOUT_D_SUMMARY.json'
            with tmp.open('w') as f:f.write(dump(z)+'\n');f.flush();os.fsync(f.fileno())
            os.replace(tmp,p);fsync_dir(self.wal.out);return z
    def stop(self,reason):
        with self.lock:self.wal.add('halt',reason=reason);self.halt=reason
    def quota(self,probe):
        # 未发送的本进程预留不阻塞重探测；已发送和恢复后的未知预留仍须排空。
        with self.lock:
            now=self.clock()
            if (self.active & self.dispatched or self.pending.keys()-self.active
                    or now<self.pause_until or now<self.probe_retry):raise Paused('quota_wait_drain_or_300s')
            try:
                q=probe();a=q['five_hour'];b=q['week']
                if type(a) not in (int,float) or type(b) not in (int,float) or not 0<=a<=1 or not 0<=b<=1:raise ValueError()
                now=self.clock()  # 查询完成后起算暂停，50秒超时不缩短300秒等待。
                good=a<.80 and b<.90
                self.wal.add('quota',successful=self.success,five_hour=a,week=b,ok=good,checked=now)
            except Exception:
                now=self.clock()
                self.wal.add('quota',successful=self.success,ok=False,reason='query_failed',checked=now);good=False
            if not good:
                self.need_probe=True;self.probe_retry=now+300;raise Paused('quota_unavailable_or_threshold')
            self.need_probe=False;self.probe_retry=0;self.quota_bucket=self.success//100
    @property
    def mode(self):return self._mode
    def _send_gate(self,seed):
        if self.mode=='live':
            require_live_release(self._explicit_live)
            if seed not in (116064001,116064002):
                if not self.smoke:raise Paused('human_smoke_release_required')
                try:validate_smoke(self.smoke_approval,self.wal.events,self.runner_version)
                except (ValueError,KeyError,TypeError) as ex:raise Paused('human_smoke_release_invalid') from ex
    def reserve(self,seed,probe):
        with self.lock:
            self._send_gate(seed)
            if self.halt:raise Paused(self.halt)
            if self.attempts>=12000:self.stop('attempt_cap_12000');raise Paused(self.halt)
            if self.success>=9000:self.stop('successful_cap_9000');raise Paused(self.halt)
            if self.pending.keys()-self.active:raise Paused('unknown_inflight')
            now=self.clock()
            boundary=self.quota_bucket!=self.success//100
            if self.need_probe or boundary:
                self.quota(probe)
            if now<self.pause_until or self.need_probe:raise Paused('service_pause_300s')
            if len(self.active)>=2 or self.success+len(self.pending)>=9000:raise Paused('reservation_capacity')
            # At 99, only one request may be in flight, so 100 cannot be crossed without probing.
            if self.success%100+len(self.pending)>=100:raise Paused('hundred_boundary_barrier')
            t=self.tasks[seed]
            if t.terminal:raise Paused('terminal_no_rerun')
            rid=t.rid
            if any(x['rid']==rid for x in self.pending.values()):raise Paused('same_logical_request_inflight')
            body=t.body();attempt=self.attempt_by_rid.get(rid,0)+1;call=self.attempts+1
            ev=self.wal.add('reserve',call=call,rid=rid,seed=seed,step=t.step,review=t.review,attempt=attempt,
                            body=base64.b64encode(body).decode(),body_sha256=sha(body))
            self.attempts=call;self.attempt_by_rid[rid]=attempt;self.requests[call]=ev;self.pending[call]=ev;self.active.add(call)
            return ev,body
    def claim_dispatch(self,reservation,body):
        """全局锁内取得一次性发送权；持久化不确定性，不释放预算。"""
        with self.lock:
            self._send_gate(reservation['seed'])
            call=reservation['call']
            if call in self.dispatched:raise Paused('dispatch_already_started')
            if self.halt:raise Paused(self.halt)
            if (self.pending.get(call)!=reservation or call not in self.active
                    or reservation['body_sha256']!=sha(body)):
                raise ValueError('live_transport_reservation_mismatch')
            # 发送边界重查最新全局暂停；不消费发送权、不释放或新增预算。
            if (self.clock()<self.pause_until or self.clock()<self.probe_retry
                    or self.need_probe or self.quota_bucket!=self.success//100):
                raise DispatchPaused('dispatch_wait_pause_or_probe')
            # 即使 fsync 抛错，同一进程也不能再消费这个发送权。
            self.dispatched.add(call)
            try:self.wal.add('dispatch_started',call=call,rid=reservation['rid'],body_sha256=sha(body))
            except BaseException:
                self.active.discard(call);self.halt='unknown_inflight';raise
    def receive(self,call,status,raw=b'',reason=None,crash_after_response=False):
        with self.lock:
            if call not in self.active:raise IntegrityError('unexpected_response')
            if self.mode=='live' and call not in self.dispatched:raise IntegrityError('response_without_dispatch')
            finished=self.clock()
            # This durable raw response is authoritative even if process dies on the next instruction.
            self.wal.add('response',call=call,status=status,raw=base64.b64encode(raw).decode(),raw_sha256=sha(raw),reason=reason,finished=finished)
            self.active.remove(call);req=self.pending.pop(call)
            if 200<=status<300:
                self.success+=1;self._consume_usage(raw)
                if crash_after_response:raise SystemExit('fault_injection_after_response_fsync')
                try:record=self.tasks[req['seed']].apply(guard(raw))
                except GuardError:self.stop('envelope_model_usage_token_guard');return
                self.wal.add('action',call=call,record=record,recovered=False);self.applied.add(call)
            elif status==0 or status==408 or 500<=status<=599:
                self.failed+=1;self.pause_until=max(self.pause_until,finished+300);self.need_probe=True
                self.wal.add('pause',reason='service_failure',until=self.pause_until,call=call)
            else:self.failed+=1;self.stop('other_http')
            self.summary()
    def request(self,seed,transport,probe):
        if self.mode=='mock' and getattr(transport,'testing',None) is False:raise Paused('mock_cannot_use_live_transport')
        if self.mode=='live' and getattr(transport,'testing',None) is not False:raise Paused('mock_transport_forbidden_live')
        with self.lock:
            self._send_gate(seed)
            if self.halt:raise Paused(self.halt)
            # 暂停时未消费的预留由调度器下次 request 原样继续，不再 reserve。
            ev=next((r for c,r in self.pending.items()
                     if r['seed']==seed and c in self.active and c not in self.dispatched),None)
            if ev is None:ev,body=self.reserve(seed,probe)
            else:
                if self.need_probe or self.quota_bucket!=self.success//100:self.quota(probe)
                body=base64.b64decode(ev['body'],validate=True)
        # Unknown exceptions stay reserved; never guess if the server executed a request.
        try:
            if self.mode=='live':status,raw,reason=transport.send(body,engine=self,reservation=ev)
            else:
                self.claim_dispatch(ev,body)
                status,raw,reason=transport.send(body)
        except BaseException as ex:
            with self.lock:
                if isinstance(ex,DispatchPaused) and ev['call'] not in self.dispatched:raise
                self.active.discard(ev['call']);self.stop('unknown_inflight')
            raise
        self.receive(ev['call'],status,raw,reason)
    def approve_smoke(self,approval):
        with self.lock:
            if self.mode!='live':raise Paused('mock_cannot_approve_human_review')
            try:validate_smoke(approval,self.wal.events,self.runner_version)
            except (ValueError,KeyError,TypeError) as ex:raise Paused('human_smoke_evidence_invalid:'+str(ex)) from ex
            self.wal.add('smoke_approved',approval=approval);self.smoke=True;self.smoke_approval=approval
    def smoke_review_template(self):
        # 人工自行填写；生成器只提供未批准且未勾选的模板。
        from evidence import CHECKS
        snap=smoke_snapshot(self.wal.events,self.runner_version)
        return dict(schema='HOLDOUT_D_AMEND3_HUMAN_REVIEW_V1',approved=False,reviewer='',reviewed_at='',
                    reviewer_declaration='',snapshot_digest=sha(dump(snap).encode()),
                    review_method='manual',checked_items={k:False for k in CHECKS},
                    snapshot=snap)
    def close(self):self.wal.close()

    def reconcile_response(self,call,evidence_path):
        """只接收可核验原始服务响应；不提供“推测未发送”释放接口。

        evidence 是项目内服务原始记录导出，需独立人工核验来源；本函数验证字节与请求身份。
        不把本地自制记录当成第三方可信证据。
        """
        with self.lock:
            p=Path(evidence_path).resolve()
            if not p.is_relative_to(ROOT/'HOLDOUT_D_runner_v5'):raise IntegrityError('evidence_outside_runner')
            z=json.loads(p.read_text());req=self.pending.get(call)
            if not req or self.active:raise Paused('no_unknown_or_active_requests')
            if z.get('rid')!=req['rid'] or z.get('body_sha256')!=req['body_sha256'] or z.get('verified_by') in (None,'') or z.get('source') in (None,''):raise IntegrityError('reconcile_evidence_identity')
            raw=base64.b64decode(z['raw'],validate=True)
            if sha(raw)!=z['raw_sha256'] or type(z['status']) is not int or not 200<=z['status']<300:raise IntegrityError('only_verified_return_reconciliation_supported')
            self.wal.add('reconcile',call=call,evidence=str(p.relative_to(ROOT)),evidence_sha256=sha(p.read_bytes()),previous_halt=self.halt)
            self.active.add(call)
            # Preserve all unrelated safety stops.
            if self.halt=='unknown_inflight':self.halt=None
            self.receive(call,z['status'],raw,reason='verified_original_response')
