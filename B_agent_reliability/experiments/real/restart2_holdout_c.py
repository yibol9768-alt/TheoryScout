"""Restart engineering adapter: replay durable responses, never repeat scored work.

No frozen source is changed. Requests use the frozen client plus the disclosed
408 adapter. All reconstructed actions are compared in memory and suppressed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

import holdout_c as h
import continue_holdout_c as correction
import reliability_core as d

TERMINAL = {'success', 'unrepaired_after_2_reviews'}
ORIGINAL_JOURNAL = d.Journal


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify():
    h.verify()
    amendment = json.loads((h.ROOT/'FREEZE_AMENDMENT_408_HOLDOUT_C.json').read_text())
    assert h.sha(h.ROOT/'FREEZE_HOLDOUT_C.json') == amendment['original_freeze_sha256']
    assert all(h.sha(h.ROOT/name) == value for name, value in amendment['sha256'].items())


def ident(row):
    return f"{row['family']}_d{row['level']}_{row['seed']}"


class ReplayTrace:
    """Compare existing applied actions; append only after the durable frontier."""
    def __init__(self, path, dry=False):
        self.path = path
        self.old = events(path) if path.exists() else []
        self.actions = [x for x in self.old if x['event'] == 'action']
        self.index = 0
        self.live = not self.old
        self.dry = dry
        self.f = None
        if not dry:
            self.f = path.open('a' if self.old else 'x')

    def add(self, event, **kw):
        if not self.live:
            if event == 'task':
                assert self.old[0]['event'] == 'task' and self.old[0]['task'] == kw['task']
            elif event == 'action':
                assert self.index < len(self.actions)
                original = self.actions[self.index]
                assert kw == {k: v for k, v in original.items() if k not in ('event', 'utc')}
                self.index += 1
            elif event == 'result':
                assert self.index == len(self.actions)
                self.reconstructed_result = kw['result']
            return
        assert not self.dry
        ORIGINAL_JOURNAL.add(self, event, **kw)


class OffsetTrace:
    def __init__(self, trace, offset):
        self.trace, self.offset, self.f = trace, offset, trace.f

    def add(self, event, **kw):
        if event in ('request', 'service_error'):
            kw['attempt'] += self.offset
        self.trace.add(event, **kw)


class ReplayClient:
    def __init__(self, old, client=None):
        self.client = client
        self.budget = client.budget if client else None
        self.responses = {x['call']: x['response'] for x in old if x['event'] == 'response'}
        self.requests = [x for x in old if x['event'] == 'request' and x['call'] in self.responses]
        unresolved = [x for x in old if x['event'] == 'request' and x['call'] not in self.responses]
        self.pending = unresolved[-1] if unresolved else None
        self.index = 0
        self.frontier = None

    def request(self, messages, trace, context, expected=None):
        if self.index < len(self.requests):
            old = self.requests[self.index]
            assert old['messages'] == messages and all(old[k] == v for k, v in context.items())
            self.index += 1
            return self.responses[old['call']], old['call']
        assert trace.index == len(trace.actions)
        offset = 0
        if self.frontier is None:
            self.frontier = dict(context, messages_sha256=hashlib.sha256(d.dump(messages).encode()).hexdigest())
            if self.pending:
                assert self.pending['messages'] == messages
                assert all(self.pending[k] == v for k, v in context.items())
                offset = self.pending['attempt'] + 1
                self.frontier['original_call'] = self.pending['call']
            if self.client is None:
                raise d.Stop('offline_recovery_frontier')
            trace.live = True
            trace.add('restart_exact_state_recovery', **self.frontier)
        if self.client is None:
            raise AssertionError('offline replay beyond frontier')
        return self.client.request(messages, OffsetTrace(trace, offset), context, expected)


def run(row, client=None, dry=False):
    path = h.OUT/(ident(row)+'.jsonl')
    trace = ReplayTrace(path, dry=dry)
    replay = ReplayClient(trace.old, client)
    # d.run_task expects its trace factory to own/close f. Dry files stay local.
    if dry:
        trace.f = (h.ROOT/'RESTART_REPLAY_SINK.tmp').open('w')
    local.trace = trace
    try:
        result = d.run_task(row, replay, h.OUT)
    finally:
        local.trace = None
    assert replay.index == len(replay.requests) and trace.index == len(trace.actions)
    return result, replay.frontier


local = threading.local()


class RoutedJournal:
    def __new__(cls, path=None):
        if path is None:
            return ORIGINAL_JOURNAL.__new__(ORIGINAL_JOURNAL)
        assert local.trace.path == path
        return local.trace


def preflight():
    verify()
    summary = json.loads((h.OUT/'summary.json').read_text())
    budget = events(h.OUT/'budget.jsonl')
    reserved = [x['call'] for x in budget if x['event'] == 'reserve']
    outcomes = [x for x in budget if x['event'] == 'outcome']
    assert reserved == list(range(1, len(reserved)+1))
    assert len({x['call'] for x in outcomes}) == len(outcomes)
    assert set(reserved) == {x['call'] for x in outcomes}, 'Unaccounted in-flight call requires separate recovery audit'
    assert all(x.get('reason') == 'http_408' for x in budget if x['event'] == 'halt')
    rows, partial = [], []
    for row in h.schedule():
        path = h.OUT/(ident(row)+'.jsonl')
        if not path.exists():
            continue
        old = events(path)
        result, frontier = run(row, dry=True)
        if result['status'] in TERMINAL:
            assert old[-1]['event'] == 'result' and old[-1]['result'] == result
            rows.append(result)
        else:
            assert result['status'] == 'offline_recovery_frontier'
            partial.append(dict(id=ident(row), replayed_actions=sum(x['event']=='action' for x in old), **frontier))
    (h.ROOT/'RESTART_REPLAY_SINK.tmp').unlink(missing_ok=True)
    assert all(any(x == r for r in rows) for x in summary['results'])
    assert len(partial) <= 2
    return dict(utc=d.now(),calls=len(reserved),successful=sum(not x['failed'] for x in outcomes),
                failed=sum(x['failed'] for x in outcomes),terminal=rows,partial=partial,
                frozen_verified=True,amendment_verified=True)


def execute():
    audit = preflight()
    summary = json.loads((h.OUT/'summary.json').read_text())
    gate = json.loads((h.OUT/'SMOKE_APPROVED.json').read_text())
    assert gate['approved'] is True and gate['reviewer'] and gate['checked_items']
    assert summary['stage'] == 'formal' and summary['halt'] is None
    archive = h.OUT/'RESTART2_ORIGINAL_20261106'
    archive.mkdir(exist_ok=False)
    originals = [h.OUT/'summary.json',h.OUT/'budget.jsonl',*sorted(h.OUT.glob('sheet_*.jsonl'))]
    for path in originals:
        (archive/path.name).write_bytes(path.read_bytes())
    audit['original_sha256'] = {p.name:h.sha(p) for p in originals}
    audit['adapter_sha256'] = h.sha(Path(__file__))
    h.write(h.ROOT/'RESTART2_AUDIT_HOLDOUT_C.json',audit)
    budget = correction.Budget(h.OUT,resume=True)
    pauses = [datetime.fromisoformat(x['utc']).timestamp()+x['seconds'] for x in events(h.OUT/'budget.jsonl') if x['event']=='pause']
    budget.pause_until = time.monotonic()+max(0,max(pauses,default=0)-time.time())
    budget.need_probe = True
    client = correction.Client(budget)
    budget.journal.add('machine_restart_recovery',audit='RESTART2_AUDIT_HOLDOUT_C.json',
                       calls=budget.calls,successful=budget.successful,failed=budget.failed,
                       terminals_preserved=len(audit['terminal']),partial=audit['partial'])
    summary['results'] = audit['terminal']
    summary['restart_recovery'] = dict(audit='RESTART2_AUDIT_HOLDOUT_C.json',utc=audit['utc'])

    def save():
        summary.update(calls=budget.calls,successful_calls=budget.successful,service_failures=budget.failed,
                       halt=budget.halt,updated_utc=d.now())
        h.write(h.OUT/'summary.json',summary)

    save()
    done = {x['id'] for x in summary['results']}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0,40,2):
            futures = [pool.submit(run,row,client) for row in h.schedule()[start:start+2] if ident(row) not in done]
            for future in as_completed(futures):
                result, _ = future.result()
                summary['results'].append(result)
                done.add(result['id'])
                save()
                print(d.dump(result),flush=True)
            if budget.halt:
                break
    summary['stage'] = 'stopped' if budget.halt else 'finished'
    if not budget.halt:
        assert len(summary['results']) == 40
        summary['finished_utc'] = d.now()
        if budget.successful % 100 == 0:
            with budget.lock:
                budget.quota()
    save()
    budget.journal.f.close()
    print('STAGE',summary['stage'],'SUCCESSFUL',budget.successful,'ATTEMPTS',budget.calls,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['check','formal'])
    args = parser.parse_args()
    with (h.OUT/'RESTART_RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        d.Journal = RoutedJournal
        if args.action == 'check':
            audit = preflight()
            print(d.dump(audit))
        else:
            execute()
