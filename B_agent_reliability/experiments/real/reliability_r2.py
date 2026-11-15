"""R2 and CALIB1: immutable protocol, bounded calls, durable natural traces."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import reliability_core as d

ROOT = d.ROOT
CODE = ('reliability_core.py', 'reliability_r2.py', 'report_reliability_r2.py', 'test_reliability_r2.py')
PREREGS = {'r2': 'PREREG_DIFFICULTY_R2.md', 'calib1': 'PREREG_CALIB1.md'}
FREEZES = {'r2': 'FREEZE_DIFFICULTY_R2.json', 'calib1': 'FREEZE_CALIB1.json'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def hashes(phase):
    return {n: sha(ROOT/n) for n in (*CODE, PREREGS[phase], 'difficulty.py')}

def verify(phase):
    frozen = json.loads((ROOT/FREEZES[phase]).read_text())
    if frozen['sha256'] != hashes(phase):
        raise d.Stop('freeze_mismatch')
    return frozen

def r2_schedule():
    return [dict(r, seed=r['seed']+1000000, length=16) for r in d.schedule()]

def read_results(out):
    return json.loads((out/'summary.json').read_text())['results']

def selection():
    old = read_results(ROOT/'difficulty_real')
    new = read_results(ROOT/'difficulty_r2_real')
    stats = []
    for family in d.KNOBS:
        for level, knob in enumerate(d.KNOBS[family]):
            batches = [[r for r in rs if (r['family'], r['level']) == (family, level)] for rs in (old, new)]
            n = sum(r['advances'] for batch in batches for r in batch)
            e = sum(r['seed_errors'] for batch in batches for r in batch)
            complete = any(len(batch) == 4 and all(r['status'] == 'success' for r in batch)
                           and sum(r['advances'] for r in batch) == 64 for batch in batches)
            if n:
                p = e/n
                stats.append(dict(family=family, level=level, knob=knob, e=e, n=n,
                                  p=p, distance=max(0, .02-p, p-.10), complete=complete))
    key = lambda s: (s['distance'], abs(s['p']-.06), s['level'], list(d.KNOBS).index(s['family']))
    target = sorted([s for s in stats if s['complete'] and s['distance'] == 0], key=key)
    chosen = []
    for s in target:
        if s['family'] not in [x['family'] for x in chosen]:
            chosen.append(s)
        if len(chosen) == 2:
            break
    reason = 'complete_target'
    if not chosen:
        complete = [s for s in stats if s['complete']]
        chosen = sorted(complete or stats, key=key)[:1]
        reason = 'complete_outside_target_fallback' if complete else 'incomplete_fallback'
    return dict(reason=reason, selected=chosen, cells=stats)

class Budget:
    def __init__(self, out, phase, mock=False):
        self.lock = threading.RLock()
        self.calls = 0
        self.halt = None
        self.real = not mock
        self.phase = phase
        self.cap = 1200 if phase == 'r2' else 4800
        self.quota_at = -1
        self.journal = d.Journal(out/'budget.jsonl')
        self.task_calls = collections.Counter()
        self.config_calls = collections.Counter()
        self.fail_streak = 0
        self.cooldown_until = 0
        self.recent = collections.deque(maxlen=20)

    def stop(self, reason):
        with self.lock:
            self.halt = reason
            self.journal.add('halt', reason=reason, calls=self.calls)

    def quota(self):
        if not self.real or self.quota_at == self.calls:
            return
        try:
            p = subprocess.run(['python3', '/home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py'],
                               capture_output=True, text=True, timeout=50, check=True)
            parts = p.stdout.split()
            if len(parts) != 2:
                raise ValueError()
            a, b = map(float, parts)
            if not (0 <= a <= 100 and 0 <= b <= 100):
                raise ValueError()
            self.journal.add('quota', calls=self.calls, five_hour=a, week=b)
            self.quota_at = self.calls
            print(f'quota calls={self.calls}: {a:g}% {b:g}%', flush=True)
            if max(a, b) >= 80:
                self.stop('quota_80')
        except Exception:
            self.stop('quota_unavailable')
        if self.halt:
            raise d.Stop(self.halt)

    def reserve(self, ident, config):
        while True:
            with self.lock:
                if self.halt:
                    raise d.Stop(self.halt)
                remaining = self.cooldown_until-time.monotonic()
                if remaining <= 0:
                    if self.calls % 100 == 0:
                        self.quota()
                    if self.calls >= self.cap:
                        self.stop('hard_cap')
                        raise d.Stop(self.halt)
                    if self.phase == 'calib1':
                        if self.task_calls[ident] >= 48:
                            raise d.Stop('task_cap_48')
                        if self.config_calls[config] >= 2400:
                            raise d.Stop('config_cap_2400')
                    self.calls += 1
                    self.task_calls[ident] += 1
                    self.config_calls[config] += 1
                    self.journal.add('reserve', call=self.calls, task=ident, config=config)
                    return self.calls
            time.sleep(min(30, remaining))

    def outcome(self, failed, call):
        with self.lock:
            self.recent.append(bool(failed))
            self.fail_streak = self.fail_streak+1 if failed else 0
            self.journal.add('outcome', call=call, failed=failed, streak=self.fail_streak)
            if self.fail_streak >= 5:
                self.cooldown_until = max(self.cooldown_until, time.monotonic()+600)
                self.fail_streak = 0
                self.journal.add('cooldown', call=call, seconds=600)
                print('service cooldown: 600 seconds', flush=True)

    def boundary(self, config, attempted):
        with self.lock:
            failures = sum(self.recent)
            self.journal.add('service_boundary', config=config, attempted=attempted,
                             recent_count=len(self.recent), failures=failures)
            if failures >= 4:
                self.stop('service_boundary_stop')

class Client:
    def __init__(self, budget, mock=False):
        self.budget, self.mock = budget, mock
        if not mock and not os.environ.get('SOLKEY'):
            raise d.Stop('missing_SOLKEY')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, messages, trace, context, expected=None):
        ident = Path(trace.f.name).stem
        config = ident.rsplit('_', 1)[0]
        for attempt in range(5):
            if attempt:
                delay = (10, 30, 60, 120)[attempt-1]
                trace.add('retry_wait', attempt=attempt, seconds=delay)
                time.sleep(delay)
            call = self.budget.reserve(ident, config)
            trace.add('request', call=call, attempt=attempt, model=d.MODEL, messages=messages, **context)
            started = time.monotonic()
            try:
                if self.mock:
                    response = {'model': d.MODEL, 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                                'choices': [{'finish_reason': 'stop', 'message': {'content': d.dump(expected)}}]}
                else:
                    body = d.dump(dict(model=d.MODEL, messages=messages, max_completion_tokens=4096)).encode()
                    req = urllib.request.Request(d.URL, body, {'Content-Type': 'application/json',
                                                            'Authorization': 'Bearer '+os.environ['SOLKEY']})
                    try:
                        with self.opener.open(req, timeout=90) as r:
                            raw = r.read()
                    except urllib.error.HTTPError as e:
                        if 500 <= e.code <= 599:
                            raise d.ServiceError(f'http_{e.code}') from None
                        self.budget.stop(f'http_{e.code}')
                        raise d.Stop(f'http_{e.code}') from None
                    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                        raise d.ServiceError('transport') from None
                    try:
                        response = json.loads(raw)
                    except (ValueError, UnicodeError):
                        self.budget.stop('invalid_api_json')
                        raise d.Stop('invalid_api_json') from None
                trace.add('response', call=call, response=response, seconds=time.monotonic()-started)
                if not isinstance(response, dict) or response.get('model') != d.MODEL:
                    self.budget.stop('model_mismatch')
                    raise d.Stop('model_mismatch')
                u = response.get('usage')
                if not isinstance(u, dict) or any(type(u.get(k)) is not int or u[k] < 0 for k in
                                                ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                    self.budget.stop('missing_usage')
                    raise d.Stop('missing_usage')
                if u['completion_tokens'] > 4096 or u['prompt_tokens'] > 16000:
                    self.budget.stop('token_guard')
                    raise d.Stop('token_guard')
                try:
                    content = response['choices'][0]['message']['content']
                    finish = response['choices'][0]['finish_reason']
                    if not isinstance(content, str) or not isinstance(finish, str):
                        raise ValueError()
                except (ValueError, KeyError, IndexError, TypeError):
                    self.budget.stop('invalid_response_shape')
                    raise d.Stop('invalid_response_shape')
                self.budget.outcome(False, call)
                return response, call
            except d.ServiceError as e:
                trace.add('service_error', call=call, attempt=attempt, reason=str(e), seconds=time.monotonic()-started)
                self.budget.outcome(True, call)
                if attempt == 4:
                    raise d.Stop('service_retry_exhausted') from None
        raise AssertionError()

def execute(phase, mock=False):
    verify(phase)
    if phase == 'r2':
        rows = r2_schedule()
    else:
        r2 = json.loads((ROOT/'difficulty_r2_real/summary.json').read_text())
        if r2['halt']:
            raise d.Stop('r2_global_halt_blocks_calibration')
        chosen = selection()
        frozen = dict(utc=d.now(), **chosen, input_sha256={
            'difficulty_real/summary.json': sha(ROOT/'difficulty_real/summary.json'),
            'difficulty_r2_real/summary.json': sha(ROOT/'difficulty_r2_real/summary.json')})
        with (ROOT/'FREEZE_SELECTION_CALIB1.json').open('x') as f:
            f.write(json.dumps(frozen, ensure_ascii=False, indent=2)+'\n')
        rows = []
        for s in chosen['selected']:
            for rep in range(1, 51):
                seed = 3000000+10000*list(d.KNOBS).index(s['family'])+1000*s['level']+rep
                rows.append({k:s[k] for k in ('family', 'level', 'knob')} | dict(seed=seed, length=32))
    out = ROOT/(('difficulty_r2_' if phase == 'r2' else 'calib1_')+('mock' if mock else 'real'))
    out.mkdir(exist_ok=False)
    budget = Budget(out, phase, mock)
    client = Client(budget, mock)
    summary = dict(started_utc=d.now(), phase=phase, mode='mock' if mock else 'real', model=d.MODEL,
                   concurrency=2, cap=budget.cap, schedule=rows, frozen_sha256=hashes(phase), results=[])
    if phase == 'calib1':
        summary['selection'] = frozen
    def save():
        summary.update(calls=budget.calls, halt=budget.halt, updated_utc=d.now(),
                       config_calls=dict(budget.config_calls))
        tmp = out/'summary.pending'
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        tmp.replace(out/'summary.json')
    save()
    cursor = 0
    smoke_at = 3 if phase == 'r2' else 2
    counts = collections.Counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        while cursor < len(rows) and not budget.halt:
            if cursor == smoke_at and not mock:
                save()
                gate = out/'SMOKE_APPROVED.json'
                print(f'SMOKE_READY {phase}: awaiting {gate.name}', flush=True)
                while not gate.exists() and not budget.halt:
                    time.sleep(2)
                if budget.halt:
                    break
                approval = json.loads(gate.read_text())
                if not approval.get('approved'):
                    budget.stop('smoke_rejected')
                    break
                budget.journal.add('smoke_approved', approval=approval, calls=budget.calls)
            row = rows[cursor]
            config = (row['family'], row['level'])
            limit = min(len(rows), cursor+2)
            if cursor < smoke_at:
                limit = min(limit, smoke_at)
            if phase == 'calib1':
                completed = counts[config]
                next_boundary = next(x for x in (10, 25, 50) if x > completed)
                limit = min(limit, cursor+next_boundary-completed)
                while limit > cursor+1 and (rows[limit-1]['family'], rows[limit-1]['level']) != config:
                    limit -= 1
            futures = [pool.submit(d.run_task, r, client, out) for r in rows[cursor:limit]]
            for f in as_completed(futures):
                try:
                    result = f.result()
                except Exception:
                    budget.stop('unexpected_local_exception')
                    save()
                    raise RuntimeError('local execution failure; details omitted to protect secrets') from None
                summary['results'].append(result)
                counts[(result['family'], result['level'])] += 1
                save()
                print(d.dump(result), flush=True)
            cursor = limit
            if phase == 'calib1' and counts[config] in (10, 25, 50):
                budget.boundary(f'{config[0]}_d{config[1]}', counts[config])
                save()
    with budget.lock:
        if budget.calls % 100 == 0 and not budget.halt:
            try:
                budget.quota()
            except d.Stop:
                pass
    summary['finished_utc'] = d.now()
    save()
    budget.journal.f.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['r2', 'calib1'])
    ap.add_argument('--mock', action='store_true')
    args = ap.parse_args()
    execute(args.phase, args.mock)
