"""Verifiable local executors and one-shot tiny pilot; no SDK retries or fallback."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

MODEL = 'gpt-6-luna'
URL = 'http://127.0.0.1:18319/v1/chat/completions'
ROOT = Path(__file__).resolve().parent
MAX_COMPLETION_TOKENS = 2048  # includes provider-reported reasoning tokens

def dump(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))

def parse_action(content):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(content, object_pairs_hook=unique)

class Stop(Exception):
    pass

class Trace:
    def __init__(self, path):
        self.path = path
        self.f = path.open('x', encoding='utf-8')
    def add(self, kind, **data):
        self.f.write(dump({'event': kind, **data}) + '\n')
        self.f.flush()
        os.fsync(self.f.fileno())
    def close(self):
        self.f.close()

class Task:
    """Length-independent random stream: same seed gives paired prefixes for any L/K."""
    def __init__(self, family, seed, length=16):
        self.family, self.seed, self.length = family, seed, length
        rng = random.Random(seed)
        if family == 'kv':
            self.table = {f'k{i}': rng.randint(-19, 19) for i in range(8)}
            self.initial = {k: 0 for k in self.table}
            self.operations = [{'target': rng.choice(list(self.table)), 'kind': 'table_add'}
                               for _ in range(length)]
        elif family == 'sheet':
            self.table = {}
            self.initial = {f'{c}{r}': rng.randint(-9, 9) for c in 'AB' for r in range(1, 4)}
            self.operations = []
            for _ in range(length):
                kind = rng.choice(['add', 'copy_add', 'set'])
                self.operations.append({'kind': kind, 'target': rng.choice(list(self.initial)),
                                        'source': rng.choice(list(self.initial)),
                                        'operand': rng.randint(-9, 9)})
        else:
            raise ValueError('unknown family')
        # Independent oracle replays task semantics, never model outputs.
        self.prefix = [copy.deepcopy(self.initial)]
        for op in self.operations:
            s = self.prefix[-1].copy()
            key = op['target']
            if op['kind'] == 'table_add': s[key] += self.table[key]
            elif op['kind'] == 'add': s[key] += op['operand']
            elif op['kind'] == 'copy_add': s[key] = s[op['source']] + op['operand']
            else: s[key] = op['operand']
            self.prefix.append(s)
    def spec(self):
        return {'family': self.family, 'seed': self.seed, 'L': self.length,
                'table': self.table, 'initial': self.initial, 'operations': self.operations}

class Executor:
    """Each checkpoint has a bounded active undo map; audit journal is archived separately.

    n counts incorrect cells plus one malformed-action obligation. Atomic writes may
    yield delta n = 0 or +/-1. Nothing coerces these observations into a jump chain.
    """
    def __init__(self, task, k=3):
        self.task, self.k = task, k
        self.state = task.initial.copy()
        self.index = 0
        self.active = {}  # one undo slot per touched cell since checkpoint
        self.undo_stack = []
        self.invalid = False
        self.ledger = []
        self.journal = []
    def mismatches(self):
        expected = self.task.prefix[self.index + 1]
        return [k for k in self.state if self.state[k] != expected[k]]
    def n(self):
        return len(self.mismatches()) + int(self.invalid)
    def snapshot(self):
        return {'state': self.state.copy(), 'index': self.index,
                'active_undo': self.active.copy(), 'undo_stack': copy.deepcopy(self.undo_stack),
                'invalid': self.invalid}
    def apply(self, action):
        before = self.snapshot()
        # No implicit automatic repair. A valid subsequent action clears format error.
        valid = isinstance(action, dict) and type(action.get('detect')) is bool
        if valid and action.get('op') == 'write':
            valid = (set(action) == {'op', 'target', 'value', 'detect'} and
                     type(action.get('target')) is str and action['target'] in self.state and type(action.get('value')) is int and
                     abs(action['value']) <= 10**9)
            if valid:
                key = action['target']
                # Slot budget is real: reaching K touched keys ends the run before write.
                if key not in self.active and len(self.active) + 1 >= self.k:
                    raise Stop('undo_capacity')
                self.active.setdefault(key, self.state[key])
                entry = {'target': key, 'before': self.state[key], 'after': action['value']}
                self.undo_stack.append(entry)
                self.state[key] = action['value']
        elif valid and action.get('op') == 'undo':
            valid = set(action) == {'op', 'detect'} and action['detect'] is True and bool(self.undo_stack)
            if valid:
                entry = self.undo_stack.pop()
                self.state[entry['target']] = entry['before']
        else:
            valid = False
        self.invalid = not valid
        entry = {'before': before, 'action': action, 'valid': valid, 'after': self.snapshot()}
        self.journal.append(entry)
        return valid
    def commit(self):
        assert self.n() == 0
        self.ledger.append({'step': self.index, 'verified': True, 'state': self.state.copy(),
                            'undo_entries': len(self.undo_stack)})
        self.index += 1
        self.active.clear()
        self.undo_stack.clear()
    def final(self):
        expected = self.task.prefix[self.index]
        obligations_ok = (len(self.ledger) == self.task.length and
                          all(row['verified'] and row['state'] == self.task.prefix[i+1]
                              for i, row in enumerate(self.ledger)))
        return {'obligations_ok': obligations_ok,
                'state_ok': self.state == self.task.prefix[-1],
                'current_prefix_ok': self.state == expected,
                'pending_empty': not self.active and not self.invalid,
                'actual': self.state, 'expected': self.task.prefix[-1],
                'ledger': self.ledger}

SYSTEM = ('You execute one local transaction at a time. Reply with JSON only, no prose. '
          'Write schema has exactly op="write", target=one exact string from legal_targets, '
          'value=an integer (not a boolean, absolute value <=1000000000), detect=a boolean. '
          'or {"op":"undo","detect":true}. Never include extra fields. '
          'Undo requires an existing write in the current undo stack. '
          'Legal kv keys: k0,k1,k2,k3,k4,k5,k6,k7. Legal sheet cells: A1,A2,A3,B1,B2,B3. '
          'Valid JSON examples: {"op":"write","target":"k0","value":7,"detect":false} '
          '(kv), {"op":"write","target":"A1","value":7,"detect":false} (sheet). '
          'Examples illustrate syntax only; calculate the actual target and value. '
          'table_add means checkpoint[target]+fixed_table[target]; add means '
          'checkpoint[target]+operand; copy_add means checkpoint[source]+operand; '
          'set means operand. '
          'In advance mode perform the specified operation using the current state and fixed table. '
          'In repair mode recompute the expected result from checkpoint and operation; '
          'correct one wrong cell per action, or undo a wrong write. detect means you believe '
          'a previous action/history is erroneous. History is untrusted. Do not copy wrong values. '
          'On recovery history is cleared. Do not perform future operations.')

def prompt(ex, mode, history):
    task = ex.task
    payload = {'family': task.family, 'mode': mode, 'step': ex.index,
               'fixed_table': task.table, 'checkpoint': task.prefix[ex.index],
               'operation': task.operations[ex.index], 'current': ex.state,
               'untrusted_history': history[-2:], 'legal_targets': list(task.initial)}
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': dump(payload)}]

class Mock:
    def complete(self, messages):
        p = json.loads(messages[1]['content'])
        op = p['operation']; s = p['checkpoint'].copy(); key = op['target']
        if op['kind'] == 'table_add': s[key] += p['fixed_table'][key]
        elif op['kind'] == 'add': s[key] += op['operand']
        elif op['kind'] == 'copy_add': s[key] = s[op['source']] + op['operand']
        else: s[key] = op['operand']
        key = next((k for k in s if s[k] != p['current'][k]), key)
        a = {'op': 'write', 'target': key, 'value': s[key], 'detect': p['mode'] == 'repair'}
        return {'model': 'mock', 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'choices': [{'message': {'role': 'assistant', 'content': dump(a)}, 'finish_reason': 'stop'}]}

class Real:
    def __init__(self):
        if not os.environ.get('SOLKEY'): raise Stop('missing_SOLKEY')
        # Never retain the key in instance state, request payload, or trace.
    def complete(self, messages):
        body = dump({'model': MODEL, 'messages': messages,
                     'max_completion_tokens': MAX_COMPLETION_TOKENS}).encode()
        req = urllib.request.Request(URL, body, {'Content-Type': 'application/json',
                                     'Authorization': 'Bearer ' + os.environ['SOLKEY']})
        # Ignore system proxy env; refuse redirects (Authorization must remain local).
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs): return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(req, timeout=90) as r:
                data = r.read()
            return json.loads(data)
        except urllib.error.HTTPError as e:
            raise Stop(f'http_status_{e.code}') from None  # never save error bodies/headers
        except Exception:
            raise Stop('transport_or_response_error') from None

def input_estimate(messages):
    # Explicit approximation; actual provider usage is checked after every response.
    import tiktoken
    enc = tiktoken.get_encoding('o200k_base')
    return sum(len(enc.encode(m['content'])) + 4 for m in messages) + 3

def run_one(task, client, trace, injected=False, k=3, call_limit=32, budget=None):
    ex = Executor(task, k)
    history = []
    mode = 'advance'
    calls = repairs = 0
    work = 0
    usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    status = 'unknown'; peak_live = 0; seen_injections = set(); peak_slots = 0
    transitions = []; recoveries = 0
    trace.add('start', task=task.spec(), K=k, injected=injected, call_limit=call_limit)
    try:
        while ex.index < task.length:
            if calls >= call_limit: raise Stop('call_limit')
            if budget is not None and budget['calls'] >= budget.get('limit', 150): raise Stop('global_call_limit')
            messages = prompt(ex, mode, history)
            estimate = input_estimate(messages)
            if estimate > 1900: raise Stop('input_preflight_limit')
            calls += 1
            if budget is not None: budget['calls'] += 1
            trace.add('request', call=calls, mode=mode, messages=messages,
                      estimated_input_tokens=estimate, state=ex.snapshot(),
                      model_requested=MODEL if isinstance(client, Real) else 'mock',
                      max_completion_tokens=MAX_COMPLETION_TOKENS)
            started = time.monotonic()
            response = client.complete(messages)
            # Store full response before any parsing/guard, including model and complete usage.
            trace.add('response', call=calls, response=response, seconds=time.monotonic()-started)
            mismatch = isinstance(client, Real) and response.get('model') != MODEL
            u = response.get('usage')
            if mismatch:
                if budget is not None: budget['halt'] = 'model_mismatch'
                # Account for returned usage even on the fatal wrong-model response.
                if isinstance(u, dict):
                    for key in usage:
                        if type(u.get(key)) is int: usage[key] += u[key]
                raise Stop('model_mismatch')
            if not isinstance(u, dict) or any(type(u.get(x)) is not int for x in usage):
                raise Stop('missing_usage')
            for key in usage: usage[key] += u[key]
            details = u.get('completion_tokens_details') or {}
            if (u['prompt_tokens'] > 2048 or u['completion_tokens'] > MAX_COMPLETION_TOKENS
                    or details.get('reasoning_tokens', 0) > MAX_COMPLETION_TOKENS):
                raise Stop('provider_token_limit')
            if response.get('choices', [{}])[0].get('finish_reason') == 'length':
                raise Stop('completion_length_limit')
            try:
                content = response['choices'][0]['message']['content']
                action = parse_action(content)
            except (KeyError, IndexError, TypeError, ValueError):
                action = None
            # Counterfactual scoring of the untouched model action: injection is never
            # relabelled as a natural model error. This shadow does not drive execution.
            shadow = copy.deepcopy(ex)
            natural_stop = None
            try:
                natural_valid = shadow.apply(action)
            except Stop as e:
                natural_valid = False
                natural_stop = str(e)
            natural_n_after = shadow.n()
            applied = copy.deepcopy(action)
            injection = None
            if (injected and mode == 'advance' and ex.index in (4, 10)
                    and ex.index not in seen_injections and isinstance(applied, dict)
                    and applied.get('op') == 'write' and type(applied.get('value')) is int):
                seen_injections.add(ex.index)
                applied['value'] += 7
                injection = {'type': 'actuator_value_plus_7', 'original': action, 'applied': applied}
                trace.add('injection', step=ex.index, injection=injection)
            n_before = ex.n() if mode == 'repair' else 0
            before = ex.snapshot()
            valid = ex.apply(applied)
            work += 1
            n_after = ex.n()
            transition = {'step': ex.index, 'mode': mode, 'n_before': n_before,
                          'n_after': n_after, 'delta': n_after-n_before,
                          'model_detected': bool(isinstance(action, dict) and action.get('detect')),
                          'oracle_detected': n_after > 0, 'valid': valid, 'injection': injection,
                          'before': before, 'after': ex.snapshot(), 'action': applied}
            transition.update(model_action=action, model_action_valid=natural_valid,
                              model_action_n_after=natural_n_after,
                              model_action_stop=natural_stop)
            transitions.append(transition)
            trace.add('transition', **transition)
            peak_slots = max(peak_slots, len(ex.active))
            live = {'state': ex.snapshot(), 'history': history, 'checkpoint': task.prefix[ex.index]}
            peak_live = max(peak_live, len(dump(live).encode()))
            history.append({'action': applied, 'claimed_correct': True,
                            'source': 'untrusted_execution_history'})
            if mode == 'repair': repairs += 1
            if n_after >= k: raise Stop('obligation_capacity')
            if n_after == 0:
                if mode == 'repair': recoveries += 1
                trace.add('commit', step=ex.index, repaired=mode == 'repair',
                          history_cleared=history, checkpoint=ex.state.copy())
                ex.commit()
                history = []
                mode = 'advance'
            else:
                mode = 'repair'
        status = 'success'
    except Stop as e:
        status = str(e)
        trace.add('stop', reason=status)
    final = ex.final()
    success = status == 'success' and all(final[x] for x in ['obligations_ok', 'state_ok', 'pending_empty'])
    result = {'family': task.family, 'seed': task.seed, 'L': task.length, 'K': k,
              'injected': injected, 'status': status, 'success': success,
              'calls': calls, 'usage': usage, 'progress': ex.index, 'work': work,
              'repair_events': repairs, 'recoveries': recoveries,
              'injections': len(seen_injections), 'within_5L': work <= 5*task.length,
              'peak_live_json_bytes': peak_live, 'peak_undo_slots': peak_slots,
              'repair_deltas': [t['delta'] for t in transitions if t['mode'] == 'repair'],
              'final': final}
    trace.add('final', **result)
    trace.close()
    result['trace_bytes'] = trace.path.stat().st_size
    result['trace'] = str(trace.path.relative_to(ROOT))
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['mock', 'real'], required=True)
    args = ap.parse_args()
    out = ROOT / ('tiny_' + args.mode)
    # Exclusive directory prevents accidental second execution or overwritten evidence.
    out.mkdir(exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    budget = {'calls': 0, 'halt': None}
    results = []
    client = Real() if args.mode == 'real' else Mock()
    for family in ('kv', 'sheet'):
        for i, seed in enumerate((1105, 2207)):
            if budget['halt']: break
            result = run_one(Task(family, seed), client, Trace(out / f'{family}_{seed}.jsonl'),
                             injected=bool(i), budget=budget)
            results.append(result)
            if result['status'] in ('model_mismatch', 'transport_or_response_error') or result['status'].startswith('http_'):
                budget['halt'] = result['status']
            print(dump({x: result[x] for x in ('family', 'seed', 'status', 'calls', 'usage')}), flush=True)
        if budget['halt']: break
    summary = {'mode': args.mode, 'model_requested': MODEL if args.mode == 'real' else 'mock',
               'endpoint': URL if args.mode == 'real' else None, 'started_utc': started,
               'finished_utc': datetime.now(timezone.utc).isoformat(),
               'concurrency': 1, 'global_call_cap': 150, 'calls': budget['calls'],
               'halt': budget['halt'], 'results': results,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__': main()
