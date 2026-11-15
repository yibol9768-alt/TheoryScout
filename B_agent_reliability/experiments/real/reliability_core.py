"""Frozen difficulty scan. Standard library only; real actions, no injections."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
MODEL = 'gpt-6-luna'
URL = 'http://127.0.0.1:18319/v1/chat/completions'
CAP = 1200
KNOBS = {'sheet': [2, 4, 8, 16], 'indirect': [1, 3, 6, 10], 'memory': [4, 8, 12, 16]}
SOURCES = ()  # Phase-specific manifests are checked by the runner.

def dump(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))

def now():
    return datetime.now(timezone.utc).isoformat()

def hashes():
    return {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES}

class Stop(Exception):
    pass

class ServiceError(Exception):
    pass

def schedule():
    # Distinct tasks across all cells; counterbalance family order by level.
    out = []
    families = list(KNOBS)
    for level in range(4):
        for rep in range(4):
            for f in families[(level+rep)%3:]+families[:(level+rep)%3]:
                fi = list(KNOBS).index(f)
                out.append(dict(family=f, level=level, knob=KNOBS[f][level],
                                seed=901001+1000*fi+100*level+rep))
    return out

def task(row):
    rng = random.Random(row['seed'])
    f, k = row['family'], row['knob']
    n = k if f == 'memory' else (6 if f == 'sheet' else 8)
    initial = {f'r{i}': rng.randrange(997) for i in range(n)}
    if f == 'indirect':
        initial['cursor'] = rng.randrange(n)
    if f == 'memory':
        initial['checksum'] = 0
    table = {str(i): [rng.randrange(n), rng.randrange(n)] for i in range(n)} if f == 'indirect' else {}
    ops = []
    for step in range(row.get("length", 16)):
        if f == 'sheet':
            instructions = []
            for _ in range(k):
                a, b, c = rng.sample(range(n), 3)
                instructions.append(dict(a=a, b=b, c=c, m=rng.choice([2, 3, 5, 7]), d=rng.randint(-37, 37)))
            op = {'instructions': instructions}
        elif f == 'indirect':
            op = {'hops': k, 'delta': [rng.randint(-31, 31) for _ in range(k)], 'start': rng.randrange(n)}
        else:
            op = {'a': rng.randrange(n), 'b': rng.randrange(n), 'shift': rng.randrange(1, n),
                  'm': rng.choice([2, 3, 5]), 'd': rng.randint(-37, 37)}
        ops.append(op)
    t = dict(row, initial=initial, table=table, operations=ops)
    t['prefix'] = [initial.copy()]
    for op in ops:
        t['prefix'].append(transition(t, t['prefix'][-1], op))
    return t

def transition(t, before, op):
    s = before.copy()
    f = t['family']
    if f == 'sheet':
        for x in op['instructions']:
            a, b, c = (f'r{x[z]}' for z in ('a', 'b', 'c'))
            # Read current values, then sequentially propagate changes.
            s[a] = (s[a]+x['m']*s[b]-s[c]+x['d']) % 997
            s[c] = (s[c]+s[a]) % 997
        s['F0'] = (s['r0']+2*s['r1']-s['r2']) % 997
        s['F1'] = (3*s['F0']+s['r3']-s['r4']) % 997
        s['F2'] = (s['F1']+s['r5']*2) % 997
    elif f == 'indirect':
        cursor = (s['cursor']+op['start']) % 8
        for d in op['delta']:
            a = f'r{cursor}'
            nxt = t['table'][str(cursor)][s[a] % 2]
            b = f'r{nxt}'
            s[a] = (s[a]+s[b]+d) % 997
            s[b] = (s[b]+2*s[a]) % 997  # aliases intentional; after first write
            cursor = (nxt+s[a] % 3) % 8
        s['cursor'] = cursor
    else:
        a, b = f"r{op['a']}", f"r{op['b']}"
        s[a] = (s[a]+op['m']*s[b]+op['d']) % 997
        c = f"r{(op['a']+op['shift']) % t['knob']}"
        s[c] = (s[c]-s[a]+s[b]) % 997
        s['checksum'] = (7*s['checksum']+s[a]+3*s[c]+s[b]) % 997
    return s

COMMON = ('You execute verifiable integer transactions. Return JSON only with exactly '
          '{"writes":{key:integer,...},"detect":boolean}. No extra fields, duplicate keys, or prose. '
          'All arithmetic uses Python modulo: x%997 is in 0..996 even for negative x. '
          'Assignments are sequential, including aliases: each RHS reads the latest state. '
          'Writes must contain exactly required_writes keys, even unchanged ones. '
          'detect means you believe the previous applied action was wrong; false on the first step. '
          'Do only the current transaction. Do not call tools. ')
RULES = {
    'sheet': ('State has r0..r5, derived cells F0,F1,F2. For each instruction in order: '
              'r[a]=(r[a]+m*r[b]-r[c]+d)%997; then r[c]=(r[c]+r[a])%997. '
              'After ALL instructions: F0=(r0+2*r1-r2)%997; F1=(3*F0+r3-r4)%997; F2=(F1+2*r5)%997. '
              'Output all six registers and all three formula cells.'),
    'indirect': ('Eight registers r0..r7 and cursor. Set local cursor=(state.cursor+start)%8. '
                 'For each d in delta: a is r[cursor]; select nxt=table[str(cursor)][value(a)%2] BEFORE writes; '
                 'b is r[nxt]; assign a=(a+b+d)%997; then b=(b+2*a)%997; '
                 'then cursor=(nxt+value(a)%3)%8 using latest value(a), including aliases. '
                 'After all hops store cursor. Output all registers and cursor.'),
    'memory': ('Initial state appears only in the first user turn. Maintain it using the complete interaction history. '
               'No information is removed. For each transaction: a=r[operation.a], b=r[operation.b]; '
               'assign a=(a+m*b+d)%997; c=r[(operation.a+shift)%width]; '
               'assign c=(c-a+b)%997; checksum=(7*checksum+a+3*c+b)%997. '
               'All reads use latest values, including aliases. Output only a,c,checksum (distinct keys).')}

def required(t, op):
    if t['family'] == 'memory':
        return sorted({f"r{op['a']}", f"r{(op['a']+op['shift'])%t['knob']}", 'checksum'})
    return list(t['prefix'][1])

def parse(content, keys):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('duplicate_key')
            d[k] = v
        return d
    try:
        a = json.loads(content, object_pairs_hook=unique)
        if (type(a) is not dict or set(a) != {'writes', 'detect'} or type(a['detect']) is not bool
                or type(a['writes']) is not dict or set(a['writes']) != set(keys)
                or any(type(v) is not int or not 0 <= v <= 996 for v in a['writes'].values())):
            return None, 'schema'
        return a, None
    except (ValueError, TypeError):
        return None, 'json'

class Journal:
    def __init__(self, path):
        self.f = path.open('x')
    def add(self, event, **kw):
        self.f.write(dump(dict(event=event, utc=now(), **kw))+'\n')
        self.f.flush()
        os.fsync(self.f.fileno())

def run_task(row, client, out):
    t = task(row)
    ident = f"{row['family']}_d{row['level']}_{row['seed']}"
    trace = Journal(out/(ident+'.jsonl'))
    trace.add('task', task=t)
    history = [{'role': 'system', 'content': COMMON+RULES[t['family']]}]
    state = t['initial'].copy()
    result = dict(row, id=ident, status='unknown', advances=0, seed_errors=0, repair_attempts=0,
                  repaired=0, repair_reerrors=0, error_detects=0, unsolicited_detects=0,
                  successful_advance_detects=0, steps=0)
    try:
        for step, op in enumerate(t['operations']):
            keys = required(t, op)
            p = dict(step=step, operation=op, required_writes=keys)
            if t['family'] == 'memory':
                p['width'] = t['knob']
                if step == 0:
                    p['initial'] = t['initial']
            else:
                p['checkpoint'] = state.copy()
                p['table'] = t['table']
            history.append({'role': 'user', 'content': dump(p)})
            expected = t['prefix'][step+1]
            start = state.copy()
            messages = history if t['family'] == 'memory' else [history[0], history[-1]]
            for review in range(3):
                gold = {'writes': {k: expected[k] for k in keys}, 'detect': review > 0}
                response, call = client.request(messages, trace, {'step': step, 'review': review}, gold)
                try:
                    content = response['choices'][0]['message']['content']
                    finish = response['choices'][0]['finish_reason']
                except (KeyError, IndexError, TypeError):
                    client.budget.stop('invalid_response_shape')
                    raise Stop('invalid_response_shape')
                action, error_type = parse(content, keys)
                before = state.copy()
                if action:
                    state.update(action['writes'])  # apply actual model action, including wrong writes
                n_after = sum(state.get(k) != v for k, v in expected.items()) + int(action is None)
                correct = n_after == 0
                if not correct and error_type is None:
                    error_type = 'numeric'
                if finish == 'length' and not correct:
                    error_type = 'truncated_'+str(error_type)
                if review == 0:
                    result['advances'] += 1
                    result['seed_errors'] += int(not correct)
                    if action and action['detect']:
                        result['unsolicited_detects'] += 1
                        result['successful_advance_detects'] += int(correct)
                else:
                    result['repair_attempts'] += 1
                    result['repair_reerrors'] += int(not correct)
                    result['error_detects'] += int(bool(action and action['detect']))
                    result['repaired'] += int(correct)
                trace.add('action', call=call, step=step, review=review, action=action,
                          content=content, before=before, after=state.copy(), expected=expected,
                          n_before=0 if review == 0 else sum(before.get(k) != v for k, v in expected.items()),
                          n_after=n_after, correct=correct, error_type=error_type)
                history.append({'role': 'assistant', 'content': content or ''})
                if correct:
                    result['steps'] += 1
                    break
                if review == 2:
                    raise Stop('unrepaired_after_2_reviews')
                # Oracle-triggered review is EXPLICIT assisted feedback, never spontaneous q.
                repair = dict(mode='validator_triggered_review', instruction='The applied transaction failed validation. Recompute and correct the required writes. State whether you detect a previous error.',
                              step=step, operation=op, required_writes=keys, applied_state=state.copy())
                if t['family'] != 'memory':
                    repair.update(checkpoint=start, table=t['table'])
                history.append({'role': 'user', 'content': dump(repair)})
                messages = history if t['family'] == 'memory' else [history[0],
                            {'role': 'user', 'content': dump(p)},
                            {'role': 'assistant', 'content': content or ''}, history[-1]]
        result['status'] = 'success'
    except Stop as e:
        result['status'] = str(e)
    trace.add('result', result=result)
    trace.f.close()
    return result

