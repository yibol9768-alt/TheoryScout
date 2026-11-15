"""执行完整性校验；任何不一致只撤销确认资格，不抛弃可读逐格数据。"""
import json
import math
import re
from common import OUT, jobs, sha

ALLOWED_STOPS = {'planned_complete', 'cpu_soft', 'wall_soft'}

def read_json(path):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result: raise ValueError('duplicate JSON key: '+k)
            result[k] = v
        return result
    def constant(v): raise ValueError('nonfinite JSON: '+v)
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=constant)

def stage_ok(state, program):
    return (isinstance(state, dict) and state.get('program') == program
            and type(state.get('returncode')) is int and state['returncode'] == 0
            and state.get('wall_hard_killed') is False)

def valid_usage(u):
    return (isinstance(u, dict) and set(u) == {'cpu_seconds','peak_rss_mib','monotonic'}
            and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in u.values()))

def inspect(data, mh, cfg, readable, hashes):
    errors = []; started = set(); completed = []; committed = {}; active = None; stopped = None
    plan = list(jobs(cfg['blocks'])); plan_set = set(plan); position = 0
    previous_usage = None
    def error(code, **detail): errors.append(dict(code=code, **detail))
    def load(name):
        try:
            value = read_json(data/name)
            if not isinstance(value, dict): raise ValueError('expected object')
            return value
        except Exception as exc:
            error('missing_or_invalid_record', file=name, error=repr(exc)); return {}
    launch = load('launch.json'); start = load('start.json')
    runner = load('runner_status.json'); exit_state = load('run_exit.json')
    if not stage_ok(exit_state, 'run.py'): error('runner_not_normal', record=exit_state)
    if not (type(launch.get('schema')) is int and launch['schema'] == 1 and launch.get('manifest_sha256') == mh
            and launch.get('cpu_hard_split') == [21340,50,10]
            and launch.get('preparation_cpu_reserve') == cfg['preparation_cpu_reserve']
            and launch.get('wall_hard_total') == cfg['total_wall_hard']
            and valid_usage(launch.get('usage'))): error('launch_record_mismatch')
    if not (type(start.get('schema')) is int and start['schema'] == 1 and start.get('manifest_sha256') == mh
            and start.get('streams_sha256') == sha(OUT/'streams.json')
            and start.get('program_sha256') == sha(OUT/'run.py')
            and start.get('cfg') == cfg and start.get('no_draw_before_this_record') is True
            and valid_usage(start.get('usage'))): error('start_record_mismatch')
    if valid_usage(start.get('usage')): previous_usage = start['usage']
    try: lines = (data/'events.jsonl').read_text().splitlines()
    except Exception as exc:
        lines = []; error('missing_or_unreadable_events', error=repr(exc))
    if not lines: error('empty_event_log')
    def key(value):
        if not (isinstance(value,list) and len(value)==3 and all(type(v) is int for v in value)):
            raise ValueError('cell must be three integers')
        k = tuple(value)
        if k not in plan_set: raise ValueError('cell outside frozen plan')
        return k
    for line_no, line in enumerate(lines, 1):
        try:
            # 与文件JSON使用相同的严格解析，拒绝重复字段和NaN。
            class Line:
                def read_text(self): return line
            event = read_json(Line())
            if not isinstance(event,dict): raise ValueError('event must be object')
            state = event.get('state')
            fields = {'started': {'schema','state','cell','usage'},
                      'complete': {'schema','state','cell','sha256','usage'},
                      'stopped': {'schema','state','reason','incomplete_cell','error','usage'}}
            if state not in fields or set(event) != fields[state] or type(event['schema']) is not int or event['schema'] != 1:
                raise ValueError('illegal event schema')
            if not valid_usage(event['usage']): raise ValueError('invalid event usage')
            u = event['usage']
            if previous_usage and (u['cpu_seconds'] < previous_usage['cpu_seconds'] or u['monotonic'] < previous_usage['monotonic']):
                error('nonmonotonic_usage', line=line_no)
            previous_usage = u
            if stopped is not None: error('event_after_stopped', line=line_no)
            if state == 'started':
                k = key(event['cell']); started.add(k)
                if active is not None: error('started_while_active', line=line_no)
                if position >= len(plan) or k != plan[position]: error('started_out_of_order_or_duplicate', line=line_no, cell=list(k))
                position += 1; active = k
            elif state == 'complete':
                k = key(event['cell'])
                if active != k: error('complete_without_matching_start', line=line_no)
                if k in committed: error('duplicate_complete', line=line_no)
                h = event['sha256']
                if not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None: raise ValueError('invalid SHA256 schema')
                name = f'd{k[0]}_r{k[1]}_s{k[2]}.npz'
                if hashes.get(name) != h: error('commit_hash_mismatch_or_missing_npz', line=line_no, file=name, committed=h, actual=hashes.get(name))
                completed.append(k); committed[k] = h; active = None
            else:
                if stopped is not None: error('duplicate_stopped', line=line_no)
                stopped = event
                if event['reason'] not in ALLOWED_STOPS or event['error'] is not None: error('abnormal_stop', line=line_no)
                incomplete = None if event['incomplete_cell'] is None else key(event['incomplete_cell'])
                if incomplete != active: error('stopped_incomplete_cell_mismatch', line=line_no)
                if event['reason'] == 'planned_complete' and (completed != plan or active is not None): error('premature_planned_complete')
        except Exception as exc:
            error('invalid_event', line=line_no, error=repr(exc))
    if stopped is None: error('missing_stopped')
    if set(completed) != set(readable):
        error('completed_readable_set_mismatch', missing_readable=[list(k) for k in sorted(set(completed)-set(readable))],
              unlogged_readable=[list(k) for k in sorted(set(readable)-set(completed))])
    expected_names = {f'd{d}_r{ir}_s{s}.npz' for d,ir,s in completed}
    if set(hashes) != expected_names: error('completed_npz_set_mismatch', unlogged=sorted(set(hashes)-expected_names), missing=sorted(expected_names-set(hashes)))
    if not (type(runner.get('schema')) is int and runner['schema']==1
            and type(runner.get('reason')) is str and runner['reason'] in ALLOWED_STOPS and runner.get('error') is None
            and valid_usage(runner.get('usage'))): error('runner_status_invalid')
    try:
        if not isinstance(runner.get('completed'),list): raise ValueError('completed must be list')
        for k in runner['completed']: key(k)
        if runner.get('incomplete_cell') is not None: key(runner['incomplete_cell'])
    except Exception as exc:
        error('runner_cell_schema_invalid', error=repr(exc))
    if runner.get('completed') != [list(k) for k in completed]: error('runner_completed_mismatch')
    if stopped is not None:
        for field in ['reason','error','incomplete_cell']:
            if field not in runner or runner[field] != stopped[field]: error('runner_stopped_mismatch', field=field)
    wall = runner.get('wall_seconds')
    if type(wall) not in (int,float) or not math.isfinite(wall) or not 0 <= wall <= cfg['runner_wall_hard']:
        error('runner_wall_budget_invalid')
    if valid_usage(runner.get('usage')):
        u = runner['usage']
        if u['cpu_seconds'] > cfg['runner_cpu_hard']: error('runner_cpu_budget_exceeded')
        if previous_usage and (u['cpu_seconds'] < previous_usage['cpu_seconds'] or u['monotonic'] < previous_usage['monotonic']): error('status_usage_precedes_events')
    return dict(valid=not errors, errors=errors, started=started, completed=completed,
                runner_exit=exit_state, runner_status=runner)
