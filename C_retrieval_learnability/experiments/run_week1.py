"""Run FORMULATION.md section 5, without network access or external APIs.

Usage: /home/liuyibo/venvs/theory/bin/python experiments/run_week1.py
The stdlib supervisor sets environment, a 3.5 GB address-space limit, and 55-min
wall-time cap. All numerical configurations execute sequentially in one child.
"""
import argparse
import datetime
import hashlib
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent


def supervisor():
    env = dict(os.environ, OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               MPLCONFIGDIR='/tmp/theory-week1-matplotlib', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES='')
    out = ROOT / 'experiments' / 'week1'
    out.mkdir(exist_ok=True)
    start = time.perf_counter()
    with (out / 'execution.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=3300)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code = 124
    import json
    (out / 'supervisor.json').write_text(json.dumps({
        'returncode': code, 'wall_seconds': time.perf_counter() - start,
        'timeout_seconds': 3300, 'address_space_limit_bytes': 3500000000,
        'interpreter': sys.executable}, indent=2) + '\n')
    print((out / 'execution.log').read_text())
    raise SystemExit(code)


def worker():
    resource.setrlimit(resource.RLIMIT_AS, (3500000000, 3500000000))
    import numpy as np
    import scipy
    from scipy.optimize import minimize_scalar, brentq
    from scipy.special import logsumexp, expit
    import torch
    torch.set_num_threads(3)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    from week1_common import (OUT, V0, STAR, Y0, core_nce, core_metrics,
                              save, memory, parameters, objective, diagnostics,
                              normalize, train)

    global_start = time.perf_counter()
    meta = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python': sys.version, 'executable': sys.executable,
            'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__},
            'seeds': [0, 1, 2], 'rng': 'numpy.random.Generator(PCG64)',
            'dtype': 'float64', 'device': 'cpu', 'torch_num_threads': torch.get_num_threads(),
            'environment': {key: os.environ.get(key) for key in
                            ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']},
            'rlimit_as_bytes': resource.getrlimit(resource.RLIMIT_AS),
            'budgets_seconds': {'A': 120, 'B': 300, 'C': 1200, 'D': 1080, 'E': 600},
            'external_api_calls': 0, 'budget_changes': [], 'group_seconds': {},
            'hashes_at_start': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [ROOT / 'FORMULATION.md', ROOT / 'RESULTS_WEEK1.md',
                                          Path(__file__).resolve(), ROOT / 'experiments/week1_common.py']},
            'memory_at_start': memory()}
    save('metadata.json', meta)

    def finish_group(group, start):
        meta['group_seconds'][group] = time.perf_counter() - start
        meta['memory_latest'] = memory()
        meta['elapsed_seconds'] = time.perf_counter() - global_start
        save('metadata.json', meta)
        print(group, 'finished', round(meta['group_seconds'][group], 3), 'seconds', flush=True)

    # Full parameter gradients against independently expressed PyTorch objectives,
    # plus centered finite differences on all parameter group types.
    validation_start = time.perf_counter()
    validations = []
    for loss in ['nce', 'sigmoid', 'pe']:
        biases = ['global', 'row', 'zero'] if loss == 'sigmoid' else ['global']
        for bias in biases:
            for beta in [16., 'train']:
                A, pos, neg, p = parameters(3, 6, 0, loss, bias, beta)
                val, grads = objective(p, A, pos, neg, loss, beta, bias)
                tp = {key: torch.tensor(v, requires_grad=True) for key, v in p.items()}
                tA = torch.tensor(A)
                temp = tp['a'].exp()[0] if 'a' in tp else beta
                tz = temp * (tp['U'] @ tp['V'].T)
                if loss == 'nce':
                    tv = (torch.logsumexp(tz, 1) - tz[tA].reshape(len(A), 3).mean(1)).mean()
                elif loss == 'sigmoid':
                    logits = tz + (tp['b'][:, None] if bias == 'row' else tp.get('b', 0.))
                    tv = torch.nn.functional.softplus(-(2. * tA - 1.) * logits).mean()
                else:
                    zp = tz[torch.arange(len(A))[:, None], torch.tensor(pos)]
                    zn = tz[torch.arange(len(A))[:, None], torch.tensor(neg)]
                    delta = zn[:, None, :] - zp[:, :, None]
                    tv = torch.logsumexp(torch.cat([torch.zeros((len(A), 3, 1)), delta], 2), 2).mean()
                tv.backward()
                autograd_error = max(float(np.max(np.abs(grads[key] - tp[key].grad.numpy()))) for key in p)
                fd_error = 0.
                for key, arr in p.items():
                    for flat in sorted(set([0, arr.size // 2, arr.size - 1])):
                        idx = np.unravel_index(flat, arr.shape)
                        old = arr[idx]
                        arr[idx] = old + 1e-6
                        plus = objective(p, A, pos, neg, loss, beta, bias)[0]
                        arr[idx] = old - 1e-6
                        minus = objective(p, A, pos, neg, loss, beta, bias)[0]
                        arr[idx] = old
                        fd_error = max(fd_error, abs((plus - minus) / 2e-6 - grads[key][idx]))
                validations.append({'loss': loss, 'bias': bias, 'beta': beta,
                                    'loss_error': abs(val - tv.item()),
                                    'autograd_max_abs_error': autograd_error,
                                    'finite_difference_max_abs_error': fd_error})
    A, pos, neg, p = parameters(3, 6, 0, 'nce', 'global', 16.)
    diag = diagnostics(p, A, pos, 16.)
    decomposition_error = abs(objective(p, A, pos, neg, 'nce', 16.)[0] - np.log(3) - diag['D'] - diag['R'])
    # Differentiate D/R with torch and project to validate causal diagnostics.
    tu, tv = torch.tensor(p['U'], requires_grad=True), torch.tensor(p['V'], requires_grad=True)
    tz = 16 * (tu @ tv.T)
    tA = torch.tensor(A)
    ld = (torch.logsumexp(tz.masked_fill(~tA, -torch.inf), 1) - tz[tA].reshape(len(A), 3).mean(1) - np.log(3)).mean()
    lr = (torch.logsumexp(tz, 1) - torch.logsumexp(tz.masked_fill(~tA, -torch.inf), 1)).mean()
    ts = tu @ tv.T
    gamma = (ts.masked_fill(~tA, torch.inf).min(1).values - ts.masked_fill(tA, -torch.inf).max(1).values).min()
    def tproject(g, x):
        return g - (g * x).sum(1, keepdim=True) * x
    hg = torch.autograd.grad(gamma, [tu, tv], retain_graph=True)
    hg = [tproject(g, x) for g, x in zip(hg, [tu, tv])]
    diagnostic_error = 0.
    for label, value in [('D', ld), ('R', lr)]:
        gg = torch.autograd.grad(value, [tu, tv], retain_graph=True)
        gg = [tproject(g, x) for g, x in zip(gg, [tu, tv])]
        product = -sum((g * h).sum() for g, h in zip(gg, hg)).item()
        diagnostic_error = max(diagnostic_error, abs(product - diag['minus_g' + label + '_dot_h']))
    gate = (max(row['autograd_max_abs_error'] for row in validations) < 1e-10 and
            max(row['finite_difference_max_abs_error'] for row in validations) < 1e-7 and
            decomposition_error < 1e-10 and diagnostic_error < 1e-10)
    save('validation.json', {'passed': gate, 'objectives': validations,
                             'decomposition_error': decomposition_error,
                             'diagnostic_autograd_error': diagnostic_error,
                             'elapsed_seconds': time.perf_counter() - validation_start})
    assert gate, 'Implementation gradient validation failed; no experiments run.'
    print('Gradient validation passed', flush=True)

    start = time.perf_counter()
    zgrid = np.arange(30001, dtype=np.float64) * .001
    formulas, boundaries = [], []
    for r in [2, 8, 32]:
        for q in [.01, .02, .05, .1, .2, .4]:
            exact = np.logaddexp(np.log(np.cosh(q * zgrid)), np.log(r / 2) - zgrid)
            scores = np.column_stack([q * zgrid, -q * zgrid] + [-zgrid] * r)
            direct = logsumexp(scores, axis=1) - np.log(2.)
            lo, hi = 0., 30.
            for _ in range(80):
                mid = (lo + hi) / 2
                if q * np.sinh(q * mid) < r / 2 * np.exp(-mid):
                    lo = mid
                else:
                    hi = mid
            zstar = (lo + hi) / 2
            formulas.append({'r': r, 'q': q, 'optimal_z_bisection': zstar,
                             'optimal_z_grid': float(zgrid[np.argmin(exact)]),
                             'optimal_E': float(np.log(np.cosh(q * zstar) + r / 2 * np.exp(-zstar))),
                             'formula_max_abs_error': float(np.max(np.abs(exact - direct))),
                             'grid_min_E': float(exact.min())})
        for target in [.01, .05, .1]:
            xmax = np.arccosh(np.exp(target))
            def boundary(x):
                return x / np.log((r / 2) / (np.exp(target) - np.cosh(x)))
            sol = minimize_scalar(lambda x: -boundary(x), bounds=(1e-12, xmax * (1 - 1e-10)),
                                  method='bounded', options={'xatol': 1e-14})
            qc = boundary(sol.x)
            def optimal(q):
                z = brentq(lambda z: q * np.sinh(q * z) - r / 2 * np.exp(-z), 0., 100.)
                return float(np.log(np.cosh(q * z) + r / 2 * np.exp(-z)))
            q_other = brentq(lambda q: optimal(q) - target, 1e-8, 10.)
            tests = []
            for q in [qc - .006, qc + .006]:
                assert q > 0
                value = optimal(q)
                tests.append({'q': q, 'distance_from_qc': abs(q - qc),
                              'optimal_E': value, 'predicted_feasible': bool(q <= qc),
                              'observed_feasible': bool(value <= target)})
            boundaries.append({'r': r, 'target': target, 'qc_formula4': qc,
                               'maximizer_x': float(sol.x), 'qc_independent_root': q_other,
                               'boundary_q_error': abs(qc - q_other), 'tests': tests})
    q1 = next(x for x in formulas if x['r'] == 8 and x['q'] == .05)
    monotonic = all(all(x['optimal_E'] <= y['optimal_E'] for x, y in zip(
        sorted([a for a in formulas if a['q'] == q], key=lambda a: a['r']),
        sorted([a for a in formulas if a['q'] == q], key=lambda a: a['r'])[1:]))
        for q in [.01, .02, .05, .1, .2, .4])
    a_gate = (max(x['formula_max_abs_error'] for x in formulas) < 1e-10 and
              max(abs(x['optimal_z_bisection'] - x['optimal_z_grid']) for x in formulas) <= .001 and
              all(t['predicted_feasible'] == t['observed_feasible'] for x in boundaries for t in x['tests']))
    save('A.json', {'passed_preregistered_gate': a_gate, 'grid': {'start': 0, 'stop': 30, 'step': .001},
                    'formulas': formulas, 'boundaries': boundaries, 'Q1': q1,
                    'Q2_negative_count_monotonic': monotonic})
    finish_group('A', start)
    assert a_gate, 'A failed preregistered gate; stop extensions.'

    start = time.perf_counter()
    variants = [('clean', None, V0.copy())]
    for seed in [0, 1, 2]:
        rng = np.random.Generator(np.random.PCG64(seed))
        variants.append((f'perturb{seed}', seed, V0 + rng.normal(scale=.001, size=V0.shape)))
    frozen, slow = [], []
    for name, seed, v in variants:
        for loss in ['nce', 'sigmoid']:
            for h in ([.05, .025] if loss == 'nce' else [.1, .05]):
                steps = round((300 if loss == 'nce' else 4000) / h)
                theta = np.zeros(2 if loss == 'nce' else 3)
                x = np.column_stack([v, np.ones(7)])
                history = []
                for step in range(steps + 1):
                    if loss == 'nce':
                        value, grad, _, _ = core_nce(theta, v)
                    else:
                        signed = Y0 * (x @ theta)
                        value = float(np.logaddexp(0., -signed).mean())
                        grad = -(x.T @ (Y0 * expit(-signed))) / 7
                    if step % 20 == 0 or step == steps:
                        w = theta if loss == 'nce' else theta[:2]
                        row = {'step': step, 'time': step * h, 'loss': value,
                               'gradient_norm': float(np.linalg.norm(grad)), **core_metrics(w, v)}
                        if loss == 'sigmoid':
                            row['min_signed_logit'] = float(signed.min())
                        history.append(row)
                    if step != steps:
                        theta -= h * grad
                item = {'variant': name, 'seed': seed, 'loss': loss, 'h': h,
                        'final': history[-1], 'theta': theta.tolist(),
                        'V_initial': v.tolist(), 'distance_to_star': float(np.linalg.norm(theta[:2] - STAR))}
                filename = f'B_frozen_{name}_{loss}_{h}.json'
                save(filename, {**item, 'history': history})
                item['raw_file'] = filename
                frozen.append(item)
    print('B frozen finished', round(time.perf_counter() - start, 2), 's', flush=True)
    for name, seed, initial_v in variants:
        for init in ['zero', 'star']:
            for eps in [.04, .02, .01, .005, .0025]:
                for h in [.05, .025]:
                    w = np.zeros(2) if init == 'zero' else STAR.copy()
                    v = initial_v.copy()
                    hit, confirmed, history, min_eigen = None, False, [], float('inf')
                    previous_gap = core_metrics(w, v)['gap']
                    for step in range(round(4 / (eps * h)) + 1):
                        value, gw, gv, probs = core_nce(w, v)
                        gap = float((v @ w)[:3].min() - (v @ w)[3:].max())
                        if gap > 1e-4 and hit is None:
                            hit = step * h
                        if hit is not None and gap <= 1e-4:
                            hit = None
                        if hit is not None and eps * (step * h - hit) >= .02 - 1e-14:
                            confirmed = True
                        if step % 20 == 0 or confirmed:
                            centered = v - probs @ v
                            eig = float(np.linalg.eigvalsh((centered.T * probs) @ centered)[0])
                            min_eigen = min(min_eigen, eig)
                            history.append({'step': step, 'time': step * h, 'tau': eps * step * h,
                                            'loss': value, 'gradient_w_norm': float(np.linalg.norm(gw)),
                                            'gradient_V_norm': float(np.linalg.norm(gv)),
                                            'V_displacement': float(np.linalg.norm(v - initial_v)),
                                            'hessian_min_eigenvalue': eig, **core_metrics(w, v)})
                        if confirmed or step == round(4 / (eps * h)):
                            break
                        w -= h * gw
                        v -= eps * h * gv
                    item = {'variant': name, 'seed': seed, 'initial_query': init,
                            'epsilon': eps, 'h': h, 'hit_time': hit,
                            'epsilon_times_hit': eps * hit if hit is not None else None,
                            'sustained_pass': confirmed, 'confirmation_delta_tau': .02,
                            'final': history[-1], 'min_hessian_eigenvalue': min_eigen,
                            'final_w': w.tolist(), 'final_V': v.tolist(), 'V_initial': initial_v.tolist()}
                    filename = f'B_slow_{name}_{init}_{eps}_{h}.json'
                    save(filename, {**item, 'history': history})
                    item['raw_file'] = filename
                    slow.append(item)
        print('B variant', name, 'finished', round(time.perf_counter() - start, 2), 's', flush=True)
    frozen_gate = all((x['final']['gradient_norm'] < 1e-7 and
                       (x['distance_to_star'] < 1e-5 and abs(x['final']['gap'] + np.log(2)) < 1e-5
                        if x['variant'] == 'clean' else x['final']['gap'] < 0))
                      if x['loss'] == 'nce' else x['final']['min_signed_logit'] > 0 for x in frozen)
    discretization = []
    h1 = []
    for name, _, _ in variants:
        for init in ['zero', 'star']:
            fine = sorted([x for x in slow if x['variant'] == name and x['initial_query'] == init and x['h'] == .025],
                          key=lambda x: -x['epsilon'])
            for fine_run in fine:
                coarse = next(x for x in slow if x['variant'] == name and x['initial_query'] == init and
                              x['epsilon'] == fine_run['epsilon'] and x['h'] == .05)
                error = abs(coarse['epsilon_times_hit'] / fine_run['epsilon_times_hit'] - 1) if coarse['hit_time'] and fine_run['hit_time'] else None
                discretization.append({'variant': name, 'initial_query': init, 'epsilon': fine_run['epsilon'],
                                       'epsilon_T_relative_difference': error})
            tail = fine[1:]
            if all(x['hit_time'] for x in tail):
                slope = float(np.polyfit(np.log([x['epsilon'] for x in tail]), np.log([x['hit_time'] for x in tail]), 1)[0])
                products = [x['epsilon_times_hit'] for x in tail]
                spread = max(products) / min(products) - 1
                supported = (-1.1 <= slope <= -.9 and spread <= .1 and all(x['sustained_pass'] for x in tail))
            else:
                slope, spread, supported = None, None, False
            h1.append({'variant': name, 'initial_query': init, 'h': .025,
                       'smallest_four_slope': slope, 'epsilon_T_relative_spread': spread,
                       'supported_preregistered_H1': supported})
    b_gate = frozen_gate and all(x['epsilon_T_relative_difference'] is not None and x['epsilon_T_relative_difference'] < .01 for x in discretization)
    save('B.json', {'passed_preregistered_gate': b_gate, 'frozen_gate': frozen_gate,
                    'frozen': frozen, 'slow': slow, 'discretization': discretization, 'H1': h1})
    finish_group('B', start)
    assert b_gate, 'B failed preregistered gate; stop C/D extensions.'

    def config(id_, k, d, seed, loss, ratio=1., beta=16., bias='global'):
        return dict(id=id_, k=k, d=d, seed=seed, loss=loss, ratio=ratio, beta=beta, bias=bias)

    c_configs = [config(f'C_k{k}_d{d}_s{seed}_{loss}', k, d, seed, loss)
                 for k in [2, 3] for d in [2 * k, 4 * k] for seed in [0, 1, 2]
                 for loss in ['nce', 'sigmoid', 'pe']]
    d_configs = []
    for ratio in [.01, .1]:
        for loss in ['nce', 'sigmoid', 'pe']:
            d_configs.append(config(f'D1_r{ratio}_{loss}', 3, 6, 0, loss, ratio=ratio))
    for bias in ['zero', 'row']:
        d_configs.append(config(f'D2_b{bias}_sigmoid', 3, 6, 0, 'sigmoid', bias=bias))
    for beta in [4., 64.]:
        for loss in ['nce', 'sigmoid', 'pe']:
            d_configs.append(config(f'D3_beta{beta}_{loss}', 3, 6, 0, loss, beta=beta))
    for loss in ['nce', 'sigmoid', 'pe']:
        d_configs.append(config(f'D4_trainbeta_{loss}', 3, 6, 0, loss, beta='train'))
    save('plan.json', {'C': c_configs, 'D_new': d_configs,
                       'D_reused_C': [x['id'] for x in c_configs if x['k'] == 3 and x['d'] == 6 and x['seed'] == 0],
                       'E_selection_rule': 'min(sigmoid_success-NCE_success,PE_success-NCE_success), max group; plus top two C settings by median paired advantage; seed 0; deterministic ties'})
    all_runs = []
    def run_group(group, configs):
        start = time.perf_counter()
        # Measure 100 steps of the largest PE problem; only timing is used.
        probe_config = config('timing_only', 3, 12, 0, 'pe')
        probe = train(probe_config, steps=100, record=False)
        estimated = probe['elapsed_seconds'] / 100 * 1500 * len(configs) * 1.25
        steps = 1500
        if estimated > meta['budgets_seconds'][group]:
            steps = 750
            meta['budget_changes'].append({'group': group, 'reason': '100-step conservative timing estimate',
                                           'estimated_seconds': estimated, 'remaining_configs_steps': 750})
        save(f'{group}_timing.json', {'probe_seconds': probe['elapsed_seconds'], 'estimated_seconds': estimated,
                                     'steps_per_run': steps, 'configs': len(configs)})
        runs = []
        for index, cfg in enumerate(configs):
            if steps == 1500 and index and (time.perf_counter() - start) / index * (len(configs) - index) > meta['budgets_seconds'][group] - (time.perf_counter() - start):
                steps = 750
                meta['budget_changes'].append({'group': group, 'reason': 'elapsed-time projection',
                                               'after_config_count': index, 'remaining_configs_steps': 750})
            result = train(cfg, steps=steps)
            save(cfg['id'] + '.json', result)
            compact = {key: val for key, val in result.items() if key != 'history'}
            compact['raw_file'] = cfg['id'] + '.json'
            runs.append(compact)
            print(group, index + 1, '/', len(configs), cfg['id'], 'success', round(result['final']['success_fraction'], 4),
                  'gap', round(result['final']['min_gap'], 6), flush=True)
        save(group + '.json', {'runs': runs})
        finish_group(group, start)
        return runs
    c_runs = run_group('C', c_configs)
    d_runs = run_group('D', d_configs)

    # Preregistered E selection. Group only configurations that differ in loss.
    import json
    baseline = [x for x in c_runs if x['config']['k'] == 3 and x['config']['d'] == 6 and x['config']['seed'] == 0]
    all_runs = c_runs + d_runs
    def group_key(run):
        cfg = run['config']
        return (cfg['k'], cfg['d'], cfg['seed'], cfg['ratio'], str(cfg['beta']), cfg['bias'])
    groups = {}
    for run in all_runs:
        groups.setdefault(group_key(run), {})[run['config']['loss']] = run
    complete = [g for g in groups.values() if set(g) == {'nce', 'sigmoid', 'pe'}]
    def advantage(g):
        base = g['nce']['final']['success_fraction']
        return min(g['sigmoid']['final']['success_fraction'] - base, g['pe']['final']['success_fraction'] - base)
    complete.sort(key=lambda g: (-advantage(g), not g['nce']['config']['id'].startswith('C'), g['nce']['config']['id']))
    selected = [complete[0]]
    setting_scores = []
    for k in [2, 3]:
        for d in [2 * k, 4 * k]:
            gs = [g for g in complete if g['nce']['config']['id'].startswith('C') and
                  g['nce']['config']['k'] == k and g['nce']['config']['d'] == d]
            setting_scores.append((float(np.median([advantage(g) for g in gs])), k, d))
    setting_scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    for _, k, d in setting_scores[:2]:
        group = next(g for g in complete if g['nce']['config']['id'].startswith('C') and
                     g['nce']['config']['k'] == k and g['nce']['config']['d'] == d and g['nce']['config']['seed'] == 0)
        if group_key(group['nce']) not in [group_key(g['nce']) for g in selected]:
            selected.append(group)
    selection = [{'ids': [g[loss]['config']['id'] for loss in ['nce', 'sigmoid', 'pe']],
                  'paired_min_success_advantage': advantage(g)} for g in selected]
    save('E_selection.json', {'selected': selection, 'C_setting_scores': setting_scores})
    print('E selection', selection, flush=True)
    start = time.perf_counter()
    e_runs = []
    for group in selected:
        for loss in ['nce', 'sigmoid', 'pe']:
            base = group[loss]
            for method in ['adam', 'gd']:
                cfg = dict(base['config'])
                cfg['id'] = 'E_' + method + '_' + cfg['id']
                result = train(cfg, steps=2 * base['steps_requested'] if method == 'adam' else base['steps_requested'],
                               method=method, half_lr=(method == 'adam'))
                result['baseline_id'] = base['config']['id']
                save(cfg['id'] + '.json', result)
                compact = {key: val for key, val in result.items() if key != 'history'}
                compact['raw_file'] = cfg['id'] + '.json'
                e_runs.append(compact)
                print('E', cfg['id'], 'success', round(result['final']['success_fraction'], 4), 'gap', round(result['final']['min_gap'], 6), flush=True)
    save('E.json', {'selection': selection, 'runs': e_runs})
    finish_group('E', start)
    meta['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta['elapsed_seconds'] = time.perf_counter() - global_start
    meta['memory_final'] = memory()
    meta['complete'] = True
    save('metadata.json', meta)
    print('All A-E experiments complete', round(meta['elapsed_seconds'], 3), 's', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker()
    else:
        supervisor()
