"""Run only preregistered timing and numerical implementation checks."""
from core import *


def main():
    start = time.time()
    timing = []
    for n, k, d in SETTINGS:
        for loss in LOSSES:
            A, pos, neg, p = init(n, k, d, 10)
            _, g = obj(p, A, pos, neg, loss)
            rates, _ = calibrate(p, g)
            opt = Adam(p, rates)
            begin = time.perf_counter()
            for t in range(1000):
                _, g = obj(p, A, pos, neg, loss)
                opt.update(p, g, .5*(1+np.cos(np.pi*t/100000)))
            seconds = time.perf_counter()-begin
            timing.append(dict(n=n, k=k, d=d, loss=loss, steps=1000, seconds=seconds))
    estimate = 5*100*sum(x['seconds'] for x in timing)*1.5
    T = 100000 if estimate<=6000 else max(1000, 1000*int(100000*6000/estimate/1000))
    decision = dict(timing=timing, conservative_serial_estimate_100k_seconds=estimate,
                    steps=T, workers=2, reason='full budget feasible' if T==100000 else
                    'uniform proportional reduction by preregistered timing-only rule',
                    preparation_start_epoch=start, memory=memory())
    save(OUT/'budget_decision.json', decision)
    # Finite differences at a small independent synthetic configuration.
    A, pos, neg, p = init(7, 3, 4, 913)
    errors = {}
    for loss in LOSSES:
        _, g = obj(p, A, pos, neg, loss)
        worst = 0.
        for key, index in [('U', (3, 1)), ('V', (2, 0)), ('a', (0,))]:
            plus, minus = copy.deepcopy(p), copy.deepcopy(p)
            plus[key][index] += 1e-6
            minus[key][index] -= 1e-6
            fd = (obj(plus, A, pos, neg, loss)[0]-obj(minus, A, pos, neg, loss)[0])/2e-6
            worst = max(worst, abs(fd-g[key][index]))
        errors[loss] = worst
    D, gd = d_gradient(p, A, pos)
    # Independently evaluate R via positive / all partition functions.
    Z = np.exp(p['a'][0])*(p['U']@p['V'].T)
    R = float(np.mean(logsumexp(Z, axis=1)-logsumexp(np.where(A, Z, -np.inf), axis=1)))
    residual = abs(obj(p, A, pos, neg, 'nce')[0]-np.log(3)-D-R)
    dg_error = 0.
    for key, index in [('U', (3, 1)), ('V', (2, 0)), ('a', (0,))]:
        plus, minus = copy.deepcopy(p), copy.deepcopy(p)
        plus[key][index] += 1e-6
        minus[key][index] -= 1e-6
        fd = (d_gradient(plus, A, pos)[0]-d_gradient(minus, A, pos)[0])/2e-6
        dg_error = max(dg_error, abs(fd-gd[key][index]))
    _, g = obj(p, A, pos, neg, 'nce')
    rates, moves = calibrate(p, g)
    opt = Adam(p, rates)
    opt.update(p, g)
    original_p, original_opt = copy.deepcopy(p), copy.deepcopy(opt)
    cf = counterfactual(p, opt, A, pos, neg, .5)
    untouched = all(np.array_equal(p[key], original_p[key]) for key in p)
    untouched &= opt.t==original_opt.t and all(np.array_equal(opt.m[key], original_opt.m[key])
                                               and np.array_equal(opt.v[key], original_opt.v[key]) for key in p)
    # A cloned full-gradient branch exactly equals a genuine next NCE update.
    q, qo = copy.deepcopy(p), copy.deepcopy(opt)
    qo.update(q, obj(q, A, pos, neg, 'nce')[1], .5)
    branch_residual = abs(metrics(q, A)['min_gap']-cf['branches']['with_D']['metrics']['min_gap'])
    checks = dict(gradient_max_abs_errors=errors, D_gradient_max_abs_error=dg_error,
                  decomposition_error=residual, counterfactual_does_not_mutate=bool(untouched),
                  counterfactual_full_branch_residual=branch_residual,
                  calibration=moves, unit_norm_max_error=max(float(np.max(abs(np.linalg.norm(q[key], axis=1)-1)))
                                                            for key in ('U', 'V')), memory=memory())
    save(OUT/'validation.json', checks)
    assert max(errors.values())<1e-7 and dg_error<1e-7 and residual<1e-12
    assert untouched and branch_residual<1e-14 and checks['unit_norm_max_error']<1e-12
    with (ROOT/'PREREG_ROUND2.md').open('a') as f:
        f.write('\n## 计时后、主实验前的预算冻结\n\n')
        f.write(f'18段1000步总计 {sum(x["seconds"] for x in timing):.6f} 秒；'
                f'带1.5倍安全因子的单进程全扫描估计 {estimate:.3f} 秒。'
                f'按事先规则冻结全部90次为 **{T}步**，比例 {T/100000:.3f}。'
                '计时不含指标，未依据学习结果决策。完整计时见 experiments/round2/budget_decision.json。'
                '实现核验通过后启动主实验；主实验期间不再修改本预注册。\n')
    print(json.dumps(dict(steps=T, estimate_seconds=estimate, validation=checks), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
