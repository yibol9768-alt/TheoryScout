"""Offline design only. No HTTP, subprocess, clients, runners, or model calls.

Run under the resource-constrained systemd scope documented in HOLDOUT_D_PREREG.md.
Creates new HOLDOUT_D_ files exclusively; refuses overwriting any output.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import ast
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'experiments/real/analysis_length'
N, LIMIT, LENGTHS = 144, 9000, [8, 16, 32, 64]
SEEDS = list(range(116064001, 116064001 + N))
RNG = {'llr': 116990001, 'cost_D': 116990002, 'cost_A': 116990003,
       'calibration': 116990004, 'future_bootstrap': 116990005,
       'mechanism_bootstrap_power': 116990006}
M = {'E': dict(p=333/1463, r1=261/333, r2=38/72, h=34/1463),
     'T': dict(p=289/1339, r1=233/289, r2=42/56, h=14/1339)}
A = dict(p=283/1279, r1=222/283, r2=42/61, h=19/1279)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    assert name.startswith('HOLDOUT_D_') and '/' not in name
    with (ROOT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def survival(L, model='D'):
    return ((1-M['E']['h'])**min(L, 16) * (1-M['T']['h'])**max(0, L-16)
            if model == 'D' else (1-A['h'])**L)


def cells(model):
    return -np.diff([1.] + [survival(L, model) for L in LENGTHS] + [0.])


def binom_pmf(n, p):
    p = np.atleast_1d(p)
    k = np.arange(n+1)
    logcomb = np.array([math.lgamma(n+1)-math.lgamma(int(x)+1)-math.lgamma(n-int(x)+1) for x in k])
    # Bootstrap estimates here are strictly between zero and one.
    return np.exp(logcomb + k*np.log(p[:, None]) + (n-k)*np.log1p(-p[:, None]))


def interval(p):
    cdf = binom_pmf(N, p).mean(0).cumsum()
    return [int(np.searchsorted(cdf, q)) for q in (.005, .995)]


def certified_cutoff(n, w, pa):
    # U=0.001*sum ceil(w/0.001) >= true log LR on every outcome.
    # Exact finite lattice convolution up to float64 summation error.
    u = np.ceil(w/.001).astype(int)
    low = int(u.min())
    shift = u-low
    dist = np.ones(1)
    for _ in range(n):
        nxt = np.zeros(len(dist)+int(shift.max()))
        for j, p in zip(shift, pa):
            nxt[j:j+len(dist)] += p*dist
        dist = nxt
    tails = np.cumsum(dist[::-1])[::-1]-dist
    k = int(np.flatnonzero(tails <= .025-1e-12)[0])
    return dict(critical=round((k+n*low)*.001, 9), strict='T > critical',
                alpha=.025, upper_rounded_weights=u.tolist(), lattice_unit=.001,
                null_tail_upper=float(tails[k]), numerical_margin=1e-12,
                total_mass=float(dist.sum()),
                derivation='U>=T pathwise; finite convolution under fixed A; P_A(T>c)<=P_A(U>c)<=.025')


def audit_seeds():
    # Conservative search also reserves every standalone integer seen in old evidence.
    # Includes plan slots, stopped runs, mock runs, archived restart copies and week1.
    old_values, manifest, seed_fields = set(), {}, set()
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.is_symlink() or path.name.startswith('HOLDOUT_D_'):
            continue
        if path.suffix not in ('.json', '.jsonl', '.csv', '.md', '.py', '.yaml', '.yml', '.txt'):
            continue
        content = path.read_text(errors='replace')
        manifest[str(path.relative_to(ROOT))] = sha(path)
        old_values.update(int(x) for x in re.findall(r'(?<![A-Za-z0-9])\d{1,12}(?![A-Za-z0-9])', content))
        seed_fields.update(int(x) for x in re.findall(r'"seed"\s*:\s*(\d+)', content))
    # Explicitly cover the complete pre-existing schedules, even never executed slots.
    planned = {5100000+10000*L+r for L in (16, 64) for r in range(1,21)}
    planned |= {7100000+10000*L+r for L in (16, 64) for r in range(1,21)}
    planned |= {3000000+10000*f+1000*l+r for f in range(3) for l in range(4) for r in range(1,51)}
    planned |= {b+1000*f+100*l+r for b in (901001, 1901001) for f in range(3) for l in range(4) for r in range(4)}
    planned |= set(range(101280001, 101280401))  # old, never-run D draft reserved too
    planned |= {1105,2207,3301,3307,3313,3319,3323,3329}
    proposed = set(SEEDS) | set(RNG.values())
    assert len(proposed) == N+len(RNG)
    assert not proposed & (old_values | planned | seed_fields)
    # Extract only two pure functions from the frozen source; never import runner/client.
    source = ROOT/'experiments/real/reliability_core.py'
    tree = ast.parse(source.read_text())
    module = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef)
                             and x.name in ('task', 'transition')], type_ignores=[])
    ns = {'random': random}
    exec(compile(module, str(source), 'exec'), ns)
    count = 0
    for seed in SEEDS:
        row = dict(family='sheet', level=0, knob=2, seed=seed)
        long = ns['task'](dict(row, length=64))
        for L in LENGTHS:
            short = ns['task'](dict(row, length=L))
            assert short['initial'] == long['initial'] and short['table'] == long['table']
            assert short['operations'] == long['operations'][:L]
            assert short['prefix'] == long['prefix'][:L+1]
            # Independent arithmetic replay; no call to source transition.
            state = short['initial'].copy()
            for k, op in enumerate(short['operations']):
                for ins in op['instructions']:
                    a,b,c = (f'r{ins[key]}' for key in ('a','b','c'))
                    newa = (state[a]+ins['m']*state[b]-state[c]+ins['d']) % 997
                    state[a] = newa
                    state[c] = (state[c]+newa) % 997
                state['F0']=(state['r0']+2*state['r1']-state['r2'])%997
                state['F1']=(3*state['F0']+state['r3']-state['r4'])%997
                state['F2']=(state['F1']+2*state['r5'])%997
                assert state == short['prefix'][k+1]
            count += 1
    return dict(old_seed_field_unique=len(seed_fields), conservative_integer_unique=len(old_values),
                explicit_reserved_unique=len(planned), collisions=[], all_unique=True,
                checked_seed_range=[SEEDS[0], SEEDS[-1]], auxiliary_seeds=RNG,
                prefix_cases=count, independent_oracle_pass=True,
                scope='All local structured/text batch evidence; all explicit old schedule slots incl. unexecuted; reserved old D draft; logs excluded',
                source_hashes=manifest)


def calibration():
    tasks = list(csv.DictReader((BASE/'ANALYSIS_LENGTH_C_tasks.csv').open()))
    steps = list(csv.DictReader((BASE/'ANALYSIS_LENGTH_C_steps.csv').open()))
    rows, groups = [], defaultdict(list)
    for i,t in enumerate(tasks):
        z=[s for s in steps if s['batch']==t['batch'] and s['id']==t['id'] and int(s['resolved'])]
        row=[]
        for early in (True,False):
            q=[s for s in z if (int(s['step'])<16)==early]
            row.extend([len(q),sum(int(s['fatal']) for s in q)])
        rows.append(row)
        groups[(t['batch'],t['length'])].append(i)
    rows=np.array(rows)
    assert rows.sum(0).tolist() == [1463,34,1339,14]
    rng=np.random.default_rng(RNG['calibration'])
    v=np.zeros((10000,4))
    for ids in groups.values():
        v += rows[np.array(ids)[rng.integers(len(ids),size=(10000,len(ids)))]].sum(1)
    he,ht=v[:,1]/v[:,0],v[:,3]/v[:,2]
    return {str(L): dict(S_D=survival(L), S_A=survival(L,'A'),
                        count99_D=interval(survival(L)), count99_A=interval(survival(L,'A')),
                        calibration95_D=np.quantile((1-he)**min(L,16)*(1-ht)**max(0,L-16),[.025,.975]).tolist(),
                        mixture_count99_D=interval((1-he)**min(L,16)*(1-ht)**max(0,L-16)))
            for L in LENGTHS}


def expected_exposures():
    e=sum((1-M['E']['h'])**t for t in range(16))
    t=survival(16)*sum((1-M['T']['h'])**k for k in range(48))
    return np.array([e,t])


def simulate_tasks(rng, count, model):
    # Exact aggregate sampler for the stopped, four-outcome repair protocol.
    # Each segment: geometric fatal position; multinomial nonfatal outcomes conditional on survival.
    shape=(count,N)
    kernels=[M['E'],M['T']] if model=='D' else [A,A]
    reached=np.ones(shape,dtype=bool)
    out=[]
    completed=np.zeros(shape,dtype=int)
    for K,L in zip(kernels,(16,48)):
        fatal_at=rng.geometric(K['h'],size=shape)
        exposure=np.minimum(fatal_at,L)*reached
        death=(fatal_at<=L)&reached
        nonfatal=exposure-death
        err_survive=K['p']*(K['r1']+(1-K['r1'])*K['r2'])/(1-K['h'])
        errors=rng.binomial(nonfatal,err_survive)
        repaired2=rng.binomial(errors,(1-K['r1'])*K['r2']/(K['r1']+(1-K['r1'])*K['r2']))
        r2n=repaired2+death
        out.append(np.stack([exposure,death,errors+death,r2n,repaired2],axis=-1).astype(float))
        completed+=nonfatal
        reached &= ~death
    return np.stack(out,axis=2), completed


def normal_mechanism(t):
    # Delta-method cluster SE; planning approximation, not the frozen final bootstrap test.
    total=t.sum(1)
    hp=total[:,:,1]/total[:,:,0]
    rp=total[:,:,4]/np.maximum(total[:,:,3],1)
    hs=(t[:,:,:,1]-hp[:,None,:]*t[:,:,:,0])/total[:,None,:,0]
    rs=(t[:,:,:,4]-rp[:,None,:]*t[:,:,:,3])/np.maximum(total[:,None,:,3],1)
    hse=np.sqrt(N/(N-1)*((hs[:,:,0]-hs[:,:,1])**2).sum(1))
    rse=np.sqrt(N/(N-1)*((rs[:,:,1]-rs[:,:,0])**2).sum(1))
    eligible=(total[:,:,3]>=30).all(1)&(total[:,:,1]>=10).all(1)
    hpass=(hp[:,0]-hp[:,1]>1.959963984540054*hse)
    rpass=(rp[:,1]-rp[:,0]>1.959963984540054*rse)
    return eligible,hpass,rpass


def bootstrap_power_check(t):
    # 300 synthetic experiments, each evaluated by the actual 10,000 whole-task bootstrap.
    rng=np.random.default_rng(RNG['mechanism_bootstrap_power'])
    passed=np.zeros(4,dtype=int)
    for row in t:
        total=row.sum(0)
        eligible=bool((total[:,3]>=30).all() and (total[:,1]>=10).all())
        v=np.zeros((10000,2,5))
        for start in range(0,10000,250):
            ids=rng.integers(N,size=(250,N))
            v[start:start+250]=row[ids].sum(1)
        valid=(v[:,:,0]>0).all(1)&(v[:,:,3]>0).all(1)
        if valid.sum()<9900:
            continue
        h=v[valid,:,1]/v[valid,:,0]
        r=v[valid,:,4]/v[valid,:,3]
        hpass=np.quantile(h[:,0]-h[:,1],.025,method='linear')>0
        rpass=np.quantile(r[:,1]-r[:,0],.025,method='linear')>0
        passed+=np.array([eligible,eligible and hpass,eligible and rpass,eligible and hpass and rpass])
    return dict(experiments=len(t), bootstraps_each=10000, seed=RNG['mechanism_bootstrap_power'],
                eligible=float(passed[0]/len(t)), fatal_difference=float(passed[1]/len(t)),
                r2_difference=float(passed[2]/len(t)), joint=float(passed[3]/len(t)),
                note='Unconditional ideal fixed-D power; MC uncertainty is substantial at only 300 outer replicates')


def protocol_mc(model,w,cut,pred):
    rng=np.random.default_rng(RNG['cost_'+model])
    costs=[];totals=[];tests=[];examples=[]
    for start in range(0,100000,1000):
        t,completed=simulate_tasks(rng,1000,model)
        cost=t[:,:,:,0].sum((1,2))+t[:,:,:,2].sum((1,2))+t[:,:,:,3].sum((1,2))
        # fatal step k has k correctly completed steps; success=64.
        cell=np.searchsorted(LENGTHS,completed,side='right')
        ll=w[cell].sum(1)
        successes=np.stack([(completed>=L).sum(1) for L in LENGTHS],axis=1)
        cal=np.ones(1000,dtype=bool)
        for j,L in enumerate(LENGTHS):
            lower,upper=pred[str(L)]['count99_D']
            cal&=(successes[:,j]>=lower)&(successes[:,j]<=upper)&(np.abs(successes[:,j]/N-survival(L))<=.10)
        eligible,hpass,rpass=normal_mechanism(t)
        complete=cost<=LIMIT
        tests.append(np.column_stack([complete,ll>cut,ll>math.log(40),cal,
                                     complete&(ll>cut),complete&cal&(ll>cut),eligible,
                                     eligible&hpass,eligible&rpass,eligible&hpass&rpass,
                                     complete&cal&(ll>cut)&eligible&hpass&rpass]))
        costs.extend(cost.tolist());totals.append(t.sum(1))
        if model=='D' and start==0:
            examples=t[:300].copy()
        if start%25000==0:
            print('protocol',model,start,flush=True)
    labels=['budget_complete','llr','llr40','calibration','budget_and_llr',
            'budget_calibration_llr','mechanism_eligible','fatal_cluster_normal',
            'r2_cluster_normal','joint_mechanism_cluster_normal','all_gates_normal_approx']
    rates=np.concatenate(tests).mean(0)
    result=dict(simulations=100000, seed=RNG['cost_'+model],
                cost_mean=float(np.mean(costs)),cost_sd=float(np.std(costs,ddof=1)),
                cost_quantiles={str(q):float(np.quantile(costs,q)) for q in (.5,.95,.99,.999)},
                cap_exceed_count=int(np.count_nonzero(np.array(costs)>LIMIT)),
                rates=dict(zip(labels,map(float,rates))),
                mean_segment_counts=np.concatenate(totals).mean(0).tolist(),
                segment_columns=['exposures','fatal','natural_errors','r2_attempts','r2_successes'])
    if model=='D':
        result['actual_bootstrap_power']=bootstrap_power_check(examples)
    return result


def requirements():
    ex=expected_exposures()
    m2=ex*np.array([M[k]['p']*(1-M[k]['r1']) for k in ('E','T')])
    result={}
    for target in (.8,.9):
        z=NormalDist().inv_cdf(.975)+NormalDist().inv_cdf(target)
        output={}
        for name,prob,per_task in [('fatal',[M['E']['h'],M['T']['h']],ex),
                                   ('r2',[M['E']['r2'],M['T']['r2']],m2)]:
            diff=abs(prob[1]-prob[0])
            ratio=per_task[1]/per_task[0]
            early=z*z*(prob[0]*(1-prob[0])+prob[1]*(1-prob[1])/ratio)/(diff*diff)
            output[name]=dict(early_required=math.ceil(early),tail_required=math.ceil(early*ratio),
                              tasks_required=math.ceil(early/per_task[0]))
        result[str(target)]=output
    return dict(expected_exposure_per_task=ex.tolist(), expected_r2_per_task=m2.tolist(),
                expected_exposures_N=(N*ex).tolist(),expected_r2_N=(N*m2).tolist(),
                expected_fatal_N=(N*ex*np.array([M['E']['h'],M['T']['h']])).tolist(),
                minimum_eligibility=dict(r2_attempts_each_segment=30,fatal_each_segment=10,
                                         bootstrap_valid=9900,bootstrap_total=10000),
                normal_planning_requirements=result,
                note='Normal approximation at expected exposure ratio, not a guaranteed sufficient sample size. Final rule is whole-task bootstrap; failed denominator eligibility means 信息不足.')


def main():
    for name in ('HOLDOUT_D_DESIGN.json','HOLDOUT_D_OFFLINE_RESULTS.json','HOLDOUT_D_SEED_AUDIT.json'):
        assert not (ROOT/name).exists(), 'No overwrite: '+name
    audit=audit_seeds()
    pa,pd=cells('A'),cells('D');w=np.log(pd/pa)
    cutoff=certified_cutoff(N,w,pa)
    pred=calibration()
    sensitivity={}
    rng=np.random.default_rng(RNG['llr'])
    for n in (46,128,140,144,150,160):
        c=certified_cutoff(n,w,pa)
        ta=rng.multinomial(n,pa,size=100000)@w
        td=rng.multinomial(n,pd,size=100000)@w
        sensitivity[str(n)]=dict(cutoff=c,type1_MC=float(np.mean(ta>c['critical'])),
                                 power_MC=float(np.mean(td>c['critical'])),
                                 power_log40_MC=float(np.mean(td>math.log(40))))
    print('predictions and fixed multinomial tests ready',flush=True)
    simD=protocol_mc('D',w,cutoff['critical'],pred)
    simA=protocol_mc('A',w,cutoff['critical'],pred)
    ex=expected_exposures()
    expected_D=N*sum(ex[j]*(1+M[k]['p']+M[k]['p']*(1-M[k]['r1'])) for j,k in enumerate(('E','T')))
    expected_A=N*sum((1-A['h'])**t for t in range(64))*(1+A['p']+A['p']*(1-A['r1']))
    req=requirements()
    design=dict(status='FROZEN_DESIGN_ONLY_NO_LLM_NO_BATCH',N=N,max_length=64,
                nested_prefix_endpoints=LENGTHS, seeds=SEEDS,auxiliary_seeds=RNG,
                old_draft_replaced='400 x 128 replaced; original files remain untouched',
                models=dict(M_D=M,A=A,fit_note='M_D exploratory CALIB1+B+C; A CALIB1 only; no refit using D'),
                predictions=pred,conditional_S64_given16=survival(64)/survival(16),
                conditional_A_S64_given16=survival(64,'A')/survival(16,'A'),
                comparison=dict(cell_labels=['failure completed<8','failure completed8..15',
                                              'failure completed16..31','failure completed32..63','success64'],
                                cells_A=pa.tolist(),cells_D=pd.tolist(),weights_log_D_over_A=w.tolist(),
                                statistic='sum(cell_counts * weights)',cutoff=cutoff,
                                secondary_log40_threshold=math.log(40),
                                primary='single fixed N=144 five-cell simple-vs-simple test, no optional stopping'),
                budget=dict(successful_return_hard_limit=LIMIT,http_attempt_hard_limit=12000,concurrency=2,
                            successful_includes='all HTTP successful model returns incl schema/content errors and guard-failing envelopes; no repeats omitted',
                            expected_D=expected_D,expected_A=expected_A,
                            expected_active_hours_D=expected_D/600,expected_active_hours_A=expected_A/600,
                            max_active_hours_at_observed_rate=15,
                            worst_case_uncapped=3*N*64,guaranteed_completion_worst_case_N=LIMIT//(3*64),
                            cap_protocol='atomic successful + in_flight/reserved + unresolved_reservations <=9000; write-ahead durable reservations; no budget refill or seed replacement',
                            completion_not_guaranteed=True,
                            incomplete_rule='information insufficient; no confirmation; report observed prefix results, bounds and censored likelihood only'),
                mechanism=req, calibration=dict(bootstrap=10000,seed=RNG['calibration'],
                    stratification='batch and planned length, whole tasks',
                    primary='fixed central99 count intervals at all four prefixes plus absolute error<=.10',
                    simultaneous_fixed_coverage_lower=.96,
                    auxiliary='calibration95 and mixture99 never replace primary thresholds'),
                future_mechanism=dict(bootstrap=10000,seed=RNG['future_bootstrap'],
                    quantile_method='numpy.quantile method=linear .025/.975',
                    paired_resampling='same sampled seed indices for both segments',
                    fatal_difference='h_E-h_T CI95 lower>0',r2_difference='r2_T-r2_E CI95 lower>0',
                    eligible='each segment >=30 second reviews AND >=10 fatal outcomes; >=9900 valid paired replicates',
                    decision='both intervals lower>0 supports position association only; eligibility failure=信息不足; eligible but CI crosses0=未达到机制门槛'),
                engineering=dict(model='gpt-6-luna',endpoint='http://127.0.0.1:18319/v1/chat/completions',
                    max_completion_tokens=4096,timeout_seconds=90,prompt_token_limit=16000,
                    service_retry_http=[408,500,502,503,504],all_5xx_retry=True,
                    connection_and_timeout='global pause, preserve original body/state/step/review; wait>=300s then actual quota check; retry same request',
                    quota=dict(five_hour_pause_ge=.80,week_pause_ge=.90,poll_seconds=300,
                               probe_failure='pause; no stale quota fallback',
                               provider='must use separately authorized quota probe within this project or permission for old external command; never read other projects under present authorization'),
                    other_http_or_envelope_guard='halt safely, retain missing; never assign model failure',
                    schema_error='model action error, consumes successful budget, no transport retry',
                    launch='setsid nohup systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0; stdin=/dev/null; threads<=2; durable output; verify SID/cgroup/flock',
                    recovery='raw immutable request/response/action WAL is authoritative; deterministic oracle+actual-action replay, stable request id, body hash, same step/review; completed tasks untouched; ambiguous in-flight remains reserved and paused until reconciled',
                    smoke='first two seeds included in N; future actual execution requires explicit user authorization and human smoke release; no smoke now',
                    runner_delivered=False,execution_ready=False),
                decision_order=['engineering/integrity and complete144','four-prefix calibration','primary five-cell LLR','both eligible mechanism CIs'],
                conclusion='Only all gates pass: one independent batch supports position-associated piecewise repair risk in this feedback protocol. Separate component results otherwise; no causal, natural detection, or permanent plateau claims.',
                seed_audit='HOLDOUT_D_SEED_AUDIT.json', offline_results='HOLDOUT_D_OFFLINE_RESULTS.json')
    result=dict(sensitivity=sensitivity,protocol_D=simD,protocol_A=simA,
                mechanisms=req,MC_se_max_100000=math.sqrt(.25/100000),
                mechanism_bootstrap_MC_se_max_300=math.sqrt(.25/300),
                limitations=['fixed-parameter IID simulations; no transport/quota delay',
                             'model/parameter uncertainty is not integrated into ideal power',
                             'mechanism normal approximation distinct from exact frozen bootstrap',
                             'quota pause increases wall time; 15h is active-return estimate only'],
                no_network=True,no_llm=True,no_formal_or_smoke_batch=True)
    assert all(abs(x['h']-x['p']*(1-x['r1'])*(1-x['r2']))<1e-15 for x in [A,*M.values()])
    assert abs(sum(pa)-1)<1e-14 and abs(sum(pd)-1)<1e-14
    assert simD['rates']['llr']>.94 and simA['rates']['llr']<.027
    # Every pre-existing input audited above must remain byte-identical.
    assert all(sha(ROOT/p)==digest for p,digest in audit['source_hashes'].items())
    save('HOLDOUT_D_SEED_AUDIT.json',audit)
    save('HOLDOUT_D_OFFLINE_RESULTS.json',result)
    save('HOLDOUT_D_DESIGN.json',design)
    print(json.dumps(dict(N=N,cutoff=cutoff,expected_D=expected_D,expected_A=expected_A,
                         D=simD,A=simA,mechanism=req),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
