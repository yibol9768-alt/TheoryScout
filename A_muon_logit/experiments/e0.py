"""Preregistered scalar Monte Carlo tests; streaming across time."""
import math
import time
import numpy as np
from common import OUT, save, ci, binomial_ci


def slopes(x,y):
    return float(np.polyfit(np.log(x),np.log(y),1)[0])


def run_e0(budget):
    started = time.monotonic()
    rng = np.random.default_rng(20261105)
    repeats,dh = 2048,16
    grid,phase = [],[]
    for eta in [.01,.02,.04]:
        for lam in [.1,.2,.4]:
            budget.check(new_run=True)
            c = 1-eta*lam
            vstar = eta/(lam*(2-eta*lam))
            mstar = math.sqrt(eta*lam/2)
            biases = np.array([0,mstar,3*mstar,1.])
            steps = math.ceil(10/(eta*lam))
            a = np.zeros((4,repeats))
            tail_a,tail_a2 = np.zeros_like(a),np.zeros_like(a)
            coherent,noise,expected = np.zeros(4),np.zeros(4),np.zeros(4)
            count = 0
            trace = []
            for t in range(1,steps+1):
                a = c*a+eta*(2*(rng.random(a.shape)<(1+biases[:,None])/2)-1)
                mean,var = a.mean(axis=1),a.var(axis=1,ddof=1)
                second = np.mean(a*a,axis=1)
                trace.append([t,mean.tolist(),second.tolist()])
                if t>steps//2:
                    tail_a += a
                    tail_a2 += a*a
                    coherent += mean*mean-var/repeats
                    noise += var
                    mu = biases*(1-c**t)/lam
                    expected += (mu*mu+(1-biases*biases)*vstar*(1-c**(2*t)))/4
                    count += 1
                if t%1000 == 0:
                    budget.check()
            coherent,noise,expected = coherent/count,noise/count,expected/count
            q = noise[1]/(coherent[1]/mstar**2)
            estimated_boundary = math.sqrt(q/(1+q))
            rows = []
            for i,m in enumerate(biases):
                row = {'m':m,'logit_tail':ci(tail_a2[i]/(count*4)),
                       'analytic_finite_tail':expected[i],
                       'analytic_stationary':(m*m/lam**2+(1-m*m)*vstar)/4,
                       'coherent_empirical_a2':coherent[i], 'noise_empirical_a2':noise[i],
                       'per_repeat_tail_mean_a':tail_a[i]/count,
                       'per_repeat_tail_mean_a2':tail_a2[i]/count}
                row['relative_error'] = abs(row['logit_tail']['mean']/expected[i]-1)
                rows.append(row)
            grid.append({'eta':eta,'lambda':lam,'c':c,'steps':steps,
                         'mstar':mstar,'estimated_boundary':estimated_boundary,
                         'boundary_relative_error':abs(estimated_boundary/mstar-1),
                         'rows':rows,'trace_columns':['step','ensemble_mean_a_by_m','ensemble_second_a_by_m'],
                         'trace':trace})

            # Same innovations in linear and genuinely projected recursions.
            T,N = math.ceil(5/(eta*lam)),math.ceil(10/(eta*lam))
            R = 2.5
            a = np.zeros(repeats)
            p = a.copy()
            records = []
            ever_coherent,ever_noise = np.zeros(repeats,bool),np.zeros(repeats,bool)
            for t in range(1,T+1):
                a = c*a+eta
                z = c*p+eta
                events = np.abs(z)>R
                ever_coherent |= events
                p = np.clip(z,-R,R)
                records.append([t,float(a.mean()),float(np.mean(a*a)),float(p.mean()),
                                float(np.mean(p*p)),float(events.mean()),float(np.mean(np.abs(a)>R))])
            AT = float(a[0])
            for n in range(1,N+1):
                xi = rng.integers(0,2,repeats)*2-1
                a = c*a+eta*xi
                z = c*p+eta*xi
                events = np.abs(z)>R
                ever_noise |= events
                p = np.clip(z,-R,R)
                records.append([T+n,float(a.mean()),float(np.mean(a*a)),float(p.mean()),
                                float(np.mean(p*p)),float(events.mean()),float(np.mean(np.abs(a)>R))])
                if n%1000 == 0:
                    budget.check()
            records = np.asarray(records)
            ns = np.arange(1,N+1)
            excess_theory = c**(2*ns)*(AT**2-vstar)/4
            mask = excess_theory>=10*vstar/4
            observed_excess = records[T:,2]/4-vstar/4
            fit_mask = mask & (observed_excess>0)
            fit = float(np.polyfit(ns[fit_mask],np.log(observed_excess[fit_mask]),1)[0])
            phase.append({'eta':eta,'lambda':lam,'T':T,'N':N,'R':R,
                          'peak_logit':AT**2/4,'vstar':vstar,
                          'expected_slope':2*math.log(c),'observed_slope':fit,
                          'slope_relative_error':abs(fit/(2*math.log(c))-1),
                          'fit_n_range':[int(ns[fit_mask][0]),int(ns[fit_mask][-1])],
                          'fit_n_count':int(fit_mask.sum()),
                          'excluded_nonpositive_in_fixed_range':int((mask & ~fit_mask).sum()),
                          'clip_any_coherent':ci(ever_coherent.astype(float)),
                          'clip_any_post_switch':ci(ever_noise.astype(float)),
                          'trace_columns':['step','mean_a','second_a','mean_projected_a',
                                           'second_projected_a','trigger_probability','linear_threshold_exceedance'],
                          'trace':records})
            print(f'E0 grid eta={eta} lambda={lam} finished',flush=True)

    # No decay: the exact time-exponent test.
    a = np.zeros((2,repeats))
    p1trace = []
    for t in range(1,101):
        a[0] += .02*(rng.integers(0,2,repeats)*2-1)
        a[1] += .02
        p1trace.append([t,*np.mean(a*a,axis=1)/4])
    p1trace = np.asarray(p1trace)
    p1 = {'eta':.02,'lambda':0,'repeats':repeats,'trace':p1trace,
          'slope_m0':slopes(p1trace[9:,0],p1trace[9:,1]),
          'slope_m1':slopes(p1trace[9:,0],p1trace[9:,2])}

    eta,lam,R,N,burn,delta = .02,.2,2.5,2500,142,.05
    c = 1-eta*lam
    vstar = eta/(lam*(2-eta*lam))
    rn = math.sqrt(2*vstar*math.log(2*N/delta))
    assert c**burn*R+rn<R
    p = np.full(repeats,R)
    late,early = np.zeros(repeats,bool),np.zeros(repeats,bool)
    cliptrace = []
    for n in range(1,N+1):
        z = c*p+eta*(rng.integers(0,2,repeats)*2-1)
        events = np.abs(z)>R
        if n<burn:
            early |= events
        else:
            late |= events
        p = np.clip(z,-R,R)
        cliptrace.append([n,float(events.mean()),float(p.mean()),float(np.mean(p*p))])
    k = int(late.sum())
    p5 = {'eta':eta,'lambda':lam,'R':R,'N':N,'n0':burn,'delta':delta,
          'r_N':rn,'envelope_at_n0':c**burn*R+rn,'repeats':repeats,
          'early_any_count':int(early.sum()),'late_any_count':k,
          'late_any_fraction':k/repeats,'late_any_CI95':binomial_ci(k,repeats),
          'late_event_per_repeat':late,'trace':cliptrace}
    p = np.zeros(128)
    counts = np.zeros(128,int)
    recurrence_trace = []
    for n in range(1,20001):
        z = .9*p+.2*(rng.integers(0,2,128)*2-1)
        counts += np.abs(z)>1
        p = np.clip(z,-1,1)
        if n%100 == 0:
            recurrence_trace.append([n,int(counts.min()),float(np.median(counts)),int(counts.max())])
    recurrence = {'eta':.2,'lambda':.5,'R':1,'steps':20000,'repeats':128,
                  'forcing_K':11,'counts_per_repeat':counts,'trace':recurrence_trace}

    peak_slopes = []
    floor_lambda_slopes = []
    floor_eta_slopes = []
    for eta in [.01,.02,.04]:
        peak_slopes.append(slopes([p['lambda'] for p in phase if p['eta']==eta],
                                  [p['peak_logit'] for p in phase if p['eta']==eta]))
        floor_lambda_slopes.append(slopes([g['lambda'] for g in grid if g['eta']==eta],
                                          [g['rows'][0]['logit_tail']['mean'] for g in grid if g['eta']==eta]))
    for lam in [.1,.2,.4]:
        floor_eta_slopes.append(slopes([g['eta'] for g in grid if g['lambda']==lam],
                                       [g['rows'][0]['logit_tail']['mean'] for g in grid if g['lambda']==lam]))
    result = {'seed':20261105,'dh':16,'independent_repeats':repeats,
              'grid':grid,'two_stage':phase,'P1':p1,'P5':p5,'recurrence':recurrence,
              'P3':{'peak_lambda_slopes':peak_slopes,'floor_lambda_slopes':floor_lambda_slopes,
                    'floor_eta_slopes':floor_eta_slopes},
              'elapsed_seconds':time.monotonic()-started}
    save(OUT/'e0.json',result)
    return result
