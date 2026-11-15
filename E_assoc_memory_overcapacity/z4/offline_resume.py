"""只重核持久化规划的种子/输入/功效，补充门控功效；不重做抽样或规划。"""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(80,90))
import math
import json
import subprocess
import numpy as np
from scipy.stats import norm, t, nct
from common_v2 import ROOT,OUT,read,dump,sha,entropy,config,environment,usage,PAIRS,METHODS
from model import analytic_alpha,predict

def main():
    start=usage();cfg=config();errors=[]
    baseline=read(OUT/'resume_baseline.json')
    changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    if changed: errors.append(dict(existing_changed=changed))
    inputs=read(OUT/'input_hashes.json')
    changed_inputs=[p for p,h in inputs.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    if changed_inputs: errors.append(dict(inputs_changed=changed_inputs))
    if environment()!=read(OUT/'environment.json'): errors.append('environment changed')
    streams=read(OUT/'streams.json');old=read(OUT/'old_entropies.json')
    expected=[entropy(0,d,m,s,k) for d in cfg['dims'] for m in range(4) for s in range(cfg['n_cal']) for k in range(4)]
    expected += [entropy(1,d,-1,s,k) for d in cfg['dims'] for s in range(cfg['n_eval']) for k in range(4)]
    if streams!=expected: errors.append('stream plan mismatch')
    def state(e):return tuple(map(int,np.random.SeedSequence(e).generate_state(8)))
    new={state(e) for e in streams}
    collision=sum(state(e) in new for e in old)
    if collision or len(new)!=len(streams):errors.append('state collision')
    # Detect additions to the old RNG-source inventory after the interrupted audit.
    paths=subprocess.check_output(['rg','--files','--hidden','-g','*.py','-g','!z4/**','-g','!.git/**'],cwd=ROOT,text=True).splitlines()
    tokens=['SeedSequence','default_rng','RandomState','random.seed','manual_seed','Sobol(']
    known={x['file'] for x in read(OUT/'rng_inventory.json')}
    new_sources=[]
    for rel in paths:
        p=ROOT/rel
        if p.is_symlink():continue
        if any(x in p.read_text(errors='replace') for x in tokens) and rel not in known:
            new_sources.append(rel)
    if new_sources: errors.append(dict(new_RNG_sources_require_review=new_sources))
    if (OUT/'formal').exists(): errors.append('formal directory exists')
    if not read(OUT/'resume_tests.json')['passed']:errors.append('resume tests failed')
    rows=[];ne=cfg['n_eval'];nc=cfg['n_cal'];df=ne-1
    c_new=float(t.ppf(1-.025/(2*42),df));c_old=float(t.ppf(1-.025/(2*18),df))
    def gate(sd_eval,sd_cal,bias=0.):
        half=c_new*sd_eval/math.sqrt(ne)
        room=cfg['match_band']-half
        sd=math.sqrt(sd_eval**2/ne+sd_cal**2/nc)
        return max(0.,float(norm.cdf((room-bias)/sd)-norm.cdf((-room-bias)/sd))) if room>0 else 0.
    pa=gate(.025,.03);pp=gate(.03,math.sqrt(2)*.03)
    for d in cfg['dims']:
        for target in cfg['targets']:
            for a,b in PAIRS:
                aa=analytic_alpha(a,d,target);ab=analytic_alpha(b,d,target)
                p1=predict(a,aa,d,'old');p2=predict(b,ab,d,'old')
                effect=p1['finite']-p2['finite'];shift=(p1['B']-p2['B'])/d
                direction=float(nct.sf(c_old,df,effect*math.sqrt(ne)/.03))
                # Separate the change from re-matching alphas from B/d at fixed alphas.
                la=analytic_alpha(a,d,target,False);lb=analytic_alpha(b,d,target,False)
                pop=predict(a,la,d,'old')['limit']-predict(b,lb,d,'old')['limit']
                low=[];high=[]
                for m in [a,b]:
                    for t0,arr in [(target-cfg['match_band'],low),(target+cfg['match_band'],high)]:
                        alpha=analytic_alpha(m,d,t0)
                        arr.append(predict(m,alpha,d,'old')['finite'])
                # Independent rectangular mismatch perturbations are conservative
                # relative to also requiring pairwise mismatch < .01. Model only.
                perturb=min(low[0]-high[1],high[0]-low[1])
                rows.append(dict(d=d,target=target,pair=a+'-'+b,effect=effect,
                    population_matched_limit_gap=pop,finite_matched_total_shift=effect-pop,
                    fixed_alpha_B_shift=shift,conditional_direction_power=direction,
                    matched_direction_power_union_lower=max(0.,2*pa+pp+direction-3),
                    model_min_gap_under_target_errors_plusminus_01=perturb))
    supplemental=dict(no_training_or_queries=True,unchanged_design=True,
        assumptions=dict(dataset_sd_new=.025,dataset_sd_old_pair=.030,dataset_sd_cal=.030,
            calibration_methods_independent=True,calibration_interpolation_bias_not_bounded=True,
            probability_method='normal planning error with fixed t cutoff; approximate, not a certified lower bound'),
        threshold=dict(min_conditional_direction_power=.8,matching_band=.01,finite_agreement_band=.015,
            alpha_matching=.025,alpha_old=.025,match_family=42,old_family=18,
            all_cells_unconditional_power_80_claim=False,coefficient_confirmation=False),
        zero_bias_absolute_gate_power=pa,zero_bias_pairwise_gate_power=pp,
        all42_gate_union_bound=max(0.,1-24*(1-pa)-18*(1-pp)),
        bias_sensitivity=[dict(absolute_bias=b,absolute_gate=gate(.025,.03,b),
            pairwise_gate=gate(.03,math.sqrt(2)*.03,b)) for b in [0.,.0025,.005,.01]],rows=rows,
        cost_from_preserved_plan=read(OUT/'power.json')['projected_cpu'],
        forecast_wall_from_preserved_plan=read(OUT/'power.json')['projected_wall'],
        coefficient_reason='two dimensions and small B shifts cannot establish an asymptotic coefficient; scaled residuals descriptive')
    if min(r['conditional_direction_power'] for r in rows)<.8:errors.append('conditional direction power below .8')
    dump(OUT/'power_resume.json',supplemental)
    dump(OUT/'resume_audit.json',dict(passed=not errors,errors=errors,baseline_files=len(baseline),
        checked_input_files=len(inputs),planned_streams=len(streams),old_unique=len(old),collisions=collision,
        new_rng_sources=new_sources,source_inventory_checked=len(paths),training_or_query_draws=0,
        caveat='records/static/conservative domains only; unrecorded historical streams cannot be audited',
        scientific_config_sha256=sha(OUT/'config.json'),formal_never_started=not (OUT/'formal').exists()))
    dump(OUT/'resume_usage.json',dict(start=start,end=usage(),formal_sampling=False,
        subprocesses='rg file listing only; no scientific computation subprocess'))
    print(json.dumps(dict(passed=not errors,errors=errors,absolute_gate=pa,pair_gate=pp,usage=usage())))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
