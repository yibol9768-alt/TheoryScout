"""Post hoc read-only report audit: no randomness, fitting, or gate changes."""
import csv
import json
import numpy as np
from common import OUT,begin,finish,prior_elapsed

def main():
    state=begin('tail_audit')
    summary=json.loads((OUT/'summary.json').read_text())
    rr=[]
    for line in (OUT/'moments.jsonl').read_text().splitlines():
        r=json.loads(line)
        if (r['family'],r['d'],r['load'],r['alpha'],r['method'])==('bounded',64,4,1.5,'delta_1.2'):
            rr.append({k:r[k] for k in ('seed','Q_bins','Q_cond_bins','m_bins')})
    assert len(rr)==16
    ordering=sorted(rr,key=lambda r:r['Q_bins'][1],reverse=True)
    momentstrata=[r for r in summary['moment_strata'] if r['family']=='bounded' and r['method'] in ('delta_0.8','delta_1.2')]
    oracle=list(csv.DictReader((OUT/'oracle_summary.csv').open()))
    cells=list(csv.DictReader((OUT/'cell_summary.csv').open()))
    age4=list(csv.DictReader((OUT/'moment_summary.csv').open()))
    group=[r for r in age4 if (r['family'],r['d'],r['load'],r['alpha'],r['method'])==('bounded','64','2','1.5','delta_0.4')]
    byseed=np.array([r['Q_bins'][1] for r in rr])
    out={'post_hoc':True,'no_new_random_data':True,
        'worst_Q_all_seeds_sorted':ordering,
        'worst_Q_largest_seed_share':float(byseed.max()/byseed.sum()),
        'worst_Q_second_largest':float(np.sort(byseed)[-2]),
        'Q_failure_strata':momentstrata,
        'oracle_tau16':[r for r in oracle if r['readout']=='tau16'],
        'V2_worst_point_moment_max':{f'{m}_{k}':max(float(r[f'{m}_{k}_relative_error']) for r in group)
           for m in ('m','Q') for k in ('conditional','unconditional')},
        'all_primary_coverage_failures':[r for r in cells if r['family']=='bounded' and float(r['alpha'])>=1 and int(r['d'])>=128 and r['readout']!='full' and r['P1_cell_pass']=='False'],
        'resources':[{k:json.loads(p.read_text())[k] for k in ('mode','elapsed_seconds','cpu_seconds','max_rss_kib','start_utc','threads','memory_max','swap_max')}
                     for p in sorted(OUT.glob('*_meta.json')) if p.stem!='tail_audit_meta']}
    (OUT/'tail_audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
    finish(state,cumulative_before_audit=prior_elapsed('tail_audit'))

if __name__=='__main__': main()
