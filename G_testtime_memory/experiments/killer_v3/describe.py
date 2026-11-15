"""Report extraction only, added after freeze; no random data or gate edits."""
import csv
import json
from common import OUT,begin,finish,prior_elapsed

def rows(name):
    return list(csv.DictReader((OUT/name).open()))

def main():
    state=begin('describe')
    s=json.loads((OUT/'summary.json').read_text())
    ms=rows('method_summary.csv'); cs=rows('cell_summary.csv'); qs=rows('moment_summary.csv')
    mainmethod=[r for r in ms if r['family']=='bounded' and float(r['alpha'])>=1 and r['readout']!='full']
    mainmoment=[r for r in qs if r['family']=='bounded' and float(r['alpha'])>=1]
    maincell=[r for r in cs if r['family']=='bounded' and float(r['alpha'])>=1 and r['readout']!='full']
    out={'gates':s['gates'],'coverage':s['coverage'],
        'stats':{k:{a:b for a,b in v.items() if a!='worst'} for k,v in s['stats'].items()},
        'tau4_coverage_failures':[r for r in maincell if r['readout']=='tau4' and int(r['d'])>=128 and r['P1_cell_pass']=='False'],
        'worst_conditional':s['stats']['P2_conditional']['worst'][:2],
        'worst_forecast':s['stats']['P2_forecast']['worst'][:2],
        'worst_Q_conditional':s['stats']['P6Q_conditional']['worst'][:2],
        'worst_Q_unconditional':s['stats']['P6Q_unconditional']['worst'][:2],
        'worst_gain_conditional':s['stats']['P3_conditional']['worst'][:1],
        'worst_gain_forecast':s['stats']['P3_forecast']['worst'][:1],
        'oracle':rows('oracle_summary.csv'),'strata':s['strata'],
        'Q_by_dimension_beta':[r for r in s['moment_strata'] if r['family']=='bounded' and r['method'] in ('delta_0.8','delta_1.2')],
        'V2_worst_cell_new_data':[r for r in ms if r['family']=='bounded' and int(r['d'])==64 and float(r['alpha'])==1.5 and int(r['load'])==2 and r['method']=='delta_0.4' and r['readout']=='tau16'],
        'V2_worst_cell_moments':[r for r in qs if r['family']=='bounded' and int(r['d'])==64 and float(r['alpha'])==1.5 and int(r['load'])==2 and r['method']=='delta_0.4'],
        'age_gaps':s['age_gaps'],'near_fraction':s['near_fraction'],
        'boundary_best_count':s['boundary_best_count'],'oracle_all_bounds_vacuous':s['oracle_all_bounds_vacuous'],
        'oracle_MC_max_bound':s['oracle_MC_max_bound'],
        'numerical':{k:s[k] for k in ('max_quadrature_difference','max_interpolation_difference')},
        'resources':{p.stem:json.loads(p.read_text()) for p in sorted(OUT.glob('*_meta.json')) if p.stem!='describe_meta'}}
    # Size-stratified descriptions; this does not introduce new pass criteria.
    out['conditional_size_errors']=[{'d':d,'fail':sum(float(r['abs_error'])>.03 for r in mainmethod if int(r['d'])==d),
        'total':sum(int(r['d'])==d for r in mainmethod),
        'max':max(float(r['abs_error']) for r in mainmethod if int(r['d'])==d)} for d in (64,128,256)]
    (OUT/'report_extract.json').write_text(json.dumps(out,indent=2))
    compact={k:v for k,v in out.items() if k not in ('resources','oracle','V2_worst_cell_moments','Q_by_dimension_beta')}
    print(json.dumps(compact,indent=2))
    finish(state,cumulative_before_describe=prior_elapsed('describe'))

if __name__=='__main__': main()
