"""Post-outcome descriptive tables only: no RNG, fitting, or gate recomputation.

Created after the single frozen analysis to quote its CSVs in the Chinese
report. This file is not part of the data-before-freeze implementation.
"""
import csv
import json
from common import OUT, begin, finish, prior_elapsed

def main():
    state = begin('description')
    def read(name):
        with (OUT/name).open() as f: return list(csv.DictReader(f))
    cells = read('cell_summary.csv')
    methods = read('method_summary.csv')
    moments = read('moment_summary.csv')
    summary = json.loads((OUT/'summary.json').read_text())
    primary_cells = [r for r in cells if r['family']=='bounded' and float(r['alpha'])>=1 and r['readout']!='full']
    primary_methods = [r for r in methods if r['family']=='bounded' and float(r['alpha'])>=1 and r['readout']!='full']
    primary_moments = [r for r in moments if r['family']=='bounded' and r['split']=='validation' and float(r['alpha'])>=1]
    delta_moments = [r for r in primary_moments if r['method']!='hebb']
    worst = summary['worst_top1'][0]
    def match_moment(r, target):
        return all(str(r[k])==str(target[k]) for k in ('family','method')) and all(float(r[k])==float(target[k]) for k in ('d','load','alpha'))
    data = {
        'status':'Post hoc descriptive extraction from frozen-analysis outputs; no new data or gates.',
        'tau4_P1_failed_cells':[r for r in primary_cells if r['readout']=='tau4' and int(r['d'])>=128 and r['P1_cell_pass']=='False'],
        'P3_failed_cell':[r for r in primary_cells if float(r['gain_error'])>.03],
        'worst_top1_moment_rows':[r for r in primary_moments if match_moment(r,worst)],
        'small_beta_top1':{'total':len([r for r in primary_methods if r['method']!='hebb' and float(r['method'].split('_')[1])<=.4]),
                           'fail':len([r for r in primary_methods if r['method']!='hebb' and float(r['method'].split('_')[1])<=.4 and float(r['abs_error'])>.03])},
        'feedback_Q_comparison':{
            'total':len(delta_moments),
            'with_feedback_fail':sum(float(r['Q_relative_error'])>.05 for r in delta_moments),
            'no_feedback_fail':sum(float(r['no_feedback_Q_relative_error'])>.05 for r in delta_moments),
            'with_feedback_mean_relative_error':sum(float(r['Q_relative_error']) for r in delta_moments)/len(delta_moments),
            'no_feedback_mean_relative_error':sum(float(r['no_feedback_Q_relative_error']) for r in delta_moments)/len(delta_moments)},
        'top1_by_dimension':[
            {'d':d,'total':len([r for r in primary_methods if int(r['d'])==d]),
             'fail':sum(float(r['abs_error'])>.03 for r in primary_methods if int(r['d'])==d),
             'max_error':max(float(r['abs_error']) for r in primary_methods if int(r['d'])==d)} for d in (64,128,256)],
        'cumulative_seconds_before_description':prior_elapsed('description')}
    (OUT/'descriptive.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data,indent=2),flush=True)
    finish(state)

if __name__=='__main__': main()
