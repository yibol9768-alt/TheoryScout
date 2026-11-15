"""Descriptive extraction for the report, no new data or gate selection."""
import json
import time
from run import OUT,frozen,dump
START=time.monotonic(); frozen()
s=json.loads((OUT/'summary.json').read_text()); m=json.loads((OUT/'formal/metadata.json').read_text())
small=dict(primary_cells=[c for c in s['cells'] if c['problem']==2 and c['group']=='learned_adam_cold'],
    primary_fits=[f for f in s['fits'] if f['group']=='learned_adam_cold'],
    primary_bridge=next(b for b in s['bridges'] if b['problem']==2 and b['group']=='learned_adam_cold'),
    spectral=[{k:r[k] for k in ['N','ratio','selected_delta','passed','max_width','segments']} for r in s['spectral']],
    other_bridge_exponents=[{k:b[k] for k in ['problem','group','exponents','zeta_range','nu_range']} for b in s['bridges']],
    learned_angle_cells=[c for c in s['cells'] if c['problem']==2 and c['group']=='learned_angle'],
    calibration=dict(median_late_abs_change=0.,max_late_abs_change=0.,selected=[]))
import numpy as np
changes=np.concatenate([abs(np.array(c['late_radius_relative_change'])) for c in m['calibration']])
small['calibration']['median_late_abs_change']=float(np.median(changes))
small['calibration']['max_late_abs_change']=float(changes.max())
small['calibration']['fraction_abs_change_over_10pct']=float(np.mean(changes>.1))
# Radius calibration explicitly retains all candidates; selected diagnostics only descriptive.
for c in s['cells']:
    if c['problem']==1 and c['ratio']==64 and '_warm' in c['group']:
        group=c['group'].replace('_warm','_cold'); idx=c['best_index']
        vals=[]
        for ca in m['calibration']:
            if ca['problem']==1:
                ii=[j for j,g in enumerate(ca['groups']) if g==group][idx]
                vals.append(ca['median_radii'][ii])
        small['calibration']['selected'].append(dict(group=c['group'],delta=c['best_value'],radius_mean=float(np.mean(vals))))
# Full positive-subset bridge predictions already saved in summary; omit from text extraction.
small['primary_bridge']={k:v for k,v in small['primary_bridge'].items() if k!='invalid_cells'}
for k in ['drift','noise']:
    if small['primary_bridge'][k]: small['primary_bridge'][k]={a:b for a,b in small['primary_bridge'][k].items() if a!='holdout_predictions'}
small['wall_seconds']=time.monotonic()-START
dump(OUT/'report_details.json',small)
print(json.dumps(small,ensure_ascii=False,indent=2))
