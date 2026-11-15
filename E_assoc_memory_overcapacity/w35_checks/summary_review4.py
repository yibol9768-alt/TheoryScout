from pathlib import Path
import json
c=json.loads(Path('w35_checks/stabilizer_certificate_review4.json').read_text())
for r in c['rows']:
 print(r['gamma'], 'F4',r['proposed_e_band_certified'])
 for d in r['differences']:print('tau',d['tau'],d['coordinate_minus_muon']['approximate'])
 if r['tau_crossing']:print('tau crossing',r['tau_crossing']['computed']['approximate'])
