"""One-time exclusive-create packaging; verifier never calls this initializer."""
import json
from pathlib import Path
import verify_m8 as v
_,rows,ends=v.saved_checks(False)
v.export_tables(rows,ends)
files=[p for p in v.BASE.iterdir() if p.is_file() and p.suffix in ('.py','.json','.csv')]
files.extend(v.ROOT/p for p in ['paper/appendix/AA_quant_rank.tex','paper/appendix/L_multistep.tex','AB1_INPUT_SHA256.txt'])
manifest={'schema':'AB1-current-paper-binding-v1','sha256':{str(p.relative_to(v.ROOT)):v.sha(p) for p in sorted(files)},
          'provenance':{'integer':'aa3_checks/AA3_centered_paths_certificate.json',
                        'decimal':'review21_checks/decimal_enclosures.json',
                        'backend':'review21_checks/decimal_enclosures.py, prefix before main(), unchanged'},
          'normalization':'J_beta/(a*p_j^2); D_beta=a^3*p_j^2*(15/16)*Jnormalized/(sqrt(q)*Lambda)',
          'scope':'four M.8 paths, r=1/4, 0<delta<=.30; 173 regions, 64 right endpoints'}
with (v.BASE/'AB1_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('PREPARE PASS',len(rows),len(ends))
