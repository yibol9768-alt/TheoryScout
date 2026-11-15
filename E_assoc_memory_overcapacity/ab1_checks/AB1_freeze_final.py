"""Freeze AB1-created inputs; only replaces our own preliminary AB1 manifest."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'ab1_checks'
files=list(B.glob('*.py'))+list(ROOT.glob('AB1_*.md'))+[ROOT/'AB1_Q_CONSTANT_TABLE.json',ROOT/'AB1_INPUT_SHA256.txt']
files += [B/name for name in ['AB1_m8_integer_certificate.json','AB1_m8_decimal_certificate.json',
          'AB1_m8_directed_envelopes.json','AB1_m8_endpoint_envelopes.json','AB1_m8_directed_envelopes.csv',
          'AB1_q_gd_example.json','AB1_q_sign_example.json','AB1_q_adam_example_v2.json','AB1_q_audit.json']]
files += [ROOT/'paper/appendix/AA_quant_rank.tex',ROOT/'paper/appendix/L_multistep.tex']
manifest={'schema':'AB1-current-paper-binding-v1','sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))},
          'provenance':{'integer':'aa3_checks/AA3_centered_paths_certificate.json','decimal':'review21_checks/decimal_enclosures.json',
                        'backend':'review21_checks/decimal_enclosures.py, exact prefix before main()'},
          'normalization':'J_beta/(a*p_j^2); derivative=a^3*p_j^2*(15/16)*Jnormalized/(sqrt(q)*Lambda)',
          'scope':'four M.8 paths, r=1/4, 0<delta<=.30; 173 regions, 64 endpoints; near-threshold roots excluded'}
# AB1_manifest.json was first created by this task, never an existing project file.
with (B/'AB1_manifest.json').open('w') as f:json.dump(manifest,f,indent=2)
print('FINAL INPUT FREEZE PASS',len(manifest['sha256']),hashlib.sha256((B/'AB1_manifest.json').read_bytes()).hexdigest())
