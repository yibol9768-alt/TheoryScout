"""Read-only validation; creates only review18_checks/final_audit.json."""
import ast
import hashlib
import json
from independent_mc import ROOT, OUT, resources, write_json

parsed=[]
for path in sorted((ROOT/'review18_checks').glob('*.py')):
    ast.parse(path.read_text())
    parsed.append(path.name)
evidence=json.loads((OUT/'input_hashes.json').read_text())
changed=[name for name,digest in evidence.items()
         if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
assert not changed
author=json.loads((ROOT/'z2b_checks/Z2B_final_audit.json').read_text())
proof_digest=hashlib.sha256((ROOT/'Z2B_BR_PROOF.md').read_bytes()).hexdigest()
assert proof_digest==author['proof_sha256']
rank=json.loads((OUT/'rank_checks.json').read_text())
analysis=json.loads((OUT/'analysis_summary.json').read_text())
max_z=max(abs(v['difference']/v['se']) for v in rank['results'])
assert rank['event_mismatch']==0
assert all(v['mean_is_covered'] for v in analysis['formal'])
assert all((OUT/f'mc_d{d}.json').exists() for d in (64,128,256,512))
assert '<!-- REVIEW18_NUMERICAL_RESULTS -->' not in (ROOT/'PROOF_REVIEW18.md').read_text()
write_json('final_audit.json',dict(source_parse=parsed,input_hashes_unchanged_since_analysis=True,
                                 own_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for p in (ROOT/'review18_checks').glob('*.py')},
                                 review_sha256=hashlib.sha256((ROOT/'PROOF_REVIEW18.md').read_bytes()).hexdigest(),
                                 reviewed_proof_matches_author_final_hash=True,
                                 reviewed_proof_sha256=proof_digest,
                                 max_direct_rank_paired_z=max_z,
                                 formal_mean_coverage=True,
                                 no_reviewed_code_imported_or_executed=True,
                                 no_llm_api_called=True,
                                 explicit_write_scope=['PROOF_REVIEW18.md','review18_checks/'],
                                 resources=resources()))
print(json.dumps(dict(source_parse=True,inputs_unchanged=True,
                      max_direct_rank_paired_z=max_z,formal_mean_coverage=True)))
