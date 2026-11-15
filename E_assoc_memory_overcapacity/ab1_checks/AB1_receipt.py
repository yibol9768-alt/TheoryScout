"""Final exclusive-create receipt, after all verification has completed."""
import hashlib,json,time
from pathlib import Path
import verify_m8 as v
start=time.monotonic();res=v.dec.resources();mh,manifest=v.bindings()
full=json.loads((v.BASE/'AB1_m8_final_run.json').read_text());qa=json.loads((v.BASE/'AB1_q_audit_final.json').read_text())
v.must(full['status']=='PASS' and full['mode']=='full_reintegration','full result')
v.must(full['reintegrated_region_count']==173 and full['reintegrated_endpoint_count']==64,'all cells')
v.must(full['manifest_sha256']==mh,'final run binding')
v.must(qa['status']=='PASS' and len(qa['methods'])==3,'Q diagnostic')
v.must(hashlib.sha256(v.canonical(dict(regions=full['regions'],endpoints=full['endpoints']))).hexdigest()==full['result_sha256'],'result checksum')
# Check original source/provenance bytes still match the initial snapshot.
for line in (v.ROOT/'AB1_INPUT_SHA256.txt').read_text().splitlines():
    digest,rel=line.split(None,1);v.must(v.sha(v.ROOT/rel.strip())==digest,'original input changed')
original=(v.ROOT/'review21_checks/decimal_enclosures.py').read_text().split('def main():',1)[0]
v.must((v.BASE/'AB1_decimal_backend.py').read_text()==original,'backend is exact independent prefix')
v.must((v.BASE/'AB1_m8_integer_certificate.json').read_bytes()==(v.ROOT/'aa3_checks/AA3_centered_paths_certificate.json').read_bytes(),'integer snapshot')
v.must((v.BASE/'AB1_m8_decimal_certificate.json').read_bytes()==(v.ROOT/'review21_checks/decimal_enclosures.json').read_bytes(),'Decimal snapshot')
files=sorted([p for p in v.ROOT.glob('AB1_*') if p.is_file()]+[p for p in v.BASE.iterdir() if p.is_file()])
receipt=dict(status='PASS',original_inputs_unchanged=True,backend_exact_snapshot=True,
             manifest_sha256=mh,result_sha256=full['result_sha256'],saved_envelopes_sha256=full['saved_envelopes_sha256'],
             source_sha256=full['paper_sha256'],full_reintegration_seconds=full['seconds'],full_reintegration_resources=full['resources'],
             Q_methods='GD,Sign,Adam outward constants + independent diagnostic PASS',
             file_sha256={str(p.relative_to(v.ROOT)):v.sha(p) for p in files},
             excluded_from_recursive_hash=['AB1_FINAL_RECEIPT.json'],seconds=time.monotonic()-start,resources=res)
with (v.ROOT/'AB1_FINAL_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps({k:val for k,val in receipt.items() if k!='file_sha256'},indent=2))
