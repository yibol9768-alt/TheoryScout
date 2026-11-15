"""Freeze R44 research, including certificate hashes and local import closure.

Creates a new archive; writes R44_HISTORY only after extracted replay passes.
Existing archives/receipts are never replaced. Standard library only.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
import time

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def members(root):
    names=set()
    for n in ['revisions/r44_shared_cdf_energy_v2/RESULT.json','revisions/r44_shared_cdf_energy_v2/SHARED_CDF.json','revisions/r44_sphere_density.json']:
        data=read(root/n);names.add(n)
        for name,digest in data['source_hashes'].items():
            assert sha(root/name)==digest,name
            names.add(name)
    names.update(str(p.relative_to(root)) for p in (root/'revisions').glob('R44_*') if p.is_file())
    names.update(str(p.relative_to(root)) for p in (root/'revisions').glob('r44_*/**/*.json') if p.is_file())
    names.update(['tools/check_shared_cdf_energy.py','tools/package_shared_cdf_research.py',
                  'revisions/CURRENT_REPRO_BUNDLE.json','history/20261108T015015.293462Z_after_r40_block_radius/manifest.json'])
    for label in ['R21','R43']:
        n=f'revisions/{label}_HISTORY.json';names.add(n);names.add(read(root/n)['path']+'/manifest.json')
    for failed in read(root/'revisions/R44_ARCHIVE_ATTEMPTS.json')['failed_archives']:names.add(failed['path']+'/manifest.json')
    pending=[n for n in names if n.endswith('.py')];visited=set()
    while pending:
        name=pending.pop()
        if name in visited:continue
        visited.add(name)
        for node in ast.walk(ast.parse((root/name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            if isinstance(node,ast.ImportFrom) and node.module:modules=[node.module]
            for module in modules:
                candidate='tools/'+module.split('.')[0]+'.py'
                if (root/candidate).is_file():names.add(candidate);pending.append(candidate)
    return {n:root/n for n in sorted(names)}

def main():
    root=Path(__file__).resolve().parents[1]
    assert not (root/'revisions/R44_HISTORY.json').exists()
    mapping=members(root)
    mapping.update({'coordination/COORDINATION_2026-11-07.md':root/'../docs/COORDINATION_2026-11-07.md',
                    'coordination/PROGRESS.md':root/'../docs/PROGRESS.md'})
    hashes={n:sha(p) for n,p in mapping.items()}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    history=root/'history'/(stamp+'_r44_research_complete');history.mkdir();archive=history/'research.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for n,p in mapping.items():tar.add(p,arcname=n,recursive=False)
    extracted=Path(tempfile.mkdtemp(prefix='theoryscout_r44_replay_'))
    with tarfile.open(archive,'r:gz') as tar:
        assert {m.name for m in tar.getmembers()}==set(hashes)
        tar.extractall(extracted,filter='data')
    for n,h in hashes.items():assert sha(extracted/n)==h,n
    start=time.monotonic();env=os.environ.copy();env['PYTHONPATH']='';env['PYTHONNOUSERSITE']='1'
    result=subprocess.run(['python3','-B','tools/check_shared_cdf_energy.py','revisions/r44_shared_cdf_energy_v2','--out','extracted_checks'],
                          cwd=extracted,env=env,text=True,capture_output=True)
    print(result.stdout,flush=True)
    if result.returncode:print(result.stderr,flush=True)
    assert result.returncode==0
    for n in ['CHECKS.json','base_audit.json']:
        assert (extracted/'extracted_checks'/n).read_bytes()==(root/'revisions/r44_shared_cdf_checks_final'/n).read_bytes(),n
    replay=dict(extracted_archive=True,all_member_hashes_verified=True,audit_files_byte_identical=True,
        density_recomputed_and_all_exact_duals_checked=True,absolute_workspace_paths_not_used=True,optimizer_required=False,
        prior_CDF_spectral_moment_C0_inputs_inherited=True,new_density_arithmetic_recomputed=True,new_joint_spectral_integrals_recomputed=0,
        no_network_calls=True,network_namespace_isolation_tested=False,wall_seconds=time.monotonic()-start)
    save(history/'manifest.json',dict(created_utc=stamp,label='r44_research_complete',
        note='Shared actual sphere CDF; complete declared inputs and transitive local Python imports; R40 paper unchanged; failed archives retained',
        archive_sha256=sha(archive),files=hashes,extracted_replay=replay))
    receipt=dict(path=str(history.relative_to(root)),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
                 files=len(hashes),all_files_verified=True,extracted_replay=replay)
    save(root/'revisions/R44_HISTORY.json',receipt);print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
