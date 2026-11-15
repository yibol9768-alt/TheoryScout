"""Build an immutable local R14 source/cache/toolchain and compact-data bundle."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import tarfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    snapshot=root/'history/20261107T202849.663492Z_after_r14_z4b_uncertainty'
    baseline=json.loads((snapshot/'manifest.json').read_text())
    for name,digest in baseline['files'].items():assert sha(root/name)==digest,name
    for name in baseline['files']:
        source=root/name;target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        assert not source.is_symlink();shutil.copy2(source,target)
    (out/'tools').mkdir()
    for name in ['audit_z4b_matching.py','check_manuscript.py']:
        shutil.copy2(root/'tools'/name,out/'tools'/name)
    for name in ['README.md','verify_bundle.py','reproduce.py']:
        shutil.copy2(root/'tools/repro_bundle'/name,out/name)
    data=out/'data/z4b_matching';data.mkdir(parents=True)
    for name in ['INPUTS.json','RESULTS.json','VERTICES.json']:
        shutil.copy2(root/'revisions/r14_z4b_replay'/name,data/name)
    provenance=out/'provenance';provenance.mkdir()
    for name in ['R14_REPORT.md','R14_VALIDATION.json','R14_Z4B_UNCERTAINTY.md']:
        shutil.copy2(root/'revisions'/name,provenance/name)
    shutil.copy2(snapshot/'manifest.json',provenance/'manuscript_manifest.json')
    toolchain=out/'toolchain';toolchain.mkdir()
    engine=Path('/home/liuyibo/.local/bin/tectonic');shutil.copy2(engine,toolchain/'tectonic')
    cache=root/'paper/.cache/Tectonic';target_cache=toolchain/'tex-cache/Tectonic'
    shutil.copytree(cache,target_cache)
    visited=set()
    def expand(path):
        path=path.resolve();assert path.is_relative_to((out/'paper').resolve());visited.add(path)
        source=re.sub(r'(?<!\\)%[^\n]*','',path.read_text())
        return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(out/'paper'/(m[1]+'.tex')),source)
    source=expand(out/'paper/main.tex')
    labels=sorted(set(re.findall(r'\\label\{((?:tab|fig):[^}]+)\}',source)))
    coverage=[dict(label=label,status='compact_statistics_and_linear_proxy_replay' if label=='tab:Z4B-propagated' else 'source_and_rendering_only') for label in labels]
    environments=[]
    for index,match in enumerate(re.finditer(r'\\begin\{(table\*?|longtable|figure\*?)\}(.*?)\\end\{\1\}',source,re.S)):
        found=re.findall(r'\\label\{([^}]+)\}',match[2])
        environments.append(dict(source_order=index+1,kind=match[1],labels=found,
             status='compact_statistics_and_linear_proxy_replay' if 'tab:Z4B-propagated' in found else 'source_and_rendering_only'))
    save(out/'COVERAGE.json',dict(active_tex_inputs=len(visited),table_and_figure_labels=coverage,
         expanded_source_environments=environments,
         full_experiment_regeneration=False,raw_ranks_included=False,complete_mathematical_verification=False))
    save(out/'BUNDLE_INFO.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),manuscript_revision='R14',
         snapshot=str(snapshot.relative_to(root)),snapshot_archive_sha256=baseline['archive_sha256'],
         tectonic_version=subprocess.check_output([str(engine),'--version'],text=True).strip(),tectonic_sha256=sha(engine),
         platform=platform.platform(),cache_files=sum(p.is_file() for p in target_cache.rglob('*')),
         body_pages=8,total_pages=145,scope='Complete manuscript build + table44 compact replay; not all experiments'))
    files=sorted(p for p in out.rglob('*') if p.is_file())
    assert all(not p.is_symlink() for p in files)
    manifest={str(p.relative_to(out)):sha(p) for p in files}
    save(out/'MANIFEST.json',dict(files=manifest))
    archive=out.with_name(out.name+'.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:tar.add(out,arcname=out.name)
    print(json.dumps(dict(directory=str(out),archive=str(archive),archive_sha256=sha(archive),files=len(manifest)),indent=2))


if __name__=='__main__':main()
