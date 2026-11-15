"""Build an immutable bundle bound to an explicit current manuscript snapshot."""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True,help='Snapshot directory, relative to E or absolute')
    parser.add_argument('--revision',required=True,help='Manuscript revision, e.g. R17')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.out.resolve()
    snapshot=(root/args.snapshot).resolve()
    if not snapshot.is_relative_to(root/'history'):raise ValueError('Snapshot must be inside this project history')
    if re.fullmatch(r'R[0-9]+',args.revision) is None:raise ValueError('Expected revision R<number>')
    baseline=json.loads((snapshot/'manifest.json').read_text())
    assert sha(snapshot/'paper.tar.gz')==baseline['archive_sha256'],'Snapshot archive hash mismatch'
    for name in baseline['files']:
        path=(root/name).resolve();assert path.is_relative_to(root/'paper') and not (root/name).is_symlink()
    for name,digest in baseline['files'].items():assert sha(root/name)==digest,name
    heading=(root/'paper/README.md').read_text().splitlines()[0]
    if re.search(r'coordinating revision '+args.revision+r'\b',heading) is None:
        raise ValueError('Requested revision differs from the snapshot manuscript heading')
    aux=(root/'paper/build/main.aux').read_text()
    body_pages=int(re.search(r'\\newlabel\{references-start\}\{\{[^}]*\}\{(\d+)\}',aux)[1])-1
    pdf_info=subprocess.check_output(['pdfinfo',str(root/'paper/build/main.pdf')],text=True)
    total_pages=int(re.search(r'Pages:\s+(\d+)',pdf_info)[1])
    check=subprocess.check_output([sys.executable,'-B',str(root/'tools/check_manuscript.py')],text=True)
    assert f'{body_pages} main-text pages, {total_pages} total pages' in check
    # Refuse stale sources/revisions before creating any output directory.
    out.mkdir(parents=True,exist_ok=False)
    for name in baseline['files']:
        source=root/name;target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        assert not source.is_symlink();shutil.copy2(source,target)
    (out/'tools').mkdir()
    for name in ['audit_z4b_matching.py','check_manuscript.py']:
        shutil.copy2(root/'tools'/name,out/'tools'/name)
    for name in ['README.md','verify_bundle.py','reproduce.py']:
        shutil.copy2(root/'tools/repro_bundle'/name,out/name)
    readme=out/'README.md'
    text=readme.read_text()
    for token,value in {'@REVISION@':args.revision,'@BODY_PAGES@':str(body_pages),'@TOTAL_PAGES@':str(total_pages)}.items():text=text.replace(token,value)
    assert re.search(r'@[A-Z_]+@',text) is None
    readme.write_text(text)
    data=out/'data/z4b_matching';data.mkdir(parents=True)
    for name in ['INPUTS.json','RESULTS.json','VERTICES.json']:
        shutil.copy2(root/'revisions/r14_z4b_replay'/name,data/name)
    provenance=out/'provenance';provenance.mkdir()
    records={'R14_REPORT.md','R14_VALIDATION.json','R14_Z4B_UNCERTAINTY.md',args.revision+'_REPORT.md',args.revision+'_VALIDATION.json'}
    for name in sorted(records):
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
    gd_checks='\\label{prop:AC-uniform-GD}' in source
    if gd_checks:
        for name in ['check_uniform_gd_geometry.py','certify_weighted_chisquare.py']:
            shutil.copy2(root/'tools'/name,out/'tools'/name)
        shutil.copy2(root/'revisions/r17_exact_geometry.json',out/'data/gd_geometry_reference.json')
    labels=sorted(set(re.findall(r'\\label\{((?:tab|fig):[^}]+)\}',source)))
    coverage=[dict(label=label,status='compact_statistics_and_linear_proxy_replay' if label=='tab:Z4B-propagated' else 'source_and_rendering_only') for label in labels]
    environments=[]
    for index,match in enumerate(re.finditer(r'\\begin\{(table\*?|longtable|figure\*?)\}(.*?)\\end\{\1\}',source,re.S)):
        found=re.findall(r'\\label\{([^}]+)\}',match[2])
        environments.append(dict(source_order=index+1,kind=match[1],labels=found,
             status='compact_statistics_and_linear_proxy_replay' if 'tab:Z4B-propagated' in found else 'source_and_rendering_only'))
    save(out/'COVERAGE.json',dict(active_tex_inputs=len(visited),table_and_figure_labels=coverage,
         expanded_source_environments=environments,
         deterministic_algebra_checks=['prop:AC-uniform-GD:72 rational matrix cases;63 recurrence/filter comparisons'] if gd_checks else [],
         full_experiment_regeneration=False,raw_ranks_included=False,complete_mathematical_verification=False))
    save(out/'BUNDLE_INFO.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),manuscript_revision=args.revision,
         snapshot=str(snapshot.relative_to(root)),snapshot_archive_sha256=baseline['archive_sha256'],
         tectonic_version=subprocess.check_output([str(engine),'--version'],text=True).strip(),tectonic_sha256=sha(engine),
         platform=platform.platform(),cache_files=sum(p.is_file() for p in target_cache.rglob('*')),
         body_pages=body_pages,total_pages=total_pages,gd_geometry_checks=gd_checks,
         scope='Complete manuscript build + table44 compact replay + declared algebra checks; not all experiments or proof review'))
    files=sorted(p for p in out.rglob('*') if p.is_file())
    assert all(not p.is_symlink() for p in files)
    manifest={str(p.relative_to(out)):sha(p) for p in files}
    save(out/'MANIFEST.json',dict(files=manifest))
    archive=out.with_name(out.name+'.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:tar.add(out,arcname=out.name)
    print(json.dumps(dict(directory=str(out),archive=str(archive),archive_sha256=sha(archive),files=len(manifest)),indent=2))


if __name__=='__main__':main()
