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


def check_sphere_archive(root,archive,expected_sha):
    if sha(archive)!=expected_sha:raise ValueError('R21 sphere replay archive hash mismatch')
    with tarfile.open(archive) as tar:
        manifest=json.load(tar.extractfile('MANIFEST.json'))
        members={m.name:m for m in tar.getmembers()}
        assert set(members)==set(manifest['files'])|{'MANIFEST.json'}
        for name,digest in manifest['files'].items():
            assert members[name].isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest,name
        for name in ['certify_sphere_threshold_cdf.py','check_sphere_threshold_cdf.py','certify_weighted_chisquare.py']:
            if sha(root/'tools'/name)!=manifest['files']['tools/'+name]:
                raise ValueError('Active sphere code differs from replay archive: '+name)
    return manifest


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
    sphere_checks=any('\\label{prop:AC-sphere-cdf}' in (root/name).read_text()
                      for name in baseline['files'] if name.endswith('.tex'))
    sphere_archive=root/'revisions/r21_numeric_replay.tar.gz'
    if sphere_checks:
        sphere_expected=json.loads((root/'revisions/R21_VALIDATION.json').read_text())['isolated_numeric_replay']['archive_sha256']
        sphere_manifest=check_sphere_archive(root,sphere_archive,sphere_expected)
    moment_checks=any('\\label{prop:AC-conditional-moments}' in (root/name).read_text()
                      for name in baseline['files'] if name.endswith('.tex'))
    moment_archives=[]
    if moment_checks:
        for revision in ['R28','R29']:
            receipt=json.loads((root/f'revisions/{revision}_HISTORY.json').read_text())
            directory=root/receipt['path'];archive=directory/'research.tar.gz'
            assert sha(archive)==receipt['archive_sha256']
            inherited=json.loads((directory/'manifest.json').read_text())
            assert inherited['archive_sha256']==receipt['archive_sha256']
            with tarfile.open(archive) as tar:
                members={m.name:m for m in tar.getmembers()}
                assert set(members)==set(inherited['files'])
                for name,digest in inherited['files'].items():
                    assert members[name].isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts
                    assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest,name
            moment_archives.append(dict(revision=revision,directory=directory,sha256=receipt['archive_sha256']))
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
    if sphere_checks:
        data_sphere=out/'data/sphere_threshold';data_sphere.mkdir()
        shutil.copy2(sphere_archive,data_sphere/sphere_archive.name)
        for name in ['R21_ANALYTIC_THRESHOLD_CDF.md','R21_REPLAY.md','R21_REPORT.md','R21_VALIDATION.json']:
            target=provenance/name
            if not target.exists():shutil.copy2(root/'revisions'/name,target)
    if moment_checks:
        destination=out/'data/conditional_moments';destination.mkdir()
        for item in moment_archives:
            revision=item['revision'];directory=item['directory']
            shutil.copy2(directory/'research.tar.gz',destination/(revision+'.tar.gz'))
            shutil.copy2(directory/'manifest.json',destination/(revision+'_MANIFEST.json'))
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
    selection_checks='\\label{cor:AC-scatter-GD}' in source
    if selection_checks:
        assert gd_checks
        shutil.copy2(root/'tools/check_scatter_gd_selection.py',out/'tools/check_scatter_gd_selection.py')
        shutil.copy2(root/'revisions/r34_selection_checks.json',out/'data/gd_scatter_selection_reference.json')
    obstruction_checks='\\label{prop:AC-energy-obstruction}' in source
    if obstruction_checks:
        shutil.copy2(root/'tools/check_energy_copula_obstruction.py',out/'tools/check_energy_copula_obstruction.py')
        shutil.copy2(root/'revisions/r38_energy_obstruction.json',out/'data/energy_obstruction_reference.json')
    block_radius_checks='\\label{prop:AC-block-radius}' in source
    if block_radius_checks:
        shutil.copy2(root/'tools/check_block_radius_geometry.py',out/'tools/check_block_radius_geometry.py')
        shutil.copy2(root/'revisions/r40_block_radius_checks_final.json',out/'data/block_radius_reference.json')
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
         deterministic_probability_replays=[dict(label='prop:AC-sphere-cdf',case='R21 represented C0 ideal Adam2',
              scope='Inherited 18 certified point bounds to analytic sphere CDF and complete random-threshold conditional LOWER',
              predecessor_point_certificates_rebuilt=False,raw_threshold_samples_used=False)] if sphere_checks else [],
         full_experiment_regeneration=False,raw_ranks_included=False,complete_mathematical_verification=False))
    if selection_checks:
        coverage_file=out/'COVERAGE.json';record=json.loads(coverage_file.read_text())
        record['deterministic_algebra_checks'].append('cor:AC-scatter-GD:36 rational residual stops;27 finite conditional selections;20 exact Wick norm cases;generic selection counterexamples')
        coverage_file.write_text(json.dumps(record,indent=2)+'\n')
    if moment_checks:
        coverage_file=out/'COVERAGE.json';record=json.loads(coverage_file.read_text())
        record['conditional_moment_and_scalar_center_replay']=dict(label='prop:AC-conditional-moments',
            research_rounds=['R28','R29'],inherited_coupling_and_spectral_bounds=True,native_residual_chain_rebuilt=False,
            Gaussian_proxy_bound_is_not_actual_recall=True)
        coverage_file.write_text(json.dumps(record,indent=2)+'\n')
    if obstruction_checks:
        coverage_file=out/'COVERAGE.json';record=json.loads(coverage_file.read_text())
        record['deterministic_algebra_checks'].append('prop:AC-energy-obstruction:exact generic joint laws, marginals, both conditional means,60 strict threshold-interval comparisons; not an actual spherical C0 law')
        coverage_file.write_text(json.dumps(record,indent=2)+'\n')
    if block_radius_checks:
        coverage_file=out/'COVERAGE.json';record=json.loads(coverage_file.read_text())
        record['deterministic_algebra_checks'].append('prop:AC-block-radius:exact event/radius and pencil comparisons,16 small-integer Beta density polynomials,36 radius-direction monomial identities; no actual C0 joint integral')
        coverage_file.write_text(json.dumps(record,indent=2)+'\n')
    save(out/'BUNDLE_INFO.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),manuscript_revision=args.revision,
         snapshot=str(snapshot.relative_to(root)),snapshot_archive_sha256=baseline['archive_sha256'],
         tectonic_version=subprocess.check_output([str(engine),'--version'],text=True).strip(),tectonic_sha256=sha(engine),
         platform=platform.platform(),cache_files=sum(p.is_file() for p in target_cache.rglob('*')),
         body_pages=body_pages,total_pages=total_pages,gd_geometry_checks=gd_checks,
         gd_scatter_selection_checks=selection_checks,
         energy_obstruction_checks=obstruction_checks,
         block_radius_geometry_checks=block_radius_checks,
         sphere_threshold_checks=sphere_checks,
         sphere_replay_archive_sha256=sphere_expected if sphere_checks else None,
         sphere_replay_files=len(sphere_manifest['files']) if sphere_checks else 0,
         conditional_moment_checks=moment_checks,
         conditional_research_archives=[dict(revision=i['revision'],sha256=i['sha256']) for i in moment_archives],
         scope='Complete manuscript build + table44 compact replay + declared GD, sphere and conditional-moment/scalar-center replays; not all experiments or proof review'))
    files=sorted(p for p in out.rglob('*') if p.is_file())
    assert all(not p.is_symlink() for p in files)
    manifest={str(p.relative_to(out)):sha(p) for p in files}
    save(out/'MANIFEST.json',dict(files=manifest))
    archive=out.with_name(out.name+'.tar.gz')
    with tarfile.open(archive,'x:gz') as tar:tar.add(out,arcname=out.name)
    print(json.dumps(dict(directory=str(out),archive=str(archive),archive_sha256=sha(archive),files=len(manifest)),indent=2))


if __name__=='__main__':main()
