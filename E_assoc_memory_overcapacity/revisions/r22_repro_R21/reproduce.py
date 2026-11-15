"""Rebuild the bound manuscript and declared checks from this immutable bundle."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode=True
from verify_bundle import verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parent;manifest=verify(root)
    info=json.loads((root/'BUNDLE_INFO.json').read_text())
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Included static Tectonic binary requires x86_64 Linux')
    if sys.version_info<(3,12):raise RuntimeError('Replay requires Python 3.12 or later')
    for command in ['pdfinfo','pdftotext']:
        if shutil.which(command) is None:raise RuntimeError('Required system tool missing: '+command)
    out=args.out.resolve()
    if out.is_relative_to(root):raise RuntimeError('Output must be outside the immutable bundle')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    shutil.copytree(root/'paper',out/'paper',ignore=shutil.ignore_patterns('build'))
    (out/'paper/build').mkdir();(out/'tools').mkdir()
    shutil.copy2(root/'tools/check_manuscript.py',out/'tools/check_manuscript.py')
    shutil.copytree(root/'toolchain/tex-cache',out/'xdg-cache');(out/'xdg-config').mkdir()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',
        XDG_CACHE_HOME=str(out/'xdg-cache'),XDG_CONFIG_HOME=str(out/'xdg-config'))
    command=[str(root/'toolchain/tectonic'),'--only-cached','--reruns','5','--untrusted',
             '--keep-intermediates','--keep-logs','--outdir','paper/build','paper/main.tex']
    with (out/'compile.log').open('x') as log:
        subprocess.run(command,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    check=subprocess.check_output([sys.executable,'-B',str(out/'tools/check_manuscript.py')],cwd=out,text=True)
    (out/'manuscript_check.txt').write_text(check)
    assert f"{info['body_pages']} main-text pages, {info['total_pages']} total pages" in check
    command=[sys.executable,'-B',str(root/'tools/audit_z4b_matching.py'),'--replay',str(root/'data/z4b_matching'),'--out',str(out/'z4b_replay')]
    with (out/'z4b_replay.log').open('x') as log:
        subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    reference=json.loads((root/'data/z4b_matching/RESULTS.json').read_text())
    replay=json.loads((out/'z4b_replay/RESULTS.json').read_text())
    assert replay==reference
    assert (out/'z4b_replay/VERTICES.json').read_bytes()==(root/'data/z4b_matching/VERTICES.json').read_bytes()
    assert replay['compact_count_statistics_recomputed']
    gd_replayed=False
    if info['gd_geometry_checks']:
        command=[sys.executable,'-B',str(root/'tools/check_uniform_gd_geometry.py'),'--out',str(out/'gd_geometry.json')]
        with (out/'gd_geometry.log').open('x') as log:
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'gd_geometry.json').read_bytes()==(root/'data/gd_geometry_reference.json').read_bytes()
        gd_replayed=True
    sphere_replayed=False
    if info.get('sphere_threshold_checks',False):
        archive=root/'data/sphere_threshold/r21_numeric_replay.tar.gz'
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==info['sphere_replay_archive_sha256']
        source=out/'sphere_source';source.mkdir()
        with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
        nested=json.loads((source/'MANIFEST.json').read_text())
        def verify_sphere_source():
            actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and p!=source/'MANIFEST.json'}
            assert actual==set(nested['files'])
            for name,digest in nested['files'].items():
                path=source/name;assert not path.is_symlink() and path.resolve().is_relative_to(source)
                assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        verify_sphere_source()
        assert not (source/'revisions/r8_random_thresholds/thresholds.npy').exists()
        sphere_out=out/'sphere_threshold';sphere_check=out/'sphere_checks'
        commands=[
            [sys.executable,'-B',str(source/'tools/certify_sphere_threshold_cdf.py'),'--out',str(sphere_out)],
            [sys.executable,'-B',str(source/'tools/check_sphere_threshold_cdf.py'),'--input',str(sphere_out),'--out',str(sphere_check)]]
        for index,command in enumerate(commands):
            with (out/f'sphere_{index}.log').open('x') as log:
                subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        expected=json.loads((source/'reference/RESULT.json').read_text())
        actual=json.loads((sphere_out/'RESULT.json').read_text())
        expected.pop('wall_seconds');actual.pop('wall_seconds');assert actual==expected
        assert (sphere_out/'RADII.json').read_bytes()==(source/'reference/RADII.json').read_bytes()
        verify_sphere_source();sphere_replayed=True
    reference_text=subprocess.check_output(['pdftotext',str(root/'paper/build/main.pdf'),'-'])
    rebuilt_text=subprocess.check_output(['pdftotext',str(out/'paper/build/main.pdf'),'-'])
    assert rebuilt_text==reference_text, 'Rebuilt PDF text differs from bound manuscript PDF'
    # Confirm the original bundle stayed immutable throughout the replay.
    verify(root)
    result=dict(bundle_revision=info['manuscript_revision'],bundle_files=len(manifest['files']),wall_seconds=time.monotonic()-start,
        clean_build=True,only_cached_tex=True,body_pages=info['body_pages'],total_pages=info['total_pages'],
        pdf_text_identical=True,pdf_text_sha256=hashlib.sha256(rebuilt_text).hexdigest(),
        pdf_bytes_identical=(out/'paper/build/main.pdf').read_bytes()==(root/'paper/build/main.pdf').read_bytes(),
        z4b_statistics_and_geometry_identical=True,immutable_bundle_reverified=True,
        gd_algebra_checks_identical=gd_replayed,
        sphere_threshold_results_identical_except_wall_time=sphere_replayed,
        sphere_radius_records_bytes_identical=sphere_replayed,
        sphere_replay_uses_threshold_samples=False if sphere_replayed else None,
        coverage='Full manuscript build; table44 compact-count replay; declared GD algebra and inherited-point sphere probability checks',
        exclusions=['Other experiments not regenerated','Raw Z4B ranks not bundled','t critical values and model slopes archived inputs',
                    'Sphere case inherits predecessor C0 point certificates; native residual chain and point integrals not rebuilt',
                    'Sphere result is a conditional LOWER at represented C0, not an unrounded Gaussian input or single-dataset guarantee',
                    'Not independent proof review, exact-matching guarantee, or complete scientific reproduction'])
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
