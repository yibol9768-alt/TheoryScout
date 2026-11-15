"""Rebuild the bound manuscript and declared checks from this immutable bundle."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode=True
from verify_bundle import verify
from fractions import Fraction as F


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
    bounded_proxy_replayed=False
    if info.get('bounded_proxy_confidence_checks',False):
        archive=root/'data/z4b_bounded/R47.tar.gz'
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==info['bounded_proxy_archive_sha256']
        nested=json.loads((root/'data/z4b_bounded/R47_MANIFEST.json').read_text())
        source=out/'z4b_bounded_source';source.mkdir()
        with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
        for name,digest in nested['files'].items():
            assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
        checked=out/'z4b_bounded_checks.json';regenerated=out/'z4b_bounded_result.json'
        commands=[[sys.executable,'-B','tools/check_z4b_bounded_proxy.py','revisions/r47_z4b_bounded_proxy.json','--out',str(checked)],
                  [sys.executable,'-B','tools/certify_z4b_bounded_proxy.py','--out',str(regenerated)]]
        for index,command in enumerate(commands):
            with (out/f'z4b_bounded_{index}.log').open('x') as log:
                subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert checked.read_bytes()==(source/'revisions/r47_z4b_bounded_checks_final.json').read_bytes()
        assert regenerated.read_bytes()==(source/'revisions/r47_z4b_bounded_proxy.json').read_bytes()
        result=json.loads(regenerated.read_text())
        def outward_pp(value,side):
            x=F(value)*100000;k=x.numerator//x.denominator if side<0 else -((-x.numerator)//x.denominator)
            return ('-' if k<0 else '')+str(abs(k)//1000)+'.'+str(abs(k)%1000).zfill(3)
        intervals=[(outward_pp(r['confidence_interval'][0],-1),outward_pp(r['confidence_interval'][1],1)) for r in result['records'] if r['kind']=='linear_proxy']
        source_table=(out/'paper/appendix/Z4B_prediction_chain.tex').read_text().split('\\label{tab:Z4B-bounded-proxy}',1)[1].split('\\end{table}',1)[0]
        assert re.findall(r'\[(-?\d+\.\d{3}),\s*(-?\d+\.\d{3})\]',source_table)==intervals
        means=[]
        for r in result['records']:
            if r['kind']=='linear_proxy':
                x=F(r['mean'])*100000;k=(2*x.numerator+x.denominator)//(2*x.denominator)
                means.append(('-' if k<0 else '')+str(abs(k)//1000)+'.'+str(abs(k)%1000).zfill(3))
        assert re.findall(r'&\s*\$(-?\d+\.\d{3})\$\s*&\s*\$\[',source_table)==means
        assert not any(r['finite_sample_bound_certifies_one_pp_gate'] for r in result['records'] if r['kind']=='matching')
        for name,digest in nested['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
        bounded_proxy_replayed=True
    gd_replayed=False
    if info['gd_geometry_checks']:
        command=[sys.executable,'-B',str(root/'tools/check_uniform_gd_geometry.py'),'--out',str(out/'gd_geometry.json')]
        with (out/'gd_geometry.log').open('x') as log:
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'gd_geometry.json').read_bytes()==(root/'data/gd_geometry_reference.json').read_bytes()
        gd_replayed=True
    selection_replayed=False
    if info.get('gd_scatter_selection_checks',False):
        command=[sys.executable,'-B',str(root/'tools/check_scatter_gd_selection.py'),'--out',str(out/'gd_scatter_selection.json')]
        with (out/'gd_scatter_selection.log').open('x') as log:
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'gd_scatter_selection.json').read_bytes()==(root/'data/gd_scatter_selection_reference.json').read_bytes()
        selection_replayed=True
    gd_schedule_replayed=False
    if info.get('gd_variable_schedule_checks',False):
        archive=root/'data/gd_schedules/R49.tar.gz'
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==info['gd_variable_schedule_archive_sha256']
        nested=json.loads((root/'data/gd_schedules/R49_MANIFEST.json').read_text())
        source=out/'gd_schedule_source';source.mkdir()
        with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
        for name,digest in nested['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
        checked=out/'gd_schedule_checks.json'
        command=[sys.executable,'-B','tools/check_gd_varying_schedule.py','--out',str(checked),'--compare','revisions/r49_gd_varying_schedule.json']
        with (out/'gd_schedule.log').open('x') as log:
            subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert checked.read_bytes()==(source/'revisions/r49_gd_varying_schedule.json').read_bytes()
        for name,digest in nested['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
        command=[sys.executable,'-B',str(root/'tools/check_gd_schedule_limits.py'),'--out',str(out/'gd_schedule_limits.json'),
                 '--compare',str(root/'data/gd_schedule_limits_reference.json')]
        with (out/'gd_schedule_limits.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'gd_schedule_limits.json').read_bytes()==(root/'data/gd_schedule_limits_reference.json').read_bytes()
        gd_schedule_replayed=True
    obstruction_replayed=False
    if info.get('energy_obstruction_checks',False):
        command=[sys.executable,'-B',str(root/'tools/check_energy_copula_obstruction.py'),'--out',str(out/'energy_obstruction.json')]
        with (out/'energy_obstruction.log').open('x') as log:
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'energy_obstruction.json').read_bytes()==(root/'data/energy_obstruction_reference.json').read_bytes()
        obstruction_replayed=True
    block_radius_replayed=False
    if info.get('block_radius_geometry_checks',False):
        command=[sys.executable,'-B',str(root/'tools/check_block_radius_geometry.py'),'--out',str(out/'block_radius.json')]
        with (out/'block_radius.log').open('x') as log:
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'block_radius.json').read_bytes()==(root/'data/block_radius_reference.json').read_bytes()
        block_radius_replayed=True
    order_density_replayed=False
    if info.get('order_density_reference_checks',False):
        command=[sys.executable,'-B',str(root/'tools/check_order_density_reference.py'),'--out',str(out/'order_density.json')]
        with (out/'order_density.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert (out/'order_density.json').read_bytes()==(root/'data/order_density_reference.json').read_bytes()
        order_density_replayed=True
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
    moment_replays=[]
    for item in info.get('conditional_research_archives',[]):
        revision=item['revision'];assert revision in ['R28','R29']
        data=root/'data/conditional_moments';archive=data/(revision+'.tar.gz')
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==item['sha256']
        nested=json.loads((data/(revision+'_MANIFEST.json')).read_text())
        source=out/(revision+'_source');source.mkdir()
        with tarfile.open(archive) as tar:
            members={m.name:m for m in tar.getmembers()};assert set(members)==set(nested['files'])
            for name,digest in nested['files'].items():
                assert members[name].isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts
                assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==digest,name
            tar.extractall(source,filter='data')
        replay_out=out/(revision+'_moments');audit_out=out/(revision+'_checks.json')
        if revision=='R28':
            reference=source/'revisions/r28_conditional_moments_v2'
            commands=[['tools/certify_conditional_moments.py','--out',str(replay_out)],
                      ['tools/check_conditional_moments.py',str(replay_out),'--compare',str(reference),'--out',str(audit_out)],
                      ['tools/check_moment_identities.py','--out',str(out/'R28_reference.json')]]
        else:
            reference=source/'revisions/r29_scalar_center_fine'
            commands=[['tools/certify_adam2_scalar_center.py','--out',str(replay_out),'--cells','8192'],
                      ['tools/check_scalar_center.py',str(replay_out),'--coarse',str(source/'revisions/r29_scalar_center'),'--out',str(audit_out)]]
        for index,command in enumerate(commands):
            with (out/f'{revision}_{index}.log').open('x') as log:
                subprocess.run([sys.executable,'-B',*command],cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=json.loads((replay_out/'RESULT.json').read_text());expected=json.loads((reference/'RESULT.json').read_text())
        actual.pop('wall_seconds');expected.pop('wall_seconds');assert actual==expected
        filenames=['INPUTS.json','TRACES.json'] if revision=='R28' else ['QUADRATURE.json']
        for filename in filenames:assert (replay_out/filename).read_bytes()==(reference/filename).read_bytes()
        if revision=='R28':
            assert json.loads((out/'R28_reference.json').read_text())==json.loads((source/'revisions/r28_moment_reference.json').read_text())
        else:
            assert json.loads(audit_out.read_text())==json.loads((source/'revisions/r29_scalar_checks_v2.json').read_text())
        for name,digest in nested['files'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
        moment_replays.append(dict(revision=revision,mathematics_identical=True,reference_checks_passed=True,inherited_numeric_prerequisites=True))
    assert bool(moment_replays)==info.get('conditional_moment_checks',False)
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
        bounded_proxy_confidence_and_manuscript_table_identical=bounded_proxy_replayed,
        gd_algebra_checks_identical=gd_replayed,
        gd_scatter_selection_checks_identical=selection_replayed,
        gd_variable_schedule_and_scope_checks_identical=gd_schedule_replayed,
        generic_energy_obstruction_checks_identical=obstruction_replayed,
        block_radius_geometry_checks_identical=block_radius_replayed,
        order_density_reference_checks_identical=order_density_replayed,
        sphere_threshold_results_identical_except_wall_time=sphere_replayed,
        sphere_radius_records_bytes_identical=sphere_replayed,
        sphere_replay_uses_threshold_samples=False if sphere_replayed else None,
        conditional_moment_and_scalar_center_replays=moment_replays,
        coverage='Full manuscript build; compact Z4B/bounded proxy table; GD geometry/selection/variable schedules; inherited-point sphere, conditional moment and scalar-center checks',
        exclusions=['Other experiments not regenerated','Raw Z4B ranks not bundled','t critical values and model slopes archived inputs',
                    'Sphere case inherits predecessor C0 point certificates; native residual chain and point integrals not rebuilt',
                    'Sphere result is a conditional LOWER at represented C0, not an unrounded Gaussian input or single-dataset guarantee',
                    'Conditional moments and scalar center replay inherit prior coupling/spectral enclosures; Gaussian proxy effect is not actual recall',
                    'Not independent proof review, exact-matching guarantee, or complete scientific reproduction',
                    'Bounded proxy coverage assumes conditionally iid datasets and independently fixed functions; no actual rematching or postselection guarantee',
                    'Variable GD references are finite algebra under scatter-only capped schedules; no rank/Haar policy, nonlinear growing time or realized-error supremum guarantee'])
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
