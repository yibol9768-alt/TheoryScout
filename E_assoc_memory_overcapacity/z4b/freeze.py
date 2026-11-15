"""排他冻结；只读核验旧字节，写新包及 Z4B 清单。"""
import os
import resource
import subprocess
import time
import sys
from common_v2 import ROOT,OUT,read,dump,sha,usage,environment

def main():
    assert not (OUT/'formal').exists() and not (ROOT/'z4/formal').exists()
    baseline=read(OUT/'baseline.json')
    protected={str(p.relative_to(ROOT)) for p in list((ROOT/'z4').rglob('*'))+list(ROOT.glob('Z4_*')) if p.is_file()}
    assert protected=={p for p in baseline if p.startswith('z4/') or p.startswith('Z4_')},'protected file set changed'
    bad=[rel for rel,h in baseline.items() if not (ROOT/rel).is_file() or sha(ROOT/rel)!=h]
    assert not bad,bad
    dump(OUT/'baseline_check.json',dict(passed=True,protected_files=len(baseline),changed=bad,
         scope='原 z4 全部文件、Z4_*、指定审计文档；其他已有文件只经限定只读输入访问，不宣称全项目并发变化不存在',usage=usage()))
    result=read(OUT/'all_tests.json');assert result['passed']
    for name in ['port_synthetic_tests.json','resume_tests.json','credential_tests.json','final_tests.json','revision_tests.json','volume_result.json']:
        assert read(OUT/name)['passed'],name
    final=read(OUT/'final_tests.json')
    assert all(sha(OUT/n)==h for n,h in final['source_sha256'].items())
    assert read(OUT/'power.json')['assumptions']['locked_model_gap_positive_given_successful_calibration'] is True
    assert read(OUT/'seed_extension.json')['passed']
    assert read(OUT/'seed_extension_test_receipt.json')['passed']
    assert environment()==read(OUT/'environment.json')
    assert all(sha(ROOT/rel)==h for rel,h in read(OUT/'input_hashes.json').items())
    paths=subprocess.check_output(['rg','--files','--hidden','-g','*.py','-g','!z4b/**','-g','!Z4B_*','-g','!.git/**'],cwd=ROOT,text=True).splitlines()
    tokens=['SeedSequence','default_rng','RandomState','random.seed','manual_seed','Sobol(']
    current={rel for rel in paths if not (ROOT/rel).is_symlink() and any(token in (ROOT/rel).read_text(errors='replace') for token in tokens)}
    assert current=={r['file'] for r in read(OUT/'rng_inventory.json')},'RNG inventory changed since preparation'
    resources=read(OUT/'resources.json');assert sum(resources['hard_allocation'][k] for k in ['preparation','runner','analysis','supervisor','observer'])==21600
    assert resources['runner_forecast_cpu']<20200
    v=read(OUT/'volume_result.json');assert v['publish_cpu']<59 and v['replay_cpu']<39 and v['report_cpu']<500
    dump(OUT/'resource_validation.json',dict(passed=True,full_scale_jobs=v['jobs'],full_scale_npz=v['npz_count'],
         report_cpu=v['report_cpu'],report_wall=v['report_wall'],report_cpu_hard=500,
         supervisor_publish_cpu=v['publish_cpu'],publish_wall=v['publish_wall'],supervisor_cpu_timer=59,supervisor_cpu_hard=60,
         cold_acceptance_replay=v['timed_cold_replay'],observer_cpu_timer=39,observer_cpu_hard=40,
         caveat='这些阶段CPU尚不含各自首次全包冻结核验；冻结后真实entry定时器路径另验并绑定清单。所有数据为伪rank。',
         future_hard_CPU_allocation=resources['hard_allocation']))
    counts={name:len(read(OUT/name).get('tests',read(OUT/name).get('checks',[]))) for name in
         ['port_synthetic_tests.json','resume_tests.json','credential_tests.json','final_tests.json','revision_tests.json']}
    # 套件续跑复用的成功记录只计费一次；不能把中断/缺失历史当0。
    unique={}
    parent_cpu=0.
    for p in sorted(OUT.glob('all_tests*.json')):
        run=read(p);parent_cpu+=run['end']['cpu']
        for row in run['scripts']:
            unique[__import__('json').dumps(row,sort_keys=True)]=row
    known=read(OUT/'build_receipt.json')['usage_cpu']+resources['preparation_end']['cpu']+parent_cpu+sum(r['child_cpu'] for r in unique.values())
    known+=read(OUT/'port_synthetic_tests_attempt1.json')['end']['cpu']
    known+=read(OUT/'seed_extension.json')['usage']['cpu']+read(OUT/'seed_extension_test_receipt.json')['child_cpu']
    dump(OUT/'preparation_usage.json',dict(recorded_cpu_lower_bound_seconds=known,
         unique_test_processes=len(unique),test_process_records=list(unique.values()),
         incomplete_usage=['首次全项目只读哈希扫描中断且无完整rusage收据',
             '两次离线prepare在校准新旧源清单时提前拒绝，未落盘完整rusage',
             '首次冻结因并发新增随机源而拒绝，及相关诊断扫描未逐项落盘完整rusage',
             '首次直接失败移植测试辅助CPU杀进程的子CPU未入该旧格式收据',
             '本冻结进程与冻结后只读核验另有使用量，尚未包含'],
         preparation_reserve=600,certified_all_history_within_reserve=False,
         failed_attempts_preserved=['port_synthetic_tests_attempt1.json','all_tests_attempt1.json','all_tests_attempt2.json','all_tests_attempt3.json','synthetic_final_attempt1/','synthetic_volume/full/pre_timing_acceptance/'],
         fixes=['先生成streams前置产物','库导入后再禁用科学RNG，避免SciPy文档初始化误拒绝',
                '反号测试的mock同步覆盖描述族F=9；生产判决不变',
                '全规模冷重放测试的90秒墙钟等待改300秒以覆盖IO；正式39秒CPU和12600秒墙钟限额未变']))
    dump(OUT/'freeze_audit.json',dict(passed=True,test_counts=counts,total_checks=sum(counts.values()),
         full_volume_jobs=5377,all_source_hashes_current=True,formal_absent=True,usage=usage(),
         note='先前失败尝试保留；此处统计最终通过套件；首次只读全目录扫描中断CPU缺收据，不作0计费'))
    files=[p for p in OUT.rglob('*') if p.is_file()]
    files += list(ROOT.glob('Z4B_*.md'))+list(ROOT.glob('Z4B_*.py'))
    assert all(not p.is_symlink() for p in files)
    manifest=ROOT/'Z4B_FROZEN.sha256'
    scan_start=usage()
    with manifest.open('x') as f:
        for p in sorted(files):f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
        f.flush();os.fsync(f.fileno())
    manifest.chmod(0o444)
    scan_end=usage()
    dump(ROOT/'Z4B_FREEZE_USAGE.json',dict(manifest_sha256=sha(manifest),
         scan_cpu=scan_end['cpu']-scan_start['cpu'],scan_wall=scan_end['mono']-scan_start['mono'],
         entries=len(files),end=scan_end,note='本次生成清单的文件哈希扫描；不是独立冷缓存上界；外置收据避免自指'))
    print(dict(entries=len(files),manifest=sha(manifest),tests=sum(counts.values()),usage=usage()))

if __name__=='__main__':main()
