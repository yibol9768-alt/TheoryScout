"""Seal a compact submission archive while preserving all local evidence."""
from pathlib import Path, PurePosixPath
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time

root = Path.cwd()
base = root/'supplement'
verification = json.loads((root/'PKG_continuation_verification.json').read_text())
assert verification['status'] == 'PASS'
folder = base/'full_experiments_continuation'
logs = base/'logs'
environment = json.loads((root/'PKG_resume_environment.json').read_text())
assert all(environment[key] == json.loads((base/'versions.json').read_text())[key] for key in ['python', 'numpy', 'scipy', 'mpmath'])
shutil.copy2(root/'PKG_resume_environment.json', logs/'continuation_environment.json')
reference = base/'experiments_replay/reference/full_replay'
reference.mkdir(exist_ok=False)
for design in ['matched', 'confirmation']:
    (reference/design).mkdir()
    for name in ['measurements.csv', 'result.json', 'population_predictions.json', 'archive_provenance.json']:
        shutil.copy2(folder/design/name, reference/design/name)
shutil.copytree(folder/'tables', reference/'tables')
shutil.copy2(root/'PKG_continuation_verification.json', logs/'full_replay_continuation.json')
shutil.copy2(root/'PKG_continuation_resources.json', logs/'continuation_resources.json')
for index, design in enumerate(['matched', 'confirmation'], 1):
    shutil.copy2(folder/f'command_{index}.log', logs/('continued_'+design+'.log'))
for path in (folder/'independent_validation').glob('*.log'):
    shutil.copy2(path, logs/('continued_'+path.name))
continuation = json.loads((folder/'timings.json').read_text())
assert len(continuation) == 2 and all(row['exit_code'] == 0 for row in continuation)
summary = json.loads((logs/'run_summary.json').read_text())
summary['full_experiments_continuation'] = dict(status='PASS', seconds=max(row['seconds'] for row in continuation), parallel=True, commands=continuation)
summary['full_query_validation_and_tables'] = dict(status='PASS', seconds=sum(row['seconds'] for row in verification['commands']), commands=verification['commands'])
q = json.loads((base/'query_pipeline_validation/timings.json').read_text())
assert all(row['exit_code'] == 0 for row in q)
summary['query_tables_default'] = dict(status='PASS', seconds=sum(row['seconds'] for row in q), commands=q)
(logs/'continued_run_summary.json').write_text(json.dumps(summary, indent=2))
(logs/'continuation_timings.json').write_text(json.dumps(continuation, indent=2))

# Distinguish externally changed manuscript snapshots from unchanged generators.
initial = json.loads((root/'PKG_original_sources.json').read_text())
baseline = json.loads((root/'PKG_resume_source_baseline.json').read_text())
current = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in initial}
generators = [name for name in initial if not name.startswith('paper/')]
assert all(initial[name] == baseline[name] == current[name] for name in generators)
audit = dict(status='PASS', numerical_sources=len(generators), all_numerical_sources_unchanged=True,
    manuscript_differences_from_initial=[name for name in initial if name.startswith('paper/') and initial[name] != current[name]],
    manuscript_differences_during_continuation=[name for name in initial if name.startswith('paper/') and baseline[name] != current[name]],
    original=initial, continuation_baseline=baseline, current=current,
    manuscript_files_written_by_this_task=False)
(root/'PKG_resume_source_audit.json').write_text(json.dumps(audit, indent=2))
(logs/'source_snapshot_status.json').write_text(json.dumps(dict(status='PASS', numerical_sources=len(generators),
    all_numerical_sources_unchanged=True, manuscript_snapshot_frozen=True,
    manuscript_files_different_from_initial=len(audit['manuscript_differences_from_initial']),
    manuscript_files_written_by_this_task=False), indent=2))

readme = base/'README.md'
shutil.copy2(readme, root/'PKG_README_before_continuation.md')
text = readme.read_text().replace('本次读取的论文公式/结论摘录', '证书生成时读取的论文公式/结论快照')
text = text.replace('当前论文源文件的原始内容哈希', '该论文快照的原始内容哈希')
text = text.replace('|约 30 秒|', '|约 30–45 秒|')
text = text.replace('它先执行 MR 的 1152 个数据集，再执行确认实验的 288 个数据集；单线程预计数小时。',
    '它在同一个资源限制中并行执行 MR 的 1152 个数据集和确认实验的 288 个数据集，每个工作进程一个数值线程；预计约 1–2 小时。')
marker = '**本次未执行这两条完整规模长时计算命令。**'
assert marker in text
text = text[:text.index(marker)]+'''**本次完整设计通过断点续算完成。** 首次完整重跑因机器重启中断，没有可用退出码，不能记作成功的单次执行。重启前完整写出的查询归档经过逐数组检查后直接复用；只补算缺失的数据集，没有重做已验证的数据集，也没有删除中断证据。两个恢复命令均正常退出，最终覆盖 MR 的全部 1152 个及确认实验的全部 288 个数据集。

恢复工具为 `experiments_replay/resume_retrieval.py`，参数为 `--design`、`--existing`（旧查询目录）与 `--output`（必须不存在的新目录）。工具不覆盖旧归档；每个复用归档都检查随机查询索引、预算、排名、成功标记、数值有限性及全部召回，并记录 SHA-256。它只用于已有断点的恢复；从零复算仍使用上面的完整重跑命令。

从全部 1440 个查询归档再次独立提取 14400 个数据集×方法×任务测量值，与恢复生成的测量逐项一致；随后分别重算论文表格，两套表格文件逐字节相同。恢复耗时、命令和资源记录见 `VALIDATION.md`、`logs/continued_run_summary.json`、`logs/full_replay_continuation.json`。全样本新测量与表格随包附在 `experiments_replay/reference/full_replay/`。

本地的 `full_experiments_recomputation/`、`full_experiments_continuation/` 和 `query_pipeline_validation/` 保留运行证据。为控制投稿体积，投稿归档不包含这些可选运行目录；所有默认输入、最小生成器、原始数学包络、代表性查询归档、全样本测量与验证日志均包含在归档中。清单只核对投稿文件，运行证据不会被删除。

论文摘录保持证书生成时的快照。续作期间检测到原论文文件另有变化，本包没有修改这些原文件，也不据此声称已经核验后续新增的解析定理。有限维修正实验仍按其原始设计和历史解释重放。
'''
readme.write_text(text)
with (base/'MATHEMATICS.md').open('a') as handle:
    handle.write('\n续作说明：上面的候选预测措辞对应原实验设计与证书生成时的论文快照；本包复算该实验及统计表格，不评判此后新增的解析证明。原论文未由本任务改动。\n')
shutil.copy2(base/'VALIDATION.md', root/'PKG_VALIDATION_before_continuation.md')
validation_text = (base/'VALIDATION.md').read_text()
validation_text = validation_text.replace('# 实际验证记录\n', '# 实际验证记录\n\n首节保留重启前的历史记录；最新完整规模结果见末尾“重启后的续算与最终验证”。\n', 1)
(base/'VALIDATION.md').write_text(validation_text)
with (base/'VALIDATION.md').open('a') as handle:
    handle.write('\n## 重启后的续算与最终验证\n\n此前的数学证书、独立全量重积分和代表性重放保持原结果，没有重复执行。完整规模训练断点采用新恢复工具完成，旧查询归档和日志全部保留。首次完整重跑被重启中断，没有退出码，不标记为单次命令 PASS。\n\n|续算任务|复用数据集|新算数据集|秒数|结果|\n|---|---:|---:|---:|---|\n')
    for design in ['matched', 'confirmation']:
        row = verification[design]
        handle.write(f"|{design}|{row['reused_datasets']}|{row['generated_datasets']}|{row['seconds']:.3f}|PASS|\n")
    memory = verification['resources']['resource_limits']
    handle.write(f"\n全部 1440 个查询归档再次独立提取测量并比较；14400 个测量的提取误差为零，所有召回与原结果相等。两条从新完整测量重算表格的命令均通过，表格文件完全相同。续算 scope 峰值内存 {int(memory['memory.peak'])/1024**3:.3f} GiB，swap 上限 0，每进程最大线程数 {verification['resources']['max_threads_per_process']}，OOM 与 OOM-kill 为零。详情见 `logs/continued_run_summary.json`。\n")

# Omit optional local evidence from the small, self-contained submission artifact.
excluded = {'full_experiments_recomputation', 'full_experiments_continuation',
    'query_pipeline_validation', 'release_validation', 'archive_validation_resume'}


def files():
    return sorted(path for path in base.rglob('*') if path.is_file()
        and path.relative_to(base).parts[0] not in excluded and path.name != 'manifest.json')


def manifest():
    paths = files()
    data = dict(format=1, files=len(paths), optional_local_evidence=sorted(excluded),
        sha256={str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (base/'manifest.json').write_text(json.dumps(data, indent=2)+'\n')
    return paths


parsed = 0
for path in files():
    assert not path.is_symlink()
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(path.relative_to(base)))
        parsed += 1
    if path.suffix in ['.py', '.md', '.tex', '.json', '.jsonl', '.csv', '.log', '.txt']:
        assert not re.search(r'/home/|/mnt/|x3_checks|y2_checks|z3_checks|experiments/x1|y3d|ICML_REVIEW|X1_PREREG', path.read_text()), str(path)
(logs/'continued_static_audit.json').write_text(json.dumps(dict(status='PASS', python_files_parsed=parsed,
    no_absolute_project_paths=True, no_internal_project_codes=True, no_symlinks=True), indent=2))
paths = manifest()
relocated = base/'release_validation/supplement'
relocated.mkdir(parents=True, exist_ok=False)
for path in paths+[base/'manifest.json']:
    output = relocated/path.relative_to(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output)
checks = [
    ('integrity', 'verify_package.py', []),
    ('independent_adam', 'two_step_adam_certificate/check_adam.py', ['--certificate-dir', str(relocated/'two_step_adam_certificate/reference'), '--output', str(relocated/'relocated_adam.json')]),
    ('polar', 'polar_certificate/check_polar.py', ['--certificate', str(relocated/'polar_certificate/reference/polar_variance.json'), '--output', str(relocated/'relocated_polar.json')]),
    ('tables', 'experiments_replay/recompute_tables.py', ['--output', str(relocated/'relocated_tables')]),
    ('matched_queries', 'experiments_replay/measurements_from_queries.py', ['--input', str(relocated/'experiments_replay/reference/small_replay/matched'), '--output', str(relocated/'relocated_matched_measurements.csv')]),
    ('confirmation_queries', 'experiments_replay/measurements_from_queries.py', ['--input', str(relocated/'experiments_replay/reference/small_replay/confirmation'), '--output', str(relocated/'relocated_confirmation_measurements.csv')]),
    ('query_tables', 'experiments_replay/recompute_tables.py', ['--matched-measurements', str(relocated/'relocated_matched_measurements.csv'), '--confirmation-measurements', str(relocated/'relocated_confirmation_measurements.csv'), '--output', str(relocated/'relocated_query_tables')])]
relocation = []
for label, script, args in checks:
    start = time.monotonic()
    proc = subprocess.run([sys.executable, '-B', str(root/'PKG_relocation_worker.py'), str(relocated), str(relocated/script), *args], capture_output=True, text=True)
    (root/('PKG_resume_relocated_'+label+'.log')).write_text(proc.stdout+proc.stderr)
    assert proc.returncode == 0, (label, proc.stdout, proc.stderr)
    relocation.append(dict(check=label, status='PASS', seconds=time.monotonic()-start, original_project_accesses=0))
(root/'PKG_resume_relocation.json').write_text(json.dumps(relocation, indent=2))
(logs/'continued_relocation_validation.json').write_text(json.dumps(relocation, indent=2))
with (base/'VALIDATION.md').open('a') as handle:
    handle.write('\n投稿文件另复制到新目录，以 audit hook 禁止读取原工程包外文件；完整性、独立 Adam 判号、polar 证书检查、全样本表格及默认查询提取管线全部通过。\n')
paths = manifest()
archive = root/'PKG_submission.tar.gz'
assert not archive.exists()
start = time.monotonic()
with tarfile.open(archive, 'w:gz', compresslevel=6) as handle:
    for path in paths+[base/'manifest.json']:
        handle.add(path, arcname='supplement/'+str(path.relative_to(base)), recursive=False)
extract = base/'archive_validation_resume'
extract.mkdir(exist_ok=False)
with tarfile.open(archive, 'r:gz') as handle:
    members = handle.getmembers()
    assert all(not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts and not m.issym() and not m.islnk() for m in members)
    handle.extractall(extract, filter='data')
proc = subprocess.run([sys.executable, '-B', str(root/'PKG_relocation_worker.py'), str(extract/'supplement'), str(extract/'supplement/verify_package.py')], capture_output=True, text=True)
assert proc.returncode == 0, proc.stdout+proc.stderr
artifact = dict(status='PASS', file=archive.name, bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=len(members),
    extracted_integrity_check=proc.stdout.strip(), seconds=time.monotonic()-start,
    optional_local_evidence_preserved=True)
(root/'PKG_archive_verification.json').write_text(json.dumps(artifact, indent=2))

report = f'''# 自包含复算包续作报告

交付：[supplement/README.md](supplement/README.md)；投稿归档：[PKG_submission.tar.gz](PKG_submission.tar.gz)，{artifact['bytes']/1024**2:.2f} MiB，{len(paths)} 个清单文件。
归档 SHA-256：`{artifact['sha256']}`。

本次从已存在证据续作，没有重算已经通过的数学证书、全量独立积分或代表性实验，没有删除已有证据。所有新文件与续写均位于 `supplement/` 或 `PKG_` 前缀文件；未调用 LLM API，也未读写其他项目。

Python 3.12.3，numpy 2.5.4，scipy 1.18.1，mpmath 1.3.0。所有计算使用指定解释器，在 `systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0` 中执行；数值线程环境均为 1。完整续算使用两个单线程数值进程，scope 内存峰值 {int(memory['memory.peak'])/1024**3:.3f} GiB，OOM 与 OOM-kill 为零。

## 已执行命令与耗时

以下任务命令为 `python reproduce.py <任务> --output <新相对目录>`；每条实际子命令、退出码和计时均保留在 `supplement/logs/run_summary.json` 与 `continued_run_summary.json`，不以预计耗时替代实测。

|任务|实测秒数|结果与来源|
|---|---:|---|
'''
labels = dict(independent_decisions='check-adam', independent_sample='check-adam-integrals', independent_full='check-adam-all', query_tables='query-tables', full_plan='full-plan')
for name, row in summary.items():
    report += f"|{labels.get(name, name)}|{row['seconds']:.3f}|PASS；{'重启前已验证，保留原证据' if name in json.loads((logs/'run_summary.json').read_text()) else '续作实际执行'}|\n"
report += '''
数学复算包括 polar 的 15 个 MSR 交点、90 条端点不等式、两步 Adam 的 1027 格精确覆盖与全部判号、64 个边界格、首次根与 u=2/5 链，以及历史阈值 1.05<e†<1.10 与两个具体反例。独立检查器不导入生成器或整数区间核心，以 Fraction 检查覆盖、mpmath.iv 重算判号；全量独立重积分已有 704.578 秒 PASS 记录，覆盖全部 1027 个父格及 1129 个精确子格，并复核边界、根、消融和反例。polar 检查器复核原始包络组合与判号；没有声称另外实现完整二维积分器。

代表性重放包含 MR 12 个、确认实验 18 个数据集，实测 319.112 秒；300 个召回测量与原结果一致。统计表格从数据集级原始测量重算，确认端点保持原完整区组设计。

## 中断恢复与完整规模结果

首次 `full-experiments` 命令因机器重启中断，没有退出码，不将该次执行标记为 PASS。其日志与查询文件保留在 `supplement/full_experiments_recomputation/`。恢复只接收逐数组校验通过的完整归档，补算缺失数据集，正常完成两组完整设计。

实际恢复命令（在 supplement 目录运行，前加上述资源限制与指定解释器）：

```bash
python experiments_replay/resume_retrieval.py --design matched --existing full_experiments_recomputation/matched --output full_experiments_continuation/matched
python experiments_replay/resume_retrieval.py --design confirmation --existing full_experiments_recomputation/confirmation --output full_experiments_continuation/confirmation
```

|设计|复用数据集|新算数据集|总数|续算秒数|结果|
|---|---:|---:|---:|---:|---|
'''
for design in ['matched', 'confirmation']:
    row = verification[design]
    report += f"|{design}|{row['reused_datasets']}|{row['generated_datasets']}|{row['datasets']}|{row['seconds']:.3f}|PASS|\n"
report += f'''
重启前日志仅保存到最后一个完整数据集的累计秒数，无法恢复被中断命令的精确终止时刻；续算秒数不包含停机时间，也不冒充从零运行的总耗时。

全部 1440 个查询归档独立提取为 14400 个测量，与恢复程序输出的全部召回、D/V 及方差协方差逐项一致。与原数据相比，所有召回误差为零，确认实验测量最大差为 {verification['confirmation']['original_measurement_max_error']:.3g}。再分别用两套新测量重算论文表格，全部文件逐字节相同：18 个主要方向、7 个总体值覆盖、22/30 个新任务等价、3 个确认端点和 10/72 个诊断等价均复现。独立提取与表格命令详见 `PKG_continuation_verification.json`。

## 自包含性、证据与范围

投稿文件复制到新目录后，用 audit hook 禁止读取原工程包外文件；完整性、Adam 独立判号、polar 包络检查、全样本表格和默认查询级管线全部通过。归档再次解压后完整性检查通过；归档成员全部为安全的相对路径，无符号链接。源码 AST、绝对工程路径和内部研究代号扫描通过。详情见 `PKG_resume_relocation.json` 与 `PKG_archive_verification.json`。

原始大规模查询、恢复新查询及此前日志全部保留。最小投稿归档省略可选本地运行目录，包含所有默认输入、原始数学包络、代表性查询级 NPZ、全样本旧/新测量与日志；不依赖被省略目录。

{len(generators)} 个原数值源码相对上次快照和续作开始均未改变。本任务没有写入 paper/；检测到 {len(audit['manuscript_differences_from_initial'])} 个论文文件相对上次生成快照变化，不能将整个论文目录宣称为“哈希全部未变”。包内摘录和证书保持已验证快照，未为外部后续修改重做已验证工作。变更清单与三个时点哈希见 `PKG_resume_source_audit.json`。

历史阈值唯一性依赖随附解析证明；有限目标转折不由小目标阈值外推。polar 仍依赖已声明的 binary64 算术假设。有限维修正统计按原实验历史解释重放，本包不新增其解析证明结论。小规模重放不代替完整统计确认。
'''
(root/'PKG_REPORT.md').write_text(report)
print(json.dumps(artifact))
print('PASS: continuation report and submission archive completed; all previous evidence retained')
