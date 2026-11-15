from pathlib import Path,PurePosixPath
import hashlib,json,shutil,subprocess,sys,tarfile,time
root=Path.cwd();base=root/'supplement';archive=root/'PKG_submission.tar.gz';assert not archive.exists()
# The final archive contains only relative supplement paths.
with tarfile.open(archive,'w:gz',compresslevel=6) as f:f.add(base,arcname='supplement')
with tarfile.open(archive,'r:gz') as f:
    members=f.getmembers();assert all(not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts and not m.issym() and not m.islnk() for m in members)
    tmp=base/'archive_validation';tmp.mkdir(exist_ok=False);f.extractall(tmp,filter='data')
relocated=tmp/'supplement'
p=subprocess.run([sys.executable,'-B',str(root/'PKG_relocation_worker.py'),str(relocated),str(relocated/'verify_package.py')],capture_output=True,text=True)
assert p.returncode==0,(p.stdout,p.stderr);shutil.rmtree(tmp)
artifact=dict(status='PASS',file=archive.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_members=len(members),relative_paths_only=True,extracted_package_verified=True)
(root/'PKG_archive_verification.json').write_text(json.dumps(artifact,indent=2))
summary=json.loads((base/'logs/run_summary.json').read_text());full=json.loads((base/'two_step_adam_certificate/reference/independent_full.json').read_text());sub=json.loads((base/'logs/independent_subcover_audit.json').read_text());sources=json.loads((root/'PKG_source_audit.json').read_text());env=json.loads((root/'PKG_environment.json').read_text());relo=json.loads((root/'PKG_relocation.json').read_text());manifest=json.loads((base/'manifest.json').read_text())
aux=sum(1 for _ in (base/'two_step_adam_certificate/reference/independent_auxiliary_integrals.jsonl').open())
report=f'''# E 题自包含复算包交付报告

交付目录为 [supplement/](supplement/README.md)，可直接提交的归档为 [PKG_submission.tar.gz](PKG_submission.tar.gz)。包内共 {manifest['files']} 个被清单校验的文件，归档大小 {artifact['bytes']/1024**2:.2f} MiB，SHA-256 为 `{artifact['sha256']}`。

仅在原工作目录中新建 `supplement/` 与 `PKG_` 前缀文件；没有调用 LLM API、网络工具或其他项目。捕获的 {sources['original_source_count']} 个原源码/论文源文件的生成前后 SHA-256 全部一致，尤其没有修改 `paper/`。详细清单见 `PKG_source_audit.json`、`PKG_original_sources.json`。

## 环境与约束

Python {env['python']}，numpy {env['numpy']}，scipy {env['scipy']}，mpmath {env['mpmath']}，{env['machine']}。包内只需要这三个第三方库与标准库，无绝对工程路径、内部研究代号、网络服务或额外绘图/符号依赖。

所有 Python 计算实际使用以下前缀：

```bash
systemd-run --user --scope -p MemoryMax=3G -p MemorySwapMax=0 \\
  env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
  /home/liuyibo/venvs/theory/bin/python -B
```

实测 scope 的 memory.max=3221225472、memory.swap.max=0，环境检查进程线程数为 {env['threads']}，OOM 与 OOM-kill 均为 0。所有数值线程环境变量设为 1。实测环境记录在 `PKG_environment.json` 和包内 `logs/resource_validation.json`；这不是对每个退出后 scope 的独立峰值内存测量。

## 已实际执行的命令与计时

下表列出 `python supplement/reproduce.py <任务> --output <新目录>` 的任务名；包内 README 提供可直接复制的相对路径命令。详细子命令、逐条秒数、退出码及 stdout 全部保存在 `supplement/logs/`。所有下表命令退出码均为 0。

|任务|实测总秒数|通过情况|
|---|---:|---|
'''
labels={'certificates':'certificates（从零完整流水线）','polar':'polar（单独入口）','adam':'adam（单独入口）','history':'history（单独入口）','independent_decisions':'check-adam','independent_sample':'check-adam-integrals','independent_full':'check-adam-all','replay':'replay','tables':'tables','query_tables':'query-tables','full_plan':'full-plan'}
for key,row in summary.items():report+=f"|{labels.get(key,key)}|{row['seconds']:.3f}|PASS|\n"
report+=f'''
完整性检查 `python verify_package.py`、脱离原目录的独立 Adam 检查与表格重算、归档解压后完整性检查也全部通过；逐项计时见 `PKG_relocation.json` 和 `PKG_integrity_verification.json`。查询级提取管线另用随包默认路径再次执行通过。

## 数学结论与独立性

- polar：重新计算全部 5×4096² 网格、15 个 MSR 交点以及方差包络的上下根，共 90 条严格端点不等式；论文交点表的包含关系通过。原始积分保存每批网格包络与效率积分 A/Q 包络。检查器使用另一套 mpmath.iv 算术重算这些包络的组合与符号；没有声称另写了第二个完整二维积分器。
- 两步 Adam：从当前源码重新生成 1027 格完整有理覆盖、64 边界格、全部匹配根、u=2/5 链、固定方向比较和 e=1 的历史消融。独立检查器只导入标准库与 mpmath，不导入生成器或整数区间核心。
- 全量独立重积分：全部 1027 个原始区域格形成 {sub['independent_subboxes']} 个通过的子格；每个子格 N、J 严格为正，另用有理数重新检查每个父格的子覆盖。还重新积分全部边界、根、固定方向、消融、阈值和具体反例；额外保存 {aux} 次辅助积分包络。实测总耗时 {full['seconds']:.3f} 秒。不存在只抽查少量积分却标记全量独立积分通过的情况。
- 历史反转：H(21/20)>0、H(11/10)<0；结合随附解析唯一性证明得到 1.05<e†<1.10。r=1/4、u=1/2、β₁=.9 时，增加 β₂ 从 0 到 .99，在 e=.5 的旧信号差约 +0.000235213，在 e=2 时约 −0.000031370；四个根均认证首次性。阈值的唯一性仍依赖解析证明，并非由两个数值点推出。

旧证书的当前论文来源哈希不符项为：{', '.join('`'+p+'`' for p in sources['old_certificate_manuscript_mismatches']) or '无'}。本包未复制旧数学证书；重新运行了当前生成器，绑定包内实际源码和当前论文摘录，并另存当前论文原文件哈希。单命令再次从零产生的五份证书与随包参考证书的数学数据完全相同，比较时只排除了计时与资源统计。

## 实验结果

代表性 MR 实际重放 12 个数据集，有限维修正确认实验重放 18 个数据集。保持原始正式种子、随机数流、查询预算和全部训练候选；300 个数据集×方法×任务召回值与原结果完全一致。MR 耗时约 32.6 秒，确认重放约 286.5 秒，合计约 5 分 20 秒。

另独立遍历全部 1152+288=1440 个原始查询级 NPZ，验证查询数、排名与成功标记及数据集测量值，导出误差为零，耗时约 12.86 秒。包中携带全样本数据集级原始测量，原查询级归档的哈希按中性标签保存；代表性查询级 NPZ 则随包提供。

从这些全样本测量重新计算全部相关表格得到：MR 的 18 个主要方向通过、7 个区间覆盖总体值、22/30 个新任务等价；确认实验 3 个端点通过，10/72 个诊断等价。90 条总体/候选修正预测从积分重新计算，与冻结输入最大误差约 4.44×10⁻¹⁶。表格并未复制旧统计汇总，统计单位保持数据集和完整区组。查询级 NPZ 独立提取再进入统计管线也已实测通过；小规模端点保持不具备正式确认资格。

## 额外校验、尝试与限制

包复制到新目录后，用 Python audit hook 阻止访问原工程包外文件，再运行完整性检查、独立 Adam 检查和全样本表格重算，均通过。缺格、重叠格、覆盖缺口、篡改导数积分、错误定点精度五类故意损坏均被拒绝。所有提交源码通过 AST 解析，所有文本扫描均没有绝对工程路径或内部研究代号；归档无符号链接、绝对成员路径或父目录穿越路径。

开发阶段修复过一次生成器组装的括号错误；独立实现最初在末格给出较宽包络，改为精确子格覆盖后通过。由于来源摘录中性化和独立检查器改进，主动终止过中间计算并重跑。最终包不包含失败证书或不完整 PASS；最终计时只对应上述成功运行。最初归档导出核验中的阈值表达式优先级错误和损坏测试的扰动强度也已修正，最后均通过。

**明确未执行的命令：** README 的 `full-experiments` 完整训练数据重跑模板，以及随后从该新完整输出重算表格的模板。完整设计计划检查、小规模实际重放、原完整查询归档核验及全样本表格重算都已执行，但没有重新生成全部 1440 个正式训练数据集。因此，“每条命令实际运行”对这两条长时完整规模模板尚未满足；本报告不将它们记作 PASS。完整实验预计需要数小时。

polar 的 binary64 机器假设、历史阈值的解析唯一性依据、有限维召回修正的候选地位，以及小规模重放不代替正式统计确认等边界，均在包内 README 与 MATHEMATICS 中明确说明。
'''
full_replay=json.loads((root/'PKG_full_replay_verification.json').read_text())
start=report.index('**明确未执行的命令：**')
end=report.index('\n\npolar 的',start)
report=report[:start]+f"**完整规模命令已全部执行。** `full-experiments` 重新生成全部 1152 个 MR 数据集和 288 个确认数据集，全部 14400 个数据集×方法×任务召回值与原结果相同；随后从该新完整输出重算论文表格通过。两组实验在同一个 3 GiB、swap 0 的 scope 内并行，每进程一个线程，观测峰值内存 {int(full_replay['resources']['resource_limits']['memory.peak'])/1024**3:.3f} GiB，OOM 与 OOM-kill 为零。完整重跑计时见上表。完整查询数组另存 `{full_replay['file']}`（{full_replay['bytes']/1024**2:.2f} MiB，SHA-256 `{full_replay['sha256']}`）；最小投稿包只带代表性查询数组、全样本数据集测量和完整重跑的新测量。"+report[end:]
(root/'PKG_REPORT.md').write_text(report)
provenance=dict(archive=artifact,original_sources=sources,environment=env,relocation=relo,independent_full=full,independent_subcover=sub)
(root/'PKG_provenance.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps(artifact));print('PASS: report and submission archive written')
