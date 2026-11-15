from pathlib import Path
p=Path('PKG_finish.py');s=p.read_text().replace("labels[key]","labels.get(key,key)")
marker="(root/'PKG_REPORT.md').write_text(report)"
insert='''full_replay=json.loads((root/'PKG_full_replay_verification.json').read_text())
start=report.index('**明确未执行的命令：**')
end=report.index('\\n\\npolar 的',start)
report=report[:start]+f"**完整规模命令已全部执行。** `full-experiments` 重新生成全部 1152 个 MR 数据集和 288 个确认数据集，全部 14400 个数据集×方法×任务召回值与原结果相同；随后从该新完整输出重算论文表格通过。两组实验在同一个 3 GiB、swap 0 的 scope 内并行，每进程一个线程，观测峰值内存 {int(full_replay['resources']['resource_limits']['memory.peak'])/1024**3:.3f} GiB，OOM 与 OOM-kill 为零。完整重跑计时见上表。完整查询数组另存 `{full_replay['file']}`（{full_replay['bytes']/1024**2:.2f} MiB，SHA-256 `{full_replay['sha256']}`）；最小投稿包只带代表性查询数组、全样本数据集测量和完整重跑的新测量。"+report[end:]
'''
assert marker in s;s=s.replace(marker,insert+marker);p.write_text(s)
