# E 论文每 10 分钟检查

用户于 2026-11-07 授权。采用本机 systemd 用户定时器，向原 Codex 会话
投递检查与改进指令，沿用该会话的模型及上下文，不创建新 agent。
这是本机定时任务，不是 ChatGPT Scheduled 云端任务。

- `theoryscout-e-review.timer`：启用后每 10 分钟触发一次。
- `tick.py`：检查原会话队列；已有同一检查待处理时跳过，避免堆积。
- `prompt.txt`：检查、实质改进、历史保存与验证要求。
- `settings.json`：目标 thread、工作目录与 CLI 路径。
- `ticks.jsonl`：每次投递或跳过记录。

查看运行状态和下次时间：

```bash
systemctl --user status theoryscout-e-review.timer --no-pager
systemctl --user list-timers theoryscout-e-review.timer --all --no-pager
journalctl --user -u theoryscout-e-review.service -n 20 --no-pager
```

停止后续触发（不会中断当前正在处理的论文步骤）：

```bash
systemctl --user disable --now theoryscout-e-review.timer
```

恢复：

```bash
systemctl --user enable --now theoryscout-e-review.timer
```

需本机、用户 systemd 和 Codex daemon 保持运行；机器休眠或关机时无法检查。
每次触发投递任务，实际处理顺序由同一会话队列决定；不是并发改写论文。
停止时若已有一条检查进入会话队列，应按用户最新停止指示跳过该任务。
