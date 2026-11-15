# 本地执行器与微型先导

解释器固定 `/home/liuyibo/venvs/theory/bin/python`。唯一额外包 `tiktoken==0.14.0`（已用 `uv pip install --python /home/liuyibo/venvs/theory/bin/python tiktoken` 安装）。HTTP 使用标准库，固定本机端点，不支持替代模型或任意URL。

```bash
/home/liuyibo/venvs/theory/bin/python -m unittest discover -s experiments/real -p 'test_*.py'
/home/liuyibo/venvs/theory/bin/python experiments/real/report.py
```

首次 mock 命令为 `.../python experiments/real/pilot.py --mode mock`；首次真实命令为 `.../python experiments/real/pilot.py --mode real`，仅从已设置的 SOLKEY 环境变量取密钥。输出目录用排他创建，本轮两个目录均已存在，因此再次调用会拒绝覆盖/重复执行。不得为重复真实实验删除证据目录。源码不打印密钥，不记录环境、鉴权头或HTTP错误正文。

`finish_tiny.py` 是本轮一次性管理修复：首个运行因服务端输出超限后，补完同批从未尝试的三项，禁止重跑；独占标记已存在，不能再次执行。它保持原提示、参数、种子、每次32调用和全局150调用限制。后续改动须在新数据前明确冻结新协议，不可在本批追加API调用。

- [PILOT_TINY.md](PILOT_TINY.md)：中文结果与是否放大。
- [DESIGN_TINY.md](DESIGN_TINY.md)：初始实施定义和限制。
- [SAMPLE_TRACES.md](SAMPLE_TRACES.md)：人工可读提示、回复、状态和恢复样例。
- [tiny_real/summary.json](tiny_real/summary.json)、[tiny_mock/summary.json](tiny_mock/summary.json)：每次结果与完整轨迹路径。
- [AUDIT.json](AUDIT.json)：离线计数、token、轨迹与提交重放校验。
- [TEST_RESULTS.txt](TEST_RESULTS.txt)：真实调用前的单元测试结果。

这里提供40次先导所需的任务生成、状态执行、恢复、模型客户端、调用硬停和轨迹底座，未执行40次，未实现/冒充完整统计假设检验或容量外推。微型结论不修改任何原预注册判据。
