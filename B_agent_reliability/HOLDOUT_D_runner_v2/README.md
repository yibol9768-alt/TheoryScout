# D 运行器 v2：AMEND2 离线交付

落实 `../AUDIT_HOLDOUT_D_AMEND1.md` 的 E1–E3。N=144、种子、协议、9000/12000 硬限、双精度权重、严格 `T>0.541` 及 G0–G4 均继承原冻结。旧目录和冻结件保持原样。本轮不含真实模型调用、模型端点连接或真实额度查询。

## 入口和真实模式

`run.py` 使用 `allow_abbrev=False`；`launch.sh` 先调用同一个解析器完成入口检查，再创建日志和独立会话。`--l/--li/--liv` 不再识别。真实 CLI、直接 `Engine`、真实 HTTP/额度客户端构造和发送均检查固定放行文件：

`HOLDOUT_D_runner_v2/HOLDOUT_D_LIVE_RELEASE.json`

**本轮不创建该文件，因此真实模式被拒。** 将来仍需用户明确授权、经人工核验的项目额度 provider 证据及这个指定文件，调用方还必须显式指定 `--live`（直接 Python 构造为 `live=True`）。不能用环境变量或其他路径代替文件。该文件由人工填写，字段严格为：`schema=HOLDOUT_D_AMEND2_LIVE_RELEASE_V1`、`explicit_user_authorization=true`、`provider_evidence`、`authorized_by`、带时区的 `authorized_at`、`runner_version`、`amend2_manifest_sha256`。后两项分别匹配代码/设计版本和本轮冻结清单 SHA256；文本字段不能为空。这些字段记录授权和核验证据，软件无法替代人工核实其真实性。

真实账本固定 `HOLDOUT_D_runner_v2/HOLDOUT_D_live`，mock 不得占用。真实模式不能使用 fixture、跳过冻结核验或使用 mock 调度/传输。真实传输还要求 Engine 的有效持久预留；裸调用 `HTTPTransport.send` 被拒。Python 发出模型请求前，在 `reserve` 层统一执行第三题及以后人工审阅门，不依赖外层调度器。

## 首两题人工审阅

首两题终局后，运行器生成 `HOLDOUT_D_SMOKE_REVIEW_TEMPLATE.json`，其中 `approved=false`、审阅人/时间为空、七项检查全部为 `false`。它是待填模板，不是放行凭据。仅人工实际审阅后才可填写并通过 `--smoke-approval` 提交；测试和重放不会生成已审凭据。

快照固定为首两题最后一个终局 `action` 的链 head，包含截至该点的**全部** reserve/response/action 及中间记录，不可任选旧前缀。凭据绑定快照序号、链 digest、完整快照字节 SHA256、两个 seed/c/status/终局事件 digest、运行器版本和本轮冻结清单 digest。人工时间不得早于终局快照或晚于当前时间。检查清单必须恰含 `evidence.CHECKS` 的七项且类型为 bool、值全 true；审阅方式必须为 manual。任意缺项、旧前缀、错误身份/终局/版本或快照均被拒。WAL 恢复时重新核验 `smoke_approved` 事件与当时完整前缀，之后每次第三题及以后的预留再核验。

## 离线评分

`score.py` 不构造 Engine、不写 WAL、不连接网络。它验证规范化原始字节、hash 链、请求正文/位置/重试序号、响应 hash、预算/并发、确定性动作与终局；不读可丢弃的 summary。它按已决推进重建每题两段 `[n,d,e,m2,s2]`，未决复查不进入任何最终机制分母。

输出五格已知计数、各题 c/终局、K/F/U 和 `[K,K+U]`、冻结校准区间、删失似然描述及 G0–G4。完整144题才输出 T；不完整时 T=null，禁止完成子集检验。mock/fixture 或工程/冻结不满足时，G0=false，G2/G3 不作确认判决，G4 写信息不足。完整 mock 的 T 仅用于工程核验。真实完整样本通过 G0/G1 后依次按原校准、严格主检验和10000次整题配对 bootstrap 判定；同索引同时用于两段/两项、共同有效重复、linear 分位数，门槛不变。

以下命令均从项目根目录运行，输出选择新的路径，避免覆盖冻结证据：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  python3 -B HOLDOUT_D_runner_v2/score.py <本项目内WAL路径>
```

将末行改为 `python3 -B HOLDOUT_D_runner_v2/score.py --c-replay` 可只读重放全部 C；改为 `python3 -B HOLDOUT_D_runner_v2/validate_all.py` 可在新时间戳目录运行全部回归；改为 `python3 -B HOLDOUT_D_AMEND2_validate.py` 可只读核验新旧冻结和旧文件完整性。所有数值线程≤2，运行器只有主线程和一个 worker。测试只允许旧 HTTP 集成的 fork 本地随机端口 mock，审计钩子拒绝18319和外部连接。

详见 `MOCK_TEST_REPORT.md`。当前没有有效人工冒烟审阅，也没有真实运行授权或真实 provider 核验；这些不能由离线通过替代。
