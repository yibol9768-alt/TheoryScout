# D 运行器 v5：AMEND5 离线交付

v5 完整复制 v4 后做最小修改：新增固定命令额度提供者与 CLI 开关；沿用 M1 发送/恢复语义、人工门、预算和评分。N=144、种子、协议、9000成功/12000尝试、五格评分、严格 T>0.541 和 G0–G4 不变。本交付全程离线，未调用模型服务、未执行真实 `plus_quota.py`，未创建真实放行文件或真实账本。

## 命令额度与 CLI

项目实际额度来源沿用 `../experiments/real/holdout_c.py` 的 `quota_values`：

```text
python3 /home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py
```

`CommandQuota(live=True)` 只在显式 live 且固定放行文件有效时可构造，每次执行前再次核验放行。命令 argv 固定为上述两项，不接受自定义命令/路径/参数，不使用 shell。`subprocess.run` 使用 `capture_output=True, text=True, timeout=50, check=True`。stdout 必须恰好包含两个以空白分隔的数，顺序为“五小时已用百分比 周已用百分比”；每个数须在闭区间 0–100，NaN/Inf 也会拒绝。返回 `five_hour=a/100, week=b/100, observed_unix=time.time()`，时间为成功解析后的当前 Unix 时间。

放行条件仍为 `five_hour < .80` 且 `week < .90`，即五小时<80%、周<90%；等于阈值也暂停。解析失败、超时、非零退出、越界和其他命令查询异常统一进入现有 Engine 的 `query_failed` 分支，记录失败且暂停 300 秒，到期重新查询，不复用旧值。`checked` 与 300 秒期限从查询结束时起算，50 秒命令超时不会缩短等待。

`run.py` 新增无参数布尔开关 `--quota-command`，与 `--quota-url URL` 互斥。前者要求 `--live`，且不能携带命令字符串；后者保留原 HTTP JSON 接口。live 必须选择其中一个提供者，未配置就拒绝。默认 mock 保持内存额度零值，不构造命令提供者。

以下仅说明未来获得有效人工放行后的 CLI 用法，**本次未运行**：

```text
HOLDOUT_D_runner_v5/launch.sh --live --quota-command
```

## 生产与测试传输

`HTTPTransport` 保持生产客户端合同：显式 live、固定放行文件和 SOLKEY，发送再次核验放行、live Engine 及持久预留；端点固定。独立 `TestHTTPTransport` 仅接受 HTTP、127.0.0.1、49152–65535 端口，构造/发送均校验，禁代理及重定向。普通 mock 使用不建 socket 的 `MockServer`。

`claim_dispatch` 在全局 RLock 内校验门、预留、active 和正文 hash，在 Request/opener 之前消费一次性发送权并 fsync `dispatch_started`。发送前复查最新暂停、探测及百次边界。暂停保留原预留、call、attempt、正文和预算，合格新探测后才发送；崩溃/未知异常保留 unknown_inflight，不自动重发。已发送失败请求等待最后失败后 300 秒及重新探测，重试产生新预留。相关 v4 实现不作重构。

## 人工放行与首两题审阅

固定放行文件为 `HOLDOUT_D_runner_v5/HOLDOUT_D_LIVE_RELEASE.json`，本交付不创建。不能用环境变量或其他路径替代。schema 字段保持：`schema=HOLDOUT_D_AMEND3_LIVE_RELEASE_V1`、`explicit_user_authorization=true`、`provider_evidence`、`authorized_by`、带时区 `authorized_at`、`runner_version`、`amend3_manifest_sha256`。**字段名保留兼容，但 v5 的 `amend3_manifest_sha256` 必须等于最终 `../HOLDOUT_D_AMEND5.sha256` 文件字节的 SHA256；不是 AMEND3/AMEND4，也不是工具脚本的 hash。** 版本 hash 纳入 v5 源码。真实账本固定在 `HOLDOUT_D_runner_v5/HOLDOUT_D_live`，不可换目录重置预算。

首两题终局后继续暂停；只生成 approved=false、身份/声明空白、七项检查全 false 的 `HOLDOUT_D_SMOKE_REVIEW_TEMPLATE.json`。凭据由人工亲自审阅并填写：

1. 从起点读到模板 snapshot.wal_seq，核对 WAL hash 链、快照字节 wal_sha256、两题终局、原始请求/返回、action 和 quota/guard。
2. 人工核对 prompt_contract、actual_error_writes、two_review_protocol、terminal_outcomes、model_usage_guards、quota_budget、wal_replay，完成才勾 true。
3. 亲手填写 reviewer、带时区 reviewed_at 和声明 `我已亲自审阅此快照绑定的首两题原始记录，并亲手填写本凭据。`，完成才把 approved 改 true；保留 snapshot/snapshot_digest，另存本版本目录后交给 `--smoke-approval`。

软件继续在恢复和后续发送时核验快照、版本、最终 AMEND5 清单、声明及七项严格 bool。人工文件是信任边界，软件不证明实际审阅行为。测试的 FORGED/SYNTHETIC 文件不能作为授权或真实 D 样本。

## 离线验收与冻结

所有计算在以下资源约束内执行，数值线程1，总线程≤2：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -B HOLDOUT_D_runner_v5/validate_all.py
```

新入口输出仅写 `amend5_checks/results/`，独占创建日志，拒绝覆盖已有结果；重跑须隔离映射新输出。命令测试在 PATH 前放入独立假的 `python3`，它验证固定 argv 后只输出合成数/退出/等待，不打开或运行真实脚本。实际 50 秒超时由该替身产生；300 秒暂停由虚拟时钟验证。审计钩子拒绝真实额度执行、全部 socket连接/DNS/sendto/bind 以及结果目录外写入。HTTP 仅用本地记录/响应桩，保留原断言、代理/重定向逻辑与 fork 退出语义。

回归包含命令额度专项、v4 的12项边界和6项恢复、v5继承48项、只读原版 v4/v1/v2/v3 的48/22/39/48项，以及六组入口/逃逸/URL/重复发送/人工材料/独立评分回归。映射只针对被测版本和隔离输出。v4 的边界与恢复断言在 v5 上原样运行，原版 v4 的48项也单独重跑。

`amend4_checks/results/`、`regression_results/`、`regression_results_initial/` 及旧 TEST_MANIFEST 都是从 v4 原样复制的历史证据；本次结果只认 `amend5_checks/results/`。`HOLDOUT_D_AMEND5_TEST_MANIFEST.txt` 保留合成测试时的临时7项源码清单；合成fixture不是最终清单的真实放行证据。

`../HOLDOUT_D_AMEND5_freeze.py` 生成最终 `../HOLDOUT_D_AMEND5.sha256`，覆盖整个 v5 与 AMEND5 前缀交付（清单不含自身）。同一资源前缀运行 `../HOLDOUT_D_AMEND5_validate.py` 可只读核验六份冻结清单、旧文件基线及新增路径范围。实际结果与逐文件 diff 见 `MOCK_TEST_REPORT.md`、`../HOLDOUT_D_AMEND5_DIFF.patch` 和 `../HOLDOUT_D_AMEND5_DIFF.json`。v1–v4 和既有冻结件保持原字节。
