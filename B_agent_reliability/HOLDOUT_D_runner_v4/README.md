# D 运行器 v4：AMEND4 离线交付

依据 `../AUDIT_HOLDOUT_D_AMEND3.md` 最小修复 M1，继承 v3 的 R1、R2。N=144、种子、协议、9000成功/12000尝试硬限、五格评分、严格 T>0.541 及 G0–G4 保持冻结设计。本交付全程离线，不调用模型、不连接18319、不查询真实额度；未创建真实放行文件、人工已审凭据或真实账本。

## 生产与测试传输

`HTTPTransport` 永远是生产客户端，`test_mock_url` 仅保留为明确拒绝的旧参数。构造需要 `live=True`、固定人工放行文件和 SOLKEY；发送再次核验放行、live Engine 和持久预留。生产 URL 固定为协议端点，修改 testing 属性也不能跳过发送检查。mock Engine 拒绝生产客户端。

独立 `TestHTTPTransport` 只用于本地 HTTP mock，解析 URL 后严格要求 scheme=http、hostname=127.0.0.1、端口49152–65535，拒绝 userinfo、fragment、空白及范围外端口；18319 不在测试范围。构造和每次发送都校验，禁代理、禁重定向。测试 URL 字符串永远不会授予生产传输豁免。普通 mock 使用不建 socket 的 `MockServer`。

## 一次性发送与恢复

`Engine.claim_dispatch` 在全局 RLock 内校验门控、预留、active 和正文 hash，消费一次性发送权，并 fsync 写入 `dispatch_started(call,rid,body_sha256)`，成功后才可进入 Request/opener。第二次 send，无论顺序调用、两线程竞争或已 receive，均拒绝。fsync 失败也会在本进程保留消费状态并停止发送。

登记发送权之前重新检查 `pause_until`、`need_probe`、`probe_retry` 和额度百次边界。暂停时抛出 `DispatchPaused`，原预留仍在 pending/active，发送权未消费；调度器下次 `request` 复用同一 call、attempt、原始正文，不新增预算。暂停结束且合格重探测后才发送。底层直接调用者须先等待、调用 `quota(probe)`，再用原预留发送。额度探测只允许已登记发送的 active 请求和恢复后的未知预留排空；本进程明确未发送的预留不阻塞探测。恢复后仍一律保留未知预留并停止，不自动重发。

发送权不释放成功预算预留。发生进程崩溃或未知异常，恢复保留 pending、重建 dispatched，active 为空并停止为 unknown_inflight；不会推断未发送或复用旧权。服务错误结束当前预留后，须等待最后失败后300秒并重新探测额度，合法重试产生新的 call、attempt、reserve 和 dispatch_started，正文保持原样。live WAL 重放和评分均拒绝缺失/重复/错身份的 dispatch_started。

## 人工操作者流程与信任边界

真实模式检查固定的 `HOLDOUT_D_runner_v4/HOLDOUT_D_LIVE_RELEASE.json`，环境变量或其他文件不能替代。本交付不创建它。未来只有在用户另行明确授权且人工核验项目额度 provider 后，操作者才可亲手填写该文件。字段恰为：schema=`HOLDOUT_D_AMEND3_LIVE_RELEASE_V1`、explicit_user_authorization=true、provider_evidence、authorized_by、带时区 authorized_at、runner_version、amend3_manifest_sha256。为最小修改保留 AMEND3 schema 和字段名，最后一项绑定本版本最终 `HOLDOUT_D_AMEND4.sha256` 的字节 SHA256。真实账本固定在本版本的 `HOLDOUT_D_live`，不得通过换输出目录重置预算。

首两题终局后暂停。运行器仅生成 `HOLDOUT_D_SMOKE_REVIEW_TEMPLATE.json`：approved=false、reviewer/reviewed_at/reviewer_declaration 为空、七项检查全 false。**凭据只能由人工操作者填写。操作者将亲自审阅首两题原始记录后亲手写凭据。** 具体步骤：

1. 保留原始 WAL，从起点逐条读到模板 snapshot.wal_seq；核对 hash 链、快照字节 wal_sha256 和两题终局，审阅 reserve 的原始正文、response 的原始返回、所有 action 及 quota/guard 记录。
2. 逐项人工检查 prompt_contract、actual_error_writes、two_review_protocol、terminal_outcomes、model_usage_guards、quota_budget、wal_replay。只有实际完成对应审阅才勾 true。
3. 亲手填写 reviewer、带时区 reviewed_at，并在 reviewer_declaration 写入：`我已亲自审阅此快照绑定的首两题原始记录，并亲手填写本凭据。` 审阅完成才把 approved 改为 true，保存为本版本目录中的新凭据文件。不得把测试目录的 FORGED/SYNTHETIC 材料作为凭据。
4. 保持模板 snapshot 及 snapshot_digest 原值；后者为规范 JSON `protocol.dump(snapshot)` 的 UTF-8 SHA256。通过 `--smoke-approval` 提交该人工文件。软件绑定当前指定快照、版本、最终冻结 digest、身份/终局/时间、声明和七项严格 bool，并在恢复和每次第三题及以后发送时再次核验。

这是一条人工文件信任边界。软件校验声明和快照 digest，不证明审阅者是否真的执行审阅，也不鉴别人类身份。明确标记的伪造测试材料用于验证这一边界；其结构通过不能解释为取得人工授权。

## 离线验证与冻结

所有计算使用以下限制；数值线程设1，调度时总线程为主线程加一个worker，≤2：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -B HOLDOUT_D_runner_v4/validate_all.py
```

`amend4_checks/` 纳入审计新增12项（只映射运行器到v4），另有6项恢复验收；然后运行 v4 继承的48项、原版 v1/v2/v3 的22/39/48项，以及6组原攻击回归。测试仅在内存映射可写路径、服务器函数及 HTTPConnection 记录桩；保留原测试断言、urllib代理/重定向逻辑与fork退出语义。所有 socket连接、DNS、sendto、bind 均由审计钩子拒绝。输出只写 `amend4_checks/results/`。

输出为保留证据，入口拒绝覆盖既有结果。重跑须创建新的隔离输出并映射，不覆盖冻结日志。`regression_results/`、`regression_results_initial/` 是随 v3 复制的历史材料；本轮结果仅认 `amend4_checks/results/`。`score.py` 只读重建原始 WAL，统计规则不变；仅同步 M1 的发送边界与排空工程检查，mock/fixture 不取得真实 G0。

用同一资源前缀运行 `HOLDOUT_D_AMEND4_validate.py` 可只读核验五份冻结清单、所有旧件及新建路径范围。最终清单由 `HOLDOUT_D_AMEND4_freeze.py` 生成，详见 `MOCK_TEST_REPORT.md`。合成fixture绑定测试时的临时清单，原字节保存在 `HOLDOUT_D_AMEND4_TEST_MANIFEST.txt`；不能作为最终清单的真实放行证据。
