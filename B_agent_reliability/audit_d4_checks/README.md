# D4 独立离线审计证据

全部新文件位于本目录，主报告位于 `../AUDIT_HOLDOUT_D_AMEND3.md`。不创建生产放行文件，不运行真实冒烟，不提交人工已审凭据。

执行入口（输出使用独占创建，复跑须另用本目录下全新输出路径）：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/audit_d4_checks" \
  python3 -B audit_d4_checks/run_all.py
```

`sitecustomize.py` 对所有测试及其子进程施加审计钩子：拒绝 Internet 连接、DNS、sendto、bind，拒绝写入本目录之外的文件。`harness.py` 把 `HTTPConnection.request/getresponse` 换成本地记录桩，保留 urllib 的代理和重定向处理。旧 HTTP 集成测试的 fork 服务器函数在内存中替换为只返回端口的桩；原状态码断言及子进程退出断言保留，不监听端口。其余发送测试使用原内存 MockServer 或 opener 记录桩。

v1/v2/v3 套件仅映射 test BASE 和 Engine 可写/调和路径到新目录；不改预算、协议、锁、发送门控、WAL、评分或测试断言。线程启动后检查 `/proc/self/task`，数值线程均为 1。

`original_*` 逐字执行 audit_d3_checks 的四份原漏洞脚本，仅通过内存 common 模块把运行器指向 v3、输出指向本目录。它们原本要求漏洞成功，因此预期被 v3 异常截断；包装器严格核对异常字符串，记录停止位置、CLI 数量及到达发送桩次数。不能把原脚本被截断描述成原来的成功断言通过。

`ported_*` 运行 v3/audit_regression 的拒绝版脚本，覆盖全部旧攻击、恢复和评分。仅映射入口哨兵路径及评分输入到本轮证据。`boundary_tests.py` 是本轮独立新增 12 项测试；生产发送测试只投影 live 身份、放行文件门并使用假密钥，不生成任何真实凭据。最后两项检查已预留但未发送的请求是否遵守另一线程新触发的服务暂停，保留真实断言结果。

所有 FORGED/SYNTHETIC 内容均是离线测试输入，不能作为真实数据或人工授权。`baseline.json`、`final_results.json` 保存旧文件完整性和四份清单校验；`progress.json`、逐组日志及结果 JSON 保存实际运行结果。若出现审计桩或包装器错误，修正与复跑单独留证，不归责于运行器。
