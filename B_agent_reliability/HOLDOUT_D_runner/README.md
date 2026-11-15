# D运行器（本次仅mock交付）

本目录实现原预注册第3、8节工程合同及审计第5节。统计门槛、样本量、种子和模型维持原冻结。当前 `execution_ready=false`：不存在真实D调用、真实额度验证或人工首两题冒烟放行。请同时读 `../HOLDOUT_D_AMEND1_PLANNING.md` 与 `MOCK_TEST_REPORT.md`。

- `protocol.py`：从冻结 `experiments/real/reliability_core.py` AST提取纯task/transition/parse/required与提示常量；不导入旧客户端。独立整数oracle核验每题64步。提示与旧C完整75动作轨迹逐条一致，推进两消息、复查四消息，不含长度。
- `engine.py`：全局RLock、排他flock、带seq/链hash/request id/body SHA256的原始WAL。预留fsync后才发送；2xx原始返回fsync后才计消费/校验/应用动作；所有2xx包括守卫失败均计成功预算。summary仅派生，恢复不读取summary。WAL尾部撕裂或hash异常直接安全停；不偷偷截断证据。
- `transport.py`：默认内存mock协议服务器；HTTP构造器显式`live=True`才有联网能力，不用代理、不跟重定向、不用SDK隐式重试。模型端点固定127.0.0.1:18319、模型gpt-6-luna、90秒超时、4096 completion tokens。唯一测试HTTP请求发向临时本地mock随机端口。
- `run.py`：两题一波，主线程与一个worker，总执行线程≤2；上一波两题都终局才进下一波。默认mock不建socket，真实模式首两题后必须有人工审阅证据。实际启动核对独立SID、无终端、stdin=/dev/null、scope内存与swap及线程环境。
- `launch.sh`：`setsid nohup systemd-run --user --scope`，持久日志，2GiB/零swap，子进程启动前同步捕获原SID。脚本在本次授权下拒绝`--live`，不构造真实客户端。
- `test_mock.py`：所有协议响应由mock服务器生成，fixture预算专供边界测试，不能进入live历史。每次新建时间戳目录，保留失败及成功证据。

在本项目根目录启动**新的离线mock**：

```bash
HOLDOUT_D_runner/launch.sh --out HOLDOUT_D_runner/HOLDOUT_D_mock_fresh
```

已有冻结测试目录不要恢复或复用。默认运行会核对原 `HOLDOUT_D_PREREG.sha256` 与补充 `HOLDOUT_D_AMEND1.sha256`。同一输出目录必须持有同一flock；同一逻辑请求不能同时预留两次，模型并发至多2。真实账本路径固定为 `HOLDOUT_D_runner/HOLDOUT_D_live`，CLI拒绝用其他输出目录启动真实模式；mock也不能写此目录。历史模式（mock/fixture/live）不能混用，不能利用新输出目录或重启重置真实预算。

`--test-unfrozen`仅用于冻结前mock/启动验证，和`--live`互斥，未来真实执行不能跳过完整性核对。当前shell启动器也拒绝`--live`，未来授权后需另行登记真实启动命令，沿用现有资源和会话隔离合同。不能仅通过填写参数认定真实执行已获授权。

预算和暂停语义：

1. 模型HTTP发出前，在同一锁内持久化正文与一次成功额度预留，同时消耗一次HTTP尝试。`success + pending <= 9000`，未知在途保留在pending；尝试≤12000。服务失败有持久结果记录才结束预留。格式/数值/截断错误是模型动作，按原协议最多两次复查，不当作传输重试。
2. 所有5xx、408、连接失败、超时进入全局暂停，截止为最后一次失败+300秒；已在途可返回并落盘。期间不发新请求。必须无在途、期限已满并重新查询额度合格才恢复，同一步序列化正文与稳定request id完全相同，仅attempt递增。
3. 启动、重启、每100成功返回边界、服务恢复均查新额度；99余量只准一个请求，阻止跨100边界。查询在全局锁内、没有在途的屏障执行。五小时>=.80或周>=.90、查询失败均暂停300秒后重查，不用旧额度。
4. `HTTPQuota`仅为项目内额度接口适配器合同：要求单独本地GET端点返回fraction单位`five_hour`、`week`与新鲜`observed_unix`（不超过60秒），且禁代理/重定向。**未声明18319实际提供该接口，也没有实测**。没有经核验的真实额度provider和显式`--live`不得调用；未配置provider的live入口直接停在客户端构造之前，不回退假额度。
5. 其他HTTP、envelope/model/usage/token异常全局停止，保留缺失；已在途仍持久化。已知usage单记，守卫异常及服务失败的未知token耗用单记次数，不推算数字。所有等待均不会重置预算。

恢复语义：重启先flock和冻结核验，为旧WAL完整前缀另写排他新建恢复档案；原WAL仅追加。逐序重建每个request的正文、step/review、attempt和动作，独立oracle核对实际错误写入。已有response但无action时离线补建一次，后续恢复不重复动作或请求。恢复记录旧档案SHA、链head、位置、下一正文hash与attempt；终局题不重跑。无持久响应的发送保留预算并停在`unknown_inflight`，两次恢复也不能放掉预留。

`reconcile_response`只接受项目内经人工核验来源的原始服务响应导出（请求身份、正文hash、响应hash、status、核验人、原始来源）；没有“猜测未发送”的释放接口。函数仅验证字节一致性，无法替代对服务记录来源的独立验证；无法调和就持续信息不足。未决响应调和不产生新模型尝试。

mock不提供科学确认结论。工程或样本不完整时，summary保留每端点K/F/U及[K,K+U]；不得以完成子集通过确认。最终统计仍严格适用原预注册G0–G4，真实首两题审阅必须在144分母内；协议实质错误停批另注册，不能调门槛、换seed或补题。
