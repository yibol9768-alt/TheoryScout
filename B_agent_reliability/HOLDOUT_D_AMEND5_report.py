"""汇总真实离线结果并生成与 v4 的逐文件 diff；不改旧件。"""
import difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'HOLDOUT_D_runner_v4'
R=ROOT/'HOLDOUT_D_runner_v5'
OUT=R/'amend5_checks/results'
def sha(raw):return hashlib.sha256(raw).hexdigest()

def diff():
    old={str(p.relative_to(OLD)):p for p in OLD.rglob('*') if p.is_file()}
    new={str(p.relative_to(R)):p for p in R.rglob('*') if p.is_file()}
    changed=[];patch=[];core=[]
    for name,p in sorted(old.items()):
        assert name in new,('deleted_copy',name)
        a=p.read_bytes();b=new[name].read_bytes()
        if a==b:continue
        changed.append(dict(path=name,before_sha256=sha(a),after_sha256=sha(b)))
        try:
            lines=list(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='HOLDOUT_D_runner_v4/'+name,tofile='HOLDOUT_D_runner_v5/'+name))
        except UnicodeDecodeError:lines=['Binary files differ: '+name+'\n']
        patch.extend(lines)
        if name in ('transport.py','run.py','engine.py','evidence.py'):core.extend(lines)
    data=dict(base='HOLDOUT_D_runner_v4',target='HOLDOUT_D_runner_v5',changed=changed,deleted=[],unchanged_copied_files=len(old)-len(changed),added=sorted(set(new)-set(old)))
    (ROOT/'HOLDOUT_D_AMEND5_DIFF.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'HOLDOUT_D_AMEND5_DIFF.patch').write_text(''.join(patch))
    return data,''.join(core)

def main():
    all_results=json.loads((OUT/'ALL_RESULTS.json').read_text());assert all_results['passed']
    rows=[];total=0
    for stem,label in [('command_quota','命令额度专项'),('boundary','v4边界（目标v5）'),('resume','v4恢复（目标v5）'),('v5','v5继承全部测试'),('v4','原版v4（只读）'),('v1','原版v1（只读）'),('v2','原版v2（只读）'),('v3','原版v3（只读）')]:
        z=json.loads((OUT/(stem+'_results.json')).read_text())
        assert z['passed'] and not z['failures'] and not z['errors']
        rows.append(f"|{label}|{z['tests']}/{z['tests']}，0失败、0错误|`amend5_checks/results/{stem}_results.json`|")
        total+=z['tests']
    timeout=json.loads((OUT/'timeout_evidence.json').read_text())
    # 报告不嵌入自身hash，先填正文，再生成完整diff。
    body='''# HOLDOUT_D v5：AMEND5 离线验收报告

日期：2026-11-06。v5 从 v4 完整复制后最小修改。**全部 {total} 项单测及六组攻击/独立评分回归通过；14 个作业退出码全部为0。** 未调用模型服务，未执行真实 plus_quota.py，未修改 tools 脚本；v1–v4及既有冻结件保持原字节。当前新证据仅为 `amend5_checks/results/`。

## 实现与失败语义

`transport.CommandQuota` 仅接受 live 标志，不接受调用者指定命令。在 live 严格为 True 且固定放行文件有效时才可构造，调用前再次核验。固定 argv 为 `['python3','/home/liuyibo/auto-llm-driving/TheoryScout/tools/plus_quota.py']`，capture_output/text/check 为 True、timeout 为50秒，不使用 shell。stdout 恰好两个0–100的数，依次五小时/周已用百分比；除100返回分数，并附成功解析后的 observed_unix 当前时间。NaN/Inf、越界、格式错误、超时、非零退出均作为查询失败。

`run.py` 提供无参数 `--quota-command`，与 `--quota-url` 互斥，命令模式要求 --live 和有效放行文件；未配置提供者仍拒绝，默认 mock 仍用内存零额度。阈值保持 `.80/.90` 的严格小于，等于或超出暂停。Engine 继续记录 query_failed、设置 need_probe/probe_retry、暂停300秒，不复用旧值。仅补查询结束时重取 checked，以保证50秒查询超时之后仍完整等待300秒。

放行 schema 保留 `HOLDOUT_D_AMEND3_LIVE_RELEASE_V1` 与字段名 `amend3_manifest_sha256`；v5 的该字段、WAL init 和首两题快照检查绑定 **HOLDOUT_D_AMEND5.sha256 文件字节的 SHA256**。真实目录、放行文件、账本和启动scope映射到v5；旧 AMEND1–4 清单核验仍保留并增验 AMEND5。无其他算法或统计定义变更。

## 实际结果

|测试组|结果|证据|
|---|---|---|
{rows}

共14组顺序执行，耗时 {seconds:.2f} 秒；总入口结果为 `amend5_checks/results/ALL_RESULTS.json`，外层日志 `../HOLDOUT_D_AMEND5_VALIDATION.log`。六组回归：attack_entries、endpoint_escape、url_resolution、repeated_dispatch、attack_smoke、check_scores 全部退出0。涵盖原入口432组、端点/URL边界、持久一次性发送、人工材料与恢复拒绝、独立评分16组和 C 的40题/1288动作，统计五格保持不变。

命令额度11项覆盖：无live/无放行/自定义命令拒绝；固定argv与timeout参数；正常数值与observed_unix；0/100合法范围及80/90严格边界；畸形/NaN/Inf；越界；有效stdout但退出7；真实假命令50秒超时；执行前放行撤销；CLI互斥/live门；AMEND4 digest拒绝及AMEND5 digest绑定；CLI正确选择命令或HTTP提供者。每类异常验证query_failed、无旧额度字段、300秒前不探测也不发请求，到期重新取新值。

测试用 PATH 前置的独立 `amend5_checks/results/command_bin/python3` 替身，只验证固定参数并输出合成数/退出/等待，从不打开真实脚本。超时替身设置等待55秒，父进程在50秒终止；实测耗时 {timeout_elapsed:.3f} 秒，证据 `timeout_evidence.json`。引擎虚拟时刻1000开始查询，1050记录失败、1350才可重探测，1349.999仍拒绝。300秒等待使用虚拟时钟，50秒子进程超时为实际发生。

## 与 v4 的 diff

完整逐行diff：`../HOLDOUT_D_AMEND5_DIFF.patch`；所有复制文件前后hash、新增列表：`../HOLDOUT_D_AMEND5_DIFF.json`。未删除任何v4复制文件。既有文件修改范围如下：

|文件|变更|
|---|---|
|transport.py|只新增subprocess导入和CommandQuota。|
|run.py|只新增互斥开关/live校验/提供者选择及v5目录映射。|
|engine.py|只增加AMEND5清单校验/绑定、v5目录映射、查询完成时重取checked。|
|evidence.py|只改v5目录及AMEND5清单绑定；schema/字段名保留。|
|score.py|只改v5输出目录及增验AMEND5；统计和工程重放公式不变。|
|launch.sh|只改v5启动目录、日志和scope名称。|
|HOLDOUT_D_cli_sentinel.py、audit_regression/entry_sentinel.py|构造哨兵补覆盖CommandQuota。|
|audit_regression/common.py|只映射v5运行器/输出目录。|
|audit_regression/check_scores.py|只改合成fixture绑定AMEND5。|
|test_mock.py|只改BASE为v5，原断言保留。|
|validate_all.py|只把离线入口切换到amend5_checks。|
|README.md、MOCK_TEST_REPORT.md|v5合同、命令格式、实测结果及diff。|

`protocol.py`、`test_amend2.py`、`test_amend3.py`、`audit_originals/`、复制的 `amend4_checks/` 和其他未列明旧复制文件逐字节不变。新增 `amend5_checks/` 源自v4 harness，保留全部边界/恢复原断言，只映射目标/隔离输出并纳入命令专项和原版v4套件；新sitecustomize另行禁止真实额度脚本执行。新增AMEND5测试清单、根目录AMEND5前缀基线/准备/编辑/报告/完整性/冻结交付。以下为核心源码实际统一diff（其他路径/文档/测试入口变更见完整patch）：

```diff
{core_diff}```

## 离线与冻结证据

全部计算使用 `systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0`，OMP/OPENBLAS/MKL/NUMEXPR线程设1。结果资源记录 memory.max=2147483648、memory.swap.max=0；线程启动监测≤2。HTTP仅用本地记录/响应桩，审计钩子阻断socket连接、DNS、sendto、bind及结果目录外写入。子进程只能通过测试目录中的独立替身触发固定额度argv，不能执行真实工具；生产入口仅在测试中投影门分支，未创建真实放行文件或真实账本。

复制的 `amend4_checks/results/`、`regression_results/`、`regression_results_initial/` 为历史材料，不计入本次结果。新合成fixture绑定测试时7项源码临时清单，原字节保存在 `HOLDOUT_D_AMEND5_TEST_MANIFEST.txt`；最终冻结清单另含报告/日志，不把旧合成fixture解释为最终真实授权或真实G0。

旧件基线 `../HOLDOUT_D_AMEND5_BASELINE.json` 覆盖本次工作前全部旧文件。`../HOLDOUT_D_AMEND5_validate.py` 检查六份清单及基线、新增路径范围、v4原断言与真实状态缺席；仅允许外层Codex日志追加且原前缀不变。冻结前记录 `../HOLDOUT_D_AMEND5_INTEGRITY.json`；最终 `../HOLDOUT_D_AMEND5.sha256` 由freeze脚本生成，覆盖全部v5与AMEND5前缀文件，自身不入清单。最终只读验收使用同一资源前缀运行：

```text
python3 -B HOLDOUT_D_AMEND5_validate.py
```
'''
    # 只取与报告正文无关的核心diff，避免循环嵌入。
    core=''.join(''.join(difflib.unified_diff(
        (OLD/name).read_text().splitlines(True),(R/name).read_text().splitlines(True),
        fromfile='HOLDOUT_D_runner_v4/'+name,tofile='HOLDOUT_D_runner_v5/'+name))
        for name in ('engine.py','evidence.py','run.py','transport.py'))
    (R/'MOCK_TEST_REPORT.md').write_text(body.format(total=total,rows='\n'.join(rows),seconds=all_results['seconds'],timeout_elapsed=timeout['elapsed_seconds'],core_diff=core))
    data,_=diff()
    print(json.dumps(dict(unit_tests=total,jobs=len(all_results['results']),changed_files=len(data['changed']),unchanged_copied_files=data['unchanged_copied_files'],added_files=len(data['added'])),ensure_ascii=False))
if __name__=='__main__':main()
