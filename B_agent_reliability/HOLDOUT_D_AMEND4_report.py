"""从本轮实际结果生成报告和相对 v3 的逐文件 diff；须在受限 scope 内运行。"""
import difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT/'HOLDOUT_D_runner_v3';NEW=ROOT/'HOLDOUT_D_runner_v4'
OUT=NEW/'amend4_checks/results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    all_results=json.loads((OUT/'ALL_RESULTS.json').read_text());assert all_results['passed']
    cases=[('审计新增边界（原失败11、12已通过）','boundary',12),('M1等待与恢复专项','resume',6),
           ('v4继承全部测试','v4',48),('原版v1','v1',22),('原版v2','v2',39),('原版v3','v3',48)]
    rows=[]
    for label,name,count in cases:
        p=OUT/(name+'_results.json');z=json.loads(p.read_text())
        assert z['passed'] and z['tests']==count and not z['errors'] and not z['failures'],p
        assert all(n<=2 for n in z['thread_counts'])
        rows.append(f'|{label}|{count}/{count}，0失败、0错误|`amend4_checks/results/{p.name}`|')
    direct=json.loads((OUT/'pause_direct_findings.json').read_text())
    normal=json.loads((OUT/'pause_normal_findings.json').read_text())
    assert all(x['sends_during_pause']==0 for x in direct)
    assert normal['sends_during_pause']==0 and normal['halt'] is None
    changed=[str(p.relative_to(OLD)) for p in sorted(OLD.rglob('*')) if p.is_file() and p.read_bytes()!=(NEW/p.relative_to(OLD)).read_bytes()]
    # 报告本身将被替换，始终列入 diff。
    changed=sorted(set(changed)|{'MOCK_TEST_REPORT.md'})
    descriptions={
      'engine.py':'M1发送边界复核；DispatchPaused保留待发预留；request复用；额度排空区分已发送/未知与明确待发；v4路径及AMEND4清单绑定。',
      'evidence.py':'仅v4目录和AMEND4清单绑定；保留AMEND3 schema/字段名及人工门合同。',
      'score.py':'同步M1发送边界/探测排空工程检查，识别跨恢复的未知预留；v4路径、增验AMEND4清单；统计公式不变。',
      'run.py':'仅运行输出/审批目录从v3映射到v4。',
      'launch.sh':'仅运行目录、日志路径及scope名称更新为v4。',
      'test_mock.py':'仅BASE输出目录更新为v4，原断言不变。',
      'audit_regression/common.py':'仅运行器与输出目录从v3映射到v4。',
      'audit_regression/check_scores.py':'仅合成fixture的清单绑定改为AMEND4。',
      'validate_all.py':'切换到严格离线的新回归入口，所有HTTP使用本地记录桩。',
      'README.md':'说明M1等待/恢复语义、离线入口、历史证据及AMEND4清单绑定。',
      'MOCK_TEST_REPORT.md':'本轮实际验收报告，含完整diff对应表。'}
    assert set(changed)==set(descriptions),(changed,descriptions.keys())
    diffrows='\n'.join('|`'+name+'`|'+descriptions[name]+'|' for name in changed)
    resource=all_results['resources']
    report=f'''# HOLDOUT_D v4：AMEND4 离线修复与 mock 验收报告

日期：2026-11-06。依据 `../AUDIT_HOLDOUT_D_AMEND3.md` 修复 M1。**新增12项、恢复6项、v4继承48项及原版v1/v2/v3全部109项均通过，共175项单测；6组攻击/独立评分回归也全部通过。** 全程未调用模型服务、未查询真实额度，所有HTTP发送均由本地记录桩接管。

## M1最小修复

`claim_dispatch` 在原全局RLock内、消费发送权和写入dispatch_started之前，重新检查最新 `pause_until`、`need_probe`、`probe_retry` 与 `quota_bucket`。暂停只抛出专用 `DispatchPaused`，不写dispatch、不进入opener、不释放pending、不增加call/attempt或预算。人工门、身份、正文hash及重复发送检查继续保留。

`request` 对此明确未发送的异常保留原预留，调度器下次调用时从原WAL预留恢复相同正文，沿用同一call、rid、attempt，再探测合格后发送。实际发送过的失败请求仍结束旧预留，合法重试仍使用新预留；未知异常、fsync失败、崩溃及恢复仍保守保留unknown_inflight，不能自动重发。

为防恢复死锁，`quota` 等待已取得发送权的active请求与恢复后的未知预留排空；同进程明确尚未取得发送权的预留不阻塞探测。未到最后服务失败后300秒、额度暂停未满300秒、已发送请求未排空，均不能探测。评分仅同步上述工程守卫；N=144、种子、协议、9000成功/12000尝试、五格、T及G0–G4统计定义均保持冻结设计。

## 实际测试结果

|测试组|结果|证据|
|---|---|---|
{chr(10).join(rows)}

单入口共{len(all_results['results'])}组，全部退出码0，耗时{all_results['seconds']:.2f}秒，见 `amend4_checks/results/ALL_RESULTS.json` 和 `../HOLDOUT_D_AMEND4_VALIDATION.log`。本轮结果仅认 `amend4_checks/results/`；随v3原样复制的 `regression_results/`、`regression_results_initial/` 为历史材料。

审计12项的源文件为 `../audit_d4_checks/boundary_tests.py`；纳入副本仅将目标运行器v3改为v4，全部原断言逐字保留，完整性校验器另行核对。01–10继续覆盖地址/端口/代理/重定向、竞争、崩溃、fsync、正文及URL变更；11在408/500/503/599后，虚拟时刻1000、pause_until=1300、need_probe=true下额外发送均为0；12正常request两线程交错也为0，halt=null，原预留保持待发。见 `pause_direct_findings.json`、`pause_normal_findings.json` 与对应WAL。

额外6项验证：连接失败/408/500/501/502/503/504/599后原预留不变；299.999秒不探测；300秒到达仍须新探测；额度阈值不合格延长300秒且probe_retry独立拦截；合格后同一预留/正文恰发送一次、attempts始终为2；正常两线程路径暂停后再request不新增reserve；已发送请求未排空不探测、最后失败重新起算300秒；待发期间崩溃后连续恢复仍unknown_inflight；百次边界与底层quota后原预留发送。见 `resume_tests.py`、`resume_results.json` 及 `results/resume/`。

六组移植攻击继续覆盖入口432组、端点逃逸及恢复、危险URL构造、5项持久发送权回归、42项负向人工材料/6次无批准发送/4次非法恢复，以及独立评分16组和C的40题/1288动作。评分64步题五格仍为[5,2,1,2,10]。详见各 `ported_*/*.json` 与日志。

## 与v3的逐文件diff

先完整复制v3，再仅修改下列{len(changed)}份既有文件；未删除任何复制文件：

|文件|变更|
|---|---|
{diffrows}

`protocol.py`、`transport.py`、`test_amend2.py`、`test_amend3.py`、`audit_originals/` 以及未列出的v3复制文件均逐字节保留。新增 `amend4_checks/`、`HOLDOUT_D_AMEND4_TEST_MANIFEST.txt` 及本目录外AMEND4前缀基线/报告生成/完整性/冻结文件。上述既有文件的逐行diff为 `../HOLDOUT_D_AMEND4_DIFF.patch`；逐文件前后hash及新增文件清单为 `../HOLDOUT_D_AMEND4_DIFF.json`。

保留AMEND3 schema和字段名避免合同迁移，但v4的 `amend3_manifest_sha256` 绑定最终AMEND4清单；v4输出、人工放行及真实账本路径均独立。未创建真实放行文件、真实账本或人工已审凭据。FORGED/SYNTHETIC字段仅是离线分支fixture，不能作为人工授权或真实D样本。合成评分fixture绑定测试时的临时7项源码清单，其原字节保存在 `HOLDOUT_D_AMEND4_TEST_MANIFEST.txt`；最终完整冻结包含报告与日志，不能用最终digest把旧合成fixture解释为真实G0。

## 离线、资源与完整性

全部测试/哈希/报告计算使用指定scope：memory.max={resource['memory_max']}、memory.swap.max={resource['swap_max']}；OMP/OPENBLAS/MKL/NUMEXPR线程均1。各套件监测实际线程，最多2（主线程和一个worker）。300秒等待均为虚拟时钟，不声称真实等待。

`sitecustomize.py` 对测试和子进程拒绝socket连接、DNS、sendto、bind，限制写入到新版本results。HTTPConnection.request/getresponse使用本地记录/响应桩；原HTTP服务器函数只返回测试端口，不监听socket。保留urllib代理/重定向处理、原状态码断言和真实fork退出断言。生产分支只投影live门并使用假密钥；预算、锁、发送权、WAL是被测实现。没有外部连接或真实额度查询。

旧文件基线为 `../HOLDOUT_D_AMEND4_BASELINE.json`。`../HOLDOUT_D_AMEND4_INTEGRITY.json` 核对旧冻结清单和原文件；只容许外层Codex日志追加且原前缀hash不变。最终冻结 `../HOLDOUT_D_AMEND4.sha256` 覆盖全部v4文件及AMEND4交付；最终只读验收核对五份清单、旧件及新增路径范围：

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \\
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\
  python3 -B HOLDOUT_D_AMEND4_validate.py
```

完整回归入口为相同资源前缀下的 `HOLDOUT_D_runner_v4/validate_all.py`；拒绝覆盖已有证据。v1/v2/v3、既有冻结件和审计目录均未由本轮改写。
'''
    (NEW/'MOCK_TEST_REPORT.md').write_text(report)
    patch='';changes=[]
    for rel in changed:
        a=OLD/rel;b=NEW/rel
        changes.append(dict(path=rel,before_sha256=sha(a),after_sha256=sha(b)))
        patch+=''.join(difflib.unified_diff(a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile='HOLDOUT_D_runner_v3/'+rel,tofile='HOLDOUT_D_runner_v4/'+rel))
    added=sorted(str(p.relative_to(NEW)) for p in NEW.rglob('*') if p.is_file() and not (OLD/p.relative_to(NEW)).exists())
    (ROOT/'HOLDOUT_D_AMEND4_DIFF.patch').write_text(patch)
    (ROOT/'HOLDOUT_D_AMEND4_DIFF.json').write_text(json.dumps(dict(modified=changes,added=added,deleted=[],unchanged_copied_files=sum(p.is_file() for p in OLD.rglob('*'))-len(changed)),ensure_ascii=False,indent=2)+'\n')
    print('Report generated; 175 unit tests and 6 attack groups passed;',len(changed),'modified copied files')
if __name__=='__main__':main()
