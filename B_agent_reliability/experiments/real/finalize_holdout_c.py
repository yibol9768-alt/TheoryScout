"""Offline C presentation adapter; original frozen report and all raw data retained."""
from datetime import datetime
import json
import holdout_c as h
import report_holdout_c as r
raw_events=r.events

def normalized_events(path):
    ev=raw_events(path)
    archive=path.parent/('STOP_408_ORIGINAL_'+path.name)
    if path.name.startswith('sheet_') and archive.exists():
        old=raw_events(archive)
        pending=[x for x in old if x['event']=='request' and x['call']==22]
        assert len(pending)==1
        stop=old[-1]
        assert stop['result']['status']=='http_408'
        synthetic=dict(event='service_error',utc=stop['utc'],call=22,attempt=0,reason='http_408',
                       seconds=0,reconstructed_from='original_http_408_halt')
        insertion=next(i for i,x in enumerate(ev) if x['event']=='result')
        ev=ev[:insertion]+[synthetic]+ev[insertion:]
        recovery=next(x for x in ev if x['event']=='exact_state_recovery')
        following=next(x for x in ev if x['event']=='request' and x['call']>22)
        assert following['messages']==pending[0]['messages']
        assert (following['step'],following['review'])==(pending[0]['step'],pending[0]['review'])
        assert datetime.fromisoformat(following['utc']).timestamp()-datetime.fromisoformat(stop['utc']).timestamp()>=300
    return ev

def finalize():
    r.events=normalized_events
    obj=r.report()
    note='\n\n## HTTP 408工程实施偏差及解释限制\n\n首轮冒烟call22的HTTP408本应依用户要求算超时暂停，却被原实现的普通4xx分支停止。原冻结件和原始停止记录均保留；另加恢复器将408转入既有transport暂停分支。以原14个有效响应确定性重建状态，再核对原待发请求并恢复；没有重跑已评分动作、更换种子或补填B。\n\n原始STOP_408_ORIGINAL_*快照、修正文档、恢复器与修正SHA清单均保留。修正文档和代码先写入，SHA清单首次写出因缺少json导入失败；实际补写时恢复已发出请求，不宣称该修正在恢复前完成哈希冻结。\n\n因此G5原冻结实现存在偏差，严格意义不能宣称G0–G5全部通过或无偏差的确认性认证。修正后的操作审计通过也不能消除该限制。该限制不改变原评分、CALIB1参数、G2缺失规则或G3门槛。\n\n离线分析器仅在内存将原HTTP408 halt对应为服务失败，来源显式标注，不回写原始轨迹。服务失败不算模型失败或单题W。\n'
    obj['G5_observed_audit']=False
    obj['G5_implementation_deviation']='HTTP408分类遗漏；恢复前修正文档/代码存在，修正哈希清单晚于恢复请求。'
    obj['correction_operational_audit']=True
    h.write(h.ROOT/'ANALYSIS_HOLDOUT_C.json',obj)
    for path in (h.ROOT/'HOLDOUT_C.md',h.ROOT.parent.parent/'RESULTS_HOLDOUT_C.md'):
        text=path.read_text().replace('- G5：已发生部分冻结、原始动作重放、硬限、并发、额度与服务规则审计通过','- G5：原冻结实现有HTTP408实施偏差；修正后原始动作重放、硬限、并发、额度与服务规则操作审计通过')
        path.write_text(text+note)
    return obj

if __name__=='__main__':finalize()
