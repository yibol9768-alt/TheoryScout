"""Transparent post-run interpretation; offline only, preserves frozen selection."""
import collections
import json
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parent

def main():
    summary=json.loads((ROOT/'difficulty_real/summary.json').read_text())
    analysis=json.loads((ROOT/'ANALYSIS_DIFFICULTY.json').read_text())
    report=(ROOT/'DIFFICULTY.md').read_text()
    assert summary['calls']==267 and analysis['recommended']==[]
    assert next(x for x in analysis['stats'] if x['family']=='indirect' and x['level']==0)['e']==6
    report=report.replace('完成 48/48 题，成功 6 题。',
                          '48个计划槽均已尝试，只有6题完成全部16步验证；15题两次复查仍错，27题服务重试耗尽。其中22题有动作暴露，26题完全没有可评分动作。')
    report=report.replace('停止原因 `None`。', '批次按既定48槽结束（没有触发额度或1200硬限）；不是48条完整轨迹。')
    report=report.replace('## 各变体与难度',
        '已找到一个**目标错误率的探索性锚点**：一跳间接寻址为6/64＝9.38%，95%参考区间3.52%–19.30%；尚未通过冻结实现中“四题全部成功”的更严格推荐门槛。因此本轮不宣称已有正式入选配置，也不将服务缺失补成零。\n\n## 各变体与难度',1)
    report=report.replace('## 各变体与难度\n',
        '## 各变体与难度\n\n每类四档、每档四个新题、计划L=16。sheet旋钮为每步子事务数2/4/8/16（6基值加3层公式）；indirect为动态条件寻址跳数1/3/6/10（8值加cursor）；memory为寄存器宽度4/8/12/16（初始状态只给首轮，历史完整保留）。后两档全部受服务故障影响，不能判断更难或更易。\n',1)
    report=report.replace('## 真实错误与复查\n',
        '## 真实错误与复查\n\n手工可读解释：sheet_d0_901001 第6步的基值和F0=804均正确，但模型把 `(3×804+47−53)%997` 算成406，正确是412；F2随之错6。两次明确失败反馈后仍给同样错误。indirect_d0_902001 第8步所有寄存器正确，仅cursor输出7、正确为6，次一复查修好；第9步再出现r6与cursor错误，第一复查仍错，第二复查修好；第16步再次错且两次复查未修好。memory_d0_903002 第1步r0=416、r1=36均正确，checksum输出750、正确560；750恰等于使用旧r1=226时的416+3×36+226，而最新值语义要求416+3×36+36。该模式与读取别名旧值一致，但不能据输出断言模型内部推理。\n\n以下均为真实原动作，含操作、动作前状态、期望状态和真实复查响应：\n',1)
    report=report.replace('## 服务、额度与资源\n',
        '## 服务、额度与资源\n\n57次均为约90秒的传输超时；首次失败30次、第二次失败27次；30次重试中3次成功、27次仍失败。逻辑请求237个，其中210个取得响应；真实HTTP尝试267次。前三次分散失败发生于调用96、108、121，随后重试成功；调用214–267连续54次超时，没有HTTP状态码，因此只能记录传输故障，不能诊断具体反代或上游根因。27个耗尽重试题中26题完全无动作，另1题已推进3步；服务停止与难度高低、运行时序混杂。\n',1)
    report=report.replace('## 推荐与后续校准\n',
        '## 推荐与后续校准\n\n**按冻结规则正式入选：0个。按后续投入优先级，只建议一个待复核候选：indirect、一跳、8寄存器、模997，保持原动作schema和提示。** 这是结果后的研究建议，不是改写冻结选择结果。其64次推进已覆盖四题全部种子动作，因此9.38%不受未观察尾部的缺失影响；但只有3/4题最后完全修复成功，且错误集中在两题（逐题4/16、2/16、0/16、0/16）。首次错前2/52＝3.85%，其后的4/12＝33.33%是受选择与反馈干预影响的条件描述，不能称作历史因果效应。四子事务工作表5/36＝13.89%可作“偏难”锚点，不推荐直接正式校准；记忆版本目前主要暴露顺序别名问题，未隔离记忆效应。\n\n建议下一轮首先对这一候选做**独立准入校准**：50个全新任务、L=32，最多2次明确反馈复查；每题至多48次HTTP尝试（含服务重试），单配置全局硬限2400，理论干净推进1600、其余最多800用于复查/重试，触顶或单题48即停并如实删失。并发≤2、只用gpt-6-luna、每100尝试查额度、任一≥80停；服务恢复后的第一个新任务仍在该50题与2400预算内，不重跑本轮失败槽。事先登记在10/25/50题边界检查服务可用性；最近20次尝试传输失败≥4则管理停批并保留全部分母，不能像本轮连续耗费27个失败槽。主报告首次错误前风险和干净前缀总体率、整题区间、实际错格变化及反馈修复率；目标仍为2%–10%点估计，至少观察30个自然错误片段且反馈后检测/修复必须分开估计。若种子率偏离、任务相关性强、修复核或状态定义不可迁移，就不启动容量留出验证。**本轮没有执行这2400次的后续预算。**\n',1)
    report=report.replace('没有完整格符合预注册的2%–10%点估计条件，不能宣称找到了合格配置。本轮到此停止，不临时追加等级或种子。',
        '没有格同时满足完整成功与2%–10%点估计两个冻结条件。本轮到此停止，不临时追加等级或种子。')
    report=report.replace('下一轮先为每个入选配置登记50个独立新题、L=32、纯自然轨迹；每个错误最多2次明确反馈复查，单题最多48调用、每配置硬上限2400调用，两配置合计≤4800（含服务重试，触顶即停），并发≤2，仍只用gpt-6-luna。该预算为建议，不在本轮执行。',
        '若独立准入校准后将来另选第二配置，须另登记其至多2400预算；两个配置总条件性上限4800，不能把它当成本轮或当前单候选预算。')
    report=report.replace('离线重现：`/home/liuyibo/venvs/theory/bin/python experiments/real/report_difficulty.py`。',
        '离线完整重现按顺序执行 `report_difficulty.py`、`finalize_difficulty.py`、`audit_difficulty.py`（使用 `/home/liuyibo/venvs/theory/bin/python`，均位于experiments/real，不调用模型）。后两者是运行后解释/离线统计补充，未修改任何冻结执行器、预注册或冻结报告器；各自SHA见ARTIFACTS_DIFFICULTY.json。')
    report+='\n## 工程验收与设计限制\n\n运行前9项单测全部通过，完整mock为48/48题、768步；独立oracle mock前缀/动作复核1536项。见 [TEST_DIFFICULTY.txt](TEST_DIFFICULTY.txt)、[difficulty_mock/summary.json](difficulty_mock/summary.json)、[ANALYSIS_DIFFICULTY_MOCK.json](ANALYSIS_DIFFICULTY_MOCK.json)。冻结时间2026-11-05T20:46:43.955893+00:00，早于真实启动；FREEZE_DIFFICULTY.json包含运行器、测试、报告器与运行前预注册全部SHA-256。\n\n全部33次自然种子错误均是合法JSON的numeric状态错误；schema/JSON错误0、输出截断0、人工注入0。最大实际输入989、输出705（含推理）token，故不能把这些错误归因于极小输出预算。53次反馈复查中18次当次修好、35次仍错；首次33次复查detect=true为29/33、一次修好13/33；第二复查修好5/20。没有“模型自发发现当前错误”的无提示复查对照，自发q不可估计。\n\n未观察高档没有自然核证据。memory宽度还同时改变别名概率：宽度增加虽提高记忆负担，却降低随机别名冲突频率，不能拟合单调记忆难度曲线。sheet与indirect每步输出整状态、memory只输出稀疏写，负担并非等价；批事务可一次产生多个错格，不能把每事务种子率直接代入原单键出生/死亡模型。\n'
    (ROOT/'DIFFICULTY.md').write_text(report)
    print('offline interpretation added; frozen recommendation remains empty')

if __name__=='__main__':main()
