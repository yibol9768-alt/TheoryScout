"""冻结前只给新账本补记已有判决所需的规划条件，不改设计和数字。"""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/'z4b/power.json'
o=json.loads(p.read_text())
o['assumptions']['locked_model_gap_positive_given_successful_calibration']=True
o['assumptions']['sign_condition_caveat']='目标根预测为正不证明随机锁定都为正；主情景以成功校准时模型正号为额外未验证条件；实际反号严格拒绝确认'
p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n')
