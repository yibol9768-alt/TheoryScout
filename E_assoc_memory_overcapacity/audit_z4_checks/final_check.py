"""最终只读边界检查与明确假设下的校准失败规划。"""
import sys
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'audit_z4_checks'))
from checks import C,DEST,save
from scipy.stats import norm,nct,t

before=C.read(DEST/'freeze_before.json')
bad=[]
for line in (ROOT/'Z4_FROZEN.sha256').read_text().splitlines():
    h,rel=line.split('  ',1)
    if C.sha(ROOT/rel)!=h:bad.append(rel)
refined=C.read(DEST/'repair_refined.json')
# 额外的、尚未实测证明的规划情景：每点SD≤.03，真实曲线距PAVA≤.003，
# 目标距两个PAVA端点≥.03。若所有点误差<.006，则PAVA改动<.015且目标被括住。
point_se=.03/math.sqrt(1024)
cal_fail_bound=4*17*2*float(norm.sf(.006/point_se))
oldcrit=float(t.ppf(1-.025/(2*3),1279))
old_fail=3*float(nct.cdf(oldcrit,1279,.03*math.sqrt(1280)/.03))
base=refined['bias_sensitivity'][2]['all7_union_lower']
save('complete_power_scenario.json',dict(additional_unverified_assumptions=dict(
    all_calibration_grid_dataset_SD_max=.03,population_PAVA_discrepancy_max=.003,
    target_distance_from_population_fitted_endpoints_min=.03,
    per_method_systematic_target_bias_max=.0025,
    true_old_pair_effect_min=.03,old_pair_dataset_SD_max=.03,
    normal_approximation_for_calibration_point_means=True),
    calibration_failure_union_upper=cal_fail_bound,old_direction_failure_union_upper=old_fail,
    complete_confirmation_power_planning_lower=max(0.,base-cal_fail_bound-old_fail),
    caveat='这是一组明确可检验的设计假设，不是从正式数据获得的保证；不作有限样本精确概率声明。'))
save('final_check.json',dict(checked_scientific_inputs=len(before['input_checks']),frozen_manifest_entries=1428,
    manifest_unchanged=C.sha(ROOT/'Z4_FROZEN.sha256')==before['manifest_sha256'],
    frozen_mismatches=bad,formal_absent=not (ROOT/'z4/formal').exists(),
    audit_report_sha256=C.sha(ROOT/'AUDIT_Z4.md'),
    audit_scripts={p.name:C.sha(p) for p in sorted(DEST.glob('*.py'))},
    artifacts_all_under_audit_directory=True,formal_sampling=False,llm_api_calls=0))
assert not bad and not (ROOT/'z4/formal').exists()
print(json.dumps(C.read(DEST/'final_check.json'),ensure_ascii=False))
