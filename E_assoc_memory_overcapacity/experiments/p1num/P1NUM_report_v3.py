"""Formatting refinement: always show both epsilon paths and all target exclusions.
Adds only already-preregistered interval endpoints/decisions; no new tests or data.
"""
from pathlib import Path
source=Path(__file__).with_name('P1NUM_report.py').read_text().replace('../../P1NUM_ANALYSIS_NOTE.md','P1NUM_ANALYSIS_NOTE.md')
needle=" text+='锚点d128/γ64两法恒等"
insert=""" extra=[]
 for cc in contr:
  dd=int(cc['d']);gg=int(cc['gamma'])
  rp=next(r for r in A if int(r['d'])==dd and int(r['gamma'])==gg and r['label']=='physical_epsilon')
  re=next(r for r in A if int(r['d'])==dd and int(r['gamma'])==gg and r['label']=='0.06')
  extra.append([dd,gg,pp(rp['mean']),f\"[{pp(rp['lower'])}, {pp(rp['upper'])}]\",rp['decision'],pp(re['mean']),f\"[{pp(re['lower'])}, {pp(re['upper'])}]\",re['decision']])
 text+=table(['d','γ','物理ε的R/pp','区间/pp','方向','固定e的R/pp','区间/pp','方向'],extra)+'\\n\\n'
"""
assert source.count(needle)==1
source=source.replace(needle,insert+needle)
needle2=" text+='功效使用旧同维24种子方差×2"
insert2=""" text+='P2相对极限的有符号偏差及同时区间（百分点；未完成格仅描述）：\\n\\n'
 text+=table(['κ','偏差/pp','偏差同时区间/pp','区间排除极限','覆盖完整'],[[r['kappa'],pp(r['mean']-r['target']),f\"[{pp(r['lower']-r['target'])}, {pp(r['upper']-r['target'])}]\",not(r['lower']<=r['target']<=r['upper']),r['complete']] for r in p2])+'\\n\\n'
"""
assert source.count(needle2)==1
source=source.replace(needle2,insert2+needle2)
needle3=" text+=f'''新六项校正相容"
insert3=""" text+='P3的显著残留标记（不与0.5pp精度内相容混同）：\\n\\n'
 text+=table(['κ','方法/指标','D区间不含0','A下界>0'],[[r['kappa'],r['name']+'/'+r['metric'],r['lower']>0 or r['upper']<0,r['A_lower']>0] for r in p3])+'\\n\\n'
"""
assert source.count(needle3)==1
source=source.replace(needle3,insert3+needle3)
exec(compile(source,str(Path(__file__).with_name('P1NUM_report.py')),'exec'))
