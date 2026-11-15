"""Bookkeeping correction only. Original frozen analysis remains byte-for-byte intact.
Written during formal execution, before inspecting any outcome: incomplete P2/P3
must say 未完成, as preregistered, and incomplete direction is not confirmed.
No change to estimators, confidence levels, thresholds, grids, samples or fits.
"""
from pathlib import Path
source=Path(__file__).with_name('P1NUM_analyze.py').read_text()
old="  mg.append(row)"
new="  if not complete:\n   row['status']='未完成'\n   if kind=='P2':row['direction_confirmed']=False\n  mg.append(row)"
assert source.count(old)==1
exec(compile(source.replace(old,new),str(Path(__file__).with_name('P1NUM_analyze.py')),'exec'))
