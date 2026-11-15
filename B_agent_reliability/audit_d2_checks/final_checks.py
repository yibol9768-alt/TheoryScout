"""最终只读核验：已有文件不变；原生 summary 与独立五格重建对照。"""
import hashlib
import json
import os
from pathlib import Path
from common import ROOT, OUT, dump_file, install_guards, resource_info, sha_file

install_guards()
before=json.loads((OUT/'baseline.json').read_text())['files']
changed=[];missing=[]
for name,digest in before.items():
    p=ROOT/name
    if not p.is_file():missing.append(name)
    elif sha_file(p)!=digest:changed.append(name)
extra=[]
for p in ROOT.rglob('*'):
    if p.is_file() and not p.is_relative_to(OUT) and str(p.relative_to(ROOT)) not in before and p.name!='AUDIT_HOLDOUT_D_AMEND1.md':
        extra.append(str(p.relative_to(ROOT)))
append_only_external_log={}
for name in changed:
    if name!='codex_daudit2.log':continue
    raw=(ROOT/name).read_bytes();h=hashlib.sha256();match=None
    for i,byte in enumerate(raw,1):
        h.update(bytes([byte]))
        if h.hexdigest()==before[name]:match=i;break
    if match is not None:
        writers=[]
        ancestor=os.getppid()
        while ancestor>1:
            proc=Path('/proc')/str(ancestor)
            try:
                if os.readlink(proc/'fd/1')==str(ROOT/name):
                    writers.append(dict(pid=int(proc.name),cmdline=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')[:160]))
                ancestor=int(next(line.split()[1] for line in (proc/'status').read_text().splitlines() if line.startswith('PPid:')))
            except (FileNotFoundError,PermissionError,OSError):break
        append_only_external_log[name]=dict(baseline_prefix_bytes=match,observed_bytes=len(raw),stdout_writers=writers)
unexpected_changes=[x for x in changed if x not in append_only_external_log]
print(json.dumps(dict(changed=changed,missing=missing,extra=extra,external_append=append_only_external_log),ensure_ascii=False),flush=True)
assert not unexpected_changes and not missing and not extra

summary=json.loads((OUT/'original_mock_r2/mock_full/HOLDOUT_D_SUMMARY.json').read_text())
design=json.loads((ROOT/'HOLDOUT_D_DESIGN.json').read_text())
cells=[0,144,0,0,0]
T=sum(n*w for n,w in zip(cells,design['comparison']['weights_log_D_over_A']))
keys=sorted(summary)
scoring=dict(native_summary_keys=keys,terminal=summary['terminal'],prefix_bounds=summary['prefix_bounds'],
             independently_reconstructed_cells=cells,independently_reconstructed_T=T,
             native_five_cells_present=any(k in summary for k in ('cells','five_cells','cell_counts','n_cells')),
             native_T_present=any(k in summary for k in ('T','primary_statistic','llr')),
             note='此 mock 不作科学确认；仅说明完整144题时原生输出仍没有五格/T/G0–G4评分。')
assert summary['terminal']==144 and not scoring['native_five_cells_present'] and not scoring['native_T_present']

launch=json.loads((ROOT/'HOLDOUT_D_runner/HOLDOUT_D_LAUNCH_TEST_FINAL/HOLDOUT_D_LAUNCH_ENV.json').read_text())
launch_check=dict(memory_max=launch['memory_max'],swap_max=launch['swap_max'],sid=launch['sid'],
                  original_sid=launch['original_sid'],ppid=launch['ppid'],stdin=launch['stdin'],
                  tty=launch['tty'],flock_held=launch['flock_held'])
assert launch_check['memory_max']=='2147483648' and launch_check['swap_max']=='0'
assert launch_check['sid']!=launch_check['original_sid'] and launch_check['ppid']==1 and launch_check['stdin']=='/dev/null'
result=dict(resources=resource_info(),original_files_checked=len(before),changed=changed,
            unexpected_changes=unexpected_changes,external_append_only_log=append_only_external_log,missing=missing,
            unexpected_new_files=extra,scoring=scoring,frozen_launch_evidence=launch_check)
dump_file('final_checks.json',result)
print(json.dumps(result,ensure_ascii=False))
