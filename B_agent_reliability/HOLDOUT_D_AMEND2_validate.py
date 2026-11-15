"""只读复核本轮新冻结与全部旧文件；无网络、无模型/额度客户端。"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
    rel=Path('/proc/self/cgroup').read_text().strip().split('0::')[-1]
    cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
    resources=dict(memory_max=(cg/'memory.max').read_text().strip(),swap_max=(cg/'memory.swap.max').read_text().strip(),
                   threads=len(list(Path('/proc/self/task').iterdir())))
    assert resources==dict(memory_max='2147483648',swap_max='0',threads=1),resources
    baseline=json.loads((ROOT/'HOLDOUT_D_AMEND2_BASELINE.json').read_text());changed=[];append_only=[]
    for name,digest in baseline.items():
        p=ROOT/name
        if not p.is_file():changed.append(name);continue
        raw=p.read_bytes()
        if sha(raw)==digest:continue
        # 外层 Codex 会话会自动追加它自己的 stdout 日志；确认旧前缀逐字节未变。
        if name=='codex_dfix2.log':
            h=hashlib.sha256();matched=None
            for i,b in enumerate(raw):
                h.update(bytes([b]))
                if h.hexdigest()==digest:matched=i+1;break
            if matched is not None:
                append_only.append(dict(file=name,original_bytes=matched,current_bytes=len(raw),original_sha256=digest));continue
        changed.append(name)
    manifests={}
    for name in ('HOLDOUT_D_PREREG.sha256','HOLDOUT_D_AMEND1.sha256','HOLDOUT_D_AMEND2.sha256'):
        errors=[];entries=0
        for line in (ROOT/name).read_text().splitlines():
            digest,path=line.split('  ',1);p=(ROOT/path).resolve();entries+=1
            if not p.is_relative_to(ROOT) or not p.is_file() or sha(p.read_bytes())!=digest:errors.append(path)
        manifests[name]=dict(entries=entries,errors=errors)
    permitted=lambda p:p.is_relative_to(ROOT/'HOLDOUT_D_runner_v2') or p.name.startswith('HOLDOUT_D_AMEND2_') or p.name=='HOLDOUT_D_AMEND2.sha256'
    unexpected=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and str(p.relative_to(ROOT)) not in baseline and not permitted(p)]
    release=ROOT/'HOLDOUT_D_runner_v2/HOLDOUT_D_LIVE_RELEASE.json'
    live_ledger=ROOT/'HOLDOUT_D_runner_v2/HOLDOUT_D_live'
    z=dict(resources=resources,baseline_files=len(baseline),unchanged=len(baseline)-len(changed)-len(append_only),
           changed=changed,outer_session_append_only=append_only,manifests=manifests,unexpected_new_files=unexpected,
           live_release_absent=not release.exists(),live_ledger_absent=not live_ledger.exists())
    print(json.dumps(z,ensure_ascii=False,indent=2))
    assert not changed and not unexpected and not release.exists() and not live_ledger.exists()
    assert all(not m['errors'] for m in manifests.values())

if __name__=='__main__':main()
