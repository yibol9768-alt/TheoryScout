"""离线基线与 v4 原样复制；只写 AMEND5 和 v5。"""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    target=ROOT/'HOLDOUT_D_runner_v5'
    assert not target.exists()
    baseline={}
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and not (p.parent==ROOT and p.name.startswith('HOLDOUT_D_AMEND5')):
            raw=p.read_bytes();baseline[str(p.relative_to(ROOT))]=dict(sha256=hashlib.sha256(raw).hexdigest(),size=len(raw))
    with (ROOT/'HOLDOUT_D_AMEND5_BASELINE.json').open('x') as f:json.dump(baseline,f,indent=2);f.write('\n')
    shutil.copytree(ROOT/'HOLDOUT_D_runner_v4',target)
    print('baseline',len(baseline),'files; copied v4 to v5')
if __name__=='__main__':main()
