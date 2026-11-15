"""冻结 v3 和 AMEND3 新交付；不修改任何旧件。须在受限 scope 内执行。"""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parent
def main():
    paths=sorted([p for p in (ROOT/'HOLDOUT_D_runner_v3').rglob('*') if p.is_file()]+
                 [p for p in ROOT.glob('HOLDOUT_D_AMEND3_*') if p.is_file()])
    raw=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in paths)
    (ROOT/'HOLDOUT_D_AMEND3.sha256').write_text(raw)
    print('AMEND3 frozen',len(paths),'files')
if __name__=='__main__':main()
