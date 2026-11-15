"""冻结 v4 与 AMEND4 新交付，不写入其他版本。须在受限 scope 内运行。"""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    paths=sorted([p for p in (ROOT/'HOLDOUT_D_runner_v4').rglob('*') if p.is_file()]+
                 [p for p in ROOT.glob('HOLDOUT_D_AMEND4_*') if p.is_file()])
    raw=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in paths)
    (ROOT/'HOLDOUT_D_AMEND4.sha256').write_text(raw)
    print('AMEND4 frozen',len(paths),'files; manifest SHA256',hashlib.sha256(raw.encode()).hexdigest())
if __name__=='__main__':main()
