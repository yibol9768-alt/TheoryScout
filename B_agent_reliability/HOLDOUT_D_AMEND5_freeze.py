"""冻结 v5/AMEND5；--test-manifest 仅用于离线合成评分fixture。"""
import hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT/'HOLDOUT_D_runner_v5'
def main():
    if sys.argv[1:]==['--test-manifest']:
        names=('engine.py','protocol.py','transport.py','run.py','launch.sh','evidence.py','score.py')
        paths=[R/n for n in names]
    else:
        assert not sys.argv[1:]
        paths=[p for p in R.rglob('*') if p.is_file()]+[p for p in ROOT.glob('HOLDOUT_D_AMEND5_*') if p.is_file()]
    raw=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(paths))
    (ROOT/'HOLDOUT_D_AMEND5.sha256').write_text(raw)
    if sys.argv[1:]:
        with (R/'HOLDOUT_D_AMEND5_TEST_MANIFEST.txt').open('x') as f:f.write(raw)
    print('AMEND5 frozen',len(paths),'files; manifest SHA256',hashlib.sha256(raw.encode()).hexdigest())
if __name__=='__main__':main()
