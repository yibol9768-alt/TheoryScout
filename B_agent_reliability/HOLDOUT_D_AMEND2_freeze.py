"""排他新建本轮最终清单；所有运行进程结束、报告完成后调用一次。"""
import hashlib
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    rel=Path('/proc/self/cgroup').read_text().strip().split('0::')[-1]
    cg=Path('/sys/fs/cgroup')/rel.lstrip('/')
    assert (cg/'memory.max').read_text().strip()=='2147483648'
    assert (cg/'memory.swap.max').read_text().strip()=='0'
    paths=sorted([p for p in ROOT.glob('HOLDOUT_D_AMEND2_*') if p.is_file()]+
                 [p for p in (ROOT/'HOLDOUT_D_runner_v2').rglob('*') if p.is_file()])
    assert not any(p.is_symlink() or p.suffix in ('.tmp','.pyc') for p in paths)
    assert not (ROOT/'HOLDOUT_D_runner_v2/HOLDOUT_D_LIVE_RELEASE.json').exists()
    text=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in paths)
    with (ROOT/'HOLDOUT_D_AMEND2.sha256').open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
    print('AMEND2 frozen entries:',len(paths))
if __name__=='__main__':main()
