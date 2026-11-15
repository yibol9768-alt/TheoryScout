"""Hash and verify the frozen documents, code and final artifacts, in-project."""
from util import HERE, ROOT, sha, verify_prereg


def main():
    verify_prereg()
    output=HERE/'ARTIFACTS.sha256'
    fixed=[ROOT/'PREREG_ROUND3.md',ROOT/'PREREG_ROUND3.sha256',ROOT/'RESULTS_ROUND3.md',
           ROOT/'FORMULATION_V2.md',ROOT/'RESULTS_WEEK1.md',
           ROOT/'experiments/model.py',ROOT/'experiments/offline_v2.py',ROOT/'experiments/analysis_notes.md',
           ROOT/'experiments/results/v2_offline/diagnostics.json']
    paths=sorted(set(fixed+[p for p in HERE.rglob('*') if p.is_file() and p!=output and '__pycache__' not in p.parts and p.suffix!='.log']))
    lines=[f'{sha(path)}  {path.relative_to(ROOT)}\n' for path in paths]
    output.write_text(''.join(lines))
    for line in output.read_text().splitlines():
        checksum,name=line.split('  ',1)
        assert sha(ROOT/name)==checksum,name
    verify_prereg()
    print(f'Frozen prereg unchanged; {len(paths)} artifact checksums verified')


if __name__=='__main__':
    main()
