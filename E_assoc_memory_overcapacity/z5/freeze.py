"""One-shot freeze of Z5 read-only reanalysis; never starts an experiment."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def digest(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main():
    frozen = ROOT / 'Z5_FROZEN.sha256'
    assert not frozen.exists(), 'Z5 already frozen; refusing overwrite'
    hashes = json.loads((ROOT / 'z5/existing_audit/input_hashes.json').read_text())
    for rel, expected in hashes.items():
        assert digest(rel) == expected, ('read input changed', rel)
    reviewed = ['ICML_REVIEW_SIM8.md', 'Z4B_PREREG.md', 'Z4B_RESULTS.md',
                'Z4B_SEED_PROOF.md', 'z4b/model.py', 'z4b/seed_domain.py',
                'Y3D_PREREG.md', 'Y3D_RESULTS.md', 'y3d/config.json',
                'y3d/parameters.json', 'y3d/predictions.csv', 'y3d/predict.py',
                'RESULTS_EXP1.md', 'RESULTS_EXP2.md', 'FT_RESULTS.md',
                'P1NUM_RESULTS.md', 'W4_PREREG.md', 'W4_RESULTS.md',
                'W4_FREEZE.sha256', 'W4_ARTIFACTS.sha256',
                'paper/sections/main_results.tex', 'paper/sections/experiments.tex',
                'paper/appendix/H_experiments.tex', 'paper/appendix/L_multistep.tex',
                'paper/appendix/M_nonlinear_finite_d.tex']
    with (ROOT / 'z5/reviewed_inputs.sha256').open('x') as f:
        f.writelines(f'{digest(rel)}  {rel}\n' for rel in reviewed)
    files = ['Z5_EXISTING.md', 'Z5_READY.md', 'z5/audit_existing.py',
             'z5/freeze.py', 'z5/reviewed_inputs.sha256',
             'z5/existing_audit/report.json', 'z5/existing_audit/cdf_grid.csv',
             'z5/existing_audit/input_hashes.json']
    with frozen.open('x') as f:
        f.writelines(f'{digest(rel)}  {rel}\n' for rel in files)
    print(json.dumps(dict(frozen_files=len(files), reviewed_inputs=len(reviewed),
                          manifest_sha256=digest('Z5_FROZEN.sha256'), new_draws=0)))


if __name__ == '__main__':
    main()
