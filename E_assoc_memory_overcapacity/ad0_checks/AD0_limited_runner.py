"""Load the new AD0 module, then enforce one numerical worker before numpy import."""
import importlib.util
import os
from pathlib import Path

path = Path(__file__).with_name('AD0_smoke.py')
spec = importlib.util.spec_from_file_location('ad0_smoke', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
module.main()
