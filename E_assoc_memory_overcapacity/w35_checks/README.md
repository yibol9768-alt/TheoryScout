# W35 independent numerical checks

Research conclusions and full arguments are in `../W35_MATCHING.md` and `../W35_STABILIZER.md`.
No old experiment code is imported. Every output is created exclusively; existing results are never overwritten.

Run commands from the E project root, **sequentially**:

```bash
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python w35_checks/check_w35.py matching

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python w35_checks/check_w35.py stabilizer

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python w35_checks/certify_stabilizer.py

systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /home/liuyibo/venvs/theory/bin/python w35_checks/audit_w35_v2.py
```

These commands are already completed. Reproduction requires a fresh directory with the same scripts, or new W35-prefixed output names: do not delete/overwrite the saved results merely to rerun them.

Artifacts:

| File | Content |
|---|---|
| `input_hashes.sha256` | Eight relevant pre-existing inputs, recorded before checks |
| `matching_population.json` | 10 scalar configurations, floating roots, fixed-24 replay, derivative grid diagnostics |
| `matching_rows.json` | 160 real trajectories, each with a 49-point grid and same-data 36-step search |
| `matching_summary.json` | Every dimension / optimizer-family summary, including nonmonotone finite-dimension errors |
| `matching_meta.json` | Resource audit for matching; 6.55 s |
| `stabilizer_forecasts.json` | Deterministic quadrature, variance bounds, sufficient bands; displayed small decimals are not interval certificates |
| `polar_rows.json` | 40 actual centered polar datasets; exact commutator / projection / dense-score checks |
| `polar_summary.json` | Dimension trends; no finite-dimensional enclosure assertion |
| `stabilizer_meta.json` | Resource audit for actual polar; 3.10 s |
| `stabilizer_certificate.json` | Outward integer intervals certifying F5 signs and F4 broad sufficient bands |
| `certificate_meta.json` | Resource audit for rational certificate; 19.09 s |
| `audit_w35.json` | Coverage, integer-sign verification, input-drift audit, hashes of the delivered artifacts |

All successful numerical runs enforce cgroup MemoryMax=2 GiB, MemorySwapMax=0 and loaded BLAS thread counts <=2. No GPU or LLM API is used. The only files read outside this project are the explicitly requested Python installation/runtime and OS `/proc`/cgroup metadata; no other research project is accessed.

The first final-audit attempt (`audit_w35.py`) stopped because two pre-existing inputs changed during this task: `paper/appendix/F.tex` (mtime 05:10:40 -0800) and `paper/sections/main_results.tex` (mtime 05:15:52 -0800), after the baseline hash snapshot at 05:09:15 -0800. This task's tool writes did not target either file. They were not reverted or overwritten. The other six tracked original inputs remained unchanged. `audit_w35_v2.py` retains the original snapshot, records both baseline/current hashes and mtimes, and completes the independent numerical-output audit; it does not label the input-integrity check as passed. A failed preliminary audit is not hidden as a successful run.

The matching grids are diagnostics of a theorem proved in the document, not a numerical proof of a continuous supremum. Adam T=4 root uniqueness at 25% loss reduction remains unproved: the derivative grid does not certify all intermediate parameters. The exact-polar variance theorem is an asymptotic probability limit; several small-dimension observations fall outside its limiting enclosure and are all retained.
