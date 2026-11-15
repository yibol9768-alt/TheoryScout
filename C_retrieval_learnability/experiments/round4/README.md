# Round 4 reproduction

All experiment files and inputs stay inside this project. No LLM API, network, GPU, or external project imports. Interpreter: `/home/liuyibo/venvs/theory/bin/python`; dependencies: NumPy, SciPy; compiler: gcc. Optional reporting uses matplotlib. Standard installed libraries are used; no torch.

Preregistration: `../../PREREG_ROUND4.md`, immutable checksum: `../../PREREG_ROUND4.sha256`. Formal code/inputs: `manifest.json`. Formal seeds 20–24; smoke seeds901/902. Complete grid: dimensions6/7 × five seeds × zero/b/theta/solve/b_legacy × train/fixed temperature =100 runs. T=100000. All results, including failures and censored runs, remain in the grid.

Every computation must be wrapped as follows (from the project root):

```sh
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 gcc -O3 -shared -fPIC experiments/round4/bias_solver.c -o experiments/round4/bias_solver.so -lm
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round4/smoke.py
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round4/freeze.py
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round4/run.py
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round4/assess.py
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round4/report.py
```

Do not overwrite the delivered artifacts to reproduce; first create a new subdirectory/copy **within this project** and adapt `OUT` or preserve the original outputs. `freeze.py` records hashes only before formal training; do not rerun it over completed outputs to hide changes. Recompiled native binary can have a different hash on another toolchain; retain source and explicitly document this rather than editing original manifest.

`experiment.py`: standalone float64 analytic sigmoid gradients and projected Adam. `bias_solver.c`: safeguarded scalar Newton per row with bracket/bisection fallback, tolerance5e-11 in sum-gradient units. Python root reference is retained for independent comparison. Bias root is determined by sum sigmoid=3; label positions affect geometry gradients but not the intercept optimum given fixed scores and positive count. All calls including root work are included in wall/CPU times. Solver iteration count sums row iterations rather than batch iterations.

Matching details: all branches share raw U,V; b/theta/solve also share initial solved effective threshold at beta16. U/V initial coordinate RMS steps are independently calibrated0.001. Bias learning rates for b/theta use the **zero-bias reference point**, targeting effective threshold RMS0.001. Actual first threshold move at the solved initial point is often near zero and is reported separately. Solve performs exact elimination; its threshold movement is not a gradient group matched to0.001. The zero and legacy branches start with zero bias, so are not initial-effective-score-matched to the solved trio. A b-vs-theta comparison isolates coordinates more cleanly than zero-vs-solve. Legacy-vs-zero is used for the specified temperature intervention and anomaly replication.

Temperature fixed path is16*256**(t/T), shared by all fixed branches. Train temperature uses log(beta) Adam. b- and theta-coordinate temperature gradients differ by b*g_b as required by the chain rule. Solve uses the envelope derivative at fixed optimized b. All branches use the same cosine schedule and steps, not the same eventual temperature or total scalar operations.

Outputs:

- `fixed_geometry.json/.npz`:30 old terminal geometries × native/16/256/4096; all rows retained, comparing original intercept, midpoint, logistic optimum.
- `raw/`, `weights/`:per-trajectory metrics/checkpoints and terminal weights, including beta and eliminated b.
- `assessment.json`, `tables.md`:frozen H1/H1b/H2/H2s/H3/H3d6/H4 adjudication, independent NPZ metric replay, norm and matching checks, complete per-seed table.
- `execution.json`:supervisor wall time and exit codes. Each run records wall time, CPU time, max RSS.
- `implementation_log.md`, `smoke_attempt*/`:preserved budget-infeasible implementation attempts before formal launch; only implementation efficiency changed, no preregistration edits or threshold changes.
- `slowdown_theory.md`:derivation draft only, not another experiment or a new convergence theorem.

Success is all560 queries above geometric margin1e-10; strict >0 results are also retained. First-hit and sustained-hit times refer only to1000-step sampling and final state. A hit at the last sample counts as sustained over the observed suffix, so it does not establish future stability. Censored times are101000 for paired summaries; report censoring explicitly. Robust threshold1e-4 applies to effective geometric scores, not temperature-scaled logits. No query is treated as an independent replicate; the paired seed is the unit.

Run the first-step reporting audit after formal completion:

```sh
systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 /home/liuyibo/venvs/theory/bin/python experiments/round4/first_step_audit.py
```

The frozen `initial_actual_bias_effective_move` field actually holds a coordinate-only displacement at beta16. `first_step_audit.json` corrects the report by reconstructing the first update with actual beta(1), including temperature-induced threshold drift and the profiled solve. This does not alter any update, metric or decision threshold. See implementation_log.md.

Fixed-geometry comparisons at counterfactual beta values retain the saved **logit-unit b** as the original-bias baseline; they do not rescale it to hold geometric theta constant. The midpoint and optimal intercept are recomputed for each beta. Native-beta diagnosis is the direct test of the saved terminal state. Counts across beta settings and geometries are descriptive row-instance counts, not independent statistical samples.
