# Round 3 reproduction

Use `/home/liuyibo/venvs/theory/bin/python`. No network/API/GPU.

In a fresh copy/directory, preserve existing results before rerunning:

1. `python experiments/round3/prepare.py`: gradient checks; timing only; freezes T, appends timing decision to preregistration, hashes all training/assessment inputs.
2. `python experiments/round3/run.py`: supervisor + two workers; five exact round-2 NCE replays and mandatory threshold/lift checks, then 45 paired random-initialization runs and 15 NCE-lift warm starts.
3. `python experiments/round3/summarize.py`: recompute final metrics from NPZ, verify hashes, evaluate frozen P1–P4 and write assessment.json.
4. `python experiments/round3/report.py`: descriptive tables, trajectory plot and resource report. This reporting helper and theorem_note.md were added while the frozen training ran; they do not define or modify the pass/fail rules.

The experiment module imports the unchanged round2/core.py and week1_common.py. Environment variables and 3.5GB RLIMIT_AS are set before NumPy import. OMP_NUM_THREADS=3; NumPy BLAS, MKL, NUMEXPR each use one thread. Two workers plus the supervisor are the only concurrent experiment Python processes. Preparation and reporting run separately.

Artifacts: manifest.json (SHA256), validation.json, budget_decision.json, execution.json, replay/ (old seeds10–14), raw/ (new seeds15–19), weights/ (all new final parameters), assessment.json, reporting_resources.json, tables.md, trajectories.png.

`d` in filenames/configs is the base dimension; `actual_d` is d+1 for sigmoid_lift. Warm-start sigmoid resets beta to16 and Adam state, calibrates fresh and runs T more steps; it therefore costs an extra NCE training stage and is a feasibility/initialization control rather than an equal-total-budget algorithm comparison. NPZ weights permit independent verification of every final row's rank and effective sign margin. Technical failures remain in raw results and are not dropped from the requested grid.

Post-run reporting correction: run `python experiments/round3/adjudicate.py` to distinguish one preregistered unavailable lift from technical failures. The frozen summarizer gates all training comparisons on exactly60 completed runs; its unchanged assessment.json is retained, while adjudication.json evaluates observable P2/P3 comparisons and explicitly marks P4 coverage incomplete. See RESULTS_ROUND3.md for the interpretation. checkpoint_description.json is a descriptive extract of already fixed checkpoints, not another training experiment.
