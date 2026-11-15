# Round 2 local CPU experiment

Research decisions are frozen in `../../PREREG_ROUND2.md`. The experiment uses
NumPy float64 analytic gradients and imports the unchanged objective definitions
from `../week1_common.py`; there are no remote services or downloads.

Files:

- `core.py`: data, metrics, displacement root finding, Adam, training, and cloned-state counterfactuals.
- `prepare.py`: the preregistered 18 timing segments, numerical checks, and budget decision. Appends the timing decision to the preregistration **before** main training.
- `run.py`: two workers and one supervisor, hard address-space limit, shared deadline, atomic per-run JSON writes, and pre-training hashes.
- `summarize.py`: checks fingerprints and hashes, recomputes endpoints from row observations, applies the frozen criteria, and generates the Chinese report and four PNGs.
- `budget_decision.json`, `validation.json`, `manifest.json`, `execution.json`, `reporting_resources.json`: timing, checks, frozen configuration, and resources.
- `raw/*.json`: all 90 runs, including all checkpoint counterfactuals, unsuccessful comparisons, and per-row terminal margins.
- `worker0.log`, `worker1.log`: completion events and any worker errors.

To regenerate plots and the numeric report from the frozen raw observations:

```bash
/home/liuyibo/venvs/theory/bin/python experiments/round2/summarize.py
```

The report's interpretation is stored in `interpretation.md` after reviewing the
numeric output and included by the summary generator. No additional training is
performed by summarization.

To repeat training, first preserve this entire project in a separate writable copy;
these commands write the same output paths:

```bash
/home/liuyibo/venvs/theory/bin/python experiments/round2/prepare.py
/home/liuyibo/venvs/theory/bin/python experiments/round2/run.py
/home/liuyibo/venvs/theory/bin/python experiments/round2/summarize.py
```

Preparation applies the timing-only rule again, so another machine may select a
different common step budget. After fresh training, review `interpretation.md`
against the new observations before publishing a new report. This run selected 84,000 steps, with the same cosine
schedule compressed to that budget. Both parameter groups have an initial actual
per-coordinate RMS displacement of 0.001. The reported Frobenius displacement
therefore depends on group size; it is not expected to match between U and V.

Metric conventions:

- Ranking margin: `min_positive score - max_negative score` within each query.
- Zero-threshold sign margin: `min_i label_i * score_i`; the global margin is the minimum over queries.
- Geometric margins use unit vectors and unscaled scores; scaled sign margins are also saved separately.
- Quantiles use NumPy's default linear interpolation at `[0,.05,.1,.25,.5,.75,.9,.95,1]`.
- Zero ranking/sign margin is failure. Recall averages uniform random breaking of exact ties.

Each counterfactual compares a full NCE gradient with `grad(NCE) - grad(D)` at
identical parameters, optimizer moments, step number, and cosine rate. It includes
the temperature gradient. It discards the branches afterwards. The main mechanism
criterion uses NCE trajectories only; PE and sigmoid checkpoint interventions are
saved as additional diagnostics. Removing D is not the PE objective, and the local
effect is not a guarantee about a separately trained trajectory.
