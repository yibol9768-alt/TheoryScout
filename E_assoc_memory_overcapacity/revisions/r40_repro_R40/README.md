# E manuscript R40: local offline build and diagnostic replay

This bundle contains the complete 8-page body / 153-page manuscript source,
reference PDF, figures, fonts/styles, the pinned static Tectonic 0.15.0 engine
and cached TeX inputs. It also contains compact Z4B dataset success counts
and the exact-rational matching-uncertainty calculation for table 44.

On x86_64 Linux with Python 3.12 or later and Poppler's `pdfinfo` / `pdftotext`:

```sh
python3 -B verify_bundle.py
python3 -B reproduce.py --out /absolute/path/to/NEW_output
```

The output directory must not exist and must be outside this bundle. Rebuilds
start without any previous PDF/auxiliary files and use only the bundled TeX
cache. The bundle remains immutable. The run checks manuscript sources and
layout, compares the entire rebuilt PDF text against the reference, and
reproduces the archived Z4B statistics and rational polytope extrema.
If the bound manuscript contains the uniform-GD proposition, it additionally
replays its 72 deterministic rational-matrix cases, with 63 finite-step
recurrence/filter comparisons. These are algebra checks, not independent
mathematical review or random experimental evidence.
If it contains the scatter-measurable GD corollary, it also checks 36 exact
residual-based stops, 27 finite conditional-selection identities and 20
Gaussian Wick-pairing norm formulas. Generic bounded-loss counterexamples
separate unconditional and conditional risk envelopes and rank peeking;
they are not failure examples for GD or empirical stopping experiments.
If the manuscript contains the radial sphere-threshold proposition, the run
also unpacks the hash-bound R21 numerical sub-bundle, replays its 36 analytic
CDF brackets and complete random-threshold conditional lower bounds at one
represented C0, and runs the associated 110-digit arithmetic diagnostics.
It requires no sampled threshold bank. The 18 point probability lower bounds
are inherited certified inputs; their native C0 residual chain and quadratic
integrals are not reconstructed by this entry point. The result retains the
R9--R13 numerical assumptions and is neither an unrounded Gaussian input
certificate nor a guarantee for all queries of one realized dataset.
PDF byte identity is recorded separately; it is not assumed from text identity.

If the manuscript contains the conditional score--energy moment proposition,
the default run also replays the R28 exact-trace/moment propagation and R29
scalar efficiency quadrature, including their exact reference checks. These
hash-bound research subarchives inherit the prior spectral/coupling enclosures;
the native residual chain and original Gaussian construction are not rebuilt.
The R29 unit-Gaussian center-proxy effect is not an actual training-recall bound.

If it contains the energy-attribution obstruction, the run also reconstructs
its exact rational joint laws, identical marginals and conditional means,
and 60 strict threshold-interval comparisons. This is a generic probability
counterexample with a different threshold law, not a realization of the
spherical C0 model or a claim about its signed energy effect.

If it contains the exact block-radius comparison, the run checks its
rational signed-threshold event algebra, joint-pencil sign comparisons,
small-integer Beta CDF density polynomials and radius/direction monomial
factorizations. It does not compute the actual high-dimensional joint
integral or certify the actual C0 energy remainder sign.

On Linux allowing unprivileged namespaces, a stronger network-disabled test is:

```sh
unshare -Urn python3 -B reproduce.py --out /absolute/path/to/NEW_output
```

`COVERAGE.json` lists active table/figure labels and expanded-source table,
longtable and figure environments, including unlabeled ones. These are source
inventory entries, not a claim about the number of captioned PDF objects.
Except table 44's compact
diagnostic, their presence means source/rendering availability, not a rerun
of their scientific experiments. The 1280-by-4-by-2 Z4B integer success counts
reproduce the seven matching and three old-contrast means/SEs, using archived
Student-t critical values. Raw rank NPZ files are represented by provenance
hashes; they are not included. Model slopes are archived coefficients, not
newly certified derivatives. All local-proxy and nominal-inference limits
in the manuscript still apply.

Other experiments, exact matching, other random-threshold cases, predecessor
C0 certificate construction and mathematical correctness are outside this replay. This is a local research
artifact, with OS/runtime prerequisites stated above; it is not a complete
OS image or a claim of scientific replication of all 153 pages.
