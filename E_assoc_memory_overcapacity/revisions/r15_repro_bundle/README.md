# E manuscript R14: local offline build and Z4B diagnostic replay

This bundle contains the complete 8-page body / 145-page manuscript source,
reference PDF, figures, fonts/styles, the pinned static Tectonic 0.15.0 engine
and cached TeX inputs. It also contains compact Z4B dataset success counts
and the exact-rational matching-uncertainty calculation for table 44.

On x86_64 Linux with Python 3 and Poppler's `pdfinfo` / `pdftotext`:

```sh
python3 -B verify_bundle.py
python3 -B reproduce.py --out /absolute/path/to/NEW_output
```

The output directory must not exist and must be outside this bundle. Rebuilds
start without any previous PDF/auxiliary files and use only the bundled TeX
cache. The bundle remains immutable. The run checks manuscript sources and
layout, compares the entire rebuilt PDF text against the reference, and
reproduces the archived Z4B statistics and rational polytope extrema.
PDF byte identity is recorded separately; it is not assumed from text identity.

On Linux allowing unprivileged namespaces, a stronger network-disabled test is:

```sh
unshare -Urn python3 -B reproduce.py --out /absolute/path/to/NEW_output
```

`COVERAGE.json` lists every active table/figure label. Except table 44's compact
diagnostic, their presence means source/rendering availability, not a rerun
of their scientific experiments. The 1280-by-4-by-2 Z4B integer success counts
reproduce the seven matching and three old-contrast means/SEs, using archived
Student-t critical values. Raw rank NPZ files are represented by provenance
hashes; they are not included. Model slopes are archived coefficients, not
newly certified derivatives. All local-proxy and nominal-inference limits
in the manuscript still apply.

Other experiments, exact matching, full random-threshold certification and
mathematical correctness are outside this replay. This is a local research
artifact, with OS/runtime prerequisites stated above; it is not a complete
OS image or a claim of scientific replication of all 145 pages.
