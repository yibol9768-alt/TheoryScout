"""Post hoc threshold diagnostic on archived complete-dataset recalls.

The independent-normal candidate model is a diagnostic, not a proved
finite-dimensional approximation to the dependent training scores.
No optimizer, efficiency, load, or prediction parameter is fitted.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, log_ndtr, ndtr, ndtri
from scipy.stats import binom


def gaussian_threshold_prediction(m, length, mu):
    # Integrate over the correct score, then independently over the threshold.
    def correct_integrand(g):
        return math.exp(-g*g/2)/math.sqrt(2*math.pi) * binom.cdf(
            length-1, m, ndtr(-mu-g))

    log_coefficient = (gammaln(m+1)-gammaln(length)-gammaln(m-length+1))

    def threshold_integrand(z):
        log_density = (log_coefficient-z*z/2-math.log(2*math.pi)/2
                       +(length-1)*log_ndtr(-z)+(m-length)*log_ndtr(z))
        return math.exp(log_density)*ndtr(mu-z)

    first, error1 = quad(correct_integrand, -12, 12, epsabs=2e-10, limit=300)
    second, error2 = quad(threshold_integrand, -12, 12, epsabs=2e-10, limit=300)
    if abs(first-second) > 2e-8:
        raise RuntimeError(f"Integral disagreement: {first}, {second}")
    return first, abs(first-second), max(error1, error2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = [root / "ac2_checks/full" / name for name in
             ("AC2_design.json", "AC2_datasets.jsonl", "AC2_results.json")]
    design = json.loads(paths[0].read_text())
    datasets = [json.loads(line) for line in paths[1].read_text().splitlines()]
    archived = json.loads(paths[2].read_text())["rows"]
    rows = []
    for old in archived:
        d, n, method = old["d"], old["n"], old["method"]
        selected = [r for r in datasets if (r["d"], r["n"], r["method"]) == (d, n, method)]
        recalls = np.asarray([r["recall"] for r in selected])
        if len(selected) != old["datasets"] or not np.allclose(
                recalls.mean(axis=0), old["mean"], rtol=0, atol=1e-12):
            raise RuntimeError("Archived dataset means failed replay")
        c = design["methods"][method]["c"]
        mu = c*d/math.sqrt(n)
        b = ndtri(1-1/(n-1))
        for index, length in enumerate(design["lists"]):
            proxy, disagreement, error = gaussian_threshold_prediction(n-1, length, mu)
            observed = float(recalls[:, index].mean())
            rows.append(dict(d=d, n=n, method=method, L=length,
                             datasets=len(selected), mu=mu, s_eff=mu-b,
                             asymptotic=float(ndtr(mu-b)), proxy=proxy,
                             observed=observed, residual_pp=100*(observed-proxy),
                             descriptive_SE=float(recalls[:, index].std(ddof=1)/math.sqrt(len(selected))),
                             integral_disagreement=disagreement, quad_error_estimate=error))
    radius = math.sqrt(math.log(2*2442/.05)/(2*16))
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(scope="Post hoc diagnostic; independent-normal proxy only; no new training data",
                  input_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  rows=rows, REP=dict(datasets=16, family=2442, alpha=.05,
                                     two_sided_Hoeffding_halfwidth=radius,
                                     grid_extension_margin=.01))
    (args.out / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Archived tail-window threshold diagnostic", "",
             "Post hoc analysis; no new training data or fitted parameters. The independent-normal proxy is not a theorem for the actual training scores.", "",
             "| d | Method | L | Observed | Normal threshold proxy | Residual (pp) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['d']} | {r['method']} | {r['L']} | {r['observed']:.6f} | {r['proxy']:.6f} | {r['residual_pp']:+.3f} |")
    lines += ["", f"REP distribution-free simultaneous grid half-width: {radius:.6f}, before the existing 0.01 grid extension. This does not certify 4pp precision.",
              "", "Both correct-score and order-statistic-density integrals are checked; quadrature errors are floating-point estimates, not interval certificates."]
    (args.out / "REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
