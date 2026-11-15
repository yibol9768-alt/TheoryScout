"""Read-only post hoc center diagnostics and deterministic radial-bound checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, log_ndtr, ndtr
from scipy.stats import binom


def radial_bound(m, ratio=1.):
    return (ratio*math.sqrt(math.log(2*m)/math.pi)
            +ratio**2*(2*math.log(2*m)+2)/math.sqrt(2*math.pi*math.e))/m


def proxy(m, length, centers, scales):
    centers, scales = np.asarray(centers), np.asarray(scales)
    if np.any(scales <= 0):
        raise ValueError("Nonpositive Gaussian scale")
    def correct(g):
        return math.exp(-g*g/2)/math.sqrt(2*math.pi)*float(np.mean(
            binom.cdf(length-1, m, ndtr(-centers-scales*g))))
    coefficient = gammaln(m+1)-gammaln(length)-gammaln(m-length+1)
    def threshold(z):
        density = math.exp(coefficient-z*z/2-math.log(2*math.pi)/2
                           +(length-1)*log_ndtr(-z)+(m-length)*log_ndtr(z))
        return density*float(np.mean(ndtr((centers-z)/scales)))
    a, e1 = quad(correct, -12, 12, epsabs=2e-10, epsrel=2e-10, limit=300)
    b, e2 = quad(threshold, -12, 12, epsabs=2e-10, epsrel=2e-10, limit=300)
    if abs(a-b) > 2e-8:
        raise RuntimeError(f"Integral disagreement: {a} versus {b}")
    return a, abs(a-b), max(e1, e2)


def sanity_checks():
    records = []
    for m in (1, 2):
        for length in range(1, m+1):
            for mu in (-1., 0., 2.):
                for ratio in (.5, 2.):
                    # Correct-score variance is one; candidate scale is ratio.
                    normal = proxy(m, length, [mu/ratio], [1/ratio])[0]
                    if m == 1:
                        sphere = float((ndtr(mu-ratio)+ndtr(mu+ratio))/2)
                    else:
                        def integrand(theta):
                            coordinates = (math.sqrt(2)*math.cos(theta), math.sqrt(2)*math.sin(theta))
                            threshold = sorted(coordinates, reverse=True)[length-1]
                            return float(ndtr(mu-ratio*threshold))/(2*math.pi)
                        sphere = quad(integrand, 0, 2*math.pi,
                                      points=[math.pi/4, 5*math.pi/4], epsabs=1e-11)[0]
                    bound = radial_bound(m, ratio)
                    if abs(sphere-normal) > bound+1e-9:
                        raise RuntimeError("Deterministic radial check failed")
                    records.append(dict(m=m, L=length, mu=mu, ratio=ratio,
                                        sphere=sphere, normal=normal, difference=abs(sphere-normal),
                                        bound=bound))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inputs = [root/"ac2_checks/full/AC2_audit.json",
              root/"ac2_checks/full/AC2_design.json",
              root/"revisions/r1_threshold_diagnostic/results.json"]
    audits = json.loads(inputs[0].read_text())["corrected_energy_rows"]
    design = json.loads(inputs[1].read_text())
    original = json.loads(inputs[2].read_text())["rows"]
    rows = []
    for old in original:
        d, n, method, length = old["d"], old["n"], old["method"], old["L"]
        selected = [r for r in audits if (r["d"], r["n"], r["method"]) == (d, n, method)]
        if len(selected) != 12:
            raise RuntimeError("Unexpected audit scatter count")
        centers = np.array([r["mu"] for r in selected])
        scales = np.array([r["centered_D_SD"] for r in selected])
        mu0 = design["methods"][method]["c"]*d/math.sqrt(n)
        center, discrepancy1, error1 = proxy(n-1, length, centers, np.ones(12))
        scale, discrepancy2, error2 = proxy(n-1, length, centers, scales)
        rows.append(dict(d=d,n=n,method=method,L=length,audit_scatters=12,
                         observed_from_separate_complete_datasets=old["observed"],
                         baseline_proxy=old["proxy"], mu0=mu0,
                         mean_audit_center=float(centers.mean()),
                         mean_center_shift=float(centers.mean()-mu0),
                         center_only_proxy=center,
                         center_effect_pp=100*(center-old["proxy"]),
                         center_only_residual_pp=100*(old["observed"]-center),
                         center_and_empirical_correct_scale_proxy=scale,
                         center_scale_residual_pp=100*(old["observed"]-scale),
                         independent_radial_bound_pp=100*radial_bound(n-1),
                         integral_disagreement=max(discrepancy1,discrepancy2),
                         quadrature_error_estimate=max(error1,error2)))
    result = dict(scope="Post hoc, separate audit and recall samples; empirical scales are descriptive; no new training data",
                  energy_identification="RMS energy errors do not determine mean energy or its joint law with correct scores",
                  input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  rows=rows,deterministic_radial_checks=sanity_checks())
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Archived center diagnostic", "",
           "Post hoc, no new training data. Audit centers/scales use 12 separate scatters, not the complete-recall datasets. No causal attribution or confirmatory inference.", "",
           "| d | Method | L | Center shift | Center effect (pp) | Remaining residual (pp) | Radial bound (pp) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['d']} | {r['method']} | {r['L']} | {r['mean_center_shift']:+.6f} | {r['center_effect_pp']:+.3f} | {r['center_only_residual_pp']:+.3f} | {r['independent_radial_bound_pp']:.4f} |")
    lines += ["", "The radial bound compares sphere versus Gaussian order statistics only when the correct Gaussian score is independent of both. It does not bound actual training recall error.",
              "", "The archived RMS energy statistic alone is insufficient to reconstruct the joint correct-score/energy distribution."]
    (args.out/"REPORT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
