"""Small independent checks of W7: real one-step FT, population retrieval matching.

Run only with the resource-limited command recorded in README.md.
No project modules, external data, APIs, or author experiment seeds are used.
"""
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import ctypes
import json
from pathlib import Path
import resource
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

OUT = Path(__file__).resolve().parent


def shape_moments(name):
    if name == "GD":
        return 1.0, 1.0, 1.0
    if name == "Sign":
        return np.sqrt(2 / np.pi), 1.0, np.sqrt(2 / np.pi)
    epsilon = float(name.split("_")[1])
    density = lambda z: np.sqrt(2 / np.pi) * np.exp(-z * z / 2)
    aa = quad(lambda z: z * z / (z + epsilon) * density(z), 0, np.inf,
              epsabs=2e-12)[0]
    qq = quad(lambda z: (z / (z + epsilon)) ** 2 * density(z), 0, np.inf,
              epsabs=2e-12)[0]
    return aa, qq, aa / np.sqrt(qq)


def endpoint(name, r, target):
    aa, qq, efficiency = shape_moments(name)
    sigma = np.sqrt(2 * (1 - r))
    kk = sigma * efficiency / 2
    bb = (1 + r) * kk / np.sqrt(1 - kk * kk)
    mm = np.hypot(r, bb)
    theta = np.arctan2(bb, r) - np.arccos(target / mm)
    normalized_update = np.sin(theta) / np.sqrt(1 - kk * kk)
    normalized_initial = np.cos(theta) + kk * normalized_update
    tt = normalized_update / normalized_initial
    assert 0 < theta < np.arccos(-kk) and normalized_initial > 0
    if name == "GD":
        rate = tt / sigma
        b = rate * sigma**2
        v = rate**2 * sigma**2
    else:
        rate = tt / np.sqrt(qq)
        b = rate * sigma * aa
        v = rate**2 * qq
    q = 1 - b + v
    c0, c1 = (1 - b / 2) / np.sqrt(q), (r + b / 2) / np.sqrt(q)
    assert abs(c1 - target) < 1e-11 and abs(c0 - np.cos(theta)) < 1e-11
    return dict(rate=float(rate), b=float(b), v=float(v), q=float(q),
                c0=float(c0), c1=float(c1), c=float(efficiency),
                target=float(target))


def displacement(name, residual, rate, sigma):
    if name == "GD":
        return rate * residual
    if name == "Sign":
        return rate * np.sign(residual)
    return rate * residual / (np.abs(residual) + float(name.split("_")[1]) * sigma)


def evaluate(uu, xx, yy, query_ids, q):
    correct = np.empty(len(query_ids))
    ranks = np.empty(len(query_ids))
    candidate_ks = []
    for left in range(0, len(query_ids), 64):
        ids = query_ids[left:left + 64]
        scores = yy.T @ (uu @ xx[:, ids]) / np.sqrt(q)
        diag = scores[ids, np.arange(len(ids))]
        correct[left:left + len(ids)] = diag
        ranks[left:left + len(ids)] = np.sum(scores > diag[None, :], axis=0) / (xx.shape[1] - 1)
        if left == 0:
            for col, index in enumerate(ids[:8]):
                wrong = np.sort(np.delete(scores[:, col], index))
                ff = ndtr(wrong)
                nn = len(wrong)
                candidate_ks.append(float(max(np.max(np.arange(1, nn + 1) / nn - ff),
                                              np.max(ff - np.arange(nn) / nn))))
    return correct, ranks, float(np.mean(candidate_ks))


def cgroup_snapshot():
    line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines()
                if line.startswith("0::"))
    location = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    return {"path": str(location), "memory.max": (location / "memory.max").read_text().strip(),
            "memory.swap.max": (location / "memory.swap.max").read_text().strip()}


def blas_threads():
    found = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "openblas" not in line.lower() or "/" not in line:
            continue
        library = line.split()[-1]
        if any(entry["library"] == library for entry in found):
            continue
        handle = ctypes.CDLL(library)
        for symbol in ("openblas_get_num_threads", "openblas_get_num_threads64_",
                       "scipy_openblas_get_num_threads", "scipy_openblas_get_num_threads64_"):
            try:
                function = getattr(handle, symbol)
                function.restype = ctypes.c_int
                count = function()
                found.append({"library": library, "threads": count})
                assert count <= 2
                break
            except AttributeError:
                continue
    return found


def main():
    started = time.monotonic()
    scope = cgroup_snapshot()
    assert scope["memory.max"] == str(2 * 1024**3)
    assert scope["memory.swap.max"] == "0"
    cases = [("independent", 0., 0.), ("shared_zero", 1., 0.),
             ("shared_positive", 1., .5), ("split_positive", np.sqrt(.5), np.sqrt(.5)),
             ("shared_negative", 1., -.5), ("split_negative", np.sqrt(.5), -np.sqrt(.5))]
    methods = ["GD", "Sign", "Adam_0.1", "Adam_1.0"]
    dimensions = [32, 64, 128]
    seed_count = 4
    kappa = .5
    quantiles = np.array([.01, .025, .05, .1, .2, .3, .5, .7, .9])
    rows, cdf_rows, pools, pair_pools, forecasts = [], [], {}, {}, {}
    max_identity_error = 0.
    for case_index, (case, rx, ry) in enumerate(cases):
        r = rx * ry
        target = r + .35 * (1 - r)
        forecasts[case] = {method: endpoint(method, r, target) for method in methods}
        for d in dimensions:
            n = round(kappa * d * d)
            for seed_index in range(seed_count):
                rng = np.random.default_rng(np.random.SeedSequence([730619, case_index, d, seed_index]))
                x0 = rng.normal(size=(d, n)) / np.sqrt(d)
                y0 = rng.normal(size=(d, n)) / np.sqrt(d)
                x1 = rx * x0 + np.sqrt(1 - rx * rx) * rng.normal(size=(d, n)) / np.sqrt(d)
                y1 = ry * y0 + np.sqrt(1 - ry * ry) * rng.normal(size=(d, n)) / np.sqrt(d)
                s1 = d / n * (x1 @ x1.T)
                z0, z1 = d / np.sqrt(n) * (y0 @ x0.T), d / np.sqrt(n) * (y1 @ x1.T)
                residual = z1 - z0 @ s1
                proxy_residual = z1 - z0
                query_ids = rng.choice(n, min(256, n), replace=False)
                for method in methods:
                    pred = forecasts[case][method]
                    rate, sigma = pred["rate"], np.sqrt(2 * (1 - r))
                    uu = z0 + displacement(method, residual, rate, sigma)
                    proxy = z0 + displacement(method, proxy_residual, rate, sigma)
                    if method == "GD":
                        identity_error = np.max(np.abs((uu - proxy) + rate * z0 @ (s1 - np.eye(d))))
                        max_identity_error = max(max_identity_error, float(identity_error))
                    mse = np.mean((uu - proxy)**2)
                    two_correct = []
                    for task, xx, yy in [(0, x0, y0), (1, x1, y1)]:
                        correct, rank, candidate_ks = evaluate(uu, xx, yy, query_ids, pred["q"])
                        mu = pred[f"c{task}"] / np.sqrt(kappa)
                        sorted_correct = np.sort(correct)
                        pp = ndtr(sorted_correct - mu)
                        nn = len(correct)
                        correct_ks = max(np.max(np.arange(1, nn + 1) / nn - pp),
                                         np.max(pp - np.arange(nn) / nn))
                        rows.append(dict(case=case, rx=rx, ry=ry, r=r, d=d, n=n,
                                         seed=seed_index, method=method, task=task, rate=rate,
                                         predicted_mu=mu, correct_mean=np.mean(correct),
                                         correct_variance=np.var(correct), correct_ks=correct_ks,
                                         wrong_cdf_ks=candidate_ks, proxy_weight_mse=mse,
                                         rank_mean=np.mean(rank), predicted_rank_mean=ndtr(-mu / np.sqrt(2))))
                        for tau in quantiles:
                            cdf_rows.append(dict(case=case, d=d, seed=seed_index, method=method,
                                                 task=task, tau=float(tau), observed=float(np.mean(rank <= tau)),
                                                 predicted=float(ndtr(mu + ndtri(tau)))))
                        key = (case, d, method, task)
                        pools.setdefault(key, []).append((correct, rank, mu))
                        two_correct.append(correct)
                    pair_pools.setdefault((case, d, method), []).append(np.array(two_correct))
            print(f"finished {case} d={d}", flush=True)
    summary = []
    for (case, d, method, task), entries in pools.items():
        correct = np.concatenate([entry[0] for entry in entries])
        rank = np.concatenate([entry[1] for entry in entries])
        mu = entries[0][2]
        grid_observed = np.array([np.mean(rank <= tau) for tau in quantiles])
        grid_predicted = ndtr(mu + ndtri(quantiles))
        select = [row for row in rows if (row["case"], row["d"], row["method"], row["task"]) == (case, d, method, task)]
        joint = np.concatenate(pair_pools[(case, d, method)], axis=1)
        summary.append(dict(case=case, d=d, method=method, task=task,
                            predicted_mu=mu, correct_mean=float(np.mean(correct)),
                            correct_variance=float(np.var(correct)),
                            wrong_cdf_ks=float(np.mean([row["wrong_cdf_ks"] for row in select])),
                            rank_grid_max_error=float(np.max(np.abs(grid_observed - grid_predicted))),
                            recall_at_10pct=float(np.mean(rank <= .1)),
                            predicted_recall_at_10pct=float(ndtr(mu + ndtri(.1))),
                            correct_cross_cov=float(np.cov(joint, ddof=0)[0, 1]),
                            predicted_cross_cov=float(next(c[1]*c[2] for c in cases if c[0] == case))))
    for filename, records in [("rows.csv", rows), ("rank_cdf.csv", cdf_rows), ("summary.csv", summary)]:
        with (OUT / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    with (OUT / "predictions.json").open("w") as stream:
        json.dump(forecasts, stream, indent=2)
    meta = dict(scope=scope, blas_threads=blas_threads(), dimensions=dimensions,
                kappa=kappa, seed_count=seed_count, query_count=256,
                dataset_count=len(cases)*len(dimensions)*seed_count,
                method_task_records=len(rows), max_gd_identity_error=max_identity_error,
                seconds=time.monotonic()-started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                numpy=np.__version__, seed_entropy=730619,
                note="Pooled CDF checks are descriptive; query/seed dependence is not ignored in a significance test.")
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
