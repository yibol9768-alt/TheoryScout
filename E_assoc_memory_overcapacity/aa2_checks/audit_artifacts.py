"""Read-only audit of AA2 outputs and protected inputs; writes one NEW manifest."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    assert out.parent.resolve() == Path("aa2_checks").resolve()
    if out.exists():
        raise FileExistsError(out)
    results = {}
    protected = {}
    for name in ("retrieval_results.json", "cue_results.json", "cue_large_results.json"):
        path = Path("aa2_checks")/name
        data = json.loads(path.read_text())
        before = data.get("inputs_sha256_before", data.get("input_hashes_before"))
        after = data.get("inputs_sha256_after", data.get("input_hashes_after"))
        assert before == after and data["inputs_unchanged"]
        for source, digest in before.items():
            actual = hashlib.sha256(Path(source).read_bytes()).hexdigest()
            assert actual == digest, source
            if source in protected:
                assert protected[source] == digest
            protected[source] = digest
        dims = data["parameters"]["dims"]
        seeds = data["parameters"]["seeds"]
        factor = 4*2*data["parameters"]["noise_repeats"] if name.startswith("retrieval_") else 5
        assert len(data["records"]) == len(dims)*seeds*factor
        results[name] = dict(record_count=len(data["records"]),inputs_unchanged=True,
                             seconds=data["seconds"])
        if "roots" in data:
            for root in data["roots"]:
                g, tau, mu, nu = (root[k] for k in ("gamma","tau","mu","nu"))
                assert root["mean_root"] < root["cue_root"] < root["msr_root"]
                assert nu < 1+mu*mu/g
                t = 1+tau*tau
                cq2 = mu*mu*t/(t-mu*mu/g)
                # Squared equality of the two cue-retrieval efficiency factors.
                assert abs(cq2/(t+cq2/g)-mu*mu/t) < 2e-15
    syntax = []
    for p in sorted(Path("aa2_checks").glob("*.py")):
        ast.parse(p.read_text(),filename=str(p))
        syntax.append(str(p))
    manifest_paths = [Path("AA2_EXT_RETRIEVAL.md"), *sorted(Path("aa2_checks").glob("*"))]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_paths if p.is_file()}
    result = dict(results=results,protected_inputs_and_sources=protected,syntax_parsed=syntax,
                  artifact_sha256=hashes,python=sys.executable,
                  threads={k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
                  note="Checks consistency and integrity, not a formal proof or an interval certificate.")
    with out.open("x") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(dict(output=str(out),all_checks_passed=True,records=sum(r["record_count"] for r in results.values()))))


if __name__ == "__main__":
    main()
