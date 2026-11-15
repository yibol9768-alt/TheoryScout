"""Replay the integral certificate from recorded rational coefficients only."""
from fractions import Fraction
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import certify_weighted_chisquare as certificate
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
inputs=json.loads(Path(__file__).with_name("input.json").read_text())
record,enclosure=certificate.certify([Fraction(v) for v in inputs["coefficients"]],inputs["counts"],certificate.pi_interval(),certificate.gauss8())
with args.out.open("x") as handle:
    json.dump(record,handle,indent=2)
    handle.write("\n")
print(record["probability"])
