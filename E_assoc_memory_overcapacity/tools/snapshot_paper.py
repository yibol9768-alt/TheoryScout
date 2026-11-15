"""Archive a manuscript revision without replacing an existing snapshot."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Short revision label")
    parser.add_argument("--note", required=True, help="Why this version is saved")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain only letters, digits, underscores or hyphens")
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = root / "history" / f"{stamp}_{args.label}"
    destination.mkdir(parents=True, exist_ok=False)
    paths = []
    for path in sorted((root / "paper").rglob("*")):
        relative = path.relative_to(root / "paper")
        if not path.is_file() or any(p.startswith(".") for p in relative.parts):
            continue
        if relative.parts[0] == "build" and relative.as_posix() not in {
            "build/main.pdf", "build/main.log", "build/main.aux"
        }:
            continue
        paths.append(path)
    archive = destination / "paper.tar.gz"
    with tarfile.open(archive, "x:gz") as handle:
        for path in paths:
            handle.add(path, arcname=str(path.relative_to(root)), recursive=False)
    manifest = {
        "created_utc": stamp, "label": args.label, "note": args.note,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in paths},
    }
    with (destination / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(destination)
    print(f"Saved {len(paths)} files; archive SHA-256 {manifest['archive_sha256']}")


if __name__ == "__main__":
    main()
