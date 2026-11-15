"""Queue the user's periodic E manuscript task on the existing Codex thread."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    settings = json.loads((directory / "settings.json").read_text())
    prompt = (directory / "prompt.txt").read_text().strip()
    with (directory / "tick.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another scheduling tick is active; skip.")
            return
        queue = Path.home() / ".codex/queue_1.sqlite"
        pending = False
        if queue.exists():
            with sqlite3.connect(f"file:{queue}?mode=ro", uri=True) as connection:
                pending = any("【E 论文 10 分钟检查与改进】" in row[0]
                              for row in connection.execute(
                                  "SELECT payload_json FROM queued_items WHERE thread_id=?",
                                  (settings["thread_id"],)))
        record = {"time_utc": datetime.now(timezone.utc).isoformat(),
                  "thread_id": settings["thread_id"], "dry_run": args.dry_run}
        if pending:
            record["status"] = "skip_existing_pending_tick"
        elif args.dry_run:
            record["status"] = "ready_to_queue"
        else:
            result = subprocess.run(
                [settings["codex"], "queue", "--thread", settings["thread_id"],
                 "--message", prompt, "--resume", "--cd", settings["workspace"]],
                text=True, capture_output=True, timeout=90,
                env={**os.environ, "PATH": str(Path(settings["codex"]).parent)+":"+os.environ.get("PATH", "")})
            record.update(status="queued" if result.returncode == 0 else "failed",
                          returncode=result.returncode,
                          stdout=result.stdout, stderr=result.stderr)
            if "warning" in result.stderr.lower():
                record["status"] = "queue_warning_check_resume"
        with (directory / "ticks.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False)+"\n")
        print(json.dumps(record, ensure_ascii=False))
        if record["status"] in {"failed", "queue_warning_check_resume"}:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
