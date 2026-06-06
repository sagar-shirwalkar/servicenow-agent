"""Snapshot the current RAG bundle so a new one can be safely installed.

Operational pair with `atlas/restore.py`. Creates a timestamped
tar.gz of the current bundle, prunes old snapshots past `--keep`.

Snapshots are tar.gz archives of the bundle directory, named with
a UTC timestamp. The default retention is 5; older snapshots are
pruned automatically.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _timestamp() -> str:
    return subprocess.run(
        ["date", "+%Y%m%dT%H%M%SZ"], capture_output=True, text=True, check=True
    ).stdout.strip()


def snapshot(bundle_dir: Path, backup_dir: Path) -> Path:
    if not (bundle_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"No manifest at {bundle_dir}; nothing to back up")
    backup_dir.mkdir(parents=True, exist_ok=True)
    name = bundle_dir.name
    archive = backup_dir / f"snapshot-{_timestamp()}.tar.gz"
    print(f"  Creating {archive}...")
    subprocess.run(
        [
            "tar",
            "-czf",
            str(archive),
            "-C",
            str(bundle_dir.parent),
            name,
        ],
        check=True,
    )
    return archive


def prune(backup_dir: Path, keep: int) -> None:
    snapshots = sorted(backup_dir.glob("snapshot-*.tar.gz"), key=lambda p: p.name)
    for old in snapshots[:-keep]:
        print(f"  Pruning old snapshot: {old.name}")
        old.unlink()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Snapshot the current RAG bundle")
    p.add_argument("--bundle", required=True, type=Path, help="Current bundle directory")
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Where to put backups (default: <bundle>/.backups)",
    )
    p.add_argument("--keep", type=int, default=5, help="Snapshots to keep after prune")
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    backup_dir = args.backup_dir or (args.bundle / ".backups")
    archive = snapshot(args.bundle, backup_dir)
    print(f"  Wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    prune(backup_dir, args.keep)
    return 0


def main() -> None:
    """Script entry point. Calls ``sys.exit(_run())`` so the return
    code propagates through both ``python -m`` and the console-script
    entry points defined in ``pyproject.toml``."""
    sys.exit(_run())


if __name__ == "__main__":
    main()
