"""Restore a previously snapshotted RAG bundle.

Operational pair with `atlas/backup.py`. Lists, picks, and swaps
back. `restore.py` itself snapshots the current bundle as a safety
net before swapping, unless `--no-safety-snapshot` is set.

Lists available snapshots, lets the user pick one (or accepts
``--latest``), snapshots the current bundle as a safety net, and
replaces it with the chosen snapshot.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def list_snapshots(backup_dir: Path) -> list[Path]:
    return sorted(backup_dir.glob("snapshot-*.tar.gz"), key=lambda p: p.name)


def pick_latest(backup_dir: Path) -> Path:
    snaps = list_snapshots(backup_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots in {backup_dir}")
    return snaps[-1]


def restore(snapshot: Path, bundle_dir: Path) -> None:
    if not snapshot.is_file():
        raise FileNotFoundError(f"Snapshot not found: {snapshot}")
    parent = bundle_dir.parent
    name = bundle_dir.name
    print(f"  Extracting {snapshot.name} into {parent}/...")
    subprocess.run(
        ["tar", "-xzf", str(snapshot), "-C", str(parent)],
        check=True,
    )
    extracted = parent / name
    if not (extracted / "manifest.json").is_file():
        raise RuntimeError("Restored bundle is missing manifest.json")
    print(f"  Restored to {extracted}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore a previously snapshotted RAG bundle")
    p.add_argument("--bundle", required=True, type=Path, help="Bundle directory to write into")
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Where snapshots live (default: <bundle>/.backups)",
    )
    p.add_argument(
        "--from",
        dest="source",
        default=None,
        help="Snapshot filename (default: --latest)",
    )
    p.add_argument(
        "--list", action="store_true", help="List available snapshots and exit"
    )
    p.add_argument(
        "--no-safety-snapshot",
        action="store_true",
        help="Skip snapshotting the current bundle before restoring",
    )
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    backup_dir = args.backup_dir or (args.bundle / ".backups")
    if args.list:
        for s in list_snapshots(backup_dir):
            print(f"  {s.name}\t{s.stat().st_size / 1e6:.1f} MB")
        return 0

    if not args.no_safety_snapshot and (args.bundle / "manifest.json").is_file():
        print("  Snapshotting current bundle as a safety net...")
        subprocess.run(
            [sys.executable, "-m", "atlas.backup", "--bundle", str(args.bundle)],
            check=True,
        )

    if args.source:
        snapshot = backup_dir / args.source
    else:
        snapshot = pick_latest(backup_dir)
    print(f"  Restoring {snapshot.name}...")
    restore(snapshot, args.bundle)
    return 0


def main() -> None:
    """Script entry point. Calls ``sys.exit(_run())`` so the return
    code propagates through both ``python -m`` and the console-script
    entry points defined in ``pyproject.toml``."""
    sys.exit(_run())


if __name__ == "__main__":
    main()
