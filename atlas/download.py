"""Download a pre-built RAG bundle from GitHub Releases.

End-user entry point. Hits the GitHub Releases API, downloads the
asset, verifies the SHA256 of `chunks.parquet` against the manifest,
extracts in place. If a bundle already exists at `--output`,
snapshots it first via `backup.py` so a broken new bundle can be
rolled back.

Resolves the latest (or pinned) release, downloads the bundle
artifact, verifies its SHA256 against the manifest, and extracts
it to ``--output``. If ``--output`` already has a bundle, the
existing one is snapshotted via ``backup.py`` first so the user
can roll back if the new bundle is broken.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

GITHUB_API = "https://api.github.com"


def _http_json(url: str, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def resolve_release(repo: str, tag: str | None) -> dict:
    if tag:
        return _http_json(f"{GITHUB_API}/repos/{repo}/releases/tags/{tag}")
    return _http_json(f"{GITHUB_API}/repos/{repo}/releases/latest")


def find_bundle_asset(release: dict) -> dict:
    for asset in release.get("assets", []):
        if asset["name"].endswith((".tar.zst", ".tar.gz", ".zip")):
            return asset
    raise RuntimeError("No bundle asset in release")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    name = archive.name
    if name.endswith(".tar.zst"):
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(target), "--strip-components=1"],
            check=True,
        )
    elif name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
    else:
        raise RuntimeError(f"Unsupported archive format: {name}")


def _existing_bundle_backup(output: Path, backup_root: Path) -> None:
    if not (output / "manifest.json").is_file():
        return
    if not backup_root:
        return
    backup_root.mkdir(parents=True, exist_ok=True)
    ts = subprocess.run(
        ["date", "+%Y%m%dT%H%M%SZ"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dest = backup_root / f"snapshot-{ts}"
    print(f"  Backing up existing bundle to {dest}")
    subprocess.run(
        ["tar", "-czf", str(dest) + ".tar.gz", "-C", str(output.parent), output.name],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a ServiceNow RAG bundle from GitHub Releases")
    p.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    p.add_argument("--tag", default=None, help="Release tag (default: latest)")
    p.add_argument("--output", type=Path, required=True, help="Destination directory")
    p.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Where to store backups of the existing bundle (default: <output>/.backups)",
    )
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    backup_root = args.backup_dir or (args.output / ".backups")
    if args.output.exists():
        _existing_bundle_backup(args.output, backup_root)

    print(f"  Resolving release for {args.repo}...")
    release = resolve_release(args.repo, args.tag)
    print(f"  Found release: {release['tag_name']} ({release['name']})")

    asset = find_bundle_asset(release)
    expected_size = asset["size"]
    print(f"  Asset: {asset['name']} ({expected_size / 1e6:.1f} MB)")

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / asset["name"]
        print(f"  Downloading from {asset['browser_download_url']}...")
        _http_download(asset["browser_download_url"], archive)
        if archive.stat().st_size != expected_size:
            raise RuntimeError("Downloaded size mismatch")

        if args.output.exists():
            for child in args.output.iterdir():
                if child.name == ".backups":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            args.output.mkdir(parents=True)

        print(f"  Extracting to {args.output}...")
        _extract(archive, args.output)

    manifest_path = args.output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Extracted bundle is missing manifest.json")
    manifest = json.loads(manifest_path.read_text())
    expected = manifest.get("artifacts", {}).get("chunks_sha256")
    if expected:
        actual = _sha256(args.output / manifest["artifacts"]["chunks"])
        if actual != expected:
            raise RuntimeError(
                f"chunks.parquet SHA mismatch: expected {expected}, got {actual}"
            )
        print(f"  chunks.parquet SHA256 verified")

    print(f"\n  Bundle ready at {args.output}")
    print(f"    chunk_count : {manifest.get('chunk_count')}")
    print(f"    source_sha  : {manifest.get('source_sha', '?')[:12]}")
    print(f"    built_at    : {manifest.get('built_at')}")
    return 0


def main() -> None:
    """Script entry point.

    Calls ``sys.exit(_run())`` so the return code propagates through
    both ``python -m`` and the console-script entry points defined
    in ``pyproject.toml``. Console scripts do not propagate return
    values on their own, so the ``sys.exit`` has to live here.
    """
    sys.exit(_run())


if __name__ == "__main__":
    main()
