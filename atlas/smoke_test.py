"""End-to-end smoke test for the bundle pipeline.

Builds a tiny bundle from a handful of files, loads it via the
same code path ``atlas.rag_server`` uses, and runs a semantic
search to confirm the round-trip works. Runs in 1-2 minutes on
Apple Silicon; useful as a CI check or a post-install verification.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = REPO_ROOT / "data" / "servicenow-docs" / "ServiceNowDocs-australia"


def _run(cmd: list[str], **kw) -> None:
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def _run_tests() -> int:
    repo_path = DEFAULT_DOCS_DIR
    if not repo_path.is_dir():
        print(f"  ! {repo_path} missing; clone it first")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bundle"
        _run(
            [
                sys.executable,
                "-m",
                "atlas.make_bundle",
                "--repo-path",
                str(repo_path),
                "--output",
                str(out),
                "--limit",
                "20",
                "--prefer",
                "cpu",
            ],
            cwd=REPO_ROOT,
        )
        if not (out / "chunks.parquet").is_file():
            print("  ! chunks.parquet missing")
            return 1

        from atlas.rag_server import Bundle  # noqa: WPS433

        bundle = Bundle(out)
        info = bundle.manifest
        print(f"  Bundle chunk_count = {info['chunk_count']}")
        if info["chunk_count"] == 0:
            print("  ! No chunks in test bundle")
            return 1

        hits = bundle.search(
            "incident",
            top_k=3,
            min_score=0.0,
        )
        print(f"  search('incident') returned {len(hits)} hits")
        for h in hits:
            print(
                f"    - {h['publication']}/{h['file']} "
                f"heading={h['heading']!r} score={h['score']:.3f}"
            )
        if not hits:
            print("  ! No hits for 'incident'")
            return 1

    print("  Smoke test passed.")
    return 0


def main() -> None:
    """Script entry point. Calls ``sys.exit(_run_tests())`` so the
    return code propagates through both ``python -m`` and the
    console-script entry points defined in ``pyproject.toml``."""
    sys.exit(_run_tests())


if __name__ == "__main__":
    main()
