"""Diagnostic probe for the ServiceNow Atlas installation.

Run this once after ``uv sync`` and after any major upgrade. It
checks the platform, probes each available backend, verifies the
optional MLX weights cache, sanity-checks a bundle if one is
given, and prints a one-screen report so the user can see what
will actually run when they invoke ``atlas-rag`` or
``atlas-build``.

Caching: the result is written to ``~/.cache/atlas/diagnosis.json``
so that ``atlas-rag`` and ``atlas-build`` don't re-probe on every
startup. Use ``atlas-doctor --refresh`` to force a re-probe, or
delete the cache file.

The probe is informational and never blocks. If the optimal
backend is unavailable we fall through to ONNX+CPU silently and
print a one-line reason in the report.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

CACHE_PATH = Path.home() / ".cache" / "atlas" / "diagnosis.json"


def _probe_platform() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
        "python": platform.python_version(),
    }


def _probe_onnxruntime() -> dict[str, Any]:
    try:
        import onnxruntime
        return {
            "available": True,
            "version": getattr(onnxruntime, "__version__", "unknown"),
            "providers": list(onnxruntime.get_available_providers()),
        }
    except ImportError:
        return {"available": False}


def _probe_mlx() -> dict[str, Any]:
    try:
        import mlx.core as mx
        # try a tiny op
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        _ = a + b
        version = getattr(mx, "__version__", "unknown")
        return {
            "available": True,
            "version": version,
            "apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
        }
    except ImportError:
        return {"available": False}
    except Exception as e:
        return {"available": False, "error": str(e)}


def _probe_mlx_weights() -> dict[str, Any]:
    from atlas.embed.mlx import DEFAULT_MLX_CACHE
    if not DEFAULT_MLX_CACHE.is_dir():
        return {"cached": False, "path": str(DEFAULT_MLX_CACHE)}
    n_files = sum(1 for _ in DEFAULT_MLX_CACHE.glob("*.npy"))
    return {
        "cached": n_files > 0,
        "path": str(DEFAULT_MLX_CACHE),
        "n_files": n_files,
    }


def _probe_nvidia() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not on PATH"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return {"available": False, "reason": out.stderr.strip() or "nvidia-smi failed"}
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        return {"available": True, "devices": lines}
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"available": False, "reason": str(e)}


def _probe_bundle(bundle_dir: Path | None) -> dict[str, Any]:
    if bundle_dir is None:
        return {"checked": False}
    if not bundle_dir.is_dir():
        return {"checked": True, "exists": False, "path": str(bundle_dir)}
    manifest = bundle_dir / "manifest.json"
    chunks = bundle_dir / "chunks.parquet"
    emb = bundle_dir / "embeddings.f16.npy"
    info: dict[str, Any] = {
        "checked": True,
        "exists": True,
        "path": str(bundle_dir),
        "has_manifest": manifest.is_file(),
        "has_chunks": chunks.is_file(),
        "has_embeddings": emb.is_file(),
    }
    if manifest.is_file():
        try:
            m = json.loads(manifest.read_text())
            info["chunk_count"] = m.get("chunk_count")
            info["embedding_model"] = m.get("embedding_model")
            info["source_branch"] = m.get("source_branch")
            info["source_sha"] = (m.get("source_sha") or "")[:10]
            info["build_backend"] = m.get("embedding_backend") or ""
            info["build_provider"] = m.get("embedding_active_provider") or ""
        except json.JSONDecodeError:
            info["manifest_valid"] = False
    if chunks.is_file() and manifest.is_file():
        # SHA check
        try:
            import hashlib
            h = hashlib.sha256()
            with chunks.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            m = json.loads(manifest.read_text())
            expected = m.get("artifacts", {}).get("chunks_sha256")
            info["chunks_sha_match"] = actual == expected
        except Exception as e:
            info["chunks_sha_check_error"] = str(e)
    return info


def _probe_ripgrep() -> dict[str, Any]:
    rg = shutil.which("rg")
    if not rg:
        return {"available": False}
    try:
        out = subprocess.run([rg, "--version"], capture_output=True, text=True, timeout=3)
        version = out.stdout.split("\n", 1)[0] if out.stdout else "unknown"
        return {"available": True, "path": rg, "version": version}
    except (subprocess.TimeoutExpired, OSError):
        return {"available": True, "path": rg}


def _probe_disk_free() -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(Path.home())
        return {
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "total_gb": round(usage.total / (1024 ** 3), 1),
        }
    except OSError as e:
        return {"error": str(e)}


def run_diagnosis(bundle: Path | None = None, force: bool = False) -> dict[str, Any]:
    if not force and CACHE_PATH.is_file():
        try:
            cached = json.loads(CACHE_PATH.read_text())
            if (bundle is None) == ("bundle" not in cached):
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    from atlas.embed import resolve_backend

    plat = _probe_platform()
    onnx = _probe_onnxruntime()
    mlx = _probe_mlx()
    mlx_w = _probe_mlx_weights()
    nvidia = _probe_nvidia()
    rg = _probe_ripgrep()
    disk = _probe_disk_free()
    bundle_info = _probe_bundle(bundle)

    backend, reason = resolve_backend("auto")
    report = {
        "version": "0.3.0",
        "platform": plat,
        "onnxruntime": onnx,
        "mlx": mlx,
        "mlx_weights": mlx_w,
        "nvidia": nvidia,
        "ripgrep": rg,
        "disk_free": disk,
        "bundle": bundle_info,
        "selected_backend": backend,
        "selected_reason": reason,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(report, indent=2))
    return report


def _fmt_bool(b: bool) -> str:
    return "OK" if b else "MISS"


def print_report(report: dict[str, Any]) -> None:
    print()
    print("=" * 64)
    print("  ServiceNow Atlas v{} — installation diagnosis".format(report.get("version", "?")))
    print("=" * 64)
    print()

    plat = report["platform"]
    print(f"  Platform       : {plat['system']} {plat['release']} ({plat['machine']})")
    print(f"  Python         : {plat['python']}")
    onnx = report["onnxruntime"]
    if onnx.get("available"):
        print(f"  ONNX Runtime   : {onnx['version']}  providers: {onnx['providers']}")
    else:
        print(f"  ONNX Runtime   : NOT INSTALLED")

    print()
    print("  Backend probe:")
    onnx_ok = onnx.get("available", False)
    print(f"    ONNX+CPU       {'OK ':<3} always available" if onnx_ok else "    ONNX+CPU       MISS")
    mlx = report["mlx"]
    if mlx.get("available"):
        mlx_w = report["mlx_weights"]
        cached = "weights cached" if mlx_w.get("cached") else "weights NOT cached (run tools/convert_bge_to_mlx.py)"
        print(f"    Apple MLX      OK   {cached}")
    else:
        print(f"    Apple MLX      MISS not importable (Apple Silicon only)")
    nvidia = report["nvidia"]
    if nvidia.get("available"):
        n = len(nvidia.get("devices", []))
        print(f"    NVIDIA CUDA    OK   {n} device(s) visible")
    else:
        print(f"    NVIDIA CUDA    MISS {nvidia.get('reason', 'no GPU')}")

    print()
    print(f"  Selected       : {report['selected_backend']}")
    print(f"                  {report['selected_reason']}")

    print()
    bundle = report.get("bundle", {})
    if bundle.get("checked"):
        if bundle.get("exists"):
            print(f"  Bundle         : {bundle['path']}")
            print(f"                  manifest={_fmt_bool(bundle.get('has_manifest'))} "
                  f"chunks={_fmt_bool(bundle.get('has_chunks'))} "
                  f"embeddings={_fmt_bool(bundle.get('has_embeddings'))}")
            if bundle.get("chunk_count") is not None:
                sha_ok = bundle.get("chunks_sha_match")
                sha_str = " (SHA ok)" if sha_ok else " (SHA MISMATCH)" if sha_ok is False else ""
                backend_str = ""
                bb = bundle.get("build_backend", "")
                bp = bundle.get("build_provider", "")
                if bb:
                    backend_str = f" build={bb}/{bp}" if bp else f" build={bb}"
                print(f"                  {bundle['chunk_count']} chunks, model={bundle.get('embedding_model')}{backend_str}{sha_str}")
        else:
            print(f"  Bundle         : not found at {bundle['path']}")

    print()
    rg = report.get("ripgrep", {})
    print(f"  ripgrep        : {rg.get('version', 'NOT INSTALLED — install with `brew install ripgrep`')}")
    disk = report.get("disk_free", {})
    if "free_gb" in disk:
        print(f"  Disk free      : {disk['free_gb']} GB / {disk['total_gb']} GB")

    print()
    print("=" * 64)
    print(f"  Cached at: {CACHE_PATH}")
    print("=" * 64)
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose the ServiceNow Atlas installation")
    p.add_argument("--bundle", type=Path, help="Optional path to a RAG bundle to verify")
    p.add_argument("--refresh", action="store_true", help="Re-probe instead of using cache")
    p.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted report")
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    report = run_diagnosis(bundle=args.bundle, force=args.refresh)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


def main() -> None:
    sys.exit(_run())


if __name__ == "__main__":
    main()
