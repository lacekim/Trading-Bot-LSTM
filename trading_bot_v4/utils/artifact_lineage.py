"""Immutable hashes and manifests for model-to-signal lineage."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable


MODEL_VERSION_DIR = Path("reports/model_versions")
MODEL_VERSION_RETENTION_COUNT = 100


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_version(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(value) for value in paths), key=lambda value: str(value)):
        digest.update(str(path).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()[:16]


def _prune_model_version_manifests() -> None:
    manifests = sorted(MODEL_VERSION_DIR.glob("*.json"))
    for stale in manifests[:-MODEL_VERSION_RETENTION_COUNT]:
        stale.unlink(missing_ok=True)


def write_model_manifest(paths: Iterable[str | Path], reason: str) -> Path:
    generated = datetime.now(timezone.utc)
    artifacts = []
    for value in paths:
        path = Path(value)
        artifacts.append({
            "path": str(path), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size, "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
        })
    payload = {
        "generated_at": generated.isoformat(), "reason": reason,
        "version": artifact_version(paths), "artifacts": artifacts,
    }
    MODEL_VERSION_DIR.mkdir(parents=True, exist_ok=True)
    output = MODEL_VERSION_DIR / f"{generated.strftime('%Y%m%dT%H%M%SZ')}_{payload['version']}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _prune_model_version_manifests()
    return output
