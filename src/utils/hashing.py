"""Content fingerprints used to skip pipeline stages whose inputs did not change."""
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(paths: list[Path], extra: object = None) -> str:
    """Hash of every existing input file plus a JSON-able extra (usually the stage config)."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.exists():
            digest.update(f"{path}:{file_sha256(path)}".encode())
    digest.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return digest.hexdigest()[:16]
