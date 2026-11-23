"""Shared utilities: logging, hashing, atomic artifact writes, manifests."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

LOGGER_NAME = "stock_transformer"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project logger (or a named child logger)."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def setup_logging(log_dir: str | Path = "logs") -> Path:
    """Configure INFO logging to stdout and ``logs/pipeline_<date>.log``."""
    log_path = Path(log_dir) / f"pipeline_{datetime.now(timezone.utc):%Y-%m-%d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(logging.INFO)
    root.propagate = False

    # Re-configuring between runs (e.g. in tests) must not duplicate handlers.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stdout = logging.StreamHandler()
    stdout.setFormatter(formatter)
    root.addHandler(stdout)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return log_path


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> Path:
    """Atomically write JSON."""
    p = ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
            fh.write("\n")
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Atomically write a Parquet file."""
    p = ensure_parent(path)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".parquet")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _package_version(name: str) -> str:
    """Best-effort version lookup for a module or distribution name."""
    try:
        module = __import__(name)
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    except Exception:  # pragma: no cover - optional dependency
        pass
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception:  # pragma: no cover - optional dependency
        return "not installed"


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in (
        "pandas",
        "numpy",
        "yfinance",
        "sklearn",
        "xgboost",
        "ta",
        "backtesting",
        "transformers",
        "torch",
    ):
        versions[mod] = _package_version(mod)
    return versions


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(
    artifact_path: str | Path,
    *,
    stage: str,
    config_hash: str | None,
    inputs: Iterable[str | Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a sidecar manifest for an artifact and the run manifest entry."""
    artifact = Path(artifact_path)
    payload: dict[str, Any] = {
        "stage": stage,
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact) if artifact.exists() else None,
        "config_hash": config_hash,
        "input_hashes": {
            str(Path(i)): sha256_file(i) for i in inputs if Path(i).exists()
        },
        "library_versions": library_versions(),
        "created_utc": utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    return write_json(artifact.with_name(artifact.name + ".manifest.json"), payload)


def artifact_is_current(
    artifact_path: str | Path,
    *,
    config_hash: str,
    inputs: Iterable[str | Path] = (),
) -> bool:
    """True when the artifact exists and its manifest matches inputs and config."""
    artifact = Path(artifact_path)
    manifest_path = artifact.with_name(artifact.name + ".manifest.json")
    if not artifact.exists() or not manifest_path.exists():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("config_hash") != config_hash:
        return False
    recorded = manifest.get("input_hashes") or {}
    for inp in inputs:
        p = Path(inp)
        if not p.exists():
            return False
        if recorded.get(str(p)) != sha256_file(p):
            return False
    return True
