"""Shared artifact identity and configuration operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path


class ReleaseError(RuntimeError):
    """Report a failed release operation."""


class ValidationError(ReleaseError, ValueError):
    """Report an invalid artifact or request."""


class IntegrityError(ReleaseError):
    """Report changed artifact contents."""


class UnavailableError(ReleaseError):
    """Report unavailable registry data or models."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_document(value: dict) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def seal(value: dict) -> dict:
    return {**value, "sha256": digest_document(value)}


def verify_document(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("sha256") != digest_document(value):
        raise IntegrityError("Document digest does not match its contents")
    return value


def read_json(path: Path | str) -> dict:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise IntegrityError(f"Cannot read JSON document: {Path(path).name}") from exc
    if not isinstance(value, dict):
        raise ValidationError("Expected a JSON object")
    return value


def sha256_file(path: Path | str, progress: bool = False) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=path.stat().st_size, desc="Verifying artifact", unit="B", unit_scale=True)
    try:
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
                if bar is not None:
                    bar.update(len(block))
    finally:
        if bar is not None:
            bar.close()
    return digest.hexdigest()


def artifact_descriptor(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def secure_path(root: Path | str, relative: Path | str) -> Path:
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValidationError("Artifact paths must be relative and cannot traverse parents")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValidationError("Artifact path escapes its bundle")
    return path


def atomic_json(path: Path | str, value: dict, mode: int = 0o640) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


_RELEASE = re.compile(
    r"(?P<family>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)@(?P<version>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)\Z"
)


def parse_release_id(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or not _RELEASE.fullmatch(value):
        raise ValidationError("Use an exact release ID such as decodertcr@1.5.0")
    return tuple(value.split("@", 1))


def parse_model_id(value: str) -> tuple[str, str, str]:
    if not isinstance(value, str) or value.count(":") != 1:
        raise ValidationError("Use an exact model ID such as decodertcr@1.5.0:600M")
    release, member = value.split(":")
    family, version = parse_release_id(release)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", member):
        raise ValidationError("Invalid release member")
    return family, version, member


def config_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "decodertcr-public" / "config.json"


def registry_root(root: Path | str | None = None) -> Path:
    configured = root if root is not None else os.environ.get("DECODERTCR_PUBLIC_REGISTRY")
    if configured is None and config_path().is_file():
        configured = read_json(config_path()).get("registry")
    if configured is None:
        raise UnavailableError("No local registry is configured")
    if not isinstance(configured, (str, Path)) or not str(configured).strip():
        raise ValidationError("Registry path must be a nonempty string or Path")
    return Path(configured).expanduser().resolve()
