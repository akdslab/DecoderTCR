"""Install and run exact public reproduction environments."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tarfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from .core import IntegrityError, ValidationError, atomic_json, read_json


def _run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def _complete(target: Path, digest: str) -> bool:
    python = target / "venv" / "bin" / "python"
    stamp = target / "recipe-environment.json"
    if not python.is_file() or not stamp.is_file():
        return False
    try:
        return read_json(stamp).get("sha256") == digest
    except (IntegrityError, ValidationError):
        return False


@contextmanager
def _setup_lock(cache: Path, digest: str):
    lock = cache / f".{digest}.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValidationError(
            f"Another process is preparing this reproduction environment at {lock}. "
            "Wait for it to finish. Inspect this path if the lock remains"
        ) from exc
    try:
        yield
    finally:
        lock.rmdir()


def ensure_environment(registry, environment_id: str) -> Path:
    """Create or reuse one verified public reproduction environment."""
    manifest, bundle = registry.resolve_bundle("environments", environment_id)
    if (
        manifest.get("platform") != platform.system()
        or manifest.get("machine") != platform.machine()
    ):
        raise ValidationError("This reproduction environment targets another platform")
    required = ("package", "lock", "project", "requirements", "python_runtime", "uv")
    paths = {field: registry.verify_artifact(bundle, manifest[field]) for field in required}
    package_name = Path(manifest["package"]["path"]).name.lower()
    wheel_match = re.fullmatch(r"decodertcr-(\d+\.\d+\.\d+)-.+\.whl", package_name)
    environment_match = re.fullmatch(r"decodertcr@(\d+\.\d+\.\d+):[A-Za-z0-9._-]+", environment_id)
    if (
        wheel_match is None
        or environment_match is None
        or wheel_match.group(1) != environment_match.group(1)
    ):
        raise ValidationError("Reproduction requires its recorded public DecoderTCR wheel")
    wheel_paths = [registry.verify_artifact(bundle, item) for item in manifest["wheels"]]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = cache / "decodertcr-public" / "environments"
    cache.mkdir(parents=True, exist_ok=True)
    digest = manifest["sha256"]
    target = cache / digest
    if _complete(target, digest):
        return target / "venv" / "bin" / "python"
    with _setup_lock(cache, digest):
        if _complete(target, digest):
            return target / "venv" / "bin" / "python"
        if target.exists():
            raise ValidationError(
                f"An incomplete reproduction environment cache entry exists at {target}. "
                "Inspect this path before retrying"
            )
        target.mkdir()
        interpreter = target / "runtime"
        interpreter.mkdir()
        try:
            with tarfile.open(paths["python_runtime"], "r:gz") as archive:
                archive.extractall(interpreter, filter="data")
        except tarfile.TarError as exc:
            raise IntegrityError("The retained Python archive is invalid") from exc
        executable = interpreter / "python" / "bin" / "python3.12"
        if not executable.is_file() or not executable.resolve().is_relative_to(interpreter):
            raise IntegrityError("The retained Python archive has no safe interpreter")
        retained_uv = target / "uv"
        shutil.copyfile(paths["uv"], retained_uv)
        retained_uv.chmod(0o700)
        _run(
            [
                str(retained_uv),
                "venv",
                "--offline",
                "--python",
                str(executable),
                str(target / "venv"),
            ]
        )
        python = target / "venv" / "bin" / "python"
        find_links = sorted({str(path.parent) for path in [paths["package"], *wheel_paths]})
        command = [
            str(retained_uv),
            "pip",
            "sync",
            "--python",
            str(python),
            "--no-index",
            "--offline",
            "--require-hashes",
        ]
        for directory in find_links:
            command.extend(["--find-links", directory])
        command.append(str(paths["requirements"]))
        _run(command)
        _run(
            [
                str(retained_uv),
                "pip",
                "install",
                "--python",
                str(python),
                "--no-index",
                "--offline",
                "--no-deps",
                str(paths["package"]),
            ]
        )
        atomic_json(target / "recipe-environment.json", {"sha256": digest}, mode=0o600)
        return target / "venv" / "bin" / "python"


def run_reproduction(
    registry,
    result_id: str,
    mode: str,
    output,
    environment_id: str,
    *,
    shard_index=None,
    shard_count=None,
    shard_outputs=None,
    parent_output=None,
) -> dict:
    """Run a reproduction in its exact registered public environment."""
    if (shard_index is None) != (shard_count is None):
        raise ValidationError("Shard index and count must be supplied together")
    if shard_index is not None and shard_outputs is not None:
        raise ValidationError("Choose a shard worker or a shard merge")
    if parent_output is not None and (shard_index is not None or shard_outputs is not None):
        raise ValidationError("A parent output cannot be combined with shard operations")
    python = ensure_environment(registry, environment_id)
    destination = (
        Path(output) if output is not None else Path("results") / f"reproduce-{uuid.uuid4().hex}"
    )
    destination = destination.resolve()
    if destination.exists():
        raise ValidationError("Reproduction output directory already exists")
    removed = {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in removed and not key.startswith("DECODERTCR_")
    }
    env["DECODERTCR_PUBLIC_REGISTRY"] = registry.spec
    env["DECODERTCR_REPRO_ENV"] = environment_id
    command = [
        str(python),
        "-I",
        "-m",
        "DecoderTCR.reproduce_worker",
        result_id,
        "--mode",
        mode,
        "--output",
        str(destination),
    ]
    if shard_index is not None:
        command.extend(["--shard-index", str(shard_index), "--shard-count", str(shard_count)])
    elif shard_outputs is not None:
        command.append("--merge-shards")
        command.extend(str(Path(path).resolve()) for path in shard_outputs)
    if parent_output is not None:
        command.extend(["--parent-output", str(Path(parent_output).resolve())])
    _run(command, env=env)
    for name in ("reproduction.json", "reproduction-shard.json", "result.json", "report.json"):
        path = destination / name
        if path.is_file():
            return {**read_json(path), "output": str(destination)}
    raise IntegrityError("Reproduction did not produce its run record")


__all__ = ["ensure_environment", "run_reproduction"]
