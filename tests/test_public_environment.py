import hashlib
import io
import platform
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

from DecoderTCR import environment


def _descriptor(path: Path) -> dict:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def test_environment_is_created_at_its_final_path(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    runtime = bundle / "python.tar.gz"
    with tarfile.open(runtime, "w:gz") as archive:
        payload = b"#!/bin/sh\n"
        info = tarfile.TarInfo("python/bin/python3.12")
        info.mode = 0o755
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    files = {}
    for name, payload in {
        "uv": b"uv",
        "requirements.txt": b"",
        "uv.lock": b"lock",
        "pyproject.toml": b"project",
        "decodertcr-1.5.0-py3-none-any.whl": b"wheel",
    }.items():
        path = bundle / name
        path.write_bytes(payload)
        files[name] = path
    manifest = {
        "sha256": "a" * 64,
        "platform": platform.system(),
        "machine": platform.machine(),
        "package": _descriptor(files["decodertcr-1.5.0-py3-none-any.whl"]),
        "requirements": _descriptor(files["requirements.txt"]),
        "lock": _descriptor(files["uv.lock"]),
        "project": _descriptor(files["pyproject.toml"]),
        "python_runtime": _descriptor(runtime),
        "uv": _descriptor(files["uv"]),
        "wheels": [],
    }

    class FixtureRegistry:
        def resolve_bundle(self, kind, identity):
            assert (kind, identity) == ("environments", "decodertcr@1.5.0:linux-x86_64")
            return manifest, bundle

        def verify_artifact(self, selected_bundle, descriptor):
            assert selected_bundle == bundle
            return bundle / descriptor["path"]

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "venv" in command:
            target = Path(command[-1])
            (target / "bin").mkdir(parents=True)
            python = target / "bin/python"
            python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
            python.chmod(0o700)

    monkeypatch.setattr(environment, "_run", fake_run)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    python = environment.ensure_environment(FixtureRegistry(), "decodertcr@1.5.0:linux-x86_64")
    assert (
        python == tmp_path / "cache/decodertcr-public/environments" / ("a" * 64) / "venv/bin/python"
    )
    assert python.is_file()
    assert all(f".{manifest['sha256']}." not in " ".join(command) for command in calls)
    completed = subprocess.run(
        [str(python), "-I", "-c", "import json; print(json.dumps({'ready': True}))"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == '{"ready": true}'
