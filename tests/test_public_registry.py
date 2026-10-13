import hashlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

import DecoderTCR as dt
from DecoderTCR import cli
from DecoderTCR.core import IntegrityError, UnavailableError, seal
from DecoderTCR.registry import DEFAULT_REPOSITORY, Registry


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _registry(
    tmp_path: Path,
    *,
    minimum_package_version: str = "0.1.0",
    schema_version: int = 1,
) -> Path:
    root = tmp_path / "registry"
    members = {}
    evidence = {}
    for name, architecture in (
        ("300M", "DecoderTCRC_300M"),
        ("600M", "DecoderTCRC_600M"),
        ("6B", "DecoderTCRC_6B"),
    ):
        payload = f"fixture-{name}".encode()
        weights = root / "releases/decodertcr/0.3.0/weights" / f"{name}.safetensors"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        members[name] = {
            "backbone": "esmc",
            "arch": architecture,
            "sequence_convention": "v1",
            "weights": {
                "path": f"weights/{name}.safetensors",
                "sha256": digest,
                "size_bytes": len(payload),
            },
            "state_schema_sha256": "a" * 64,
            "license": {"name": "MIT", "text": "fixture license"},
        }
        evidence[name] = {"artifact_sha256": digest, "passed": True}
    manifest_fields = {
        "schema_version": schema_version,
        "kind": "release",
        "family": "decodertcr",
        "version": "0.3.0",
        "release_id": "decodertcr@0.3.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "minimum_package_version": minimum_package_version,
        "inventory_note": "fixture",
        "members": members,
    }
    if schema_version == 2:
        manifest_fields["declared_members"] = list(members)
    manifest = seal(manifest_fields)
    validation = seal(
        {
            "schema_version": 1,
            "manifest_sha256": manifest["sha256"],
            "passed": True,
            "members": evidence,
            "checks": ["tensor_equality", "strict_load", "logits", "masked_pll"],
        }
    )
    bundle = root / "releases/decodertcr/0.3.0"
    _write(bundle / "manifest.json", manifest)
    _write(bundle / "validation.json", validation)
    catalog = seal(
        {
            "schema_version": 2,
            "generation": 0,
            "previous_sha256": None,
            "releases": {
                "decodertcr@0.3.0": {
                    "path": "releases/decodertcr/0.3.0/manifest.json",
                    "sha256": manifest["sha256"],
                    "validation_sha256": validation["sha256"],
                    "state": "available",
                    "reason": "fixture",
                }
            },
            "datasets": {},
            "results": {"missing-result": {"path": "missing/manifest.json", "sha256": "b" * 64}},
            "environments": {},
        }
    )
    _write(root / "catalog.json", catalog)
    return root


def test_public_package_version_identities_agree():
    project_root = Path(__file__).parents[1]
    project = tomllib.loads((project_root / "pyproject.toml").read_text())
    lock = tomllib.loads((project_root / "uv.lock").read_text())
    editable = [
        package
        for package in lock["package"]
        if package["name"] == "decodertcr" and package.get("source") == {"editable": "."}
    ]

    assert project["project"]["version"] == "1.5.0"
    assert dt.__version__ == "1.5.0"
    assert [package["version"] for package in editable] == ["1.5.0"]


def test_schema_v2_legacy_minimum_is_accepted_by_current_package(tmp_path):
    root = _registry(tmp_path, minimum_package_version="0.4.0", schema_version=2)

    resolved = Registry(root).resolve("decodertcr@0.3.0:300M")

    assert resolved.manifest["minimum_package_version"] == "0.4.0"


def test_schema_v2_minimum_below_contract_floor_is_rejected(tmp_path):
    root = _registry(tmp_path, minimum_package_version="0.3.9", schema_version=2)

    with pytest.raises(dt.ValidationError, match="require package version 0.4.0 or newer"):
        Registry(root).resolve("decodertcr@0.3.0:300M")


def test_manifest_minimum_newer_than_runtime_is_rejected(tmp_path):
    root = _registry(tmp_path, minimum_package_version="1.5.1", schema_version=2)

    with pytest.raises(
        dt.ValidationError,
        match=r"Upgrade DecoderTCR to >= 1\.5\.1",
    ):
        Registry(root).resolve("decodertcr@0.3.0:300M")


def test_metadata_and_artifact_contracts_are_distinct(tmp_path):
    root = _registry(tmp_path)
    metadata = dt.artifact_info("decodertcr@0.3.0:300M", registry=root)
    assert set(metadata) == {
        "model_id",
        "sha256",
        "size_bytes",
        "backbone",
        "architecture",
        "sequence_convention",
        "manifest_sha256",
        "catalog_sha256",
    }
    artifact = dt.artifact("decodertcr@0.3.0:300M", registry=root)
    assert artifact["path"].read_bytes() == b"fixture-300M"
    assert artifact["sha256"] == metadata["sha256"]


def test_info_does_not_download_or_resolve_result_bundles(tmp_path):
    info = Registry(_registry(tmp_path)).info("decodertcr@0.3.0:300M")
    assert info["results"] == []
    assert info["member"]["arch"] == "DecoderTCRC_300M"


def test_catalog_release_key_must_match_manifest_identity(tmp_path):
    root = _registry(tmp_path)
    catalog = json.loads((root / "catalog.json").read_text())
    entry = catalog["releases"].pop("decodertcr@0.3.0")
    catalog["releases"]["decodertcr@1.5.0"] = entry
    _write(root / "catalog.json", seal({k: v for k, v in catalog.items() if k != "sha256"}))
    with pytest.raises(IntegrityError, match="identity disagrees"):
        Registry(root).resolve("decodertcr@1.5.0:300M")


def test_member_label_must_match_family_architecture(tmp_path):
    root = _registry(tmp_path)
    manifest_path = root / "releases/decodertcr/0.3.0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["members"]["300M"]["arch"] = "DecoderTCRC_6B"
    manifest = seal({key: value for key, value in manifest.items() if key != "sha256"})
    _write(manifest_path, manifest)
    validation_path = root / "releases/decodertcr/0.3.0/validation.json"
    validation = json.loads(validation_path.read_text())
    validation["manifest_sha256"] = manifest["sha256"]
    validation = seal({key: value for key, value in validation.items() if key != "sha256"})
    _write(validation_path, validation)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entry = catalog["releases"]["decodertcr@0.3.0"]
    entry["sha256"] = manifest["sha256"]
    entry["validation_sha256"] = validation["sha256"]
    _write(catalog_path, seal({key: value for key, value in catalog.items() if key != "sha256"}))

    with pytest.raises((IntegrityError, dt.ValidationError)):
        Registry(root).resolve("decodertcr@0.3.0:300M")


def test_release_family_must_be_complete(tmp_path):
    root = _registry(tmp_path)
    manifest_path = root / "releases/decodertcr/0.3.0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["members"].pop("6B")
    manifest = seal({key: value for key, value in manifest.items() if key != "sha256"})
    _write(manifest_path, manifest)
    validation_path = root / "releases/decodertcr/0.3.0/validation.json"
    validation = json.loads(validation_path.read_text())
    validation["manifest_sha256"] = manifest["sha256"]
    validation["members"].pop("6B")
    validation = seal({key: value for key, value in validation.items() if key != "sha256"})
    _write(validation_path, validation)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entry = catalog["releases"]["decodertcr@0.3.0"]
    entry["sha256"] = manifest["sha256"]
    entry["validation_sha256"] = validation["sha256"]
    _write(catalog_path, seal({key: value for key, value in catalog.items() if key != "sha256"}))

    with pytest.raises((IntegrityError, dt.ValidationError)):
        Registry(root).resolve("decodertcr@0.3.0:300M")


def test_remote_registry_fails_closed_without_a_live_head(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("DecoderTCR.registry.HfApi.repo_info", unavailable)
    with pytest.raises(UnavailableError, match="explicit local mirror"):
        Registry("hf://biohub/DecoderTCR/registry").catalog()


def _mock_hub(monkeypatch, roots, revision_state, calls, info_calls=None):
    def repo_info(*args, **kwargs):
        if info_calls is not None:
            info_calls.append((args, kwargs))
        return SimpleNamespace(sha=revision_state["revision"])

    def download(*, repo_id, filename, repo_type, revision, local_files_only):
        calls.append((repo_id, filename, repo_type, revision, local_files_only))
        return str(roots[revision] / filename.removeprefix("registry/"))

    monkeypatch.setattr("DecoderTCR.registry.HfApi.repo_info", repo_info)
    monkeypatch.setattr("DecoderTCR.registry.hf_hub_download", download)


def _replace_fixture_weights(root: Path, member_name: str, state: dict[str, torch.Tensor]) -> str:
    bundle = root / "releases/decodertcr/0.3.0"
    weights = bundle / "weights" / f"{member_name}.safetensors"
    save_file(state, str(weights))
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    descriptor = manifest["members"][member_name]["weights"]
    descriptor.update({"sha256": digest, "size_bytes": weights.stat().st_size})
    manifest = seal({key: value for key, value in manifest.items() if key != "sha256"})
    _write(manifest_path, manifest)
    validation_path = bundle / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation["manifest_sha256"] = manifest["sha256"]
    validation["members"][member_name]["artifact_sha256"] = digest
    validation = seal({key: value for key, value in validation.items() if key != "sha256"})
    _write(validation_path, validation)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    entry = catalog["releases"]["decodertcr@0.3.0"]
    entry.update({"sha256": manifest["sha256"], "validation_sha256": validation["sha256"]})
    _write(catalog_path, seal({key: value for key, value in catalog.items() if key != "sha256"}))
    return digest


def _isolate_default_selection(tmp_path, monkeypatch):
    config_home = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("DECODERTCR_PUBLIC_REGISTRY", raising=False)
    monkeypatch.setenv("DECODERTCR_REGISTRY", str(tmp_path / "internal-environment"))
    _write(
        config_home / "decodertcr/config.json",
        {"registry": str(tmp_path / "internal-configuration")},
    )


def test_hub_reads_are_all_pinned_to_one_live_revision(tmp_path, monkeypatch):
    revision = "1" * 40
    root = _registry(tmp_path)
    calls = []
    _mock_hub(monkeypatch, {revision: root}, {"revision": revision}, calls)

    models = Registry("hf://biohub/DecoderTCR/registry").models()

    assert len(models) == 3
    assert calls
    assert {call[3] for call in calls} == {revision}
    assert all(call[4] is False for call in calls)


def test_default_models_and_exact_load_use_staging_hub(tmp_path, monkeypatch):
    from DecoderTCR import runtime

    _isolate_default_selection(tmp_path, monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    revision = "3" * 40
    root = _registry(tmp_path)
    expected_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    digest = _replace_fixture_weights(root, "300M", {"weight": expected_weight})
    calls = []
    info_calls = []
    _mock_hub(
        monkeypatch,
        {revision: root},
        {"revision": revision},
        calls,
        info_calls,
    )
    monkeypatch.setattr(runtime, "model_spec", lambda backbone, arch: {"n_layers": 1})
    monkeypatch.setattr(
        runtime,
        "build_model",
        lambda backbone, arch, device: torch.nn.Linear(2, 2, bias=False, device=device),
    )

    assert len(dt.models()) == 3
    loaded = dt.load("decodertcr@0.3.0:300M", device="cpu")

    assert DEFAULT_REPOSITORY == "Decoder-CHI/DecoderTCR-staging"
    assert loaded.artifact_sha256 == digest
    torch.testing.assert_close(loaded.module.weight, expected_weight)
    assert info_calls
    assert all(len(args) == 1 for args, _ in info_calls)
    assert all(
        kwargs
        == {
            "repo_id": "Decoder-CHI/DecoderTCR-staging",
            "repo_type": "model",
            "revision": "main",
        }
        for _, kwargs in info_calls
    )
    assert calls
    assert {call[0] for call in calls} == {"Decoder-CHI/DecoderTCR-staging"}
    assert {call[3] for call in calls} == {revision}
    assert all(call[1].startswith("registry/") for call in calls)
    assert all(call[4] is False for call in calls)


def test_registry_selection_precedence_supports_local_and_hub_sources(tmp_path, monkeypatch):
    _isolate_default_selection(tmp_path, monkeypatch)
    public_config = tmp_path / "config/decodertcr-public/config.json"
    local = tmp_path / "local-registry"
    _write(public_config, {"registry": "hf://saved/example/metadata"})
    monkeypatch.setenv("DECODERTCR_PUBLIC_REGISTRY", "hf://environment/example/catalog")

    assert Registry(local).spec == str(local.resolve())
    assert Registry("hf://explicit/example/registry").spec == "hf://explicit/example/registry"
    assert Registry().spec == "hf://environment/example/catalog"
    monkeypatch.delenv("DECODERTCR_PUBLIC_REGISTRY")
    assert Registry().spec == "hf://saved/example/metadata"
    public_config.unlink()
    assert Registry().spec == "hf://Decoder-CHI/DecoderTCR-staging/registry"


def test_live_block_overrides_a_previously_verified_artifact(tmp_path, monkeypatch):
    first_revision = "1" * 40
    second_revision = "2" * 40
    first = _registry(tmp_path / "first")
    second = _registry(tmp_path / "second")
    catalog_path = second / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["releases"]["decodertcr@0.3.0"].update(
        {"state": "blocked", "reason": "withdrawn fixture"}
    )
    _write(catalog_path, seal({key: value for key, value in catalog.items() if key != "sha256"}))
    revision_state = {"revision": first_revision}
    calls = []
    _mock_hub(
        monkeypatch,
        {first_revision: first, second_revision: second},
        revision_state,
        calls,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    registry = Registry("hf://biohub/DecoderTCR/registry")
    assert registry.verify_artifact(
        registry.resolve("decodertcr@0.3.0:300M").bundle,
        registry.resolve("decodertcr@0.3.0:300M").member["weights"],
    ).is_file()

    revision_state["revision"] = second_revision
    with pytest.raises(UnavailableError, match="withdrawn fixture"):
        registry.resolve("decodertcr@0.3.0:300M")


def test_read_only_local_registry_has_no_ownership_gate(tmp_path):
    root = _registry(tmp_path)
    paths = sorted(root.rglob("*"), reverse=True)
    try:
        for path in paths:
            path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555)
        assert len(Registry(root).models()) == 3
    finally:
        root.chmod(0o755)
        for path in reversed(paths):
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_catalog_traversal_is_rejected(tmp_path):
    root = _registry(tmp_path)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["releases"]["decodertcr@0.3.0"]["path"] = "../manifest.json"
    _write(catalog_path, seal({key: value for key, value in catalog.items() if key != "sha256"}))

    with pytest.raises(dt.ValidationError, match="traverse"):
        Registry(root).models()


def test_corrupt_manifest_is_rejected(tmp_path):
    root = _registry(tmp_path)
    manifest_path = root / "releases/decodertcr/0.3.0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inventory_note"] = "tampered"
    _write(manifest_path, manifest)

    with pytest.raises(IntegrityError):
        Registry(root).models()


def test_configure_preserves_hub_spec_and_uses_public_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "public-config"))
    monkeypatch.setenv("DECODERTCR_REGISTRY", str(tmp_path / "private-registry"))
    monkeypatch.setattr(Registry, "catalog", lambda self: {"sha256": "a" * 64})

    assert cli.main(["init", "--registry", "hf://biohub/DecoderTCR/registry"]) == 0

    configured = json.loads((tmp_path / "public-config/decodertcr-public/config.json").read_text())
    assert configured == {"registry": "hf://biohub/DecoderTCR/registry"}
    assert Registry().spec == "hf://biohub/DecoderTCR/registry"


def test_cli_models_uses_default_staging_registry(tmp_path, monkeypatch, capsys):
    _isolate_default_selection(tmp_path, monkeypatch)
    selected = []

    def models(self, include_unavailable=False):
        selected.append((self.spec, include_unavailable))
        return []

    monkeypatch.setattr(Registry, "models", models)

    assert cli.main(["models"]) == 0
    assert selected == [("hf://Decoder-CHI/DecoderTCR-staging/registry", False)]
    assert json.loads(capsys.readouterr().out) == []


def test_cli_score_routes_gene_input(tmp_path, monkeypatch):
    root = _registry(tmp_path)
    source = tmp_path / "genes.csv"
    output = tmp_path / "gene-scores.csv"
    pd.DataFrame([{"value": "fixture"}]).to_csv(source, index=False)
    calls = []

    def score(frame, *, species):
        calls.append(species)
        return frame.assign(
            pll=-1.0,
            artifact_sha256="a" * 64,
            model_id="decodertcr@0.3.0:300M",
        )

    loaded = SimpleNamespace(
        score=score,
        info={"artifact_sha256": "a" * 64},
    )
    monkeypatch.setattr("DecoderTCR.runtime.load", lambda *args, **kwargs: loaded)

    status = cli.main(
        [
            "--registry",
            str(root),
            "score",
            "--model",
            "decodertcr@0.3.0:300M",
            "--input",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    )

    assert status == 0
    assert calls == ["human"]
    assert output.is_file()


@pytest.mark.parametrize("command", ["score", "profile", "design"])
def test_cli_rejects_input_format_flag(command):
    parser = cli._parser()
    common = [
        command,
        "--model",
        "decodertcr@0.3.0:300M",
        "--input",
        "input.csv",
        "--output",
        "output.csv",
    ]
    if command == "design":
        common.extend(["--length", "9"])
    args = parser.parse_args(common)
    assert not hasattr(args, "input_format")
    with pytest.raises(SystemExit):
        parser.parse_args([*common, "--input-format", "prepared"])


def test_cli_profile_does_not_advertise_all_cdr(capsys):
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["profile", "--help"])
    help_text = capsys.readouterr().out
    assert "cdr3a" in help_text
    assert "cdr3b" in help_text
    assert "{peptide,cdr3a,cdr3b,cdr3}" in help_text
