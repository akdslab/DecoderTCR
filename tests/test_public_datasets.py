import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import DecoderTCR as dt
from DecoderTCR import cli
from DecoderTCR.core import (
    IntegrityError,
    ValidationError,
    artifact_descriptor,
    seal,
)
from DecoderTCR.registry import Registry
from DecoderTCR.reproduction import create_result_bundle
from DecoderTCR.runtime import LoadedModel

ROOT = Path(__file__).parents[1]
SOURCES = {
    "demo-gene-pairs@1.0.0": {
        "path": ROOT / "Demo/sample_data/genes_pairs.csv",
        "sha256": "e46ff5f0b470829133d767245c8a14ef8d43ed6bca7013d2b8deb2631da2beb0",
        "rows": 30,
        "columns": [
            "name",
            "trav",
            "traj",
            "cdr3a",
            "trbv",
            "trbj",
            "cdr3b",
            "hla",
            "peptide",
            "label",
        ],
        "label_column": "label",
        "label_counts": {"0": 15, "1": 15},
        "usage_column": None,
        "usage_counts": {},
    },
    "immrep23-public-inputs@1.0.0": {
        "path": ROOT / "results/immrep23/immrep23_test.csv",
        "sha256": "b68b52d83d613dfdf8b1b711b6142dc99e99d4a25a6fb6c80daee7efd0b77ecb",
        "rows": 3484,
        "columns": [
            "ID",
            "Peptide",
            "HLA",
            "Va",
            "Ja",
            "TCRa",
            "CDR1a",
            "CDR2a",
            "CDR3a",
            "CDR3a_extended",
            "Vb",
            "Jb",
            "TCRb",
            "CDR1b",
            "CDR2b",
            "CDR3b",
            "CDR3b_extended",
            "Label",
            "Usage",
        ],
        "label_column": "Label",
        "label_counts": {"0": 2886, "1": 598},
        "usage_column": "Usage",
        "usage_counts": {"Private": 3217, "Public": 267},
    },
}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _registry(tmp_path: Path) -> Path:
    root = tmp_path / "registry"
    entries = {}
    for dataset_id, expected in SOURCES.items():
        bundle = root / "datasets" / dataset_id
        bundle.mkdir(parents=True)
        shutil.copyfile(expected["path"], bundle / "data.csv")
        manifest = seal(
            {
                "schema_version": 3,
                "kind": "dataset",
                "dataset_id": dataset_id,
                "input_format": "genes",
                "species": "human",
                "artifact": artifact_descriptor(bundle / "data.csv", bundle),
                "n_rows": expected["rows"],
                "columns": expected["columns"],
                "label_column": expected["label_column"],
                "label_counts": expected["label_counts"],
                "usage_column": expected["usage_column"],
                "usage_counts": expected["usage_counts"],
                "source": {
                    "name": dataset_id,
                    "public_baseline_path": str(expected["path"].relative_to(ROOT)),
                    "citation": "public fixture",
                    "rights_note": "No new rights are granted.",
                },
            }
        )
        _write(bundle / "manifest.json", manifest)
        entries[dataset_id] = {
            "path": f"datasets/{dataset_id}/manifest.json",
            "sha256": manifest["sha256"],
        }
    catalog = seal(
        {
            "schema_version": 2,
            "generation": 1,
            "previous_sha256": "a" * 64,
            "releases": {},
            "datasets": entries,
            "results": {},
            "environments": {},
        }
    )
    _write(root / "catalog.json", catalog)
    return root


def test_shipped_csv_contracts_are_exact():
    for expected in SOURCES.values():
        assert hashlib.sha256(expected["path"].read_bytes()).hexdigest() == expected["sha256"]
        frame = pd.read_csv(expected["path"], keep_default_na=False)
        assert list(frame.columns) == expected["columns"]
        assert len(frame) == expected["rows"]
        assert (
            frame[expected["label_column"]].astype(str).value_counts().to_dict()
            == expected["label_counts"]
        )
        if expected["usage_column"]:
            assert (
                frame[expected["usage_column"]].value_counts().to_dict() == expected["usage_counts"]
            )


def test_metadata_listing_and_dataframe_identity_do_not_touch_other_artifacts(tmp_path):
    root = _registry(tmp_path)
    registry = Registry(root)
    listed = registry.datasets()
    assert [entry["dataset_id"] for entry in listed] == sorted(SOURCES)
    info = registry.dataset_info("demo-gene-pairs@1.0.0")
    assert info["n_rows"] == 30
    frame = dt.dataset("demo-gene-pairs@1.0.0", registry=root)
    direct = pd.read_csv(SOURCES["demo-gene-pairs@1.0.0"]["path"], keep_default_na=False)
    pd.testing.assert_frame_equal(frame, direct)
    assert frame.attrs["dataset"]["dataset_id"] == "demo-gene-pairs@1.0.0"
    assert frame.attrs["dataset"]["artifact_sha256"] == SOURCES["demo-gene-pairs@1.0.0"]["sha256"]
    assert frame.loc[0, "label"] in (0, 1)
    assert [entry["dataset_id"] for entry in dt.datasets(registry=root)] == sorted(SOURCES)
    assert dt.dataset_info("demo-gene-pairs@1.0.0", registry=root)["n_rows"] == 30


def test_remote_dataset_uses_one_pinned_revision_and_only_selected_csv(tmp_path, monkeypatch):
    root = _registry(tmp_path)
    calls = []

    monkeypatch.setattr(
        "DecoderTCR.registry.HfApi.repo_info",
        lambda *args, **kwargs: SimpleNamespace(sha="f" * 40),
    )

    def download(*, repo_id, filename, repo_type, revision, local_files_only):
        calls.append((repo_id, filename, repo_type, revision, local_files_only))
        return str(root / filename.removeprefix("registry/"))

    monkeypatch.setattr("DecoderTCR.registry.hf_hub_download", download)
    frame = dt.dataset("demo-gene-pairs@1.0.0", registry="hf://owner/repository/registry")
    assert len(frame) == 30
    assert {call[3] for call in calls} == {"f" * 40}
    assert [call[1] for call in calls] == [
        "registry/catalog.json",
        "registry/datasets/demo-gene-pairs@1.0.0/manifest.json",
        "registry/datasets/demo-gene-pairs@1.0.0/data.csv",
    ]
    assert all("weights" not in call[1] for call in calls)


def test_dataset_failures_are_closed(tmp_path):
    root = _registry(tmp_path)
    with pytest.raises(dt.UnavailableError, match="No registered dataset"):
        dt.dataset("missing@1.0.0", registry=root)

    data = root / "datasets/demo-gene-pairs@1.0.0/data.csv"
    data.write_bytes(data.read_bytes() + b"corrupt")
    with pytest.raises(IntegrityError, match="Artifact"):
        dt.dataset("demo-gene-pairs@1.0.0", registry=root)


def test_dataset_artifact_path_cannot_traverse_bundle(tmp_path):
    root = _registry(tmp_path)
    manifest_path = root / "datasets/demo-gene-pairs@1.0.0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact"]["path"] = "../data.csv"
    manifest = seal({key: value for key, value in manifest.items() if key != "sha256"})
    _write(manifest_path, manifest)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["datasets"]["demo-gene-pairs@1.0.0"]["sha256"] = manifest["sha256"]
    _write(catalog_path, seal({key: value for key, value in catalog.items() if key != "sha256"}))
    with pytest.raises(ValidationError):
        dt.dataset("demo-gene-pairs@1.0.0", registry=root)


def test_prepared_dataset_is_not_a_raw_dataset_or_reproduction_input(tmp_path):
    root = _registry(tmp_path)
    manifest_path = root / "datasets/demo-gene-pairs@1.0.0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest["benchmark_name"] = "chris"
    manifest["sequence_convention"] = "v1"
    manifest["table"] = manifest.pop("artifact")
    manifest["example"] = manifest["table"]
    manifest["group_key"] = "clone"
    manifest["metric_type"] = "ranking"
    manifest["inventory"] = {}
    manifest["exclusions"] = manifest["table"]
    manifest["raw"] = [manifest["table"]]
    manifest["n_source_rows"] = manifest["n_rows"]
    manifest["n_excluded_source_rows"] = 0
    manifest["measured_convention"] = {
        "verdict": "v1",
        "modal_lengths": {"HLA_a": 0, "TCR_a": 0, "TCR_b": 0},
    }
    manifest["preparation"] = "complete_population_before_sharding_v1"
    for key in (
        "input_format",
        "species",
        "columns",
        "label_column",
        "label_counts",
        "usage_column",
        "usage_counts",
        "source",
    ):
        manifest.pop(key)
    manifest = seal({key: value for key, value in manifest.items() if key != "sha256"})
    _write(manifest_path, manifest)
    catalog_path = root / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["datasets"]["demo-gene-pairs@1.0.0"]["sha256"] = manifest["sha256"]
    _write(catalog_path, seal({key: value for key, value in catalog.items() if key != "sha256"}))
    with pytest.raises(ValidationError, match="prepared reproduction bundle"):
        dt.dataset("demo-gene-pairs@1.0.0", registry=root)


def test_raw_dataset_is_rejected_before_reproduction_recipe_resolution(tmp_path):
    class RawRegistry:
        def info(self, model_id):
            return {
                "member": {
                    "weights": {"sha256": "a" * 64},
                    "sequence_convention": "v2",
                }
            }

        def resolve_bundle(self, kind, identity):
            assert kind == "datasets"
            return {"schema_version": 3, "dataset_id": identity}, tmp_path

    with pytest.raises(
        ValidationError,
        match="cannot be used as reproduction bundles.*prepared reproduction dataset",
    ):
        create_result_bundle(
            RawRegistry(),
            "decodertcr@1.5.0:300M",
            "demo-gene-pairs@1.0.0",
            pd.DataFrame(),
            tmp_path / "result",
            {},
            "environment",
            {"atol": 1e-6, "rtol": 1e-6},
            result_id="raw-dataset-result",
        )


@pytest.mark.parametrize("convention", ["v1", "v2"])
def test_loaded_score_preserves_dataset_identity_and_matches_direct_rows(
    tmp_path, monkeypatch, convention
):
    frame = dt.dataset("demo-gene-pairs@1.0.0", registry=_registry(tmp_path))
    direct = pd.read_csv(SOURCES["demo-gene-pairs@1.0.0"]["path"], keep_default_na=False)
    for value in (frame, direct):
        value.loc[0, "cdr3a_extended"] = "EXTENDED"
        value.loc[1, "trbj"] = ""

    def prepare(data, convention, *, species):
        output = data.copy()
        entries = []
        for _, row in output.iterrows():
            extended = row.get("cdr3a_extended", "")
            cdr3a = extended if isinstance(extended, str) and extended else row["cdr3a"]
            entries.append(
                None
                if not row["trbj"]
                else {
                    "sequences": {
                        "Peptide": row["peptide"],
                        "TCR_a": cdr3a,
                    }
                }
            )
        output["reconstruction_ok"] = [entry is not None for entry in entries]
        output["reconstruction_reason"] = [
            "" if entry is not None else "blank" for entry in entries
        ]
        return output, entries

    monkeypatch.setattr("DecoderTCR.reconstruct.components.prepare_gene_inputs", prepare)
    monkeypatch.setattr(
        "DecoderTCR.runtime.run_pll_benchmark",
        lambda module, layers, selected, *args: [
            float(len(entry["sequences"]["TCR_a"])) for entry in selected
        ],
    )
    model = LoadedModel(
        torch.nn.Linear(1, 1),
        "fixture@1.0.0:tiny",
        "esmc",
        "tiny",
        convention,
        "b" * 64,
        {"sha256": "c" * 64},
        "d" * 64,
        torch.device("cpu"),
        1,
    )
    loaded_result = model.score(frame)
    direct_result = model.score(direct)
    pd.testing.assert_frame_equal(loaded_result, direct_result)
    assert loaded_result.loc[0, "pll"] == len("EXTENDED")
    assert pd.isna(loaded_result.loc[1, "pll"])
    assert loaded_result.attrs["dataset"] == frame.attrs["dataset"]
    assert "dataset" not in direct_result.attrs


@pytest.mark.parametrize("convention", ["v1", "v2"])
def test_immrep_loader_composes_with_real_gene_preparation(tmp_path, monkeypatch, convention):
    loaded = dt.dataset("immrep23-public-inputs@1.0.0", registry=_registry(tmp_path))
    direct = pd.read_csv(SOURCES["immrep23-public-inputs@1.0.0"]["path"], keep_default_na=False)
    loaded_subset = pd.concat([loaded.iloc[[0]], loaded.iloc[[0]], loaded.iloc[[1]]])
    direct_subset = pd.concat([direct.iloc[[0]], direct.iloc[[0]], direct.iloc[[1]]])
    loaded_subset.attrs = dict(loaded.attrs)
    loaded_subset.iloc[2, loaded_subset.columns.get_loc("Jb")] = ""
    direct_subset.iloc[2, direct_subset.columns.get_loc("Jb")] = ""

    monkeypatch.setattr(
        "DecoderTCR.runtime.run_pll_benchmark",
        lambda module, layers, selected, *args: [float(index) for index, _ in enumerate(selected)],
    )
    model = LoadedModel(
        torch.nn.Linear(1, 1),
        "fixture@1.0.0:tiny",
        "esmc",
        "tiny",
        convention,
        "b" * 64,
        {"sha256": "c" * 64},
        "d" * 64,
        torch.device("cpu"),
        1,
    )
    loaded_result = dt.score(loaded_subset, model=model)
    direct_result = model.score(direct_subset)
    pd.testing.assert_frame_equal(loaded_result, direct_result)
    assert loaded_result.index.tolist() == [0, 0, 1]
    assert loaded_result["ID"].tolist() == [0, 0, 1]
    assert loaded_result["Usage"].tolist() == ["Private", "Private", "Private"]
    assert loaded_result["Label"].tolist() == [0, 0, 0]
    assert loaded_result["reconstruction_ok"].tolist() == [True, True, False]
    assert pd.isna(loaded_result.iloc[2]["pll"])
    alpha_column = loaded_result.attrs["reconstruction_columns"]["TCR_a"]
    assert loaded_subset.iloc[0]["CDR3a_extended"] in loaded_result.iloc[0][alpha_column]
    assert loaded_result.attrs["dataset"] == loaded_subset.attrs["dataset"]


def test_cli_exports_exact_bytes_sidecar_and_refuses_overwrite(tmp_path, monkeypatch):
    root = _registry(tmp_path)
    output = tmp_path / "export.csv"
    cli.main(["--registry", str(root), "dataset", "demo-gene-pairs@1.0.0", "--output", str(output)])
    assert output.read_bytes() == SOURCES["demo-gene-pairs@1.0.0"]["path"].read_bytes()
    sidecar = output.with_suffix(".csv.json")
    assert json.loads(sidecar.read_text())["dataset_id"] == "demo-gene-pairs@1.0.0"
    assert (
        cli.main(
            ["--registry", str(root), "dataset", "demo-gene-pairs@1.0.0", "--output", str(output)]
        )
        == 2
    )

    dangling = tmp_path / "dangling.csv"
    os.symlink(tmp_path / "absent", dangling)
    assert (
        cli.main(
            ["--registry", str(root), "dataset", "demo-gene-pairs@1.0.0", "--output", str(dangling)]
        )
        == 2
    )


def test_cli_race_never_removes_competing_output(tmp_path, monkeypatch):
    root = _registry(tmp_path)
    output = tmp_path / "race.csv"
    original = Path.open

    def collide(path, mode="r", *args, **kwargs):
        if path == output and mode == "xb":
            output.write_bytes(b"other process")
            raise FileExistsError(output)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", collide)
    assert (
        cli.main(
            ["--registry", str(root), "dataset", "demo-gene-pairs@1.0.0", "--output", str(output)]
        )
        == 2
    )
    assert output.read_bytes() == b"other process"
    assert not output.with_suffix(".csv.json").exists()
