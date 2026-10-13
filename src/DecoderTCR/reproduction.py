"""Bind frozen benchmark inputs to retained scores and rerun exact recipes."""

from __future__ import annotations

import math
import os
import platform
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .benchmarks import compute_metrics, shard_rows
from .core import (
    IntegrityError,
    UnavailableError,
    ValidationError,
    artifact_descriptor,
    atomic_json,
    digest_document,
    parse_model_id,
    read_json,
    seal,
    secure_path,
    sha256_file,
    utc_now,
    verify_document,
)

_SHARD_STRATEGY = "frozen_row_stride_v1"


def _runtime_identity() -> dict:
    """Describe the installed runtime separately from the reference environment."""
    return {
        "environment": "installed",
        "package_version": __version__,
        "python_version": platform.python_version(),
        "dependencies": {
            name: version(name)
            for name in (
                "torch",
                "safetensors",
                "numpy",
                "pandas",
                "pyarrow",
                "scipy",
                "scikit-learn",
                "tqdm",
                "einops",
                "jsonschema",
            )
        },
    }


def _registry(value):
    from .registry import Registry

    return (
        value
        if isinstance(value, Registry) or hasattr(value, "resolve_bundle")
        else Registry(value)
    )


def _read_table(registry, bundle, descriptor) -> pd.DataFrame:
    path = registry.verify_artifact(bundle, descriptor)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    raise ValidationError("Recorded tables must be CSV or Parquet")


def _selected_dataset(manifest, sequence_convention, dataset_view_sha256=None):
    if manifest.get("schema_version") == 3:
        raise ValidationError(
            "Raw gene-input datasets cannot be used as reproduction bundles. "
            "Use a prepared reproduction dataset"
        )
    if manifest.get("schema_version") == 1:
        if dataset_view_sha256 is not None:
            raise IntegrityError("Legacy datasets do not have paired view identities")
        if (
            sequence_convention is not None
            and manifest.get("sequence_convention") != sequence_convention
        ):
            raise IntegrityError("Recorded dataset convention disagrees with its model")
        return manifest
    if manifest.get("schema_version") != 2:
        raise ValidationError("Upgrade DecoderTCR to read this dataset format")
    if sequence_convention not in ("v1", "v2"):
        raise ValidationError("Paired datasets require an explicit sequence convention")
    views = manifest.get("views")
    if not isinstance(views, dict) or sequence_convention not in views:
        raise ValidationError("Dataset does not provide the model's sequence convention")
    view = verify_document(views[sequence_convention])
    if view.get("sequence_convention") != sequence_convention:
        raise IntegrityError("Dataset view and sequence convention disagree")
    if dataset_view_sha256 is not None and view["sha256"] != dataset_view_sha256:
        raise IntegrityError("Recorded dataset view identity changed")
    selected = {**manifest, **view}
    selected["sha256"] = manifest["sha256"]
    selected["dataset_view_sha256"] = view["sha256"]
    return selected


def _dataset(
    registry,
    dataset_id,
    recipe,
    *,
    sequence_convention=None,
    dataset_view_sha256=None,
):
    manifest, bundle = registry.resolve_bundle("datasets", dataset_id)
    if manifest.get("kind") != "dataset" or manifest.get("dataset_id") != dataset_id:
        raise IntegrityError("Dataset identity disagrees with its catalog entry")
    manifest = _selected_dataset(manifest, sequence_convention, dataset_view_sha256)
    view = recipe.get("dataset_view", "table")
    if view not in ("table", "example"):
        raise ValidationError("Dataset view must be table or example")
    frame = _read_table(registry, bundle, manifest[view])
    if "row_id" not in frame or frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise IntegrityError("Dataset requires unique nonmissing row IDs")
    if view == "table" and len(frame) != manifest["n_rows"]:
        raise IntegrityError("Dataset row count disagrees with its manifest")
    return manifest, frame


def _align_predictions(table, predictions, model_id, artifact_sha256):
    required = {"row_id", "pll", "model_id", "artifact_sha256"}
    if not required.issubset(predictions.columns):
        raise ValidationError("Predictions require row_id, pll, model_id, and artifact_sha256")
    if predictions.row_id.duplicated().any() or set(predictions.row_id) != set(table.row_id):
        raise ValidationError("Predictions must cover the exact frozen row population once")
    if not predictions.model_id.eq(model_id).all():
        raise IntegrityError("Predictions refer to a different model ID")
    if not predictions.artifact_sha256.eq(artifact_sha256).all():
        raise IntegrityError("Predictions refer to different checkpoint contents")
    values = predictions.set_index("row_id").loc[table.row_id]
    for column in set(table.columns) & set(predictions.columns) - {"row_id"}:
        left, right = table[column].reset_index(drop=True), values[column].reset_index(drop=True)
        if not ((left == right) | (left.isna() & right.isna())).all():
            raise IntegrityError(f"Prediction input column changed: {column}")
    result = table.copy()
    for column in ("pll", "model_id", "artifact_sha256", "inference_version"):
        if column in values:
            result[column] = values[column].to_numpy()
    numeric = pd.to_numeric(result.pll, errors="raise")
    if np.isinf(numeric).any():
        raise ValidationError("Infinite prediction scores are invalid")
    return result


def _validate_tolerances(value):
    if not isinstance(value, dict) or not {"atol", "rtol"} <= value.keys():
        raise ValidationError("Record explicit atol and rtol reproduction tolerances")
    for key in ("atol", "rtol"):
        if (
            not isinstance(value[key], (int, float))
            or isinstance(value[key], bool)
            or not math.isfinite(value[key])
            or value[key] < 0
        ):
            raise ValidationError("Numerical tolerances must be finite and nonnegative")


def _default_recipe(dataset, recipe):
    supplied = dict(recipe)
    allowed = {
        "metric_type",
        "group_key",
        "score_column",
        "group_column",
        "recall_ks",
        "controls",
        "dataset_view",
        "device",
        "batch_size",
        "affinity_column",
        "affinity_transform",
    }
    if set(supplied) - allowed:
        raise ValidationError("Recipe contains unsupported settings")
    if type(supplied.get("batch_size", 1)) is not int or supplied.get("batch_size", 1) != 1:
        raise ValidationError("The retained scoring implementation uses batch_size=1")
    if "controls" in supplied and not isinstance(supplied["controls"], bool):
        raise ValidationError("controls must be a boolean")
    if "recall_ks" in supplied and (
        not isinstance(supplied["recall_ks"], list)
        or not supplied["recall_ks"]
        or any(type(k) is not int or k < 1 for k in supplied["recall_ks"])
    ):
        raise ValidationError("recall_ks must be a nonempty list of positive integers")
    if (
        supplied.get("score_column", "pll") != "pll"
        or supplied.get("group_column", "group_id") != "group_id"
    ):
        raise ValidationError("Reproduction uses the canonical pll and group_id columns")
    expected = {"metric_type": dataset["metric_type"], "group_key": dataset["group_key"]}
    for key, value in expected.items():
        if key in supplied and supplied[key] != value:
            raise ValidationError(f"Recipe {key} contradicts the dataset inventory")
    return {
        **expected,
        "score_column": "pll",
        "group_column": "group_id",
        "recall_ks": [10, 100, 1000],
        "controls": dataset["metric_type"] == "ranking",
        "dataset_view": "table",
        "device": "cuda",
        "batch_size": 1,
        **supplied,
    }


def _derived_predictions(parent_table, table, model_id, artifact_sha256):
    keys = ["group_id", "Peptide", "TCR_a", "TCR_b", "HLA_a", "HLA_b"]
    if parent_table.duplicated(keys).any():
        raise IntegrityError("Parent predictions contain ambiguous derived-view keys")
    values = parent_table[keys + ["label", "pll", "model_id", "artifact_sha256"]]
    joined = table.merge(
        values, on=keys, how="left", validate="one_to_one", suffixes=("", "_parent"), indicator=True
    )
    if not joined._merge.eq("both").all() or not joined.label.eq(joined.label_parent).all():
        raise IntegrityError("Derived dataset is not a label-consistent subset of its parent")
    predictions = joined[["row_id", "pll", "model_id", "artifact_sha256"]]
    return _align_predictions(table, predictions, model_id, artifact_sha256)


def create_result_bundle(
    registry,
    model_id,
    dataset_id,
    predictions,
    destination,
    recipe,
    environment_id,
    tolerances,
    *,
    result_id,
    parent_result_id=None,
    purpose="benchmark",
) -> dict:
    """Create a result bundle from registered model, dataset, and environment inputs."""
    registry = _registry(registry)
    parse_model_id(model_id)
    if not isinstance(result_id, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.@:-]*", result_id
    ):
        raise ValidationError("Invalid result ID")
    if purpose not in ("benchmark", "demo"):
        raise ValidationError("Result purpose must be benchmark or demo")
    _validate_tolerances(tolerances)
    info = registry.info(model_id)
    artifact_hash = info["member"]["weights"]["sha256"]
    sequence_convention = info["member"]["sequence_convention"]
    dataset_manifest, _ = registry.resolve_bundle("datasets", dataset_id)
    if dataset_manifest.get("schema_version") == 3:
        raise ValidationError(
            "Raw gene-input datasets cannot be used as reproduction bundles. "
            "Use a prepared reproduction dataset"
        )
    paired = dataset_manifest.get("schema_version") == 2
    if paired and re.search(r"-v[12]$", result_id):
        raise ValidationError(
            "Schema version 2 result identities must omit sequence convention suffixes"
        )
    recipe = _default_recipe(dataset_manifest, recipe)
    if purpose == "demo" and recipe["dataset_view"] != "example":
        raise ValidationError("Demo references require the example dataset view")
    dataset, table = _dataset(
        registry,
        dataset_id,
        recipe,
        sequence_convention=sequence_convention,
    )
    if dataset["sequence_convention"] != info["member"]["sequence_convention"]:
        raise ValidationError("Model and benchmark sequence conventions disagree")
    if not environment_id:
        raise ValidationError("Reference results require a retained execution environment")
    environment, _ = registry.resolve_bundle("environments", environment_id)
    if environment.get("environment_id") != environment_id:
        raise IntegrityError("Environment identity disagrees with its catalog entry")
    if dataset["inventory"].get("derived_from") and purpose == "benchmark":
        if not parent_result_id:
            raise ValidationError("Derived benchmarks require a parent result ID")
        parent, parent_bundle = registry.resolve_bundle("results", parent_result_id)
        parent_dataset, _ = registry.resolve_bundle("datasets", parent["dataset_id"])
        if parent_dataset.get("schema_version") == 3:
            raise ValidationError(
                "Raw gene-input datasets cannot be used as reproduction bundles. "
                "Use a prepared reproduction dataset"
            )
        if (
            parent["artifact_sha256"] != artifact_hash
            or parent["dataset_sha256"] != parent_dataset["sha256"]
        ):
            raise IntegrityError("Parent result artifact or dataset binding changed")
        if (
            parent["model_id"] != model_id
            or parent_dataset["benchmark_name"] != dataset["inventory"]["derived_from"]
        ):
            raise ValidationError("Derived benchmark parent has the wrong model or benchmark")
        parent_table = _read_table(registry, parent_bundle, parent["predictions"])
        _, parent_inputs = _dataset(
            registry,
            parent["dataset_id"],
            parent["recipe"],
            sequence_convention=parent.get("sequence_convention"),
            dataset_view_sha256=parent.get("dataset_view_sha256"),
        )
        parent_table = _align_predictions(parent_inputs, parent_table, model_id, artifact_hash)
        combined = _derived_predictions(parent_table, table, model_id, artifact_hash)
        if predictions is not None:
            supplied = _align_predictions(
                table, _coerce_table(predictions), model_id, artifact_hash
            )
            if not np.allclose(combined.pll, supplied.pll, rtol=0, atol=0, equal_nan=True):
                raise IntegrityError("Supplied derived scores disagree with parent predictions")
    else:
        if parent_result_id:
            raise ValidationError("Only derived benchmarks can reference a parent result")
        combined = _align_predictions(table, _coerce_table(predictions), model_id, artifact_hash)
    expected = compute_metrics(combined, recipe)
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    combined.to_parquet(dest / "predictions.parquet", index=False)
    if purpose == "demo":
        combined.to_csv(dest / "expected_scores.csv", index=False)
    atomic_json(dest / "metrics.json", expected)
    result_fields = {
        "schema_version": 2 if paired else 1,
        "kind": "result",
        "result_id": result_id,
        "purpose": purpose,
        "model_id": model_id,
        "artifact_sha256": artifact_hash,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset["sha256"],
        "predictions": artifact_descriptor(dest / "predictions.parquet", dest),
        "metrics": artifact_descriptor(dest / "metrics.json", dest),
        "recipe": recipe,
        "expected_metrics": expected,
        "tolerances": tolerances,
        "environment_id": environment_id,
        "environment_sha256": environment["sha256"],
        "parent_result_id": parent_result_id,
        "reference_status": "reference_only",
        "created_at": utc_now(),
    }
    if paired:
        result_fields.update(
            sequence_convention=sequence_convention,
            dataset_view_sha256=dataset["dataset_view_sha256"],
        )
    manifest = seal(result_fields)
    if purpose == "demo":
        manifest = seal(
            {**manifest, "expected_scores": artifact_descriptor(dest / "expected_scores.csv", dest)}
        )
    atomic_json(dest / "manifest.json", manifest)
    return manifest


def _coerce_table(value):
    if isinstance(value, pd.DataFrame):
        return value
    if value is None:
        raise ValidationError("Predictions are required")
    path = Path(value)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _compare(actual, expected, tolerances, path="metrics"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise IntegrityError(f"Reproduction keys disagree at {path}")
        for key, value in expected.items():
            _compare(actual[key], value, tolerances, f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise IntegrityError(f"Reproduction list disagrees at {path}")
        for index, value in enumerate(expected):
            _compare(actual[index], value, tolerances, f"{path}.{index}")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, abs_tol=tolerances["atol"], rel_tol=tolerances["rtol"]
        ):
            raise IntegrityError(f"Reproduction value exceeds tolerance at {path}")
    elif actual != expected:
        raise IntegrityError(f"Reproduction value disagrees at {path}")


def _row_ids_sha256(frame: pd.DataFrame) -> str:
    values = frame.row_id.tolist()
    if any(not isinstance(value, str) or not value for value in values):
        raise IntegrityError("Frozen row IDs must be nonempty strings")
    return digest_document({"row_ids": values})


def _verify_local_artifact(root: Path, descriptor: dict) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise IntegrityError("Local artifact descriptor is invalid")
    if (
        not isinstance(descriptor["path"], str)
        or not isinstance(descriptor["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"])
        or type(descriptor["size_bytes"]) is not int
        or descriptor["size_bytes"] < 0
    ):
        raise IntegrityError("Local artifact descriptor is invalid")
    path = secure_path(root, descriptor["path"])
    if (
        not path.is_file()
        or path.stat().st_size != descriptor["size_bytes"]
        or sha256_file(path) != descriptor["sha256"]
    ):
        raise IntegrityError("Local reproduction artifact changed")
    return path


def _resolve_result(registry, result_id):
    manifest, bundle = registry.resolve_bundle("results", result_id)
    if manifest.get("kind") != "result" or manifest.get("result_id") != result_id:
        raise IntegrityError("Result identity disagrees with its catalog entry")
    return manifest, bundle


def _resolve_environment(registry, manifest):
    environment_id = manifest["environment_id"]
    environment, _ = registry.resolve_bundle("environments", environment_id)
    if environment["sha256"] != manifest["environment_sha256"]:
        raise IntegrityError("Recorded execution environment changed")
    return environment_id, environment


def _reference_tables(registry, manifest, bundle):
    tolerances = manifest["tolerances"]
    _validate_tolerances(tolerances)
    if manifest.get("schema_version") == 2 and (
        manifest.get("sequence_convention") not in ("v1", "v2")
        or not isinstance(manifest.get("dataset_view_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest["dataset_view_sha256"])
    ):
        raise IntegrityError("Paired result has no valid dataset view binding")
    dataset, table = _dataset(
        registry,
        manifest["dataset_id"],
        manifest["recipe"],
        sequence_convention=manifest.get("sequence_convention"),
        dataset_view_sha256=manifest.get("dataset_view_sha256"),
    )
    if dataset["sha256"] != manifest["dataset_sha256"]:
        raise IntegrityError("Recorded dataset identity changed")
    reference = _read_table(registry, bundle, manifest["predictions"])
    reference = _align_predictions(
        table, reference, manifest["model_id"], manifest["artifact_sha256"]
    )
    return tolerances, dataset, table, reference


def _dataset_view_binding(dataset):
    digest = dataset.get("dataset_view_sha256")
    return {"dataset_view_sha256": digest} if digest is not None else {}


def _observed_model(registry, manifest, dataset):
    observed = registry.info(manifest["model_id"])
    if observed["member"]["weights"]["sha256"] != manifest["artifact_sha256"]:
        raise IntegrityError("Result model artifact identity changed")
    if dataset["sequence_convention"] != observed["member"]["sequence_convention"]:
        raise IntegrityError("Recorded dataset convention disagrees with its model")
    return observed


def reproduce_shard(
    result_id,
    shard_index,
    shard_count,
    output,
    registry=None,
) -> dict:
    """Score one deterministic shard of a frozen full-reproduction population."""
    if (
        type(shard_index) is not int
        or type(shard_count) is not int
        or shard_count < 1
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise ValidationError("Shard index must be in the range of the positive shard count")
    if output is None:
        raise ValidationError("Shard reproduction requires an explicit output directory")
    registry = _registry(registry)
    manifest, bundle = _resolve_result(registry, result_id)
    if manifest.get("parent_result_id"):
        raise ValidationError("Derived results must reuse a verified parent reproduction")
    environment_id, environment = _resolve_environment(registry, manifest)
    if os.environ.get("DECODERTCR_REPRO_ENV") != environment_id:
        from .environment import run_reproduction

        return run_reproduction(
            registry,
            result_id,
            "full",
            output,
            environment_id,
            shard_index=shard_index,
            shard_count=shard_count,
        )
    tolerances, dataset, table, reference = _reference_tables(registry, manifest, bundle)
    if shard_count > len(table):
        raise ValidationError("Shard count cannot exceed the frozen row count")
    observed = _observed_model(registry, manifest, dataset)
    selected = shard_rows(table, shard_index, shard_count)
    reference_by_id = reference.set_index("row_id", drop=False)
    expected = reference_by_id.loc[selected.row_id].reset_index(drop=True)
    expected = _align_predictions(
        selected, expected, manifest["model_id"], manifest["artifact_sha256"]
    )
    destination = Path(output)
    if destination.exists():
        raise ValidationError("Reproduction output must be a new directory")
    destination.mkdir(parents=True)
    from .runtime import load

    model = load(
        manifest["model_id"],
        registry=registry,
        device=manifest["recipe"].get("device", "cuda"),
    )
    computed = model._score_prepared(selected)
    execution = computed.attrs.get("reproduction")
    computed = _align_predictions(
        selected, computed, manifest["model_id"], manifest["artifact_sha256"]
    )
    if not np.allclose(
        computed.pll,
        expected.pll,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        equal_nan=True,
    ):
        raise IntegrityError("Reproduced shard scores exceed the recorded numerical tolerance")
    computed.to_parquet(destination / "predictions.parquet", index=False)
    shard = {
        "strategy": _SHARD_STRATEGY,
        "index": shard_index,
        "count": shard_count,
        "row_count": len(selected),
        "row_ids_sha256": _row_ids_sha256(selected),
    }
    record = seal(
        {
            "schema_version": 1,
            "kind": "reproduction_shard",
            "status": "scored",
            "runtime": _runtime_identity(),
            "result_id": result_id,
            "reference_manifest_sha256": manifest["sha256"],
            "model_id": manifest["model_id"],
            "artifact_sha256": manifest["artifact_sha256"],
            "dataset_id": manifest["dataset_id"],
            "dataset_sha256": dataset["sha256"],
            **_dataset_view_binding(dataset),
            "sequence_convention": dataset["sequence_convention"],
            "environment_id": environment_id,
            "environment_sha256": environment["sha256"],
            "mode": "full",
            "shard": shard,
            "catalog_sha256": observed["catalog_sha256"],
            "catalog_generation": observed["generation"],
            "finished_at": utc_now(),
            "execution": execution,
            "predictions": artifact_descriptor(destination / "predictions.parquet", destination),
        }
    )
    atomic_json(destination / "reproduction-shard.json", record)
    return {
        "status": "scored",
        "result_id": result_id,
        "mode": "full",
        "shard": shard,
        "output": str(destination.resolve()),
    }


def _read_shard_output(
    root,
    manifest,
    environment,
    dataset,
    table,
):
    root = Path(root).resolve()
    record = verify_document(read_json(root / "reproduction-shard.json"))
    expected = {
        "kind": "reproduction_shard",
        "status": "scored",
        "result_id": manifest["result_id"],
        "reference_manifest_sha256": manifest["sha256"],
        "model_id": manifest["model_id"],
        "artifact_sha256": manifest["artifact_sha256"],
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": dataset["sha256"],
        **_dataset_view_binding(dataset),
        "sequence_convention": dataset["sequence_convention"],
        "environment_id": manifest["environment_id"],
        "environment_sha256": environment["sha256"],
        "mode": "full",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise IntegrityError(f"Shard record binding changed: {key}")
    shard = record.get("shard")
    if not isinstance(shard, dict) or set(shard) != {
        "strategy",
        "index",
        "count",
        "row_count",
        "row_ids_sha256",
    }:
        raise IntegrityError("Shard coverage record is invalid")
    if shard["strategy"] != _SHARD_STRATEGY:
        raise IntegrityError("Shard strategy changed")
    if (
        type(shard["index"]) is not int
        or type(shard["count"]) is not int
        or shard["count"] < 1
        or shard["index"] < 0
        or shard["index"] >= shard["count"]
        or shard["count"] > len(table)
    ):
        raise IntegrityError("Shard index or count is invalid")
    selected = shard_rows(table, shard["index"], shard["count"])
    if shard["row_count"] != len(selected) or shard["row_ids_sha256"] != _row_ids_sha256(selected):
        raise IntegrityError("Shard coverage differs from the frozen population")
    path = _verify_local_artifact(root, record.get("predictions"))
    if path.suffix != ".parquet":
        raise IntegrityError("Shard predictions must use Parquet")
    predictions = pd.read_parquet(path)
    if predictions.row_id.tolist() != selected.row_id.tolist():
        raise IntegrityError("Shard prediction order differs from its frozen coverage")
    predictions = _align_predictions(
        selected, predictions, manifest["model_id"], manifest["artifact_sha256"]
    )
    return record, predictions


def merge_reproduction_shards(
    result_id,
    shard_outputs,
    output,
    registry=None,
) -> dict:
    """Merge a complete exact shard set in the frozen dataset order."""
    if output is None:
        raise ValidationError("Shard merging requires an explicit output directory")
    if isinstance(shard_outputs, (str, Path)) or not shard_outputs:
        raise ValidationError("Shard merging requires one or more shard directories")
    registry = _registry(registry)
    manifest, bundle = _resolve_result(registry, result_id)
    if manifest.get("parent_result_id"):
        raise ValidationError("Derived results must reuse a verified parent reproduction")
    environment_id, environment = _resolve_environment(registry, manifest)
    if os.environ.get("DECODERTCR_REPRO_ENV") != environment_id:
        from .environment import run_reproduction

        return run_reproduction(
            registry,
            result_id,
            "full",
            output,
            environment_id,
            shard_outputs=shard_outputs,
        )
    tolerances, dataset, table, reference = _reference_tables(registry, manifest, bundle)
    observed = _observed_model(registry, manifest, dataset)
    shards = [
        _read_shard_output(path, manifest, environment, dataset, table) for path in shard_outputs
    ]
    counts = {record["shard"]["count"] for record, _ in shards}
    if len(counts) != 1:
        raise IntegrityError("Shard records disagree on the shard count")
    count = counts.pop()
    by_index = {record["shard"]["index"]: (record, frame) for record, frame in shards}
    if len(by_index) != len(shards):
        raise IntegrityError("Shard indexes are duplicated")
    if set(by_index) != set(range(count)):
        raise IntegrityError("Shard records do not form a complete index set")
    ordered = [by_index[index] for index in range(count)]
    row_ids = [row_id for _, frame in ordered for row_id in frame.row_id]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(table.row_id):
        raise IntegrityError("Shards are not an exact disjoint frozen population")
    combined = pd.concat([frame for _, frame in ordered], ignore_index=True)
    combined = _align_predictions(
        table, combined, manifest["model_id"], manifest["artifact_sha256"]
    )
    if not np.allclose(
        combined.pll,
        reference.pll,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        equal_nan=True,
    ):
        raise IntegrityError("Merged scores exceed the recorded numerical tolerance")
    metrics = compute_metrics(combined, manifest["recipe"])
    _compare(metrics, manifest["expected_metrics"], tolerances)
    destination = Path(output)
    if destination.exists():
        raise ValidationError("Reproduction output must be a new directory")
    destination.mkdir(parents=True)
    combined.to_parquet(destination / "predictions.parquet", index=False)
    atomic_json(destination / "metrics.json", metrics)
    summaries = [
        {
            "index": record["shard"]["index"],
            "record_sha256": record["sha256"],
            "predictions_sha256": record["predictions"]["sha256"],
            "row_count": record["shard"]["row_count"],
            "row_ids_sha256": record["shard"]["row_ids_sha256"],
        }
        for record, _ in ordered
    ]
    shard_merge = {
        "strategy": _SHARD_STRATEGY,
        "shard_count": count,
        "row_count": len(table),
        "row_ids_sha256": _row_ids_sha256(table),
        "shards": summaries,
    }
    record = seal(
        {
            "schema_version": 1,
            "result_id": result_id,
            "reference_manifest_sha256": manifest["sha256"],
            "model_id": manifest["model_id"],
            "artifact_sha256": manifest["artifact_sha256"],
            "dataset_id": manifest["dataset_id"],
            "dataset_sha256": dataset["sha256"],
            **_dataset_view_binding(dataset),
            "sequence_convention": dataset["sequence_convention"],
            "environment_id": environment_id,
            "environment_sha256": environment["sha256"],
            "mode": "full",
            "status": "reproduced",
            "catalog_sha256": observed["catalog_sha256"],
            "catalog_generation": observed["generation"],
            "finished_at": utc_now(),
            "execution": {"strategy": _SHARD_STRATEGY},
            "runtime": _runtime_identity(),
            "shard_merge": shard_merge,
            "predictions": artifact_descriptor(destination / "predictions.parquet", destination),
            "metrics": artifact_descriptor(destination / "metrics.json", destination),
        }
    )
    atomic_json(destination / "reproduction.json", record)
    return {
        "status": "reproduced",
        "result_id": result_id,
        "mode": "full",
        "output": str(destination.resolve()),
        "metrics": metrics,
        "shard_merge": shard_merge,
    }


def _verified_merged_parent(registry, parent_result_id, parent_output):
    parent, parent_bundle = _resolve_result(registry, parent_result_id)
    _, environment = _resolve_environment(registry, parent)
    tolerances, dataset, table, reference = _reference_tables(registry, parent, parent_bundle)
    root = Path(parent_output).resolve()
    record = verify_document(read_json(root / "reproduction.json"))
    expected = {
        "status": "reproduced",
        "result_id": parent_result_id,
        "reference_manifest_sha256": parent["sha256"],
        "model_id": parent["model_id"],
        "artifact_sha256": parent["artifact_sha256"],
        "dataset_id": parent["dataset_id"],
        "dataset_sha256": dataset["sha256"],
        **_dataset_view_binding(dataset),
        "sequence_convention": dataset["sequence_convention"],
        "environment_id": parent["environment_id"],
        "environment_sha256": environment["sha256"],
        "mode": "full",
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise IntegrityError(f"Parent reproduction binding changed: {key}")
    merge = record.get("shard_merge")
    if (
        not isinstance(merge, dict)
        or merge.get("strategy") != _SHARD_STRATEGY
        or merge.get("row_count") != len(table)
        or merge.get("row_ids_sha256") != _row_ids_sha256(table)
        or type(merge.get("shard_count")) is not int
        or merge["shard_count"] < 1
        or not isinstance(merge.get("shards"), list)
        or len(merge["shards"]) != merge["shard_count"]
    ):
        raise IntegrityError("Parent is not a verified complete shard merge")
    summaries = merge["shards"]
    if any(
        not isinstance(summary, dict)
        or set(summary)
        != {
            "index",
            "record_sha256",
            "predictions_sha256",
            "row_count",
            "row_ids_sha256",
        }
        for summary in summaries
    ):
        raise IntegrityError("Merged parent shard summaries are invalid")
    if (
        any(type(summary["index"]) is not int for summary in summaries)
        or [summary["index"] for summary in summaries] != list(range(merge["shard_count"]))
        or any(
            type(summary["row_count"]) is not int or summary["row_count"] < 1
            for summary in summaries
        )
        or sum(summary["row_count"] for summary in summaries) != len(table)
    ):
        raise IntegrityError("Merged parent shard coverage is invalid")
    predictions_path = _verify_local_artifact(root, record.get("predictions"))
    if predictions_path.suffix != ".parquet":
        raise IntegrityError("Parent predictions must use Parquet")
    predictions = pd.read_parquet(predictions_path)
    if predictions.row_id.tolist() != table.row_id.tolist():
        raise IntegrityError("Parent prediction order differs from the frozen population")
    predictions = _align_predictions(
        table, predictions, parent["model_id"], parent["artifact_sha256"]
    )
    if not np.allclose(
        predictions.pll,
        reference.pll,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        equal_nan=True,
    ):
        raise IntegrityError("Parent scores exceed the recorded numerical tolerance")
    metrics_path = _verify_local_artifact(root, record.get("metrics"))
    metrics = read_json(metrics_path)
    computed_metrics = compute_metrics(predictions, parent["recipe"])
    _compare(metrics, computed_metrics, tolerances, "parent_metrics")
    _compare(computed_metrics, parent["expected_metrics"], tolerances)
    return predictions, record


def reproduce(
    result_id,
    mode="metrics",
    output=None,
    registry=None,
    *,
    parent_output=None,
) -> dict:
    registry = _registry(registry)
    if mode not in ("metrics", "full"):
        raise ValidationError("Reproduction mode must be metrics or full")
    manifest, bundle = _resolve_result(registry, result_id)
    environment_id, _ = _resolve_environment(registry, manifest)
    if parent_output is not None and (mode != "full" or not manifest.get("parent_result_id")):
        raise ValidationError("A parent output applies only to full derived reproduction")
    if os.environ.get("DECODERTCR_REPRO_ENV") != environment_id:
        from .environment import run_reproduction

        if parent_output is None:
            return run_reproduction(registry, result_id, mode, output, environment_id)
        return run_reproduction(
            registry,
            result_id,
            mode,
            output,
            environment_id,
            parent_output=parent_output,
        )
    tolerances, dataset, table, reference = _reference_tables(registry, manifest, bundle)
    destination = (
        Path(output)
        if output is not None
        else Path("results")
        / (f"{result_id}-{mode}-{utc_now().replace(':', '').replace('+', '_')}")
    )
    if destination.exists():
        raise ValidationError("Reproduction output must be a new directory")
    destination.mkdir(parents=True)
    observed = registry.info(manifest["model_id"])
    if observed["member"]["weights"]["sha256"] != manifest["artifact_sha256"]:
        raise IntegrityError("Result model artifact identity changed")
    execution = None
    if mode == "full":
        if dataset["sequence_convention"] != observed["member"]["sequence_convention"]:
            raise IntegrityError("Recorded dataset convention disagrees with its model")
        if manifest.get("parent_result_id"):
            if parent_output is None:
                parent = reproduce(
                    manifest["parent_result_id"], "full", destination / "parent", registry
                )
                parent_predictions = pd.read_parquet(Path(parent["output"]) / "predictions.parquet")
            else:
                parent_predictions, parent_record = _verified_merged_parent(
                    registry, manifest["parent_result_id"], parent_output
                )
                execution = {
                    "strategy": "derived_from_verified_merged_parent_v1",
                    "parent_reproduction_sha256": parent_record["sha256"],
                }
            computed = _derived_predictions(
                parent_predictions, table, manifest["model_id"], manifest["artifact_sha256"]
            )
        else:
            from .runtime import load

            model = load(
                manifest["model_id"],
                registry=registry,
                device=manifest["recipe"].get("device", "cuda"),
            )
            computed = model._score_prepared(table)
            execution = computed.attrs.get("reproduction")
            computed = _align_predictions(
                table, computed, manifest["model_id"], manifest["artifact_sha256"]
            )
        if not np.allclose(
            computed.pll,
            reference.pll,
            atol=tolerances["atol"],
            rtol=tolerances["rtol"],
            equal_nan=True,
        ):
            raise IntegrityError("Reproduced scores exceed the recorded numerical tolerance")
    else:
        computed = reference
    metrics = compute_metrics(computed, manifest["recipe"])
    _compare(metrics, manifest["expected_metrics"], tolerances)
    computed.to_parquet(destination / "predictions.parquet", index=False)
    atomic_json(destination / "metrics.json", metrics)
    record = seal(
        {
            "schema_version": 1,
            "result_id": result_id,
            "reference_manifest_sha256": manifest["sha256"],
            "model_id": manifest["model_id"],
            "artifact_sha256": manifest["artifact_sha256"],
            "dataset_id": manifest["dataset_id"],
            "dataset_sha256": dataset["sha256"],
            **_dataset_view_binding(dataset),
            "sequence_convention": dataset["sequence_convention"],
            "environment_id": environment_id,
            "mode": mode,
            "status": "reproduced",
            "runtime": _runtime_identity(),
            "catalog_sha256": observed["catalog_sha256"],
            "catalog_generation": observed["generation"],
            "finished_at": utc_now(),
            "execution": execution,
            "predictions": artifact_descriptor(destination / "predictions.parquet", destination),
            "metrics": artifact_descriptor(destination / "metrics.json", destination),
        }
    )
    atomic_json(destination / "reproduction.json", record)
    return {
        "status": "reproduced",
        "result_id": result_id,
        "mode": mode,
        "output": str(destination.resolve()),
        "metrics": metrics,
    }


def demo(model_id, output=None, registry=None) -> dict:
    registry = _registry(registry)
    parse_model_id(model_id)
    candidates = []
    for identity in registry.catalog()["results"]:
        manifest, _ = registry.resolve_bundle("results", identity)
        if manifest.get("purpose") == "demo" and manifest.get("model_id") == model_id:
            candidates.append(identity)
    if not candidates:
        raise UnavailableError("No registered demo result exists for this exact model")
    if len(candidates) != 1:
        raise ValidationError(
            "Multiple examples are registered. Use an exact reproduction result ID"
        )
    return reproduce(candidates[0], mode="full", output=output, registry=registry)
