from types import SimpleNamespace

import pandas as pd
import pytest

from DecoderTCR import reproduction as r
from DecoderTCR.core import (
    IntegrityError,
    UnavailableError,
    ValidationError,
    artifact_descriptor,
    atomic_json,
    read_json,
    seal,
    secure_path,
    sha256_file,
    verify_document,
)

MODEL_ID = "decodertcr@1.0.0:300M"
ARTIFACT_HASH = "1" * 64


class FixtureRegistry:
    def __init__(self, root):
        self.root = root
        self.entries = {"datasets": {}, "results": {}, "environments": {}}
        self.state = "available"

    def add(self, kind, identity, manifest, bundle):
        atomic_json(bundle / "manifest.json", manifest)
        self.entries[kind][identity] = bundle

    def resolve_bundle(self, kind, identity):
        bundle = self.entries[kind][identity]
        return verify_document(read_json(bundle / "manifest.json")), bundle

    def verify_artifact(self, bundle, descriptor):
        path = secure_path(bundle, descriptor["path"])
        if sha256_file(path) != descriptor["sha256"]:
            raise IntegrityError("Artifact changed")
        return path

    def info(self, identity):
        assert identity == MODEL_ID
        return {
            "member": {"weights": {"sha256": ARTIFACT_HASH}, "sequence_convention": "v2"},
            "catalog_sha256": "2" * 64,
            "generation": 1,
            "state": self.state,
        }

    def catalog(self):
        return self.entries


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    registry = FixtureRegistry(tmp_path)
    frame = pd.DataFrame(
        {
            "row_id": ["one", "two", "three", "four"],
            "HLA_a": ["AAA"] * 4,
            "HLA_b": ["CCC"] * 4,
            "TCR_a": ["AA", "AA", "BB", "BB"],
            "TCR_b": ["CC"] * 4,
            "Peptide": ["AAA", "CCC", "AAA", "CCC"],
            "group_id": ["one", "one", "two", "two"],
            "label": [1, 0, 0, 1],
        }
    )
    dataset_bundle = tmp_path / "dataset"
    dataset_bundle.mkdir()
    frame.to_parquet(dataset_bundle / "table.parquet", index=False)
    frame.head(2).to_csv(dataset_bundle / "example.csv", index=False)
    dataset = seal(
        {
            "schema_version": 1,
            "kind": "dataset",
            "dataset_id": "fixture-v2@1.0.0",
            "benchmark_name": "fixture",
            "sequence_convention": "v2",
            "table": artifact_descriptor(dataset_bundle / "table.parquet", dataset_bundle),
            "example": artifact_descriptor(dataset_bundle / "example.csv", dataset_bundle),
            "n_rows": 4,
            "group_key": "clone",
            "metric_type": "ranking",
            "inventory": {},
        }
    )
    registry.add("datasets", dataset["dataset_id"], dataset, dataset_bundle)
    env_bundle = tmp_path / "environment"
    env_bundle.mkdir()
    environment = seal(
        {"schema_version": 1, "kind": "environment", "environment_id": "fixture-env"}
    )
    registry.add("environments", "fixture-env", environment, env_bundle)
    predictions = frame.assign(
        pll=[4.0, 3.0, 1.0, 2.0], model_id=MODEL_ID, artifact_sha256=ARTIFACT_HASH
    )
    candidate = tmp_path / "result"
    manifest = r.create_result_bundle(
        registry,
        MODEL_ID,
        dataset["dataset_id"],
        predictions,
        candidate,
        {},
        "fixture-env",
        {"atol": 1e-6, "rtol": 1e-6},
        result_id="fixture-result",
    )
    registry.add("results", "fixture-result", manifest, candidate)
    monkeypatch.setenv("DECODERTCR_REPRO_ENV", "fixture-env")
    return registry, manifest, predictions


def test_reference_is_not_labeled_reproduced_and_reproduction_is_read_only(prepared, tmp_path):
    registry, manifest, _ = prepared
    assert manifest["reference_status"] == "reference_only"
    paths = [
        path
        for values in registry.entries.values()
        for bundle in values.values()
        for path in bundle.rglob("*")
        if path.is_file()
    ]
    before = {str(path): (sha256_file(path), path.stat().st_mtime_ns) for path in paths}
    result = r.reproduce("fixture-result", output=tmp_path / "output", registry=registry)
    assert result["status"] == "reproduced"
    record = verify_document(read_json(tmp_path / "output/reproduction.json"))
    assert record["dataset_sha256"] == manifest["dataset_sha256"]
    assert record["artifact_sha256"] == ARTIFACT_HASH
    assert record["runtime"]["environment"] == "installed"
    assert record["runtime"]["package_version"] == r.__version__
    assert record["runtime"]["python_version"] == r.platform.python_version()
    assert record["runtime"]["dependencies"]["numpy"] == r.version("numpy")
    assert before == {str(path): (sha256_file(path), path.stat().st_mtime_ns) for path in paths}
    with pytest.raises(ValidationError, match="new directory"):
        r.reproduce("fixture-result", output=tmp_path / "output", registry=registry)


def test_paired_dataset_auto_selects_model_convention_and_binds_view(prepared, tmp_path):
    registry, _, predictions = prepared
    columns = [
        "row_id",
        "HLA_a",
        "HLA_b",
        "TCR_a",
        "TCR_b",
        "Peptide",
        "group_id",
        "label",
    ]
    v2 = predictions[columns].copy()
    v1 = v2.assign(row_id=lambda frame: "v1-" + frame.row_id)
    bundle = tmp_path / "paired-dataset"
    bundle.mkdir()
    views = {}
    for convention, frame in (("v1", v1), ("v2", v2)):
        directory = bundle / convention
        directory.mkdir()
        frame.to_parquet(directory / "table.parquet", index=False)
        frame.head(2).to_csv(directory / "example.csv", index=False)
        (directory / "exclusions.json").write_text("{}", encoding="utf-8")
        (directory / "raw.json").write_text("{}", encoding="utf-8")
        views[convention] = seal(
            {
                "sequence_convention": convention,
                "table": artifact_descriptor(directory / "table.parquet", bundle),
                "example": artifact_descriptor(directory / "example.csv", bundle),
                "exclusions": artifact_descriptor(directory / "exclusions.json", bundle),
                "raw": [artifact_descriptor(directory / "raw.json", bundle)],
                "n_rows": len(frame),
                "n_source_rows": len(frame),
                "n_excluded_source_rows": 0,
                "measured_convention": {
                    "verdict": convention,
                    "modal_lengths": {"HLA_a": 3, "TCR_a": 2, "TCR_b": 2},
                },
            }
        )
    dataset = seal(
        {
            "schema_version": 2,
            "kind": "dataset",
            "dataset_id": "fixture@1.0.0",
            "benchmark_name": "fixture",
            "group_key": "clone",
            "metric_type": "ranking",
            "inventory": {},
            "preparation": "complete_population_before_sharding_v1",
            "population_sha256": "a" * 64,
            "views": views,
        }
    )
    registry.add("datasets", dataset["dataset_id"], dataset, bundle)
    result_bundle = tmp_path / "paired-result"
    result = r.create_result_bundle(
        registry,
        MODEL_ID,
        dataset["dataset_id"],
        predictions,
        result_bundle,
        {},
        "fixture-env",
        {"atol": 1e-6, "rtol": 1e-6},
        result_id="paired-result",
    )
    assert result["schema_version"] == 2
    assert result["sequence_convention"] == "v2"
    assert result["dataset_view_sha256"] == views["v2"]["sha256"]

    rejected = tmp_path / "paired-result-v2"
    with pytest.raises(ValidationError, match="omit sequence convention suffixes"):
        r.create_result_bundle(
            registry,
            MODEL_ID,
            dataset["dataset_id"],
            predictions,
            rejected,
            {},
            "fixture-env",
            {"atol": 1e-6, "rtol": 1e-6},
            result_id="paired-result-v2",
        )
    assert not rejected.exists()

    registry.add("results", result["result_id"], result, result_bundle)
    r.reproduce("paired-result", output=tmp_path / "paired-output", registry=registry)
    record = verify_document(read_json(tmp_path / "paired-output/reproduction.json"))
    assert record["sequence_convention"] == "v2"
    assert record["dataset_view_sha256"] == views["v2"]["sha256"]

    changed = seal({**result, "dataset_view_sha256": views["v1"]["sha256"]})
    atomic_json(result_bundle / "manifest.json", changed)
    with pytest.raises(IntegrityError, match="view identity"):
        r.reproduce("paired-result", output=tmp_path / "wrong-view", registry=registry)


def test_metrics_remain_available_when_model_blocked(prepared, tmp_path):
    registry, _, _ = prepared
    registry.state = "blocked"
    assert (
        r.reproduce("fixture-result", output=tmp_path / "metrics", registry=registry)["status"]
        == "reproduced"
    )


def test_full_reproduction_loads_through_live_policy_and_verifies_scores(
    prepared, tmp_path, monkeypatch
):
    from DecoderTCR import runtime

    registry, _, predictions = prepared
    calls = []

    def load(model_id, *, device, registry):
        calls.append(model_id)
        if registry.state == "blocked":
            raise UnavailableError("blocked")

        def score_prepared(table):
            return predictions

        return SimpleNamespace(_score_prepared=score_prepared)

    monkeypatch.setattr(runtime, "load", load)
    result = r.reproduce("fixture-result", "full", tmp_path / "full", registry)
    assert result["status"] == "reproduced"
    assert calls == [MODEL_ID]
    registry.state = "blocked"
    with pytest.raises(UnavailableError, match="blocked"):
        r.reproduce("fixture-result", "full", tmp_path / "blocked", registry)
    assert not (tmp_path / "blocked/reproduction.json").exists()


def test_corrupt_predictions_and_changed_metrics_fail(prepared, tmp_path):
    registry, manifest, _ = prepared
    bundle = registry.entries["results"]["fixture-result"]
    changed = dict(manifest)
    changed["expected_metrics"] = {**manifest["expected_metrics"], "macro_auroc": 0.2}
    atomic_json(bundle / "manifest.json", seal(changed))
    with pytest.raises(IntegrityError, match="tolerance"):
        r.reproduce("fixture-result", output=tmp_path / "changed", registry=registry)
    atomic_json(bundle / "manifest.json", manifest)
    with (bundle / "predictions.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(IntegrityError, match="Artifact changed"):
        r.reproduce("fixture-result", output=tmp_path / "corrupt", registry=registry)


def test_result_population_and_artifact_binding_are_required(prepared, tmp_path):
    registry, manifest, predictions = prepared

    def create(frame):
        return r.create_result_bundle(
            registry,
            MODEL_ID,
            manifest["dataset_id"],
            frame,
            tmp_path / "invalid",
            {},
            "fixture-env",
            {"atol": 1e-6, "rtol": 1e-6},
            result_id="bad",
        )

    with pytest.raises(ValidationError, match="exact frozen"):
        create(predictions.head(3))
    with pytest.raises(IntegrityError, match="checkpoint contents"):
        create(predictions.assign(artifact_sha256="0" * 64))
    with pytest.raises(IntegrityError, match="input column changed"):
        create(predictions.assign(label=[0, 0, 0, 1]))
    assert not (tmp_path / "invalid").exists()


def test_environment_dispatch_is_automatic(prepared, monkeypatch):
    from DecoderTCR import environment

    registry, _, _ = prepared
    monkeypatch.delenv("DECODERTCR_REPRO_ENV")
    calls = []
    monkeypatch.setattr(
        environment, "run_reproduction", lambda *args: calls.append(args) or {"ok": True}
    )
    assert r.reproduce("fixture-result", registry=registry) == {"ok": True}
    assert calls[0][-1] == "fixture-env"


def test_recipe_settings_and_tolerances_reject_silent_coercion():
    dataset = {"metric_type": "ranking", "group_key": "clone"}
    for recipe in ({"controls": "false"}, {"batch_size": True}, {"recall_ks": [1.5]}):
        with pytest.raises(ValidationError):
            r._default_recipe(dataset, recipe)
    with pytest.raises(ValidationError):
        r._validate_tolerances({"atol": "1e-6", "rtol": 0})


def test_demo_resolves_exact_published_example(prepared, tmp_path, monkeypatch):
    from DecoderTCR import runtime

    registry, manifest, predictions = prepared
    candidate = tmp_path / "demo"
    demo = r.create_result_bundle(
        registry,
        MODEL_ID,
        manifest["dataset_id"],
        predictions.head(2),
        candidate,
        {"dataset_view": "example"},
        "fixture-env",
        {"atol": 1e-6, "rtol": 1e-6},
        result_id="fixture-demo",
        purpose="demo",
    )
    registry.add("results", "fixture-demo", demo, candidate)
    assert (candidate / "expected_scores.csv").is_file()
    calls = []

    def load(model_id, **kwargs):
        calls.append(model_id)
        return _fixture_scorer(predictions)

    monkeypatch.setattr(runtime, "load", load)
    output = tmp_path / "demo-output"
    result = r.demo(MODEL_ID, output=output, registry=registry)
    assert result["status"] == "reproduced"
    assert result["result_id"] == "fixture-demo"
    assert result["mode"] == "full"
    assert calls == [MODEL_ID]
    record = verify_document(read_json(output / "reproduction.json"))
    assert record["runtime"] == r._runtime_identity()


def test_derived_benchmark_filters_parent_instead_of_rescoring(prepared, tmp_path):
    registry, parent, predictions = prepared
    parent_dataset, parent_bundle = registry.resolve_bundle("datasets", parent["dataset_id"])
    parent_dataset["benchmark_name"] = "prp"
    parent_dataset = seal(parent_dataset)
    registry.add("datasets", parent_dataset["dataset_id"], parent_dataset, parent_bundle)
    parent = seal({**parent, "dataset_sha256": parent_dataset["sha256"]})
    registry.add(
        "results", parent["result_id"], parent, registry.entries["results"]["fixture-result"]
    )
    source = pd.read_parquet(parent_bundle / "table.parquet").head(2).copy()
    source["row_id"] = ["derived-one", "derived-two"]
    bundle = tmp_path / "derived"
    bundle.mkdir()
    source.to_parquet(bundle / "table.parquet", index=False)
    derived = seal(
        {
            **parent_dataset,
            "dataset_id": "chris-v2@1.0.0",
            "benchmark_name": "chris",
            "n_rows": 2,
            "inventory": {"derived_from": "prp"},
            "table": artifact_descriptor(bundle / "table.parquet", bundle),
        }
    )
    registry.add("datasets", derived["dataset_id"], derived, bundle)
    manifest = r.create_result_bundle(
        registry,
        MODEL_ID,
        derived["dataset_id"],
        None,
        tmp_path / "derived-result",
        {},
        "fixture-env",
        {"atol": 1e-6, "rtol": 1e-6},
        result_id="derived-result",
        parent_result_id="fixture-result",
    )
    assert manifest["parent_result_id"] == "fixture-result"
    scores = pd.read_parquet(tmp_path / "derived-result/predictions.parquet")
    assert scores.pll.tolist() == predictions.pll.head(2).tolist()
    assert scores.row_id.tolist() == ["derived-one", "derived-two"]


def _fixture_scorer(predictions):
    indexed = predictions.set_index("row_id", drop=False)

    def score_prepared(table):
        result = indexed.loc[table.row_id].reset_index(drop=True)
        result.attrs["reproduction"] = {"fixture": True}
        return result

    return SimpleNamespace(_score_prepared=score_prepared)


def test_shards_are_scored_without_a_reproduction_claim_and_merge_in_frozen_order(
    prepared, tmp_path, monkeypatch
):
    from DecoderTCR import runtime

    registry, manifest, predictions = prepared
    monkeypatch.setattr(runtime, "load", lambda *args, **kwargs: _fixture_scorer(predictions))
    outputs = []
    for index in range(2):
        output = tmp_path / f"shard-{index}"
        result = r.reproduce_shard("fixture-result", index, 2, output, registry)
        assert result["status"] == "scored"
        assert not (output / "reproduction.json").exists()
        record = verify_document(read_json(output / "reproduction-shard.json"))
        assert record["status"] == "scored"
        assert record["reference_manifest_sha256"] == manifest["sha256"]
        assert record["shard"]["strategy"] == "frozen_row_stride_v1"
        assert record["runtime"] == r._runtime_identity()
        outputs.append(output)

    merged = r.merge_reproduction_shards(
        "fixture-result", list(reversed(outputs)), tmp_path / "merged", registry
    )
    assert merged["status"] == "reproduced"
    assert merged["shard_merge"]["shard_count"] == 2
    assert [item["index"] for item in merged["shard_merge"]["shards"]] == [0, 1]
    table = pd.read_parquet(tmp_path / "merged/predictions.parquet")
    assert table.row_id.tolist() == ["one", "two", "three", "four"]
    assert table.pll.tolist() == predictions.pll.tolist()
    record = verify_document(read_json(tmp_path / "merged/reproduction.json"))
    assert record["status"] == "reproduced"
    assert record["shard_merge"]["row_count"] == 4
    assert record["runtime"] == r._runtime_identity()


def test_merge_rejects_incomplete_duplicate_and_changed_shard_coverage(
    prepared, tmp_path, monkeypatch
):
    from DecoderTCR import runtime

    registry, _, predictions = prepared
    monkeypatch.setattr(runtime, "load", lambda *args, **kwargs: _fixture_scorer(predictions))
    first, second = tmp_path / "first", tmp_path / "second"
    r.reproduce_shard("fixture-result", 0, 2, first, registry)
    r.reproduce_shard("fixture-result", 1, 2, second, registry)
    with pytest.raises(IntegrityError, match="duplicated"):
        r.merge_reproduction_shards(
            "fixture-result", [first, first], tmp_path / "duplicates", registry
        )
    with pytest.raises(IntegrityError, match="complete index"):
        r.merge_reproduction_shards("fixture-result", [first], tmp_path / "incomplete", registry)

    changed = pd.read_parquet(second / "predictions.parquet").iloc[::-1]
    changed.to_parquet(second / "predictions.parquet", index=False)
    record = read_json(second / "reproduction-shard.json")
    record["predictions"] = artifact_descriptor(second / "predictions.parquet", second)
    atomic_json(second / "reproduction-shard.json", seal(record))
    with pytest.raises(IntegrityError, match="order"):
        r.merge_reproduction_shards(
            "fixture-result", [first, second], tmp_path / "changed", registry
        )


def test_derived_full_reproduction_reuses_verified_merged_parent(prepared, tmp_path, monkeypatch):
    from DecoderTCR import runtime

    registry, parent, predictions = prepared
    parent_dataset, parent_bundle = registry.resolve_bundle("datasets", parent["dataset_id"])
    parent_dataset["benchmark_name"] = "prp"
    parent_dataset = seal(parent_dataset)
    registry.add("datasets", parent_dataset["dataset_id"], parent_dataset, parent_bundle)
    parent = seal({**parent, "dataset_sha256": parent_dataset["sha256"]})
    registry.add(
        "results", parent["result_id"], parent, registry.entries["results"]["fixture-result"]
    )

    source = pd.read_parquet(parent_bundle / "table.parquet").head(2).copy()
    source["row_id"] = ["derived-one", "derived-two"]
    bundle = tmp_path / "derived-dataset"
    bundle.mkdir()
    source.to_parquet(bundle / "table.parquet", index=False)
    derived = seal(
        {
            **parent_dataset,
            "dataset_id": "chris-v2@1.0.0",
            "benchmark_name": "chris",
            "n_rows": 2,
            "inventory": {"derived_from": "prp"},
            "table": artifact_descriptor(bundle / "table.parquet", bundle),
        }
    )
    registry.add("datasets", derived["dataset_id"], derived, bundle)
    derived_result = r.create_result_bundle(
        registry,
        MODEL_ID,
        derived["dataset_id"],
        None,
        tmp_path / "derived-reference",
        {},
        "fixture-env",
        {"atol": 1e-6, "rtol": 1e-6},
        result_id="derived-result",
        parent_result_id="fixture-result",
    )
    registry.add("results", "derived-result", derived_result, tmp_path / "derived-reference")

    monkeypatch.setattr(runtime, "load", lambda *args, **kwargs: _fixture_scorer(predictions))
    shard_dirs = []
    for index in range(2):
        output = tmp_path / f"parent-shard-{index}"
        r.reproduce_shard("fixture-result", index, 2, output, registry)
        shard_dirs.append(output)
    parent_output = tmp_path / "parent-merged"
    r.merge_reproduction_shards("fixture-result", shard_dirs, parent_output, registry)

    def refuse_load(*args, **kwargs):
        raise AssertionError("Derived reproduction must not load or rescore the parent")

    monkeypatch.setattr(runtime, "load", refuse_load)
    result = r.reproduce(
        "derived-result",
        "full",
        tmp_path / "derived-full",
        registry,
        parent_output=parent_output,
    )
    assert result["status"] == "reproduced"
    scores = pd.read_parquet(tmp_path / "derived-full/predictions.parquet")
    assert scores.row_id.tolist() == ["derived-one", "derived-two"]
    assert scores.pll.tolist() == predictions.pll.head(2).tolist()
    record = verify_document(read_json(tmp_path / "derived-full/reproduction.json"))
    assert record["execution"]["strategy"] == "derived_from_verified_merged_parent_v1"
