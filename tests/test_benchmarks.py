import json

import numpy as np
import pandas as pd
import pytest

from DecoderTCR import benchmarks as b
from DecoderTCR.core import IntegrityError, ValidationError, verify_document


def entry(peptide="AAA", clone="clone", label=1, allele="A"):
    return {
        "sequences": {
            "HLA_a": "A" * 180,
            "HLA_b": "C" * 90,
            "Peptide": peptide,
            "TCR_a": "D" * 110,
            "TCR_b": "E" * 120,
        },
        "meta_data": {
            "epitope": peptide,
            "true_epitope": peptide,
            "clone": clone,
            "label": label,
            "hla_allele": allele,
        },
        "pocket_idx": {},
    }


def convention_entry(convention, **kwargs):
    value = entry(**kwargs)
    if convention == "v1":
        value["sequences"].update(
            HLA_a="A" * 300,
            TCR_a="D" * 270,
            TCR_b="E" * 300,
        )
    return value


def paired_producer(tmp_path, monkeypatch):
    definitions = b.inventory()
    definitions["tcrvdb"]["n_entries"] = 2
    definitions["tcrvdb"]["n_pos"] = 1
    definitions["tcrvdb"]["n_neg"] = 1
    monkeypatch.setattr(b, "inventory", lambda: definitions)
    producer = tmp_path / "producer"
    (producer / "eval/scripts").mkdir(parents=True)
    (producer / "eval/data").mkdir()
    (producer / "eval/scripts/benchmark_registry.py").write_text(
        "raise RuntimeError('must not execute')\nBENCHMARKS = " + repr(definitions)
    )
    for convention in ("v1", "v2"):
        raw = {
            "a": convention_entry(convention, label=1),
            "b": convention_entry(convention, peptide="CCC", label=0),
        }
        source = producer / "eval/data" / definitions["tcrvdb"]["files"][convention]
        source.write_text(json.dumps(raw))
    return producer


def rewrite_v2_population(producer, definitions, transform):
    source = producer / "eval/data" / definitions["tcrvdb"]["files"]["v2"]
    raw = json.loads(source.read_text())
    source.write_text(json.dumps(transform(raw)))


def test_inventory_preserves_core_benchmarks_and_adds_public_grid():
    inventory = b.inventory()
    assert set(inventory) == {"tcrvdb", "immrep23", "viral-public-grid", "prp"}
    assert all(set(spec["files"]) == {"v1", "v2"} for spec in inventory.values())
    assert all(all(spec["files"].values()) for spec in inventory.values())
    assert "derives" not in inventory["prp"]
    assert "chris" not in json.dumps(b.release_inventory(inventory["prp"])).lower()


def test_freeze_rejects_unknown_convention():
    with pytest.raises(ValidationError, match="Unknown benchmark"):
        b.freeze_entries({"one": entry()}, "chris", "v2")


def test_recall_pessimistic_tie_and_requested_cutoff_names():
    values = b.recall_at_k(np.array([1, 1, 0]), np.array([0.9, 0.5, 0.5]), (1, 2, 100))
    assert values == {"recall@1": 0.5, "recall@2": 0.5, "recall@100": 1.0}
    assert b.recall_at_k(np.array([1, 0]), np.array([0.5, 0.5]), (1,))["recall@1"] == 0


def test_auroc_ap_skip_single_class_but_recall_preserves_positive_groups():
    frame = pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "b", "c", "d"],
            "label": [1, 0, 1, 0, 1, 0],
            "pll": [0.9, 0.1, 0.2, 0.8, 0.9, np.nan],
        }
    )
    metrics = b.compute_metrics(frame, {"controls": False, "recall_ks": [1, 100]})
    assert metrics["macro_auroc"] == 0.5
    assert metrics["macro_auprc"] == 0.75
    assert metrics["n_groups"] == 2
    assert metrics["n_groups_total"] == 3
    assert metrics["macro_prevalence"] == 0.5
    assert metrics["pooled_prevalence"] == 0.6
    assert metrics["macro_recall@1"] == pytest.approx(2 / 3)
    assert metrics["macro_recall@100"] == 1.0
    assert metrics["n_missing_scores"] == 1
    assert metrics["per_group"]["c"]["auroc"] is None


def test_uncrossed_design_suppresses_gain():
    frame = pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "b"],
            "label": [1, 0, 1, 0],
            "pll": [3, 1, 4, 2],
            "TCR_a": ["A", "A", "B", "B"],
            "TCR_b": ["C"] * 4,
            "Peptide": ["AA", "BB", "CC", "DD"],
        }
    )
    metrics = b.compute_metrics(frame, {"group_key": "clone"})
    assert metrics["control_used"] == "tcr_blind"
    assert metrics["control_value"] is None
    assert metrics["gain"] is None
    assert metrics["control_invalid"] == "design_not_crossed"
    assert metrics["control_subset_rows"] == 0


def test_missing_positive_scores_preserve_population_count_and_scored_metrics():
    frame = pd.DataFrame(
        {
            "group_id": ["a", "a", "a", "a"],
            "label": [1, 0, 1, 0],
            "pll": [0.9, 0.1, np.nan, np.nan],
        }
    )
    metrics = b.compute_metrics(frame, {"controls": False, "recall_ks": [1]})
    assert metrics["n_entries"] == 4
    assert metrics["n_scored"] == 2
    assert metrics["n_missing_scores"] == 2
    assert metrics["n_pos"] == 2
    assert metrics["macro_auroc"] == 1
    assert metrics["macro_auprc"] == 1
    assert metrics["macro_prevalence"] == 0.5
    assert metrics["pooled_prevalence"] == 0.5
    assert metrics["macro_recall@1"] == 1
    assert metrics["per_group"]["a"]["n_pos"] == 1


def test_crossed_marginal_axes_are_opposite():
    frame = pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "b"],
            "label": [1, 0, 0, 1],
            "pll": [4, 3, 1, 2],
            "TCR_a": ["A", "A", "B", "B"],
            "TCR_b": ["C"] * 4,
            "Peptide": ["AA", "BB", "AA", "BB"],
        }
    )
    metrics = b.compute_metrics(frame, {"group_key": "clone"})
    assert metrics["macro_auroc"] == 1
    assert metrics["tcr_blind"] == 0.5
    assert metrics["pep_blind"] == 0.5
    assert metrics["gain"] == 0.5
    assert metrics["control_degenerate"] == "exactly_chance"


def test_affinity_and_context_do_not_invent_binary_metrics():
    frame = pd.DataFrame({"pll": [-1.0, -2.0, -3.0], "log10_kd_um": [1.0, 2.0, 3.0]})
    affinity = b.compute_metrics(frame, {"metric_type": "affinity"})
    assert affinity["spearman"] == -1
    assert "macro_auroc" not in affinity
    context = b.compute_metrics(frame, {"metric_type": "context"})
    assert context["mean_pll"] == -2
    with pytest.raises(ValidationError, match="Infinite"):
        b.compute_metrics(pd.DataFrame({"pll": [np.inf]}), {"metric_type": "context"})


def test_import_copies_source_and_seals_only_relative_metadata(tmp_path, monkeypatch):
    definitions = b.inventory()
    definitions["tcrvdb"]["n_entries"] = 2
    definitions["tcrvdb"]["n_pos"] = 1
    definitions["tcrvdb"]["n_neg"] = 1
    monkeypatch.setattr(b, "inventory", lambda: definitions)
    producer = tmp_path / "producer"
    (producer / "eval/scripts").mkdir(parents=True)
    (producer / "eval/data").mkdir()
    (producer / "eval/scripts/benchmark_registry.py").write_text(
        "raise RuntimeError('must not execute')\nBENCHMARKS = " + repr(definitions)
    )
    raw = {"a": entry(label=1), "b": entry(label=0)}
    source = producer / "eval/data" / definitions["tcrvdb"]["files"]["v2"]
    source.write_text(json.dumps(raw))
    manifest = b.import_dataset(producer, "tcrvdb", "v2", tmp_path / "candidate")
    verify_document(manifest)
    assert manifest["n_rows"] == 2
    assert manifest["dataset_id"] == "tcrvdb-v2@1.0.0"
    assert (tmp_path / "candidate/raw/input.json").read_bytes() == source.read_bytes()
    assert str(producer) not in json.dumps(manifest)
    assert pd.read_parquet(tmp_path / "candidate/table.parquet").row_id.nunique() == 2
    with pytest.raises(FileExistsError):
        b.import_dataset(producer, "tcrvdb", "v2", tmp_path / "candidate")


def test_import_pair_seals_two_independent_preprocessed_views(tmp_path, monkeypatch):
    producer = paired_producer(tmp_path, monkeypatch)
    destination = tmp_path / "paired"
    manifest = b.import_dataset_pair(producer, "tcrvdb", destination)

    verify_document(manifest)
    assert manifest["schema_version"] == 2
    assert manifest["dataset_id"] == "tcrvdb@1.0.0"
    assert set(manifest["views"]) == {"v1", "v2"}
    assert len(manifest["population_sha256"]) == 64
    assert "files" not in manifest["inventory"]
    for convention, view in manifest["views"].items():
        verify_document(view)
        assert view["sequence_convention"] == convention
        assert view["n_rows"] == 2
        assert view["table"]["path"] == f"{convention}/table.parquet"
        frame = pd.read_parquet(destination / view["table"]["path"])
        assert frame.sequence_convention.unique().tolist() == [convention]
    assert manifest["views"]["v1"]["sha256"] != manifest["views"]["v2"]["sha256"]


def test_import_pair_population_digest_ignores_source_ids_and_order(tmp_path, monkeypatch):
    producer = paired_producer(tmp_path, monkeypatch)
    definitions = b.inventory()
    rewrite_v2_population(
        producer,
        definitions,
        lambda raw: {"a": raw["b"], "b": raw["a"]},
    )
    manifest = b.import_dataset_pair(producer, "tcrvdb", tmp_path / "paired")
    assert len(manifest["population_sha256"]) == 64


def test_import_pair_rejects_a_changed_biological_population(tmp_path, monkeypatch):
    producer = paired_producer(tmp_path, monkeypatch)
    definitions = b.inventory()

    def change_clone(raw):
        raw["a"]["meta_data"]["clone"] = "different-clone"
        return raw

    rewrite_v2_population(producer, definitions, change_clone)
    destination = tmp_path / "paired"
    with pytest.raises(ValidationError, match="same population"):
        b.import_dataset_pair(producer, "tcrvdb", destination)
    assert not destination.exists()


def test_combine_views_preserves_frozen_artifacts_without_preprocessing(tmp_path, monkeypatch):
    producer = paired_producer(tmp_path, monkeypatch)
    legacy = {}
    for convention in ("v1", "v2"):
        root = tmp_path / f"legacy-{convention}"
        legacy[convention] = b.import_dataset(producer, "tcrvdb", convention, root)
    destination = tmp_path / "combined"
    manifest = b.combine_dataset_views(
        tmp_path / "legacy-v1",
        tmp_path / "legacy-v2",
        destination,
    )

    verify_document(manifest)
    assert manifest["dataset_id"] == "tcrvdb@1.0.0"
    assert "files" not in manifest["inventory"]
    for convention in ("v1", "v2"):
        view = verify_document(manifest["views"][convention])
        for artifact in ("table", "example", "exclusions"):
            assert view[artifact]["sha256"] == legacy[convention][artifact]["sha256"]
        assert [item["sha256"] for item in view["raw"]] == [
            item["sha256"] for item in legacy[convention]["raw"]
        ]
        original = pd.read_parquet(tmp_path / f"legacy-{convention}/table.parquet")
        combined = pd.read_parquet(destination / view["table"]["path"])
        assert combined.row_id.tolist() == original.row_id.tolist()


def test_combine_views_accepts_reordered_and_rekeyed_population(tmp_path, monkeypatch):
    producer = paired_producer(tmp_path, monkeypatch)
    definitions = b.inventory()
    rewrite_v2_population(
        producer,
        definitions,
        lambda raw: {"a": raw["b"], "b": raw["a"]},
    )
    for convention in ("v1", "v2"):
        b.import_dataset(
            producer,
            "tcrvdb",
            convention,
            tmp_path / f"legacy-{convention}",
        )
    manifest = b.combine_dataset_views(
        tmp_path / "legacy-v1",
        tmp_path / "legacy-v2",
        tmp_path / "combined",
    )
    assert len(manifest["population_sha256"]) == 64


def test_combine_views_rejects_a_changed_biological_population(tmp_path, monkeypatch):
    producer = paired_producer(tmp_path, monkeypatch)
    definitions = b.inventory()

    def change_clone(raw):
        raw["a"]["meta_data"]["clone"] = "different-clone"
        return raw

    rewrite_v2_population(producer, definitions, change_clone)
    for convention in ("v1", "v2"):
        b.import_dataset(
            producer,
            "tcrvdb",
            convention,
            tmp_path / f"legacy-{convention}",
        )
    destination = tmp_path / "combined"
    with pytest.raises(ValidationError, match="same population"):
        b.combine_dataset_views(
            tmp_path / "legacy-v1",
            tmp_path / "legacy-v2",
            destination,
        )
    assert not destination.exists()


def test_combine_views_rejects_changed_legacy_artifact_before_output(tmp_path, monkeypatch):
    producer = paired_producer(tmp_path, monkeypatch)
    for convention in ("v1", "v2"):
        b.import_dataset(
            producer,
            "tcrvdb",
            convention,
            tmp_path / f"legacy-{convention}",
        )
    with (tmp_path / "legacy-v2/table.parquet").open("ab") as stream:
        stream.write(b"changed")
    destination = tmp_path / "combined"
    with pytest.raises(IntegrityError, match="artifact changed"):
        b.combine_dataset_views(
            tmp_path / "legacy-v1",
            tmp_path / "legacy-v2",
            destination,
        )
    assert not destination.exists()


def test_static_inventory_parser_rejects_execution(tmp_path):
    (tmp_path / "eval/scripts").mkdir(parents=True)
    (tmp_path / "eval/scripts/benchmark_registry.py").write_text("BENCHMARKS = dict(**other)")
    with pytest.raises(ValidationError):
        b._producer_inventory(tmp_path)
