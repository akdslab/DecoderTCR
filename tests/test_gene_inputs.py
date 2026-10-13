from __future__ import annotations

import pandas as pd
import pytest

from DecoderTCR.core import ValidationError
from DecoderTCR.reconstruct import components

BASE = {
    "name": "sample",
    "trav": "TRAV12-1",
    "traj": "TRAJ41",
    "cdr3a": "CVVNPSNSNSGYALNF",
    "trbv": "TRBV19",
    "trbj": "TRBJ2-7",
    "cdr3b": "CASSIRASHEQYF",
    "hla": "HLA-A*02:01",
    "peptide": "GILGFVFTL",
    "label": "positive",
}


def _stitched(rows, *, species="human"):
    output = []
    for row in rows:
        output.append(
            {
                "TCR_a": "AAA" + row["cdr3a"] + "ZZZ",
                "TCR_b": "BBB" + row["cdr3b"] + "YYY",
                "ok": True,
                "reason": "",
                "warnings": "",
                "normalized_TRAV": row["trav"],
                "normalized_TRAJ": row["traj"],
                "normalized_TRBV": row["trbv"],
                "normalized_TRBJ": row["trbj"],
                "TRAV": row["trav"] + "*01",
                "TRAJ": row["traj"] + "*01",
                "TRBV": row["trbv"] + "*01",
                "TRBJ": row["trbj"] + "*01",
                "TRAC": "TRAC*01",
                "TRBC": "TRBC2*01",
                "TRA_leader": row["trav"] + "*01",
                "TRB_leader": row["trbv"] + "*01",
            }
        )
    return output


@pytest.fixture
def isolated_reconstruction(monkeypatch):
    monkeypatch.setattr(components, "stitch_tcrs", _stitched)
    monkeypatch.setattr(
        components,
        "lookup_hla",
        lambda allele, convention, *, species="human": ("HLAA", "B2M"),
    )


def test_aliases_extended_cdr3_priority_and_original_values(isolated_reconstruction):
    row = {
        "tcra_v": "TRAV12-1",
        "tcra_j": "TRAJ41",
        "cdr3a": "IGNORED",
        "cdr3a_extended": "CVVNPSNSNSGYALNF",
        "tcrb_v": "TRBV19",
        "tcrb_j": "TRBJ2-7",
        "cdr3b": "IGNORED",
        "cdr3b_extended": "CASSIRASHEQYF",
        "allele": "HLA-A*02:01",
        "epitope": "GILGFVFTL",
        "binder": "keep-me",
    }
    frame, entries = components.prepare_gene_inputs(row, "v1")
    assert frame.loc[0, "cdr3a"] == "IGNORED"
    assert frame.loc[0, "binder"] == "keep-me"
    assert entries[0]["sequences"]["TCR_a"] == "AAACVVNPSNSNSGYALNFZZZ"
    assert entries[0]["meta_data"]["label"] == "keep-me"


def test_ambiguous_aliases_and_case_duplicate_headers_fail(isolated_reconstruction):
    ambiguous = pd.DataFrame([{**BASE, "allele": BASE["hla"]}])
    with pytest.raises(ValidationError, match="Ambiguous aliases for hla"):
        components.prepare_gene_inputs(ambiguous, "v1")
    duplicate = pd.DataFrame([[*BASE.values(), BASE["trav"]]], columns=[*BASE, "TRAV"])
    with pytest.raises(ValidationError, match="Duplicate case-insensitive"):
        components.prepare_gene_inputs(duplicate, "v1")


def test_missing_column_is_global_error_but_blank_row_is_retained(isolated_reconstruction):
    missing = dict(BASE)
    del missing["trbj"]
    with pytest.raises(ValidationError, match="Missing required gene input columns"):
        components.prepare_gene_inputs(missing, "v1")

    blank = pd.DataFrame([BASE, {**BASE, "name": "blank", "trbj": ""}])
    frame, entries = components.prepare_gene_inputs(blank, "v1")
    assert len(frame) == len(entries) == 2
    assert entries[0] is not None
    assert entries[1] is None
    assert not frame.iloc[1]["reconstruction_ok"]
    assert "blank required TCR fields" in frame.iloc[1]["reconstruction_reason"]


def test_duplicate_user_index_names_order_and_labels_are_preserved(isolated_reconstruction):
    source = pd.DataFrame([BASE, {**BASE, "name": "second", "label": 7}], index=[4, 4])
    frame, entries = components.prepare_gene_inputs(source, "v1")
    assert frame.index.tolist() == [4, 4]
    assert frame["name"].tolist() == ["sample", "second"]
    assert frame["label"].tolist() == ["positive", 7]
    assert [entry["meta_data"]["label"] for entry in entries] == ["positive", 7]


def test_row_species_mismatch_is_retained_and_mixed_rows_do_not_override_call(
    isolated_reconstruction,
):
    source = pd.DataFrame(
        [
            {**BASE, "name": "human", "species": "human"},
            {**BASE, "name": "mouse", "species": "mouse"},
            {**BASE, "name": "unsupported", "species": "dog"},
        ],
        index=[5, 5, 9],
    )
    frame, entries = components.prepare_gene_inputs(source, "v1", species="human")
    assert frame.index.tolist() == [5, 5, 9]
    assert frame["name"].tolist() == ["human", "mouse", "unsupported"]
    assert entries[0] is not None
    assert entries[1:] == [None, None]
    assert "does not match" in frame.iloc[1]["reconstruction_reason"]
    assert "unsupported row species" in frame.iloc[2]["reconstruction_reason"]


def test_species_and_organism_conflict_and_invalid_global_options_fail(
    isolated_reconstruction,
):
    with pytest.raises(ValidationError, match="Ambiguous aliases for species"):
        components.prepare_gene_inputs({**BASE, "species": "human", "organism": "mouse"}, "v1")
    with pytest.raises(ValidationError, match="species must"):
        components.prepare_gene_inputs(BASE, "v1", species="dog")
    with pytest.raises(ValidationError, match="selected model does not support mouse"):
        components.prepare_gene_inputs(BASE, "v1", species="mouse")


def test_generated_columns_do_not_overwrite_original_fields(isolated_reconstruction):
    source = pd.DataFrame([{**BASE, "TCR_a": "user supplied"}])
    frame, entries = components.prepare_gene_inputs(source, "v1")
    assert frame.loc[0, "TCR_a"] == "user supplied"
    assert frame.loc[0, "reconstruction_TCR_a"].startswith("AAA")
    assert frame.attrs["reconstruction_columns"]["TCR_a"] == "reconstruction_TCR_a"
    assert entries[0]["sequences"]["TCR_a"].startswith("AAA")


def test_empty_typed_frame_and_all_failed_frame_have_stable_outputs(isolated_reconstruction):
    empty = pd.DataFrame(columns=list(BASE))
    empty_result, empty_entries = components.prepare_gene_inputs(empty, "v1")
    assert empty_result.empty
    assert empty_entries == []
    assert "reconstruction_ok" in empty_result

    failed = pd.DataFrame([{**BASE, "peptide": ""}, {**BASE, "hla": ""}])
    frame, entries = components.prepare_gene_inputs(failed, "v1")
    assert entries == [None, None]
    assert frame["reconstruction_ok"].tolist() == [False, False]


def test_entry_ranges_use_the_public_inclusive_end_contract(isolated_reconstruction):
    frame, entries = components.prepare_gene_inputs(BASE, "v1")
    entry = entries[0]
    assert entry["annotation_contract"] == "inclusive-end"
    assert entry["pocket_idx"]["CDRa"] == [[3, 3 + len(BASE["cdr3a"]) - 1]]
    assert entry["pocket_idx"]["CDRb"] == [[3, 3 + len(BASE["cdr3b"]) - 1]]
    assert frame.attrs["reconstruction"] == entry["reconstruction"]


def test_csv_and_parquet_paths_match_dataframe_input(isolated_reconstruction, tmp_path):
    source = pd.DataFrame([BASE])
    csv_path = tmp_path / "genes.csv"
    parquet_path = tmp_path / "genes.parquet"
    source.to_csv(csv_path, index=False)
    source.to_parquet(parquet_path, index=False)
    csv_frame, _ = components.prepare_gene_inputs(csv_path, "v1")
    parquet_frame, _ = components.prepare_gene_inputs(parquet_path, "v1")
    assert csv_frame[list(BASE)].equals(parquet_frame[list(BASE)])
