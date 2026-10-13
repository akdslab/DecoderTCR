from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from DecoderTCR.core import IntegrityError, ValidationError
from DecoderTCR.reconstruct import (
    _resources,
    components,
    normalize_gene,
    stitch_tcrs,
    tcr,
)
from DecoderTCR.reconstruct.hla import list_alleles, lookup_hla, normalize_allele_key

GENE_ROW = {
    "name": "V01",
    "trav": "TRAV12-1",
    "traj": "TRAJ41",
    "cdr3a": "CVVNPSNSNSGYALNF",
    "trbv": "TRBV19",
    "trbj": "TRBJ2-7",
    "cdr3b": "CASSIRASHEQYF",
}


def _fake_thimble(command, **kwargs):
    input_path = Path(command[command.index("-in") + 1])
    output_path = Path(command[command.index("-o") + 1])
    rows = list(csv.DictReader(input_path.open(), delimiter="\t"))
    fields = [
        "TCR_name",
        "TRA_aa",
        "TRB_aa",
        "TRAV",
        "TRAJ",
        "TRBV",
        "TRBJ",
        "TRAC",
        "TRBC",
        "TRA_leader",
        "TRB_leader",
        "Warnings/Errors",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "TCR_name": row["TCR_name"],
                    "TRA_aa": "LEADER" + row["TRA_CDR3"] + "CONSTANT",
                    "TRB_aa": "LEADER" + row["TRB_CDR3"] + "CONSTANT",
                    "TRAV": row["TRAV"] + "*01",
                    "TRAJ": row["TRAJ"] + "*01",
                    "TRBV": row["TRBV"] + "*01",
                    "TRBJ": row["TRBJ"] + "*01",
                    "TRAC": "TRAC*01",
                    "TRBC": "TRBC2*01",
                    "TRA_leader": row["TRAV"] + "*01",
                    "TRB_leader": row["TRBV"] + "*01",
                    "Warnings/Errors": "allele fallback",
                }
            )
    return subprocess.CompletedProcess(command, 0, "", "")


def test_normalize_allele_key_public_forms():
    expected = "B2705__B2M"
    for value in ("HLA-B*27:05", "B*27:05", "B27:05", "B2705", "B2705__B2M"):
        assert normalize_allele_key(value) == expected


def test_v1_explicit_class_ii_pairs_are_strict_and_order_independent():
    expected = "DQA10301__DQB10201"
    assert normalize_allele_key("HLA-DQA1*03:01/HLA-DQB1*02:01") == expected
    assert normalize_allele_key("DQB1*02:01/DQA1*03:01") == expected
    hla_a, hla_b = lookup_hla("DQA1*03:01/DQB1*02:01", "v1")
    assert hla_a and hla_b
    assert lookup_hla(expected, "v1") == (hla_a, hla_b)
    with pytest.raises(ValidationError, match="explicit alpha and beta"):
        normalize_allele_key("DRB1*04:01")
    with pytest.raises(ValidationError, match="Unsupported class II"):
        normalize_allele_key("DQA1*03:01/DRB1*04:01")


def test_v2_human_class_ii_pairs_are_explicit_and_order_independent():
    expected = lookup_hla("DRA*01:01/DRB1*04:01", "v2")
    assert expected == lookup_hla("DRB1*04:01/DRA*01:01", "v2")
    assert all(expected)
    with pytest.raises(ValidationError, match="not available"):
        lookup_hla("DRA*01:01/DRB1*01:02", "v2")


def test_species_specific_mhc_pairs_are_strict():
    mouse = lookup_hla("H2-AAb/H2-ABb", "v2", species="mouse")
    assert mouse == lookup_hla("H2-ABb/H2-AAb", "v2", species="mouse")
    assert all(mouse)
    assert normalize_allele_key("H2-Kb", species="mouse") == "H2KB__B2M"
    with pytest.raises(ValidationError, match="selected model does not support mouse"):
        lookup_hla("H2-Kb", "v1", species="mouse")
    with pytest.raises(ValidationError, match="not available"):
        lookup_hla("H2-AAb/H2-ABk", "v2", species="mouse")
    with pytest.raises(ValidationError, match="accepts one allele or one explicit pair"):
        normalize_allele_key("H2-AAb/H2-ABb/H2-EBk", species="mouse")
    with pytest.raises(ValidationError, match="Mouse B2M"):
        normalize_allele_key("A0201__B2M_mouse", species="human")
    with pytest.raises(ValidationError, match="Human B2M"):
        normalize_allele_key("H2-Kb__B2M_human", species="mouse")
    with pytest.raises(ValidationError, match="sequence_convention"):
        lookup_hla("A0201", "v3")
    with pytest.raises(ValidationError, match="sequence_convention"):
        lookup_hla("A0201", "V2")


def test_public_allele_normalizer_propagates_species():
    import DecoderTCR as dt

    assert dt.normalize_allele_key("H2-Kb", species="mouse") == "H2KB__B2M"


def test_every_listed_v1_key_round_trips():
    for key in list_alleles("v1"):
        hla_a, hla_b = lookup_hla(key, "v1")
        assert hla_a and hla_b


def test_v2_inventory_preserves_human_class_i_and_adds_bounded_pairs():
    source = json.loads(
        (Path(tcr.__file__).parent / "data/hla_v2.json").read_text(encoding="utf-8")
    )
    assert len(source) == 124
    human = list_alleles("v2", species="human")
    supported_source = {key: source[key] for key in source.keys() & set(human)}
    assert len(supported_source) == 123
    for key, record in supported_source.items():
        assert lookup_hla(key, "v2") == (record["HLA_a"], record["HLA_b"])
    mouse = list_alleles("v2", species="mouse")
    assert len(human) == 183
    assert len(mouse) == 12
    human_class_ii = [key for key in human if not key.endswith("__B2M")]
    assert len(human_class_ii) == 60
    assert sum(key.endswith("__B2M") for key in mouse) == 5
    assert sum(not key.endswith("__B2M") for key in mouse) == 7
    for key in human_class_ii:
        assert all(lookup_hla(key, "v2"))
    for key in mouse:
        assert all(lookup_hla(key, "v2", species="mouse"))


def test_mouse_leader_stripping_fails_closed_for_unverified_or_wrong_prefix():
    leaders = {"TRAV14-1*01": "MDKILTASFLLLGLHLAGVNGQQKEKHD"}
    assert (
        components._strip_mouse_leader(
            "MDKILTASFLLLGLHLAGVNGQQKEKHDCHAIN", "TRAV14-1*01(L)", leaders
        )
        == "CHAIN"
    )
    with pytest.raises(ValidationError, match="No verified mouse leader peptide"):
        components._strip_mouse_leader("MSEQUENCE", "TRAV3N-3*01", leaders)
    with pytest.raises(ValidationError, match="not a reconstructed-chain prefix"):
        components._strip_mouse_leader("MSEQUENCE", "TRAV14-1*01", leaders)


def test_public_gene_normalizer_fallback_categories():
    assert normalize_gene("TRBV07-09*01") == "TRBV7-9*01"
    assert normalize_gene("TCRAV08-03") == "TRAV8-3"
    assert normalize_gene("TCRAV29-05") == "TRAV29/DV5"
    assert normalize_gene("TCRBV12-X").startswith("TRBV12-")
    assert normalize_gene("TRBV7-999").startswith("TRBV7-")


def test_stitch_tcrs_uses_current_python_worker_and_bundled_data(monkeypatch):
    observed = {}

    def fake(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return _fake_thimble(command, **kwargs)

    monkeypatch.setattr(tcr, "_require_stitchr", lambda: None)
    monkeypatch.setattr(tcr.subprocess, "run", fake)
    result = stitch_tcrs([GENE_ROW])
    command = observed["command"]
    assert command[:3] == [
        sys.executable,
        "-m",
        "DecoderTCR.reconstruct._thimble_worker",
    ]
    assert command[-3:] == ["-s", "HUMAN", "-sc"]
    assert command[0] != "thimble"
    assert observed["kwargs"] == {"capture_output": True, "text": True, "check": False}
    assert result[0]["name"] == "V01"
    assert result[0]["normalized_TRBV"] == "TRBV19"
    assert result[0]["TRBV"] == "TRBV19*01"
    assert result[0]["TRB_leader"] == "TRBV19*01"
    assert result[0]["TRBC"] == "TRBC2*01"
    assert result[0]["warnings"] == "allele fallback"


def test_stitch_tcrs_selects_mouse_resources(monkeypatch):
    observed = {}

    def fake(command, **kwargs):
        observed["command"] = command
        return _fake_thimble(command, **kwargs)

    monkeypatch.setattr(tcr, "_require_stitchr", lambda: None)
    monkeypatch.setattr(tcr.subprocess, "run", fake)
    stitch_tcrs([GENE_ROW], species="mouse")
    assert observed["command"][-3:] == ["-s", "MOUSE", "-sc"]


@pytest.mark.parametrize("field,value", [("cdr3a", "A\tB"), ("trbv", "TRBV7-9,TRBV19")])
def test_stitch_tcrs_rejects_tsv_controls_and_multi_output(field, value, monkeypatch):
    monkeypatch.setattr(tcr, "_require_stitchr", lambda: None)
    row = dict(GENE_ROW)
    row[field] = value
    with pytest.raises(ValidationError):
        stitch_tcrs([row])


def test_unexpected_or_duplicate_worker_ids_fail_whole_call(monkeypatch):
    def bad_worker(command, **kwargs):
        result = _fake_thimble(command, **kwargs)
        output_path = Path(command[command.index("-o") + 1])
        rows = list(csv.reader(output_path.open(), delimiter="\t"))
        rows[1][0] = "user-facing-name"
        with output_path.open("w", newline="") as handle:
            csv.writer(handle, delimiter="\t").writerows(rows)
        return result

    monkeypatch.setattr(tcr, "_require_stitchr", lambda: None)
    monkeypatch.setattr(tcr.subprocess, "run", bad_worker)
    with pytest.raises(RuntimeError, match="unexpected row identifier"):
        stitch_tcrs([GENE_ROW])


def test_nonzero_worker_exit_is_a_whole_call_failure(monkeypatch):
    monkeypatch.setattr(tcr, "_require_stitchr", lambda: None)
    monkeypatch.setattr(
        tcr.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, "", "failure"),
    )
    with pytest.raises(RuntimeError, match="exit 2"):
        stitch_tcrs([GENE_ROW])


def test_worker_failure_remains_primary_when_temporary_cleanup_fails(monkeypatch, tmp_path):
    class CleanupFailure:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *_):
            raise OSError(39, "Directory not empty")

    monkeypatch.setattr(tcr, "_require_stitchr", lambda: None)
    monkeypatch.setattr(tcr.tempfile, "TemporaryDirectory", lambda **_: CleanupFailure())
    monkeypatch.setattr(
        tcr.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, "", "failure"),
    )

    with pytest.raises(RuntimeError, match="exit 2") as captured:
        stitch_tcrs([GENE_ROW])
    assert isinstance(captured.value.__cause__, OSError)


def test_successful_worker_does_not_hide_temporary_cleanup_failure(monkeypatch, tmp_path):
    class CleanupFailure:
        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *_):
            raise OSError(39, "Directory not empty")

    monkeypatch.setattr(tcr, "_require_stitchr", lambda: None)
    monkeypatch.setattr(tcr.tempfile, "TemporaryDirectory", lambda **_: CleanupFailure())
    monkeypatch.setattr(tcr.subprocess, "run", _fake_thimble)

    with pytest.raises(OSError, match="Directory not empty"):
        stitch_tcrs([GENE_ROW])


def test_hash_corruption_is_detected(monkeypatch, tmp_path):
    relative = "hla_v1.json"
    (tmp_path / relative).write_text("{}")
    monkeypatch.setattr(_resources, "_DATA_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="hash verification"):
        _resources.verify_resources([relative])


def test_gene_resource_verification_is_not_skipped_after_parsed_cache(monkeypatch):
    normalize_gene("TRBV19")

    def changed(_):
        raise IntegrityError("changed after cache")

    monkeypatch.setattr(tcr, "resource_identity", changed)
    with pytest.raises(IntegrityError, match="changed after cache"):
        normalize_gene("TRBV19")


def test_hla_references_preserve_class_i_b2m_and_strict_v1_pairs():
    expected_b2m = "e92bf108e2abfa0fc43ccdf0684d41b023e4368eb04b185c4578cda6a492fb46"
    for convention in ("v1", "v2"):
        alleles = list_alleles(convention)
        assert alleles
        hla_a, hla_b = lookup_hla("HLA-A*02:01", convention)
        assert hla_a
        assert hashlib.sha256(hla_b.encode()).hexdigest() == expected_b2m
    assert sum(key.endswith("__B2M") for key in list_alleles("v2")) == 123
    assert any(not key.endswith("__B2M") for key in list_alleles("v1"))
    assert "DQA10101__DRB10701" not in list_alleles("v1")


def test_manifest_has_no_development_paths():
    root = Path(tcr.__file__).parent
    for path in (root / "data").rglob("*"):
        if path.is_file():
            text = path.read_text(errors="ignore")
            assert "/mnt/" not in text
            assert "/home/" not in text


@pytest.mark.parametrize(
    "convention,expected",
    [
        (
            "v1",
            {
                "HLA_a": "f65b9ebcf3677f304926cac3e13331022ebdcf996ce416117bf390c60ecc55d0",
                "HLA_b": "e92bf108e2abfa0fc43ccdf0684d41b023e4368eb04b185c4578cda6a492fb46",
                "TCR_a": "b8627160a58f3da3c11000f62c0ba59e0b23c6ce9dc29d496a1faf4c38eff002",
                "TCR_b": "f87433968d68af7c37f7823e8b4df090f441eb4916e13d316d0896ae167ac632",
            },
        ),
        (
            "v2",
            {
                "HLA_a": "34d3c396ae02a2d90b94885b132aa28719e385d93b9a5a3441aa726f1a132112",
                "HLA_b": "e92bf108e2abfa0fc43ccdf0684d41b023e4368eb04b185c4578cda6a492fb46",
                "TCR_a": "dfc11af2868cd5447addbaa31ed716b8251d11d7a653eff8958b88acbe92679b",
                "TCR_b": "a4b362b1d3951a939a81478495bd17cd2bec7f82c3f1eee8c4371426e432e808",
            },
        ),
    ],
)
def test_real_thimble_v1_v2_golden_sequences(convention, expected):
    from DecoderTCR.reconstruct.components import prepare_gene_inputs

    frame, entries = prepare_gene_inputs(
        [{**GENE_ROW, "hla": "HLA-A*02:01", "peptide": "GILGFVFTL", "label": 1}],
        convention,
    )
    assert frame.iloc[0]["reconstruction_ok"]
    sequences = entries[0]["sequences"]
    observed = {
        key: hashlib.sha256(value.encode()).hexdigest()
        for key, value in sequences.items()
        if key != "Peptide"
    }
    assert observed == expected
    assert entries[0]["annotation_contract"] == "inclusive-end"
    assert entries[0]["reconstruction"] == frame.attrs["reconstruction"]
    assert json.loads(json.dumps(entries[0]["reconstruction"])) == entries[0]["reconstruction"]


def test_real_mouse_thimble_v2_leaders_constants_and_coordinates():
    from DecoderTCR.reconstruct.components import prepare_gene_inputs

    row = {
        "name": "mouse-golden",
        "trav": "TRAV14-1",
        "traj": "TRAJ33",
        "cdr3a": "CAASDNYQLIW",
        "trbv": "TRBV12-1",
        "trbj": "TRBJ2-7",
        "cdr3b": "CASSRANYEQYF",
        "hla": "H2-Kb",
        "peptide": "ANYKFTLV",
    }
    stitched = stitch_tcrs([{key: row[key] for key in GENE_ROW if key in row}], species="mouse")
    assert stitched[0]["TRA_leader"] == "TRAV14-1*01(L)"
    assert stitched[0]["TRB_leader"] == "TRBV12-1*01(L)"
    assert stitched[0]["TCR_a"].startswith("MDKILTASFLLLGLHLAGVNGQQKEKHD")
    assert stitched[0]["TCR_b"].startswith("MSNTVLADSAWGITLLSWVTVFLLGTSSA")
    assert stitched[0]["TRAC"] == "TRAC*01"
    assert stitched[0]["TRBC"] == "TRBC2*01"

    class_ii_row = {**row, "name": "mouse-class-ii", "hla": "H2-AAb/H2-ABb"}
    frame, entries = prepare_gene_inputs([row, class_ii_row], "v2", species="mouse")
    assert frame.iloc[0]["reconstruction_ok"]
    entry = entries[0]
    assert not entry["sequences"]["TCR_a"].startswith("MDKILTASFLLLGLHLAGVNGQQKEKHD")
    assert not entry["sequences"]["TCR_b"].startswith("MSNTVLADSAWGITLLSWVTVFLLGTSSA")
    alpha_start = entry["sequences"]["TCR_a"].index(row["cdr3a"])
    beta_start = entry["sequences"]["TCR_b"].index(row["cdr3b"])
    assert entry["pocket_idx"]["CDRa"] == [[alpha_start, alpha_start + len(row["cdr3a"]) - 1]]
    assert entry["pocket_idx"]["CDRb"] == [[beta_start, beta_start + len(row["cdr3b"]) - 1]]
    assert frame.iloc[0]["TRAC"] == "TRAC*01"
    assert frame.iloc[0]["TRBC"] == "TRBC2*01"
    assert frame.iloc[1]["reconstruction_ok"]
    assert entries[1]["sequences"]["HLA_a"]
    assert entries[1]["sequences"]["HLA_b"]
    assert (
        entries[1]["sequences"]["HLA_a"],
        entries[1]["sequences"]["HLA_b"],
    ) == lookup_hla("H2-AAb/H2-ABb", "v2", species="mouse")
