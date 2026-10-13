from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from DecoderTCR.constants import ALPHABET, CLS_IDX, EOS_IDX, MASK_IDX
from DecoderTCR.core import ValidationError
from DecoderTCR.runtime import (
    LoadedModel,
    build_model,
    model_spec,
    normalize_state_dict,
    restore_state,
)
from DecoderTCR.scoring import compute_pll, prepare_inputs, run_pll_benchmark
from DecoderTCR.tokenizer import TCRpMHCTokenizer


@pytest.fixture(autouse=True)
def small_cpu_work():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def entry():
    return {
        "sequences": {"HLA_a": "AC", "HLA_b": "D", "Peptide": "EF", "TCR_a": "GH", "TCR_b": "IK"},
        "pocket_idx": {},
    }


def _small(backbone, device="cpu"):
    config = (
        {"d_model": 32, "n_heads": 4, "n_layers": 2}
        if backbone == "esmc"
        else {"embed_dim": 32, "attention_heads": 4, "num_layers": 2}
    )
    return build_model(backbone, "direct-test-dimensions", device, architecture_config=config)


def test_tokenizer_masks_complete_peptide_with_cls_offset(entry):
    tok = TCRpMHCTokenizer(entry)
    assert tok.full_seq == "ACDEFGHIK"
    assert tok.original_ids.tolist() == [CLS_IDX] + ALPHABET.encode(tok.full_seq) + [EOS_IDX]
    assert tok.peptide_indices.tolist() == [3, 4]
    assert tok.token_indices.tolist() == [4, 5]
    assert (tok.input_ids == MASK_IDX).nonzero().flatten().tolist() == [4, 5]
    assert tok.original_ids[4:6].tolist() == ALPHABET.encode("EF")


def test_normalization_includes_all_64_channels():
    logits = torch.zeros(2, 64)
    target = torch.tensor([ALPHABET.get_idx("A"), ALPHABET.get_idx("C")])
    assert compute_pll(target, logits) == pytest.approx(-np.log(64), abs=1e-6)
    logits[:, 63] = 8
    assert compute_pll(target, logits) == pytest.approx(-np.log(63 + np.exp(8)), abs=1e-6)
    assert compute_pll(target, logits) < compute_pll(target, logits[:, :33])


@pytest.mark.parametrize("backbone", ["esm2", "esmc"])
def test_small_models_roundtrip_logits_and_pll(backbone, entry, tmp_path):
    torch.manual_seed(71)
    source = _small(backbone)
    tok = TCRpMHCTokenizer(entry)
    with torch.inference_mode():
        expected = source(tok.input_ids.unsqueeze(0))["logits"]
    destination = tmp_path / "weights.safetensors"
    state = {key: value.detach().clone().contiguous() for key, value in source.state_dict().items()}
    save_file(state, str(destination))
    from safetensors.torch import load_file

    with torch.device("meta"):
        restored = _small(backbone, device="meta")
    restored = restore_state(restored, load_file(str(destination)))
    with torch.inference_mode():
        actual = restored(tok.input_ids.unsqueeze(0))["logits"]
    assert actual.shape == (1, 11, 64 if backbone == "esmc" else 33)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    source_scores = run_pll_benchmark(source, 2, [entry])
    restored_scores = run_pll_benchmark(restored, 2, [entry])
    np.testing.assert_array_equal(source_scores, restored_scores)
    assert all(not tensor.is_meta for tensor in restored.buffers())


def test_context_cache_preserves_distinct_targets(entry):
    class Predictable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, tokens):
            self.calls += 1
            assert (tokens == MASK_IDX).sum() == 2
            values = torch.arange(64, dtype=torch.float32)
            return {"logits": values.expand(*tokens.shape, 64)}

    other = {"sequences": {**entry["sequences"], "Peptide": "YW"}}
    model = Predictable()
    cached = run_pll_benchmark(model, 2, [entry, other])
    assert model.calls == 1
    assert cached[0] != cached[1]
    fresh = run_pll_benchmark(model, 2, [entry, other], cache_size=0)
    np.testing.assert_array_equal(cached, fresh)


def test_model_handle_preserves_rows_and_records_release_identity(entry):
    model = LoadedModel(
        _small("esmc"),
        "fixture@1.0.0:a",
        "esmc",
        "fixture",
        "v1",
        "b" * 64,
        {"sha256": "c" * 64},
        "d" * 64,
        torch.device("cpu"),
        2,
    )
    frame = pd.DataFrame([{**entry["sequences"], "row_id": "one", "label": 1}])
    output = model._score_prepared(frame)
    pd.testing.assert_frame_equal(output[frame.columns], frame)
    assert output["model_id"].tolist() == ["fixture@1.0.0:a"]
    assert output["artifact_sha256"].tolist() == ["b" * 64]
    assert output.attrs["reproduction"]["normalization"] == "full-vocabulary"
    assert "global_step" not in output.attrs["reproduction"]


@pytest.mark.parametrize("sequence_convention", ["v1", "v2"])
def test_model_handle_defaults_to_genes_and_aligns_failures(monkeypatch, sequence_convention):
    from DecoderTCR import runtime
    from DecoderTCR.reconstruct import components

    model = LoadedModel(
        _small("esmc"),
        "fixture@1.0.0:a",
        "esmc",
        "fixture",
        sequence_convention,
        "b" * 64,
        {"sha256": "c" * 64},
        "d" * 64,
        torch.device("cpu"),
        2,
    )
    valid = {
        "sequences": {
            "HLA_a": "AC",
            "HLA_b": "D",
            "Peptide": "EF",
            "TCR_a": "GH",
            "TCR_b": "IK",
        },
        "pocket_idx": {},
    }
    frame = pd.DataFrame(
        [
            {
                "name": "same",
                "reconstruction_ok": True,
                "reconstruction_reason": "",
            },
            {
                "name": "same",
                "reconstruction_ok": False,
                "reconstruction_reason": "unknown allele",
            },
        ],
        index=[7, 7],
    )
    frame.attrs["reconstruction"] = {"sequence_convention": sequence_convention}
    prepare = Mock(return_value=(frame, [valid, None]))
    monkeypatch.setattr(components, "prepare_gene_inputs", prepare)
    runner = Mock(return_value=np.asarray([-2.5], dtype=np.float64))
    monkeypatch.setattr(runtime, "run_pll_benchmark", runner)

    rows = [{"row": 1}, {"row": 2}]
    output = model.score(rows)
    repeated = model.score(rows)

    assert [item.args for item in prepare.call_args_list] == [
        (rows, sequence_convention),
        (rows, sequence_convention),
    ]
    assert runner.call_count == 2
    pd.testing.assert_frame_equal(output, repeated)
    assert output.index.tolist() == [7, 7]
    assert output.name.tolist() == ["same", "same"]
    assert output.reconstruction_ok.tolist() == [True, False]
    assert output.reconstruction_reason.tolist() == ["", "unknown allele"]
    assert output.pll.iloc[0] == -2.5
    assert np.isnan(output.pll.iloc[1])
    assert output.attrs["reconstruction"] == {"sequence_convention": sequence_convention}


def test_model_handle_rejects_input_format_keyword(entry):
    model = LoadedModel(
        _small("esmc"),
        "fixture@1.0.0:a",
        "esmc",
        "fixture",
        "v1",
        "b" * 64,
        {"sha256": "c" * 64},
        "d" * 64,
        torch.device("cpu"),
        2,
    )
    with pytest.raises(TypeError, match="input_format"):
        model.score([entry], input_format="raw")


def test_prepared_inputs_preserve_missing_chains_and_enforce_convention(entry):
    row = {"hla": "", "b2m": None, "peptide": "ACD", "tcra": np.nan, "tcrb": pd.NA}
    frame, entries = prepare_inputs([{**row, "row_id": "r1"}], "v2")
    assert frame.row_id.tolist() == ["r1"]
    assert entries[0]["sequences"] == {
        "HLA_a": "",
        "HLA_b": "",
        "Peptide": "ACD",
        "TCR_a": "",
        "TCR_b": "",
    }
    with pytest.raises(ValidationError, match="convention"):
        prepare_inputs([{**row, "sequence_convention": "v1"}], "v2")
    with pytest.raises(ValidationError, match="one sequence column"):
        prepare_inputs([{**row, "hla_a": "AC"}], "v1")
    with pytest.raises(ValidationError, match="cannot be empty"):
        prepare_inputs([{**entry["sequences"], "Peptide": ""}], "v1")


def test_structured_scoring_preserves_and_validates_declarations(entry):
    structured = {
        **entry,
        "sequence_convention": "v2",
        "annotation_contract": "half-open",
        "meta_data": {
            "sequence_convention": "v2",
            "annotation_contract": "half-open",
        },
    }
    frame, entries = prepare_inputs(structured, "v2")
    assert frame.sequence_convention.tolist() == ["v2"]
    assert entries[0]["annotation_contract"] == "half-open"

    mismatched = {**structured, "sequence_convention": "v1", "meta_data": {}}
    with pytest.raises(ValidationError, match="does not match"):
        prepare_inputs(mismatched, "v2")

    metadata_mismatch = {
        **entry,
        "meta_data": {"sequence_convention": "v1"},
    }
    with pytest.raises(ValidationError, match="does not match"):
        prepare_inputs(metadata_mismatch, "v2")

    conflicting = {
        **structured,
        "sequence_convention": "v2",
        "meta_data": {"sequence_convention": "v1"},
    }
    with pytest.raises(ValidationError, match="Conflicting sequence_convention"):
        prepare_inputs(conflicting, "v2")

    conflicting_contract = {
        **structured,
        "annotation_contract": "half-open",
        "meta_data": {"annotation_contract": "inclusive-end"},
    }
    with pytest.raises(ValidationError, match="Conflicting annotation_contract"):
        prepare_inputs(conflicting_contract, "v2")

    invalid_contract = {**entry, "annotation_contract": "closed"}
    with pytest.raises(ValidationError, match="annotation_contract"):
        prepare_inputs(invalid_contract, "v2")

    _, default_entries = prepare_inputs(entry, "v2")
    assert default_entries[0].get("annotation_contract") is None


@pytest.mark.parametrize("mask_region", ["cdr", "cdr3", "bg"])
def test_inclusive_end_scoring_masks_are_rejected_for_prepared_and_genes(
    monkeypatch, entry, mask_region
):
    from DecoderTCR.reconstruct import components

    annotated = {
        **entry,
        "pocket_idx": {"CDRa": [[0, 1]], "CDRb": [[0, 1]]},
        "annotation_contract": "inclusive-end",
    }
    model = LoadedModel(
        _small("esmc"),
        "fixture@1.0.0:a",
        "esmc",
        "fixture",
        "v2",
        "b" * 64,
        {"sha256": "c" * 64},
        "d" * 64,
        torch.device("cpu"),
        2,
    )
    with pytest.raises(ValidationError, match="requires half-open"):
        model._score_prepared(annotated, mask_region=mask_region)

    frame = pd.DataFrame([{"reconstruction_ok": True}])
    prepare_genes = Mock(return_value=(frame, [annotated]))
    monkeypatch.setattr(components, "prepare_gene_inputs", prepare_genes)
    with pytest.raises(ValidationError, match="unsupported for gene-component"):
        model.score([{"trav": "TRAV1"}], mask_region=mask_region)
    prepare_genes.assert_not_called()


def test_inclusive_end_contract_preserves_peptide_scoring(entry):
    torch.manual_seed(89)
    model = _small("esmc")
    baseline = run_pll_benchmark(model, 2, [entry], mask_region="peptide")
    declared = run_pll_benchmark(
        model,
        2,
        [{**entry, "annotation_contract": "inclusive-end"}],
        mask_region="peptide",
    )
    np.testing.assert_array_equal(declared, baseline)


@pytest.mark.parametrize(
    ("column", "alias"),
    [
        ("HLA_a", "hla_a"),
        ("HLA_a", "hla"),
        ("HLA_a", "hla_chain"),
        ("HLA_a", "hla_seq"),
        ("HLA_b", "hla_b"),
        ("HLA_b", "b2m"),
        ("HLA_b", "beta2m"),
        ("Peptide", "peptide"),
        ("TCR_a", "tcr_a"),
        ("TCR_a", "tcra"),
        ("TCR_b", "tcr_b"),
        ("TCR_b", "tcrb"),
    ],
)
def test_prepared_inputs_keep_supported_aliases(entry, column, alias):
    row = dict(entry["sequences"])
    row[alias] = row.pop(column)
    _, entries = prepare_inputs(row, "v1")
    assert entries == [entry]


@pytest.mark.parametrize("column", ["HLA_a", "HLA_b", "Peptide", "TCR_a", "TCR_b"])
@pytest.mark.parametrize("input_kind", ["record", "dataframe", "structured"])
def test_prepared_inputs_reject_missing_columns(entry, column, input_kind):
    row = {key: value for key, value in entry["sequences"].items() if key != column}
    data = {
        "record": [row],
        "dataframe": pd.DataFrame([row]),
        "structured": [{"sequences": row, "meta_data": {column: "AC"}}],
    }[input_kind]
    with pytest.raises(ValidationError, match=f"Missing required sequence columns: {column}"):
        prepare_inputs(data, "v1")


def test_prepared_inputs_reject_typo_and_missing_chain_in_later_record(entry):
    typo = {**entry["sequences"], "TCR_beta": "IK"}
    del typo["TCR_b"]
    with pytest.raises(ValidationError, match="TCR_b"):
        prepare_inputs([typo], "v1")
    with pytest.raises(ValidationError, match="TCR_b"):
        prepare_inputs([entry["sequences"], typo], "v1")
    with pytest.raises(ValidationError, match="TCR_b"):
        prepare_inputs([entry, {"sequences": typo}], "v1")


def test_prepared_inputs_validate_empty_dataframe_schema(entry):
    frame = pd.DataFrame(columns=[*entry["sequences"], "sequence_convention"])
    output, entries = prepare_inputs(frame, "v1")
    pd.testing.assert_frame_equal(output, frame)
    assert entries == []
    with pytest.raises(ValidationError, match="Missing required sequence columns"):
        prepare_inputs(pd.DataFrame(columns=["Peptide"]), "v1")
    with pytest.raises(ValidationError, match="Missing required sequence columns"):
        prepare_inputs([], "v1")


def test_prepared_inputs_preserve_structured_records(entry):
    frame, entries = prepare_inputs([{**entry, "meta_data": {"row_id": "r1"}}], "v1")
    assert frame.row_id.tolist() == ["r1"]
    assert entries == [entry]
    with pytest.raises(ValidationError, match="mix flat and structured"):
        prepare_inputs([entry["sequences"], entry], "v1")
    with pytest.raises(ValidationError, match="must be mappings"):
        prepare_inputs([{"sequences": ["AC"]}], "v1")


def test_prepared_inputs_validate_csv_schema(entry, tmp_path):
    row = {**entry["sequences"], "TCR_beta": "IK"}
    del row["TCR_b"]
    csv = tmp_path / "pairs.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)
    with pytest.raises(ValidationError, match="TCR_b"):
        prepare_inputs(csv, "v1")


def test_model_handle_rejects_missing_columns_before_inference():
    class NoInference(torch.nn.Module):
        def eval(self):
            pytest.fail("Invalid input must fail before inference starts")

    model = LoadedModel(
        NoInference(),
        "fixture@1.0.0:a",
        "esmc",
        "fixture",
        "v1",
        "b" * 64,
        {"sha256": "c" * 64},
        "d" * 64,
        torch.device("cpu"),
        2,
    )
    with pytest.raises(ValidationError, match="Missing required sequence columns"):
        model._score_prepared([{"Peptide": "ACD", "TCR_beta": "IK"}])


def test_reject_injected_special_tokens_and_bad_regions(entry):
    invalid = {"sequences": {**entry["sequences"], "Peptide": "A<mask>"}}
    with pytest.raises(ValidationError, match="unsupported residue"):
        TCRpMHCTokenizer(invalid)
    invalid = {**entry, "pocket_idx": {"HLA_a": [0, 100]}}
    with pytest.raises(ValidationError, match="region annotation"):
        TCRpMHCTokenizer(invalid)


def test_state_normalization_preserves_all_tensors():
    tensor = torch.ones(2)
    state = {
        "state_dict": {
            "model.model.embed.weight": tensor,
            "model.model.sequence_head.0.weight": tensor,
        }
    }
    normalized = normalize_state_dict(state, "esmc")
    assert set(normalized) == {"embed.weight", "sequence_head.0.weight"}
    assert normalized["embed.weight"] is tensor
    with pytest.raises(ValidationError, match="named tensor"):
        normalize_state_dict({"weights": tensor, "extra": "value"}, "esm2")
    with pytest.raises(ValidationError, match="colliding"):
        normalize_state_dict(
            {"encoder.embed_tokens.weight": tensor, "embed_tokens.weight": tensor}, "esm2"
        )


def test_architecture_registry_does_not_include_fixture_models():
    assert model_spec("esmc", "DecoderTCRC_600M")["n_layers"] == 36
    with pytest.raises(ValidationError, match="Upgrade"):
        model_spec("esmc", "fixture")


def test_esmc_inference_ignores_external_flash_toggle(monkeypatch, entry):
    model = _small("esmc")
    tokens = TCRpMHCTokenizer(entry).input_ids.unsqueeze(0)
    with torch.inference_mode():
        expected = model(tokens)["logits"]
        monkeypatch.setenv("USE_FLASH_ATTN", "1")
        actual = model(tokens)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exact_load_rechecks_availability(monkeypatch, tmp_path):
    import DecoderTCR.registry as registry_module
    from DecoderTCR import runtime
    from DecoderTCR.core import UnavailableError

    source = _small("esmc")
    weights = tmp_path / "model.safetensors"
    save_file(
        {key: value.clone().contiguous() for key, value in source.state_dict().items()},
        str(weights),
    )

    class FakeRegistry:
        def __init__(self):
            self.blocked = False
            self.reads = 0

        def resolve(self, model_id):
            self.reads += 1
            if self.blocked:
                raise UnavailableError("blocked")
            return SimpleNamespace(
                bundle=tmp_path,
                manifest={"sha256": "a" * 64},
                catalog_sha256="b" * 64,
                member={
                    "backbone": "esmc",
                    "arch": "DecoderTCRC_300M",
                    "sequence_convention": "v1",
                    "weights": {"sha256": "c" * 64},
                },
            )

        def verify_artifact(self, bundle, descriptor):
            return weights

    registry = FakeRegistry()
    monkeypatch.setattr(registry_module, "Registry", FakeRegistry)
    monkeypatch.setattr(
        runtime, "build_model", lambda backbone, arch, device: _small(backbone, device)
    )
    handle = runtime.load("fixture@1.0.0:a", "cpu", registry=registry)
    registry.blocked = True
    with pytest.raises(UnavailableError, match="blocked"):
        runtime.load("fixture@1.0.0:a", "cpu", registry=registry)
    assert registry.reads == 2
    assert (
        len(
            handle._score_prepared(
                [
                    {
                        "HLA_a": "",
                        "HLA_b": "",
                        "Peptide": "ACD",
                        "TCR_a": "",
                        "TCR_b": "",
                    }
                ],
            )
        )
        == 1
    )
