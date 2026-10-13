"""Feature parity checks against the committed public algorithms.

The behavioral source is DecoderTCR public release e842d5b4.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from DecoderTCR.constants import AA20, AA20_IDS, MASK_IDX
from DecoderTCR.core import ValidationError
from DecoderTCR.design import design_peptides
from DecoderTCR.features import (
    _embed,
    _region_profile,
    embed,
    peptide_profile,
    profile_from_logits,
    region_positions,
    region_profile,
)
from DecoderTCR.runtime import LoadedModel, build_model
from DecoderTCR.tokenizer import TCRpMHCTokenizer


def _loaded(module, backbone="esmc", layers=2):
    return LoadedModel(
        module,
        "fixture@1.0.0:a",
        backbone,
        "fixture",
        "v1",
        "b" * 64,
        {"sha256": "c" * 64},
        "d" * 64,
        torch.device("cpu"),
        layers,
    )


class PositionModel(torch.nn.Module):
    def __init__(self, vocab=64, width=4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab = vocab
        self.width = width
        self.seen = []

    def forward(self, tokens, repr_layers=None, return_contacts=False):
        self.seen.append(tokens.detach().clone())
        positions = torch.arange(tokens.shape[1], dtype=torch.float32)
        logits = torch.arange(self.vocab, dtype=torch.float32).view(1, 1, -1)
        logits = logits + positions.view(1, -1, 1)
        representations = positions.view(1, -1, 1).expand(1, -1, self.width)
        key = repr_layers[0] if repr_layers else -1
        return {"logits": logits, "representations": {key: representations, -1: representations}}


@pytest.fixture
def flat_complex():
    return {
        "HLA_a": "AC",
        "HLA_b": "D",
        "Peptide": "EFG",
        "TCR_a": "HIKLM",
        "TCR_b": "NPQRS",
    }


@pytest.mark.parametrize("backbone", ["esm2", "esmc"])
def test_embed_matches_direct_final_representation_for_tiny_backbones(backbone, flat_complex):
    torch.manual_seed(31)
    config = (
        {"embed_dim": 32, "attention_heads": 4, "num_layers": 2}
        if backbone == "esm2"
        else {"d_model": 32, "n_heads": 4, "n_layers": 2}
    )
    module = build_model(backbone, "fixture", architecture_config=config)
    loaded = _loaded(module, backbone)
    entry = {"sequences": flat_complex, "pocket_idx": {}}
    tok = TCRpMHCTokenizer(entry, mask_region=None)
    with torch.inference_mode():
        direct = module(tok.original_ids.unsqueeze(0), repr_layers=[2], return_contacts=False)
    key = 2 if backbone == "esm2" else -1
    expected = direct["representations"][key][0, 1:-1].float().numpy()
    actual = _embed(loaded, flat_complex, pool=None, prepared=True)[0]
    torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))


@pytest.mark.parametrize("backbone", ["esm2", "esmc"])
def test_profile_matches_direct_aa20_logits_for_tiny_backbones(backbone, flat_complex):
    torch.manual_seed(37)
    config = (
        {"embed_dim": 32, "attention_heads": 4, "num_layers": 2}
        if backbone == "esm2"
        else {"d_model": 32, "n_heads": 4, "n_layers": 2}
    )
    module = build_model(backbone, "fixture", architecture_config=config)
    loaded = _loaded(module, backbone)
    expected_entry = {
        "sequences": {**flat_complex, "Peptide": "AAA"},
        "pocket_idx": {},
    }
    tok = TCRpMHCTokenizer(expected_entry, mask_region="peptide")
    with torch.inference_mode():
        logits = module(tok.input_ids.unsqueeze(0))["logits"][0, tok.token_indices]
    expected = torch.softmax(logits[:, AA20_IDS].double(), dim=-1).numpy()

    actual = _region_profile(loaded, flat_complex, length=3, prepared=True)[list(AA20)].to_numpy()

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_embed_region_order_and_cls_eos_exclusion(flat_complex):
    loaded = _loaded(PositionModel())
    regions = _embed(loaded, flat_complex, pool="regions", prepared=True)[0]
    assert list(regions) == ["HLA_a", "HLA_b", "Peptide", "TCR_a", "TCR_b"]
    assert regions["HLA_a"].tolist() == [1.5] * 4
    assert regions["Peptide"].tolist() == [5.0] * 4
    residues = _embed(loaded, flat_complex, pool=None, prepared=True)[0]
    assert residues[:, 0].tolist() == list(range(1, 17))


def test_profile_is_aa20_float64_and_masks_full_peptide(flat_complex):
    model = PositionModel()
    loaded = _loaded(model)
    profile = _region_profile(loaded, flat_complex, length=3, prepared=True)
    assert profile[list(AA20)].to_numpy().dtype == np.float64
    np.testing.assert_allclose(profile[list(AA20)].sum(axis=1), 1.0)
    masked = model.seen[-1][0]
    assert (masked == MASK_IDX).nonzero().flatten().tolist() == [4, 5, 6]
    assert profile.attrs["reproduction"]["model_id"] == loaded.model_id
    assert profile.attrs["reproduction"]["artifact_sha256"] == loaded.artifact_sha256


def test_documented_single_gene_context_profiles_and_designs(monkeypatch, flat_complex):
    from DecoderTCR.reconstruct import components

    context = {
        "trav": "TRAV1-1",
        "traj": "TRAJ1",
        "cdr3a": "CAVF",
        "trbv": "TRBV1",
        "trbj": "TRBJ1-1",
        "cdr3b": "CASSF",
        "hla": "A0201",
        "peptide": "EFG",
    }

    def prepare(row, convention, *, species):
        assert isinstance(row, dict)
        assert convention == "v1"
        assert species == "human"
        entry = {
            "sequences": {**flat_complex, "Peptide": row["peptide"]},
            "pocket_idx": {"HLA_a": [], "HLA_b": [], "CDRa": [], "CDRb": []},
            "annotation_contract": "inclusive-end",
        }
        return pd.DataFrame([{"reconstruction_ok": True}]), [entry]

    monkeypatch.setattr(components, "prepare_gene_inputs", prepare)
    loaded = _loaded(PositionModel())
    profile = peptide_profile(loaded, context, 3)
    candidates = design_peptides(loaded, context, 3, n=1, rescore=False, seed=42)

    assert profile.shape == (3, 21)
    assert len(candidates) == 1
    assert len(candidates.iloc[0]["sequence"]) == 3


def test_public_cdr3_profile_accepts_all_documented_single_context_forms(monkeypatch, flat_complex):
    from DecoderTCR.reconstruct import components

    context = {
        "trav": "TRAV1-1",
        "traj": "TRAJ1",
        "cdr3a": "CAVF",
        "trbv": "TRBV1",
        "trbj": "TRBJ1-1",
        "cdr3b": "CASSF",
        "hla": "A0201",
        "peptide": "EFG",
    }
    entry = {
        "sequences": flat_complex,
        "pocket_idx": {"CDRa": [[2, 4]], "CDRb": []},
        "annotation_contract": "inclusive-end",
        "reconstruction": {"sequence_convention": "v1"},
    }
    seen = []

    def prepare(row, convention, *, species):
        assert isinstance(row, dict)
        assert convention == "v1"
        assert species == "human"
        seen.append(row)
        return pd.DataFrame([{"reconstruction_ok": True}]), [entry]

    monkeypatch.setattr(components, "prepare_gene_inputs", prepare)
    forms = [context, pd.Series(context), pd.DataFrame([context])]
    profiles = [region_profile(_loaded(PositionModel()), form, "cdr3a") for form in forms]

    for profile in profiles[1:]:
        pd.testing.assert_frame_equal(profile, profiles[0])
        assert profile.attrs == profiles[0].attrs
    assert profiles[0].shape == (3, 21)
    assert [row for row in seen] == [context, context, context]


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame([["TRAV1-1", "TRAV2"]], columns=["trav", "trav"]),
        pd.DataFrame([["TRAV1-1", "TRAV2"]], columns=["trav", "TRAV"]),
        pd.Series(["TRAV1-1", "TRAV2"], index=["trav", "trav"]),
        pd.Series(["TRAV1-1", "TRAV2"], index=["trav", "TRAV"]),
    ],
)
def test_public_gene_context_rejects_duplicate_labels_before_reconstruction(monkeypatch, data):
    def unexpected_reconstruction(*args, **kwargs):
        raise AssertionError("reconstruction must not receive collapsed duplicate labels")

    monkeypatch.setattr(
        "DecoderTCR.reconstruct.components.prepare_gene_inputs", unexpected_reconstruction
    )

    with pytest.raises(ValidationError, match="Duplicate case-insensitive input header"):
        region_profile(_loaded(PositionModel()), data, "cdr3a")


def test_profile_from_logits_uses_only_aa20_channels():
    logits = torch.zeros(2, 64)
    logits[:, 63] = 100
    profile = profile_from_logits(logits)
    np.testing.assert_allclose(profile[list(AA20)], 1 / 20)
    assert profile[list(AA20)].to_numpy().dtype == np.float64


def test_operation_specific_annotation_endpoints(flat_complex):
    entry = {
        "sequences": flat_complex,
        "pocket_idx": {
            "CDRa": [[0, 1], [1, 2], [2, 4]],
            "CDRb": [[0, 1]],
        },
        "annotation_contract": "inclusive-end",
    }
    tok = TCRpMHCTokenizer(entry, mask_region=None)
    assert region_positions(tok, "cdr3a", entry).tolist() == [8, 9, 10]
    assert region_positions(tok, "cdr3b", entry).tolist() == [11, 12]
    assert region_positions(tok, "cdr3", entry).tolist() == [8, 9, 10, 11, 12]

    half_open = {**entry, "annotation_contract": "half-open"}
    assert region_positions(tok, "cdr", half_open).tolist() == [6, 7, 8, 9, 11]


@pytest.mark.parametrize("region", ["cdr", "cdr3", "cdr3a", "cdr3b"])
def test_annotated_regions_require_explicit_matching_contract(flat_complex, region):
    entry = {
        "sequences": flat_complex,
        "pocket_idx": {"CDRa": [[0, 1], [1, 2], [2, 4]], "CDRb": [[0, 1]]},
    }
    tok = TCRpMHCTokenizer(entry, mask_region=None)
    with pytest.raises(ValidationError, match="annotation_contract"):
        region_positions(tok, region, entry)
    wrong = {**entry, "annotation_contract": "inclusive-end" if region == "cdr" else "half-open"}
    with pytest.raises(ValidationError, match="annotation_contract"):
        region_positions(tok, region, wrong)


def test_private_region_profile_accepts_prepared_single_complex_forms(flat_complex):
    loaded = _loaded(PositionModel())
    series = pd.Series({key.lower(): value for key, value in flat_complex.items()})
    structured = {"sequences": flat_complex, "pocket_idx": {}}
    for data in (flat_complex, series, pd.DataFrame([flat_complex]), structured):
        assert _region_profile(loaded, data, length=2, prepared=True).shape == (2, 21)


def test_structured_feature_input_checks_metadata_convention(flat_complex):
    loaded = _loaded(PositionModel())
    structured = {
        "sequences": flat_complex,
        "pocket_idx": {},
        "meta_data": {"sequence_convention": "v2"},
    }
    with pytest.raises(ValidationError, match="convention"):
        _region_profile(loaded, structured, length=2, prepared=True)

    structured["sequence_convention"] = "v1"
    with pytest.raises(ValidationError, match="Conflicting"):
        _region_profile(loaded, structured, length=2, prepared=True)


def test_embed_accepts_omitted_conditioning_chains_and_public_aliases():
    loaded = _loaded(PositionModel())
    values = _embed(
        loaded,
        {"mhc": "ACD", "epitope": "EFG"},
        prepared=True,
    )
    assert values.shape == (1, 4)


def test_case_insensitive_duplicate_fields_fail_loudly(flat_complex):
    loaded = _loaded(PositionModel())
    duplicated = {**flat_complex, "hla_a": "AC"}
    with pytest.raises(ValidationError, match="unique ignoring capitalization"):
        _region_profile(loaded, duplicated, length=3, prepared=True)


def test_profile_rejects_unknown_regions_and_bad_scalar_inputs(flat_complex):
    loaded = _loaded(PositionModel())
    with pytest.raises(ValidationError, match="Unknown region"):
        region_profile(loaded, flat_complex, "tail")
    for length in (None, 0, 1.5, True):
        with pytest.raises(ValidationError, match="positive integer"):
            _region_profile(loaded, flat_complex, length=length, prepared=True)


def test_esmc_embedding_uses_final_representation_key(flat_complex):
    class FinalOnly(PositionModel):
        def forward(self, tokens, repr_layers=None, return_contacts=False):
            output = super().forward(tokens, repr_layers, return_contacts)
            output["representations"] = {-1: output["representations"][-1]}
            return output

    values = _embed(_loaded(FinalOnly()), flat_complex, prepared=True)
    assert values.shape == (1, 4)


@pytest.mark.parametrize("sequence_convention", ["v1", "v2"])
def test_feature_defaults_match_explicit_genes_and_use_loaded_convention(
    monkeypatch, flat_complex, sequence_convention
):
    from DecoderTCR.reconstruct import components

    loaded = _loaded(PositionModel())
    loaded.sequence_convention = sequence_convention
    entry = {
        "sequences": flat_complex,
        "pocket_idx": {"CDRa": [], "CDRb": []},
        "annotation_contract": "inclusive-end",
        "reconstruction": {"sequence_convention": sequence_convention},
    }
    prepare = Mock(return_value=(pd.DataFrame([{"reconstruction_ok": True}]), [entry]))
    monkeypatch.setattr(components, "prepare_gene_inputs", prepare)
    genes = {"trav": "TRAV1-1", "peptide": "EFG"}

    default_embedding = embed(loaded, genes)
    repeated_embedding = embed(loaded, genes)
    default_profile = peptide_profile(loaded, genes, 3)
    repeated_profile = peptide_profile(loaded, genes, 3)

    np.testing.assert_array_equal(default_embedding, repeated_embedding)
    pd.testing.assert_frame_equal(default_profile, repeated_profile)
    assert [item.args[1] for item in prepare.call_args_list] == [
        sequence_convention,
        sequence_convention,
        sequence_convention,
        sequence_convention,
    ]
