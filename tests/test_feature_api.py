"""Check exact-model routing for the integrated research APIs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

import DecoderTCR as dt
from DecoderTCR.core import ValidationError


def test_score_from_components_selects_gene_input_once(monkeypatch):
    loaded = Mock()
    loaded.score.return_value = pd.DataFrame({"pll": [-1.0]})
    loader = Mock(return_value=loaded)
    monkeypatch.setattr(dt, "load", loader)

    rows = [{"trav": "TRAV1-1", "peptide": "ACD"}]
    result = dt.score_from_components(rows, model="family@1.0.0:300M", device="cpu")

    loader.assert_called_once_with("family@1.0.0:300M", device="cpu", registry=None)
    loaded.score.assert_called_once_with(
        rows,
        species="human",
        mask_region="peptide",
        cache_size=256,
    )
    assert result.pll.tolist() == [-1.0]


def test_public_score_routes_gene_inputs_and_rejects_prepared_override():
    from DecoderTCR.runtime import LoadedModel

    loaded = object.__new__(LoadedModel)
    loaded.score = Mock(return_value=pd.DataFrame({"pll": [-1.0]}))
    rows = [{"trav": "TRAV1-1"}]

    dt.score(rows, model=loaded)
    with pytest.raises(TypeError, match="input_format"):
        dt.score(rows, model=loaded, input_format="prepared")

    assert loaded.score.call_args_list[0].args == (rows,)
    assert loaded.score.call_args_list[0].kwargs == {
        "species": "human",
        "mask_region": "peptide",
        "cache_size": 256,
    }


def test_loaded_model_is_retained_across_nested_feature_call(monkeypatch):
    from DecoderTCR.runtime import LoadedModel

    loaded = object.__new__(LoadedModel)
    feature = Mock(return_value=pd.DataFrame({"sequence": ["ACD"]}))
    monkeypatch.setattr("DecoderTCR.design.design_peptides", feature)

    result = dt.design_peptides(
        {"HLA_a": "AC"},
        3,
        model=loaded,
        n=1,
        rescore=True,
    )

    assert feature.call_args.args == (loaded, {"HLA_a": "AC"}, 3)
    assert feature.call_args.kwargs["species"] == "human"
    assert feature.call_args.kwargs["n"] == 1
    assert feature.call_args.kwargs["rescore"] is True
    assert result.sequence.tolist() == ["ACD"]


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (dt.score, ([],)),
        (dt.score_from_components, ([],)),
        (dt.embed, ([],)),
        (dt.region_profile, ({},)),
        (dt.peptide_profile, ({}, 2)),
        (dt.iegr_profile, ({},)),
        (dt.block_gibbs, ({}, ["AC"])),
        (dt.iegr, ({},)),
        (dt.design_peptides, ({}, 2)),
    ],
)
def test_public_model_operations_reject_prepared_override_before_resolution(
    monkeypatch, operation, args
):
    loader = Mock(side_effect=AssertionError("model resolution must not run"))
    monkeypatch.setattr(dt, "load", loader)
    with pytest.raises(TypeError, match="input_format"):
        operation(*args, model="family@1.0.0:300M", input_format="prepared")
    loader.assert_not_called()


def test_public_feature_wrappers_route_gene_inputs(monkeypatch):
    from DecoderTCR.runtime import LoadedModel

    loaded = object.__new__(LoadedModel)
    rows = {"HLA_a": "AC", "Peptide": "EF"}
    embedded = Mock(return_value="embedding")
    profile = Mock(return_value="profile")
    design = Mock(return_value="design")
    monkeypatch.setattr("DecoderTCR.features.embed", embedded)
    monkeypatch.setattr("DecoderTCR.features.peptide_profile", profile)
    monkeypatch.setattr("DecoderTCR.design.design_peptides", design)

    assert dt.embed(rows, model=loaded) == "embedding"
    assert dt.peptide_profile(rows, 2, model=loaded) == "profile"
    assert dt.design_peptides(rows, 2, model=loaded) == "design"

    embedded.assert_called_once_with(loaded, rows, species="human", pool="mean", layer=None)
    profile.assert_called_once_with(loaded, rows, 2, species="human")
    design.assert_called_once_with(
        loaded,
        rows,
        2,
        species="human",
        n=10,
        method="one_shot",
        temperature=None,
        seed=42,
        rescore=True,
        gibbs_rounds=10,
        gibbs_subset_size=5,
        profile_method="one_shot",
        gibbs_k=0,
        gibbs_temperature=None,
        gibbs_max_forwards=None,
        cap=1_000_000,
    )


@pytest.mark.parametrize("value", [None, object(), 7])
def test_new_model_dependent_apis_reject_nonidentities(value):
    with pytest.raises(ValidationError, match="exact model ID or LoadedModel"):
        dt.embed([], model=value)


def test_reconstruction_reads_convention_without_loading_weights(monkeypatch):
    from DecoderTCR.registry import Registry

    store = object.__new__(Registry)
    resolve = Mock(return_value=SimpleNamespace(member={"sequence_convention": "v2"}))
    monkeypatch.setattr(store, "resolve", resolve)
    reconstruct = Mock(return_value=pd.DataFrame({"ok": [True]}))
    monkeypatch.setattr("DecoderTCR.reconstruct.components.reconstruct_components", reconstruct)
    rows = [{"trav": "TRAV1-1"}]

    result = dt.reconstruct_components(rows, model="family@1.0.0:300M", registry=store)

    resolve.assert_called_once_with("family@1.0.0:300M")
    reconstruct.assert_called_once_with(rows, "v2", species="human")
    assert result.ok.tolist() == [True]


@pytest.mark.parametrize(
    ("model_id", "preparation"),
    [
        ("decodertcr@1.5.0:300M", "v2"),
        ("decodertcr@1.5.0:600M", "v2"),
        ("decodertcr@1.5.0:6B", "v2"),
        ("decodertcr@0.3.0:300M", "v1"),
        ("decodertcr@0.3.0:600M", "v1"),
        ("decodertcr@0.3.0:6B", "v1"),
    ],
)
def test_all_exact_model_ids_route_their_registered_preparation(monkeypatch, model_id, preparation):
    from DecoderTCR.registry import Registry

    store = object.__new__(Registry)
    resolve = Mock(return_value=SimpleNamespace(member={"sequence_convention": preparation}))
    monkeypatch.setattr(store, "resolve", resolve)
    reconstruct = Mock(return_value=pd.DataFrame({"ok": [True]}))
    monkeypatch.setattr("DecoderTCR.reconstruct.components.reconstruct_components", reconstruct)

    dt.reconstruct_components([{"trav": "TRAV1-1"}], model=model_id, registry=store)

    resolve.assert_called_once_with(model_id)
    assert reconstruct.call_args.args[1] == preparation


def test_loaded_reconstruction_uses_handle_metadata(monkeypatch):
    from DecoderTCR.runtime import LoadedModel

    loaded = object.__new__(LoadedModel)
    loaded.sequence_convention = "v1"
    lookup = Mock(return_value=("HLA", "B2M"))
    monkeypatch.setattr("DecoderTCR.reconstruct.hla.lookup_hla", lookup)

    assert dt.lookup_hla("HLA-B*27:05", model=loaded) == ("HLA", "B2M")
    lookup.assert_called_once_with("HLA-B*27:05", "v1", species="human")


def test_model_free_helpers_do_not_resolve_a_model(monkeypatch):
    helper = Mock(return_value="ACD")
    monkeypatch.setattr("DecoderTCR.features.consensus", helper)
    profile = SimpleNamespace()
    assert dt.consensus(profile) == "ACD"
    helper.assert_called_once_with(profile)


def test_raw_sequence_entrypoints_are_absent_from_public_surfaces():
    for name in ("score_one", "embed_chains", "tokenize_sequences"):
        assert name not in dt.__all__
        assert not hasattr(dt, name)
