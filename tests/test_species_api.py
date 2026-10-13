"""Check species routing across public inference surfaces."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

import DecoderTCR as dt
from DecoderTCR import cli
from DecoderTCR.constants import AA20
from DecoderTCR.design import design_peptides
from DecoderTCR.runtime import LoadedModel


def _loaded() -> LoadedModel:
    return LoadedModel(
        torch.nn.Linear(1, 1),
        "fixture@1.0.0:a",
        "esmc",
        "fixture",
        "v2",
        "a" * 64,
        {"sha256": "b" * 64},
        "c" * 64,
        torch.device("cpu"),
        1,
    )


def test_loaded_score_routes_species_only_to_gene_reconstruction(monkeypatch):
    from DecoderTCR import runtime
    from DecoderTCR.reconstruct import components

    model = _loaded()
    entry = {
        "sequences": {
            "HLA_a": "A",
            "HLA_b": "",
            "Peptide": "C",
            "TCR_a": "",
            "TCR_b": "",
        },
        "pocket_idx": {},
    }
    gene_frame = pd.DataFrame(
        [{"name": "same"}, {"name": "same"}],
        index=[7, 7],
    )
    prepare_genes = Mock(return_value=(gene_frame, [entry, None]))
    prepare_sequences = Mock(return_value=(pd.DataFrame([{"name": "prepared"}]), [entry]))
    monkeypatch.setattr(components, "prepare_gene_inputs", prepare_genes)
    monkeypatch.setattr(runtime, "prepare_inputs", prepare_sequences)
    monkeypatch.setattr(
        runtime,
        "run_pll_benchmark",
        Mock(side_effect=[np.asarray([-2.0]), np.asarray([-3.0])]),
    )

    genes = model.score([{"trav": "TRAV1"}], species="mouse")
    prepared = model._score_prepared([{"Peptide": "C"}])

    prepare_genes.assert_called_once_with(
        [{"trav": "TRAV1"}],
        "v2",
        species="mouse",
    )
    prepare_sequences.assert_called_once_with([{"Peptide": "C"}], "v2")
    assert genes.index.tolist() == [7, 7]
    assert genes.name.tolist() == ["same", "same"]
    assert prepared.name.tolist() == ["prepared"]


def test_private_prepared_score_has_no_species_switch():
    with pytest.raises(TypeError, match="species"):
        _loaded()._score_prepared([], species="dog")


def test_public_wrappers_retain_handle_and_propagate_species(monkeypatch):
    loaded = object.__new__(LoadedModel)
    loaded.score = Mock(return_value=pd.DataFrame({"pll": [-1.0]}))
    embedded = Mock(return_value="embedded")
    region = Mock(return_value="region")
    peptide = Mock(return_value="peptide")
    profile = Mock(return_value="iegr-profile")
    gibbs = Mock(return_value=([], {}))
    iegr = Mock(return_value=pd.DataFrame({"sequence": ["ACD"]}))
    designed = Mock(return_value=pd.DataFrame({"sequence": ["ACD"]}))
    monkeypatch.setattr("DecoderTCR.features.embed", embedded)
    monkeypatch.setattr("DecoderTCR.features.region_profile", region)
    monkeypatch.setattr("DecoderTCR.features.peptide_profile", peptide)
    monkeypatch.setattr("DecoderTCR.design.iegr_profile", profile)
    monkeypatch.setattr("DecoderTCR.design.block_gibbs", gibbs)
    monkeypatch.setattr("DecoderTCR.design.iegr", iegr)
    monkeypatch.setattr("DecoderTCR.design.design_peptides", designed)
    rows = [{"trav": "TRAV1"}]

    dt.score(rows, model=loaded, species="mouse")
    dt.embed(rows, model=loaded, species="mouse")
    dt.region_profile(rows, "peptide", model=loaded, species="mouse")
    dt.peptide_profile(rows, 3, model=loaded, species="mouse")
    dt.iegr_profile(rows, model=loaded, species="mouse")
    dt.block_gibbs(rows, ["ACD"], model=loaded, species="mouse")
    dt.iegr(rows, model=loaded, species="mouse")
    result = dt.design_peptides(rows, 3, model=loaded, species="mouse")

    assert loaded.score.call_args.kwargs["species"] == "mouse"
    assert embedded.call_args.kwargs["species"] == "mouse"
    region.assert_called_once_with(loaded, rows, region="peptide", length=None, species="mouse")
    peptide.assert_called_once_with(loaded, rows, 3, species="mouse")
    assert profile.call_args.kwargs["species"] == "mouse"
    assert gibbs.call_args.kwargs["species"] == "mouse"
    assert iegr.call_args.kwargs["species"] == "mouse"
    assert designed.call_args.kwargs["species"] == "mouse"
    assert result.sequence.tolist() == ["ACD"]


def test_composite_design_reconstructs_mouse_genes_once(monkeypatch):
    entry = {
        "sequences": {
            "HLA_a": "AAA",
            "HLA_b": "",
            "Peptide": "AAA",
            "TCR_a": "",
            "TCR_b": "",
        },
        "pocket_idx": {"CDRa": [], "CDRb": []},
        "annotation_contract": "inclusive-end",
        "reconstruction": {"sequence_convention": "v2", "species": "mouse"},
    }
    prepare = Mock(return_value=(pd.DataFrame([{"reconstruction_ok": True}]), [entry]))
    profile = pd.DataFrame([[1.0 / len(AA20)] * len(AA20)], columns=list(AA20))
    profile.attrs["reconstruction"] = entry["reconstruction"]
    profile_call = Mock(return_value=profile)
    monkeypatch.setattr("DecoderTCR.reconstruct.components.prepare_gene_inputs", prepare)
    monkeypatch.setattr("DecoderTCR.design._region_profile", profile_call)
    monkeypatch.setattr(
        "DecoderTCR.design.sample_from_profile",
        Mock(
            return_value=(
                ["A"],
                {
                    "n_requested": 1,
                    "n_returned": 1,
                    "n_draws_used": 1,
                    "saturated": False,
                    "cap_reached": False,
                    "support_exhausted": False,
                },
            )
        ),
    )
    loaded = SimpleNamespace(
        sequence_convention="v2",
        info={"model_id": "fixture@1.0.0:a"},
    )

    output = design_peptides(
        loaded,
        {"trav": "TRAV1"},
        1,
        n=1,
        rescore=False,
        species="mouse",
    )

    prepare.assert_called_once_with(
        {"trav": "TRAV1", "peptide": "A"},
        "v2",
        species="mouse",
    )
    profile_call.assert_called_once_with(
        loaded,
        entry,
        region="peptide",
        length=1,
        species="mouse",
        prepared=True,
    )
    assert output.attrs["reconstruction"]["species"] == "mouse"


@pytest.mark.parametrize("command", ["score", "profile", "design"])
def test_cli_species_default_choice_and_override(command):
    common = [
        command,
        "--model",
        "fixture@1.0.0:a",
        "--input",
        "input.csv",
        "--output",
        "output.csv",
    ]
    if command == "design":
        common.extend(["--length", "9"])

    assert cli._parser().parse_args(common).species == "human"
    assert cli._parser().parse_args([*common, "--species", "mouse"]).species == "mouse"
    with pytest.raises(SystemExit):
        cli._parser().parse_args([*common, "--species", "dog"])


@pytest.mark.parametrize("command", ["profile", "design"])
def test_cli_dispatches_mouse_species(command, tmp_path, monkeypatch):
    source = tmp_path / f"{command}.csv"
    output = tmp_path / f"{command}-output.csv"
    pd.DataFrame([{"value": "fixture"}]).to_csv(source, index=False)
    loaded = SimpleNamespace(
        model_id="fixture@1.0.0:a",
        artifact_sha256="a" * 64,
    )
    monkeypatch.setattr("DecoderTCR.runtime.load", lambda *args, **kwargs: loaded)
    result = pd.DataFrame({"sequence": ["ACD"]})
    selected = Mock(return_value=result)
    if command == "profile":
        monkeypatch.setattr("DecoderTCR.features.region_profile", selected)
    else:
        monkeypatch.setattr("DecoderTCR.design.design_peptides", selected)
    arguments = [
        command,
        "--model",
        "fixture@1.0.0:a",
        "--input",
        str(source),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--species",
        "mouse",
    ]
    if command == "design":
        arguments.extend(["--length", "3"])

    assert cli.main(arguments) == 0
    assert selected.call_args.kwargs["species"] == "mouse"
    assert output.is_file()
